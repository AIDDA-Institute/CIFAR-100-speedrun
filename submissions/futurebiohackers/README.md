# futurebiohackers: faster CIFAR-100 training

A convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
Three convolution groups of 64, 256 and 768 channels, three convolutions each with a
residual connection over the last two; the last group's first convolution (256 -> 768 on the 7x7
map, the network's largest) uses a 2x2 kernel; in the last group that residual pair runs through 512
channels (3x3 768 -> 512 -> 768), in the second group through 192 channels as a 3x3
convolution (256 -> 192) followed by a 1x1 (192 -> 256), and in the first group as two 1x1
convolutions at 64 channels, while each group's input, residual and output keep its width. The classifier reads the
per-channel maximum plus the per-channel mean of the last group's map. The first group is deliberately narrow: it
runs at 31x31 and 15x15, where each channel costs 4-16x more than in the later
groups and cuDNN reaches only 23-34% of A100 peak; the capacity sits in the last
group, which runs at 3x3 at 78-89% of peak.

Training: 11.5 epochs at batch 1024, Nesterov SGD (lr 11.5, weight decay 0.017 per 1024
examples, momentum 0.85, BatchNorm-bias lr 32x), label smoothing 0.25, logit scale
1.25/9, BatchNorm momentum 0.5, lookahead weight average. The first 15% of the
examples train on 20x20 bilinear downsamples of the images (0.39x the pixels), the next 35% on
24x24 (0.56x), the second half at 32x32. Augmentation: alternating flip, 2-pixel translation, per-image brightness
and contrast jitter of 0.2. Normalization and patch whitening of the training images
run inside the timer.

Implementation details that change the time but not the model:

- The SGD step is one fused CUDA kernel (`fused_sgd`).
- BatchNorm runs in fp16 like the rest of the network (`bn_dtype: "half"`).
- The forward pass and loss compile as one graph (`compile_step`), with static
  shapes so no recompilation happens inside a trial; every (batch, resolution)
  pair gets its own graph, warmed in `build`.
- Random crops use one gather per epoch (`crop_mode: "indexed"`) instead of 25
  masked copies, which also avoids host syncs.
- The global pooling (max + mean, described above) is written with `F.max_pool2d` and
  `F.avg_pool2d` over the whole map (`global_pool: "fullpool_avgsum"`), which `torch.compile`
  lowers to pointwise kernels; the reduction kernels it generates for `max(dim).values` and
  `mean` (PR #24) cost 0.13 s more per trial at identical results. `adaptive_max_pool2d`
  backward uses an atomic kernel (about 0.3 s per trial in the profile), and `amax` trains to
  NaN under `torch.compile` in torch 2.4.
- `prepare` takes 35-85 ms instead of 170 ms: the dirac part of the conv init is
  vectorized (`nn.init.dirac_` launches one kernel per channel), and each
  downsampling resize is two matmuls instead of `F.interpolate` (65 ms on fp16 channels-last).

## Development results

Confirmation runs for the submitted defaults (20 px for the first 15% of the examples, 24 px to the
half and 32 px after; group 3's first convolution as a 2x2 kernel; group 1's residual pair as two 1x1
convolutions, group 2's as a 3x3 then a 1x1 through 192 channels; the 512-channel pair in the last
group; max + mean pooling through the pooling kernels; 11.5 epochs), 4 October: 40 trials per seed set,
the 75% target enforced, Modal NVIDIA A100-SXM4-80GB at the 400 W power limit ("A100 SXM 400 W").
Cards at the same power limit differ by one to two percent in speed, so each seed set ran this recipe
and the previous one (PR #33) back to back in one container, both compiled cold (fresh Inductor cache)
under a hard four-CPU limit, the judges' quota; the build times of the first four rows are those real
cold builds. The PR #33, PR #29, PR #27 and PR #28 rows are also 4-CPU cold containers; the older
rows ran in 20-CPU containers. Official judging runs on an A100 80GB PCIe, where times are about
10% higher: this recipe 75.23% in 4.07 s vs PR #29 75.30% in 4.35 s (paired n=40 on seeds 0-39, both
cold at 4 CPUs in one PCIe container; PR #33 in another: 75.26% in 4.16 s vs PR #29 75.24% in 4.26 s);
the ranking between recipes carries over.

| Seeds | Recipe | Mean accuracy | Min / max trial | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | ---: | --- |
| 0-39 | **submitted defaults**, cold build 317 s at 4 CPUs | **75.24% (sd 0.27)** | 74.58% / 75.73% | **3.61 s** (sd 0.01) | yes |
| 0-39 | previous recipe (PR #33), same container, cold build 311 s | 75.24% (sd 0.28) | 74.31% / 75.79% | 3.77 s | yes |
| 40-79 | **submitted defaults**, cold build 368 s at 4 CPUs | **75.26% (sd 0.22)** | 74.87% / 75.79% | **3.63 s** (sd 0.01) | yes |
| 40-79 | previous recipe (PR #33), same container, cold build 381 s | 75.24% (sd 0.20) | 74.74% / 75.64% | 3.78 s | yes |
| 0-39 / 40-79 | PR #33 vs PR #29, both compiled cold at 4 CPUs (builds 365 / 419 s) | 75.30% / 75.20% vs 75.31% / 75.22% | | 3.78 / 3.79 s vs 3.88 / 3.88 s | yes |
| 0-39 / 40-79 | PR #29 vs PR #27, both compiled cold at 4 CPUs (builds 364 / 355 s) | 75.22% / 75.22% vs 75.11% / 75.07% | | 3.90 / 3.87 s vs 4.04 / 4.06 s | yes |
| 0-39 / 40-79 | PR #27 vs PR #26, both compiled cold at 4 CPUs (builds 319 / 391 s) | 75.16% / 75.13% vs 75.20% / 75.18% | | 4.07 / 4.08 s vs 4.26 / 4.30 s | yes |
| 0-39 / 40-79 | PR #28 (full-width group 2 pair, 24 px to 35%) vs PR #25, cold at 4 CPUs | 75.25% / 75.26% vs 75.24% / 75.27% | | 4.32 / 4.14 s vs 4.46 / 4.28 s | yes |
| 0-39 / 40-79 | PR #26 paired vs PR #23 in one container | 75.16% / 75.19% vs 75.20% / 75.22% | | 4.23 / 4.15 s vs 4.50 / 4.42 s | yes |
| 0-39 / 40-79 | PR #23 standalone, cold build 381 / 278 s | 75.19% / 75.25% | | 4.56 / 4.49 s | yes |
| 0-39 | 9.0 epochs, 28 px first half (PR #14 defaults), cold build 193 s | 75.12% (sd 0.26) | | 4.72 s (sd 0.01) | yes |

Paired differences against PR #33 on the same seeds and card: seeds 0-39 -0.00 +- 0.06 points,
-0.15 s; seeds 40-79 +0.03 +- 0.05 points, -0.15 s (+0.01 +- 0.04 points over the 80
pairs). All 160 trials of the two containers (this recipe and PR #33) finished and every run qualified
on its mean; this recipe's 80 trials never fell below 74.58%, so none diverged (this file carries
no device-side non-finite counter).

Why this recipe: two changes on PR #33. (1) The first convolution of group 3 (256 -> 768 on the 7x7
map, about 38% of the network's FLOPs) uses a 2x2 kernel instead of a 3x3 (`conv1_kernels` [3, 3, 2]):
4/9 of the FLOPs of that layer, with PyTorch's asymmetric 'same' padding so the map stays 7x7 (the
padding costs one zero-padded copy of the 7x7x256 input per step). Alone, at 11 epochs, it measured
-0.25 +- 0.06 / -0.12 +- 0.05 points for -0.31 / -0.31 s against PR #33 over 40 paired trials per seed
set (75.06% / 75.09%, below the team's 75.15% target), after a 16-trial screen on PR #29's recipe had
read -0.06 +- 0.07 points for -0.28 s. (2) Half an epoch more (11.5) buys the accuracy back along the
ladder: confirmed above at 40 trials per seed set (-0.00 / +0.03 points, -0.15 / -0.15 s against
PR #33). The same 2x2 kernel in group 2's first convolution as well measured -0.19 +- 0.10 points for
-0.37 s (16 trials on PR #29), too costly in accuracy.

The rest is PR #33, one change on PR #29, group 1's residual pair as two 1x1 convolutions instead of a
3x3 followed by a 1x1 (`inner_kernels` [[1, 1], [3, 1], 3]): the pair runs on the 15x15 map, where
the 3x3 at 64 channels is memory-bound, and the 3x3 -> 1x1 change of PR #29 had already cost no
accuracy. Screened cold at 16 paired trials: +0.08 +- 0.08 points for -0.08 s; confirmed above at 40
trials per seed set (-0.01 / -0.02 points, -0.10 / -0.09 s). The alternatives screened in the same
round lose accuracy for their time: no group 1 pair at all (depths [2, 3, 3]) -0.15 +- 0.11 points
for -0.15 s, the 24 px phase to 60% -0.15 +- 0.10 for -0.24 s (at 11.25 epochs -0.15 +- 0.09 for
-0.14 s), the 20 px phase to 25% -0.15 +- 0.08 for -0.02 s, 10.75 epochs -0.07 +- 0.09 for -0.09 s.

The rest is PR #29, two changes on PR #27 and one more epoch, every comparison made with both recipes
compiled cold in one container (which removes the one-to-two-percent speed differences between cards
and any kernel choice inherited from a warm compile cache). (1) The first resolution phase trains at
20 px instead of 24 px and ends at 15% of the examples, and the 24 px phase then runs to the half in
place of the 28 px phase: at 10 epochs this saves 0.40 s for -0.23 +- 0.07 points (8 paired trials);
one more epoch buys the points back at a net gain, since the epoch ladder is linear at about 1.05
points per second. Schedules that keep a 28 px phase (20 -> 24 -> 28 -> 32) save less (-0.19 / -0.29 s
at -0.31 / -0.10 points) and need eight compiled graphs, whose four-CPU cold build measured 441 s on
a sibling schedule ([[20, 0.15], [24, 0.3], [28, 0.5]]). (2) Group 1's residual pair becomes a 3x3 followed by a 1x1 like group 2's: -0.13 s at
+0.02 +- 0.13 points (8 paired trials); the 1x1 first and the 3x3 second saves only 0.06 s. Stacked
at 11 epochs the two changes measured -0.16 s for -0.05 +- 0.08 points over 16 paired trials before
the 40-trial confirmation above (-0.14 / -0.19 s at +0.12 / +0.16 points); at 10.75 epochs -0.24 s
for +0.01 +- 0.08 points (16 trials) and at 10.5 epochs -0.33 s for -0.07 +- 0.14 (8 trials), not
taken to 40 trials because their accuracy margin would be thin.

The rest is PR #27, one change on PR #26: the second convolution of group 2's residual pair is a 1x1
instead of a 3x3 (the first stays a 3x3, 256 -> 192). Group 2 runs on a 7x7 map, where 3x3
convolutions reach a small fraction of A100 peak, and the 1x1 has a ninth of the FLOPs: -0.21 s per
trial on both seed sets for +0.05 +- 0.06 / -0.08 +- 0.06 points (8 paired trials: -0.07 +- 0.11
points, -0.19 s; 16: +0.03 +- 0.07, -0.21 s). The same 1x1 in group 3's pair costs 0.3-0.4 points
for 0.3 s and only breaks even with half an epoch more; the 1x1 first and the 3x3 second in group 2
loses 0.1 points for the same time; two 1x1 convolutions in group 3 (an earlier experiment) lose
1.9 points. PR #26 is three changes on PR #23. The residual pair of the second group through 192
channels, together with the max + mean pooling, saves 0.13 s for about 0.05 points (PR #24: 40
paired trials per seed set, -0.07 +- 0.06 and -0.02 +- 0.06 points, -0.12 and -0.14 s; the pair
alone was screened at 8 trials: +0.07 +- 0.15 points, -0.13 s). The final pooling's max and mean
are written as `F.max_pool2d` / `F.avg_pool2d`, which `torch.compile` lowers to pointwise kernels
instead of the reduction kernels it generates for `max(dim).values` and `mean`: identical
results and another 0.13 s (the max alone through `F.max_pool2d`: -0.03 +- 0.08 points, -0.14 s
over 16 paired trials). The per-channel mean added to the max costs no time; it measured +0.30 +- 0.14
points in an 8-trial screen and about +0.06 points over 80 paired trials on PR #23. Together:
-0.26 / -0.27 s for -0.05 +- 0.06 / -0.02 +- 0.06 points on the two seed sets (16 paired trials:
-0.06 +- 0.11 points, -0.24 s). Fewer epochs do not pay: with the compiled pooling, 9.875 epochs
lost 0.10 points for 0.03 s and 9.75 epochs about 0.20 points for about 0.09 s. The residual pair of the last group is about a
third of the network's FLOPs, and running it through 512 channels saves 0.31 s for 0.35 points
(paired 8-trial screens at 400 W); half an epoch more buys the points back at a net gain of
about 0.1 s (PR #23). The 24x24 first quarter in front of the 28x28 phase saves 0.38 s at equal
accuracy, and the epoch ladder is linear at about 1.05 points per second. Every optimizer and
augmentation knob was re-screened on this schedule (learning rate, weight decay, BatchNorm-bias
learning rate, momentum, BatchNorm momentum, label smoothing, warmup, final learning rate,
lookahead period, translation, jitter, cutout, batch 512/768, batch schedules, logit scale): all
flat or worse. Under the four-CPU container quota (the official judging limit) the cold build of exactly this
configuration took 317 and 368 s in the two confirmation containers (limit 600 s; PR #33's folder
built in 311 and 381 s in the same containers, in 365 / 419 s in its own verification containers;
the six compiled graphs are the same as PR #33's, the 2x2 kernel changes the kernels inside them).
PR #33's cold build of its own configuration took 365 and 419 s in its two confirmation containers (PR #29's folder
built in 374 and 429 s in the same containers, in 355 and 364 s in its own verification containers
and in 324 / 416 s in the two 11.25-epoch containers of the same round). The six compiled graphs are the same as PR #27's
(three resolutions, two whitening-bias flags); the 20 px phase and the 1x1 change the kernels inside
them, not their number.

Earlier results for the 9-epoch recipe:

Each row pair ran back to back in one Modal A100-SXM4-80GB container on the same 40
random seeds (drawn like the organizer seed file), 75% target enforced. Seed sets
are independent draws; cards differ in speed, so compare within a pair.

| Seed set | Recipe | Mean accuracy | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | --- |
| F | PR #14 defaults (9 epochs, 28 px first half) | 75.009% (sd 0.28) | 4.646 s (sd 0.010) | yes |
| F | PR #14 as first opened (128/256/768, 8.25 ep) | 75.193% (sd 0.25) | 5.445 s (sd 0.009) | yes |
| E | PR #14 defaults before the `prepare` speedup | 75.083% (sd 0.25) | 5.321 s (sd 0.052) | yes |
| E | PR #14 as first opened | 75.150% (sd 0.23) | 6.153 s (sd 0.060) | yes |
| B | PR #14 as first opened | 75.130% (sd 0.29) | 6.048 s (sd 0.055) | yes |
| B | PR #9 defaults | 75.268% (sd 0.28) | 6.319 s (sd 0.042) | yes |

PR #14's defaults were 13.5-14.7% faster than PR #14 as first opened, which was 3.2-4.3%
faster than PR #9; their accuracy margin was thin (about 75.06% over 110 trials, an estimated
10% chance of a 40-seed draw below 75%), which the later PRs addressed with more epochs and
the changes above. The batch-512, 8-epoch variant mentioned in earlier revisions was not
adopted.

Run the defaults with the harness from `cifar100-speedrun/`:
`uv run python -m benchmark.run --submission futurebiohackers --n 40`
(`just modal 40` is the team repository's Modal wrapper around the same command).

## Experiments and progress

Tested in same-GPU comparisons against a control (8 to 40 paired trials each):

| Option | Result |
| --- | --- |
| `batch_size: 512, lr: 16` | 8 epochs match 9 epochs of batch 1024, 4% faster; lr 11.5 or 18 loses 0.2 pt; 7.5 epochs fails |
| `batch_size: 768, lr: 14`, 8.5 epochs | 2% faster at equal accuracy |
| `batch_schedule` (512 during the 28 px phase) | pending |
| `resolution_schedule: [[24, 0.33], [28, 0.67]]` | 24 px costs 0.8 pt at 10.5 epochs; worse than cutting epochs on CIFAR-100 |
| `resolution_schedule: [[28, 0.6]]`, 9.25 epochs | same as 28 px for half |
| `momentum: 0.9` | -0.08 pt, same time |
| `whiten_bias_epochs: 1` | -0.08 pt, same time |
| `widths: [64, 256, 896]` | more accurate, slower; on the same accuracy/time line |
| `widths: [48, 256, 768]` or `[64, 192, 768]` | less accurate at equal time |
| `g2_pair: "inner192"` (PR #24, PR #26) | -0.13 s for about 0.05 points; group 3 pairs at 448 / 384 sit on the accuracy/time line; a 1x1-3x3-1x1 bottleneck in group 3 loses 1.4 points for 0.7 s |
| `resolution_schedule: [[20, 0.15], [24, 0.5]]` (PR #29) | 20 px for the first 15% of the examples, 24 px to the half, 32 px after: -0.40 s at -0.23 points at 10 epochs (8 paired trials, cold); 20 -> 24 -> 28 -> 32 (eight graphs): -0.19 / -0.29 s at -0.31 / -0.10 points, with a 441 s four-CPU build on a sibling schedule; epoch ladder on the schedule: 10.5 ep -0.24 s / +0.05, 10.75 ep -0.14 s / -0.06, 11 ep -0.08 s / +0.09 points (8 trials each) |
| `conv1_kernels: [3, 3, 2]` (this PR) | group 3's first convolution as a 2x2 kernel: -0.06 +- 0.07 points / -0.28 s over 16 paired trials on PR #29 (cold); at 11 epochs vs PR #33 over 40 per seed set -0.25 / -0.12 points for -0.31 / -0.31 s; with 11.5 epochs (this PR) -0.00 / +0.03 points / -0.15 / -0.15 s; `[3, 2, 2]` (group 2 as well): -0.19 +- 0.10 points / -0.37 s |
| `inner_kernels: [[1, 1], [3, 1], 3]` (PR #33) | group 1's pair as two 1x1 convolutions: +0.08 +- 0.08 points / -0.08 s over 16 paired trials (cold), -0.01 / -0.02 points / -0.10 / -0.09 s over 40 per seed set vs PR #29; `depths: [2, 3, 3]` (no pair): -0.15 points / -0.15 s |
| `inner_kernels: [[3, 1], [3, 1], 3]` (PR #29) | group 1's pair as 3x3 then 1x1, on top of group 2's: -0.13 s at +0.02 points (8 paired trials, cold); `[[1, 3], [3, 1], 3]`: -0.06 s at +0.05 |
| `inner_kernels: [3, [3, 1], 3]` (PR #27) | group 2's pair as 3x3 then 1x1: -0.21 s at +0.05 / -0.08 points over 40 paired trials per seed set vs PR #26 (warm cache; -0.19 / -0.22 s cold at 4 CPUs); 1x1 then 3x3: -0.09 points for the same time; the same in group 3: -0.37 points for -0.30 s |
| `global_pool: "fullpool_avgsum"` (PR #25, PR #26), `"maxmean_sum"` (PR #24), `"maxmean_cat"` | the same max + mean through `F.max_pool2d` / `F.avg_pool2d`: -0.13 s at identical results; the summed max + mean measured +0.3 points in an 8-trial screen (+0.2 for the concatenated one) and about +0.06 at 40 trials |
| squeeze-and-excitation on groups 2-3, multi-scale head | break-even and -0.3 points |
| wider group 1 (`widths: [128, 256, 768]`) | on or below the accuracy/time line |
| `inductor_tuning` | coordinate-descent tuning: 0.5% faster, within noise |
| `optimizer: "muon"` (hiverge-style, batched) | 3.5 points less accurate at 8 epochs |
| `activation: "silu"` | 2% faster, 0.4 points less accurate |
| `stem: "patch4s2"` (4x4 stride-2 whitening) | 2-3 points less accurate |
| `inner_kernels: [3, 3, 1]` (1x1 convs in group 3) | 1.9 points less accurate |
| `pool_first` in group 3 | 1 point less accurate |
| `bn_recal_batches` | no gain |
| `ema_every: 0` | 0.6 points less accurate |
| Vision transformer (patch 4, dim 256, 6 layers) | 37% at 9 epochs, 58% at 30 epochs (39 s) |

Convolutions are about 70% of GPU time and Inductor's BatchNorm/activation kernels about
25% (profiles of the 10-epoch ancestors of this recipe: GPU busy 99%, no host synchronisation inside the step, about
180 kernels per step). Inductor's CUDA graphs are on; a whole-run graph would add little
since the GPU is already busy for the whole step.

`--params` exposes other resolution schedules, block widths/depths, residual-pair widths and kernels,
proxy-based hard-example selection, other pooling and optimizer settings, and an optional
Triton crop/flip kernel; these are off in the defaults (the 20/24 px schedule, the mixed residual pairs and the max + mean
pooling are on). In PR #5's tests, proxy-based selection did not improve the qualifying result,
and the unsuccessful experimental BN/GELU fusion was excluded from the submitted source.

Team-repository tooling (not part of this folder): `scripts/modal_experiments.py` runs bounded
comparisons through the unchanged competition harness, and `scripts/track_speedrun.py` writes
a live log of the experiments with each one's main changes, accuracy, trial count and paired
control.

## Data and evaluation checks

Build warms compilation and kernels using random synthetic images only. Every
trial resets model weights, BatchNorm statistics, gradients, optimizer state,
moving averages and any proxy masks. Preparation and training use only the
harness-provided training images and labels. Evaluation uses one image view and
frozen training statistics; it does not fit to test images, use test-batch
statistics or change registered model state.

The [rules](https://github.com/AIDDA-Institute/CIFAR-100-speedrun/blob/main/RULES.md)
allow comparing reported test accuracy during development. Test results never
choose a stopping point or hard-example masks within a trial. Source review found
no test-data leakage; official acceptance still requires organizer review and the
prescribed environment.

The correctness checks cover resets, immutable training inputs, evaluation state
and batch independence (13 CPU tests), plus exact crop/flip equivalence across
layouts and resolutions (36 GPU checks). The GPU benchmark applies the harness's
own output and state checks. The CPU checks live in the team repository
(`scripts/test_speedrun_recipe.py`; from `cifar100-speedrun/`:
`uv run python -m pytest -q ../scripts/test_speedrun_recipe.py`).

Adapted under the MIT license, Copyright (c) 2024 Keller Jordan. The full original
permission notice is preserved in `LICENSE.airbench`.
