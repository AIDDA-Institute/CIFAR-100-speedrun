# NAVIER-STOKED — CIFAR-100 to 75 % in about 3.5 seconds

Measured with the benchmark harness on an A100 80GB PCIe (300 W, the official card). The run used the official base image and pinned packages, 4 CPU cores, an empty compile cache and 40 fresh random seeds:

| Check | Mean accuracy | Mean time (prepare + train) |
|---|---|---|
| **This folder (8.75 epochs)**, 40 trials | **75.19 %** (lowest trial 74.58 %, SD 0.26) | **3.50 s** |

All 40 trials succeeded and the run qualified. The estimated chance of an official 40-trial mean below 75 % is about 0.05 %. For comparison, the README baseline takes 59.3 s.

## The recipe, starting from airbench

Our starting point is Keller Jordan's [airbench](https://github.com/KellerJordan/cifar10-airbench), the CIFAR-10 speedrun record line: a small conv net with a frozen patch-whitening first layer, identity initialisation and alternating flips, trained for a few epochs. Every change below was measured against a concurrent control on the same GPU type. Time effects come from same-container A/B runs, and the numbers are the measured effects.

### 1. Network

- **Structure.** Three groups of 64 → 256 → 768 channels. Each group is one convolution, a 2×2 max pool, then a residual pair of convolutions:
  - group 1's pair is two 1×1 convs;
  - group 2's pair is a 3×3 down to 192 channels and a 1×1 back;
  - group 3's pair is two 3×3 convs through 512 channels;
  - group 3's first convolution uses a 2×2 kernel.

  This structure follows team futurebiohackers' submission (PR #2), re-implemented in our own code.
- **Why this shape.** Convolutions on the large early maps (31×31, 15×15) run at a small fraction of the GPU's peak, while the wide last group runs at 3×3, where they are efficient. The net needs 175M multiply-adds per image against 349M for our earlier two-convs-per-group net (128 / 384 / 576). On the official card it reaches 75 % in 3.5 s against 5.4 s for that net.
- **Classifier.** Global max plus global mean of the last map, a linear head, logit scale 1.25/9.
- **From airbench:**
  - the whitening stem: a 2×2 conv set from the training images' patch statistics, computed inside the timed `prepare` (+1.1 pp);
  - identity (dirac) initialisation of the convolutions (+1.5 pp);
  - SiLU instead of ReLU (+0.9 pp on the earlier net). SiLU is also what our fused kernels implement.

### 2. Optimizer

- **Muon** on all convolution filters: each update is orthogonalised with three Newton–Schulz iterations, and the filters are renormalised. Nesterov SGD trains the classifier and the BatchNorm parameters.
  - On this network, Muon reached 76.25 % at 11.5 epochs, where futurebiohackers' SGD + lookahead reports 75.24 %.
  - Our own SGD + lookahead version reached only 73.5 %.
  - That extra ≈ 1 pp per epoch is what lets us train for 8.75 epochs instead of 11.5 (section 5).
- **Batch 1024** with raised learning rates (SGD 1.38, Muon 0.18). Larger batches mean fewer, fuller GPU steps. They only paid off once Muon's learning rate was raised with them; untuned, they had lost 0.7 pp.
- **Schedule:** linear warm-up over the first 25 % of steps, then linear decay to zero.

### 3. Data and regularisation

- **Progressive resizing:** the first 15 % of steps train on 20×20 images and the steps up to the half on 24×24, the rest at 32×32. These cost 39 % and 56 % of full-size compute.
- **Augmentation:** random shifts of up to 2 pixels, and airbench's alternating flips (each image is seen both ways).
- **Label smoothing 0.25.**

### 4. Systems: making the GPU do only useful work

- **The whole training step is a CUDA graph.** `build()` is untimed. On synthetic data it records the complete step (augmentation, forward, backward, both optimizers) once per image size, and training only replays these graphs. Before this, Python needed nearly as long to *issue* a step (5.9 ms) as the GPU needed to *run* it (7.5 ms).
- **Compiled code.**
  - The forward step (augmentation + network + loss) is one `torch.compile` graph: −0.08 s.
  - The Muon update is compiled into fused kernels: −0.09 s. On its own it was ≈ 1.6 ms of uncompiled elementwise work per step.
- **Custom fused Triton kernels** (`fused.py`):
  - one kernel does max pool + BatchNorm statistics + normalisation + SiLU, and another BatchNorm + SiLU, each with its own backward;
  - every activation array is read and written fewer times, and the pool's backward writes its gradient straight into the input gradient;
  - they match PyTorch numerically, including running statistics (worst relative error 3.7e-3), and fall back to plain PyTorch in eval mode;
  - on this network they are 0.08 s faster than Inductor's generated kernels and build 83 s faster (−0.24 s on the earlier net).
- **Cheap global max pool** (`flatten().max()`): PyTorch's `AdaptiveMaxPool2d` backward was 2.5 % of each step, and `amax` trains to NaN under `torch.compile`.
- **Other details:**
  - everything stays on the GPU in bf16 with channels-last memory layout;
  - output buffers are allocated directly in channels-last layout, because `empty().contiguous(channels_last)` copied every buffer once;
  - `prepare()` resets every weight, statistic and optimizer buffer, so trials are independent. Accuracy shows no drift with trial order, and reversing the seed order gave the same results.

### 5. Choosing the number of epochs

A failed 40-trial mean disqualifies the entry, so the length is the one knob tuned to the official card. Per-seed accuracy varies by about ±0.25 pp. The chance that the official 40-trial mean falls below 75 % combines our estimate's error with the official run's own sampling noise: z = (mean − 75) / √(sd²/n + sd²/40). Same card, same 40 seeds, harness, 4 CPUs:

| Epochs | Mean accuracy (40 trials) | Lowest trial | Time | P(fail) |
|---|---|---|---|---|
| 8.5 | 75.05 % | 74.35 % | 3.48 s | ≈ 17 % |
| **8.75** | **75.19 %** | 74.58 % | **3.50 s** | **≈ 0.05 %** |
| 9.0 | 75.33 % | 74.77 % | 3.67 s | ≈ 0 |
| 10 | 75.8 % (8 seeds, SXM) | | | |
| 11.5 | 76.1–76.25 % | | 4.56 s | |

We train for **8.75 epochs**.

## Everything else we tested (no gain)

- **Network:**
  - ResNet9/18, wide ResNets, VGG and ensembles of small nets;
  - a 3×3 whitening stem;
  - pooling *before* convolutions (−0.7 to −1.9 pp);
  - on the earlier net: linear-bottleneck entry convs (−0.95 pp), a third residual conv in the last block (+0.16 pp for +0.48 s), and an average + max head (no effect);
  - starting resolutions of 16 or 20 px on the earlier net (−0.9 to −1.2 pp).
- **Optimizer and schedule:**
  - lookahead/EMA, SWA, and a 35/65 final/average weight blend;
  - warm-up-stable-decay and warm-up/hold/decay schedules;
  - a learning-rate floor and other warm-up lengths;
  - Muon momentum (0.5 / 0.7 / warm-up) and weight-decay changes;
  - a separate classifier learning rate (×0.5 to ×4) and label smoothing 0.2 / 0.3 combined with it;
  - per-resolution batch sizes.
- **Data:**
  - cutout and batch augmentation;
  - 1-pixel instead of 2-pixel shifts;
  - no augmentation in the final steps;
  - online label smoothing;
  - training only on the hardest 80 % of images, by loss or by "learnability": it saves time but costs about as much accuracy.
- **Systems:**
  - fp16 weights, and the `max-autotune` / `reduce-overhead` compile modes;
  - Inductor's `aggressive_fusion` (no effect);
  - coordinate-descent tuning: −0.13 s before the fused kernels; after them, tuning from a cold cache picked slower kernels (+0.1 s) and added minutes of build time;
  - `benchmark_fusion`: builds over 15 minutes, beyond the 600 s limit;
  - freezing the first group or the BatchNorm statistics late in training (−0.04 to −0.09 s for −0.17 to −0.36 pp).
- **Not allowed:** test-time augmentation (RULES.md §3).

Many small "wins" seen on 1–8 seeds vanished at 32 seeds. Accuracy-changing choices were therefore confirmed on 32+ seeds, and the final recipe on 40 fresh seeds.

## How it was validated

- The benchmark harness (`python -m benchmark.run --submission NAVIER-STOKED`) on a rented **A100 80GB PCIe (300 W, MIG off)**.
- The environment matches the official Dockerfile: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, build-essential, uv 0.10.8, Python 3.12.10, `uv sync --frozen --no-dev`, PyTorch 2.4.0, CUDA 12.4.
- The run was limited to **4 CPU cores** and started from an **empty compile cache**, with **40 fresh random seeds** drawn like the organizers' seed file.
- **One limitation:** the rental host does not allow network namespaces, so `--official`'s network-isolation check could not pass there. The same harness ran in development mode, which has identical timing.
- **Build:** the cold build (compilation, warm-up and graph capture, untimed) stays well under the 600 s limit. Evaluation takes about 0.08 s against its 5 s limit.

## Files

- `submission.py`: `build` / `prepare` / `train`, the schedule and the CUDA-graph training step.
- `model.py`: the network and the whitening stem.
- `optim.py`: Muon (compiled) and graph-safe Nesterov SGD.
- `fused.py`: the fused pool/BatchNorm/SiLU Triton kernels, as `torch.library` custom ops with their own backward.
- `LICENSE.airbench`, `LICENSE.muon`: the MIT notices of the adapted code.

## Credits and licenses

- **Keller Jordan's [cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)** (MIT): the overall design and dirac initialisation. The whitening-stem initialisation (`WhiteningStem.fit` in `model.py`) and the alternating flips (`_augment` in `submission.py`) are adapted from it. Full notice in `LICENSE.airbench`.
- **Keller Jordan's [Muon](https://github.com/KellerJordan/Muon)** (MIT): `zeropower_newton_schulz` and the Muon update in `optim.py` are adapted from it. Full notice in `LICENSE.muon`.
- **Team futurebiohackers (PR #2 in this repository):** the network structure (widths, residual three-conv groups, mixed kernels, max + mean head) and the 20 / 24 / 32 px schedule, re-implemented in our own code. No code was copied.
- **Inspiration (no code copied):** David Page's "How to train your ResNet" and tysam-code's hlb-CIFAR10.

Everything else in this folder, including the CUDA-graph training step, the compiled Muon update and the fused Triton kernels, is our own work, contributed under the repository's MIT License.
