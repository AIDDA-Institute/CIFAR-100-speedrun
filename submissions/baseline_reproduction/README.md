# Baseline reproduction

This is a reconstruction of the README's documented ResNet9-style calibration
recipe. The original organizer calibration source was kept in an ignored
`.local/` directory and is not available in the repository history.

The default recipe uses:

- a width-64 ResNet9;
- 40 epochs with batch size 512;
- random reflected crops, horizontal flips, 10-pixel Cutout, and light Mixup;
- 24-pixel training for the first 10 epochs, then full 32-pixel training;
- SGD with Nesterov momentum and weight decay;
- a fastai-style one-cycle schedule; and
- BF16 training with channels-last tensors, `torch.compile` reduce-overhead
  mode, and synthetic build-time graph warmup.

The locked combined recipe qualified over 32 untouched validation seeds at
75.470% mean accuracy with a 0.057-point standard error. Excluding the first
trial, median preparation-plus-training time was 42.606 seconds on an
uncontended A100 80GB PCIe. Synthetic warmup keeps compilation in the untimed
build phase. This is 29.4% faster than the 60.314-second reproduced baseline.
An additional disjoint 16-seed `reduce-overhead` control averaged 75.474%,
bringing the combined 48-seed accuracy evidence to 75.471%. Amortizing one
44.125-second first trial across an official 40-trial run gives an estimated
42.644-second scored mean.

## Validation policy

- Historical seeds 0-6 are retired because they were used during recipe
  selection.
- Powered screening used seeds 20-39.
- Seeds 40-55 were consumed once for an interim audit.
- Seeds 100-131 were consumed once to validate the width-72 safe fallback.
- Seeds 200-231 were consumed once for final validation of the locked width-64
  fast candidate.
- Seeds 300-331 remain reserved for an unbiased audit of any future recipe.

An accuracy-costing speed change must not be adopted unless a disjoint held-out
mean remains at least 75.35%. Small margin changes require a powered paired
test, normally at least 16 seeds; patch whitening is only eligible if its
implementation uses training data inside the timed phases and measures its full
preparation-plus-training cost.

## Compilation and rejected failures

`build()` runs once per worker, while its compiled graphs persist across the
worker's trial loop. Synthetic BF16 forward/backward warmup covers both 24- and
32-pixel inputs at batch sizes 512 and 336. Step instrumentation found lazy
compilation only for the first occurrence of each shape, not once per trial.
Warmup restores parameters, buffers, gradients, module modes, and CPU/CUDA RNG
state before any real training data or trial seed is available.

`max-autotune` was re-screened on 16 paired seeds after treating its 316-second
build as unscored. It did not improve timed execution: its steady median was
42.619 seconds versus 42.605 seconds for `reduce-overhead`, with a wider timing
IQR, so `reduce-overhead` remains the default.

## Known future work

The final accuracy estimate leaves about 0.29 percentage points above the
modeled 1%-risk threshold. A full optimizer-step CUDA graph could potentially
trade some of that margin for additional speed, but dynamic augmentation RNG,
OneCycle scheduler state, two batch shapes, and two resolutions make correct
capture and reset substantially riskier than the model-only compilation used
here. It was intentionally left out of the verified submission.

Two historical results are not candidate evidence: the 0% run was a CPU
synthetic smoke test, and the 19.08% run used an unsuitable EMA decay of 0.999.
The EMA setting reproduced at 17.51% over eight real-data seeds and was excluded
from every fusion candidate.
