# it_compiles: airbench + Muon with progressive resolution

An airbench-style CNN (ported from Keller Jordan's
[airbench94_muon](https://github.com/KellerJordan/cifar10-airbench)) trained with the Muon
optimizer, on a 16 → 20 → 24 → 32 pixel resolution schedule. The airbench-derived code is
used under airbench's MIT license; see [LICENSE-airbench](LICENSE-airbench).

**Development results (A100 80GB):** 75.30% mean test accuracy over 80 seeds; 6.22 s mean
training time on A100 PCIe (40 seeds).

## Recipe

| Part | Setting |
| --- | --- |
| Network | Fixed 2×2 whitening conv (fitted on 5,000 training images in `prepare`), then three conv groups (128, 512, 512 channels), each conv → max-pool → BN → GELU plus two more convs with a residual connection; global max pool; linear head |
| Optimizers | Muon (3 Newton–Schulz steps, LR 0.24, momentum 0.6) for conv filters; Nesterov SGD for BN biases, whitening bias and head (head LR 3.0) |
| Schedule | 2 epochs at 16×16, 1 at 20×20, 2 at 24×24, 3 at 32×32 (200 steps, batch 2000); low resolutions are area downsamples of each epoch's crops |
| Learning rate | 5% linear warmup, hold at peak for 30% of steps, linear decay to zero |
| Regularisation | Label smoothing 0.3, ±1 px random translation, flips (all images flipped together every other epoch), weight decay 2e-6 × batch size |
| Precision | float16 weights and activations, float32 BatchNorm and logits |
| Compilation | `torch.compile(mode="max-autotune")`; `build` warms up every graph on synthetic data, so the timed trials never compile |

## Timing rules

- `build` (untimed): creates and compiles the model, then trains briefly on synthetic data to compile every training and evaluation graph.
- `prepare` (timed): resets all weights, BatchNorm statistics and optimizer state, moves the data to the GPU, and fits the whitening layer.
- `train` (timed): runs the 200 training steps and returns the model.
- Inference is a single plain forward pass per image, with no test-time augmentation.
