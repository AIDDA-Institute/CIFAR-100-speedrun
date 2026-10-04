# AlphaBetaGammaChi: deep whitening CNN with EMA

A source-only CIFAR-100 recipe, trained from scratch on each trial using only that trial's
training split. Default settings are the submitted configuration:

- fixed, training-data-derived 2x2 whitening filter bank (computed during timed preparation);
- three convolution groups of widths 128/384/512, each with three convolutions
  (the third is residual), BatchNorm and GELU;
- batches of 2,000 held on the GPU in channels-last half precision;
- precomputed flips and two-pixel translations;
- 14 epochs with linear learning-rate decay, label smoothing 0.2;
- Muon (Newton-Schulz orthogonalised) updates for convolution filters, batched by shape;
- exponential moving average of weights and BatchNorm statistics over the last 4 epochs;
- torch.compile of the training forward/backward graph during the untimed build, warmed up on
  synthetic inputs only, with all model state reset afterwards.

Evaluation is eager, single-view at 32x32, and uses no test statistics, adaptation or
augmentation. No pretrained weights, external data, checkpoints or cross-trial state are used.
kernels.py contains an optional Triton crop kernel (off by default); the reference PyTorch path
is used.

## Development evidence (not official)

Seeds 2 and 4 on an A100 MIG 3g.40gb slice (42 SMs), 14 epochs: 75.75% and 76.04% top-1,
27.2 s preparation + training each. 12, 16 and 18 epochs on the same seeds gave 75.68-76.59%.
Official results require the organizers' 40-seed run on an A100 80GB PCIe.

## Attribution

The optimizer iteration, whitening architecture and training design are adapted from:

Keller Jordan, *CIFAR-10 Airbench*, 2024,
<https://github.com/KellerJordan/cifar10-airbench>, MIT License.

The required upstream license notice is preserved in THIRD_PARTY_LICENSES.md.
