# encode: Reflexive AutoResearch

This recipe was developed with Reflexive AutoResearch, the autoresearch framework built
during the hackathon. Each component was kept only if paired multi-seed experiments showed it
lowered the training time while keeping mean accuracy at or above 75%.

```bash
uv run python -m benchmark.run --submission encode --n 40
```

**Result:** 75.15% mean accuracy and 4.68 s mean preparation + training, measured on an
NVIDIA A100 80GB PCIe in the pinned environment with 40 fresh seeds and default settings.

## Method

- **Network:** a fixed 2×2 whitening convolution, three convolution–BatchNorm–GELU stages
  (widths 128/384/640) with residual connections, global max-pooling and a linear head.
  Stage 1's residual branch uses a 1×1 convolution.
- **Data:** random 2-pixel translations and alternating horizontal flips. The first half of
  training runs at 20 px, the rest at 32 px.
- **Optimisation:** Nesterov SGD with lookahead, label smoothing 0.4, batch 1024, 8.25 epochs.
- **Efficiency:** fp16, with the model, loss and optimiser step compiled by `torch.compile`.
  Parts of the network leave autograd once their learning rate reaches zero: the whitening
  bias after 3 epochs, and stage 1 after 80% of training.

## Compliance

- `build` uses synthetic data only.
- `prepare` resets all weights and statistics, and computes normalisation and whitening from
  the trial's own training images.
- Evaluation is a single forward pass with no test-time augmentation.
- There are no pretrained weights, no external data and no state carried between trials. The
  recipe uses only the pinned dependencies.
