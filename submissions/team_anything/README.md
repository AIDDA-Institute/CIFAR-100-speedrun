# Team Anything: airbench96 with Muon, hard-example selection and a group-1 freeze

Team: Harris, Balint, Tara, Amir.

A CIFAR-100 adaptation of Keller Jordan's airbench96 (MIT, notice in `LICENSE`), tuned with an LLM
research swarm and our own search runs. Only the recipe is submitted; the search tooling is not.
`submission.py` runs the final recipe with its defaults; it needs no parameters.

## Recipe

- 258 optimizer steps of 1024 images (`epochs=6.5`).
- Epoch 0 at 22x22, epoch 1 at 24x24 (antialiased downscale), then 32x32.
- From epoch 2, each epoch trains on the 75% of examples with the highest loss when last seen, chosen
  within the trial, with 15% of the selected examples drawn at random from the easier ones.
- Muon (3 Newton-Schulz steps) on the 3x3 conv filters, Nesterov SGD on everything else,
  with a lookahead EMA every 5 steps.
- Group 1's learning rates decay linearly to zero, then the group is frozen from epoch 5: its output is
  detached, so backward stops at group 2.
- Head logits x 1/5, label smoothing 0.4, flip + translate 2 + per-image colour jitter, no cutout,
  BatchNorm momentum 0.6 with parameters and buffers in fp16 like the rest of the network, QuickGELU.
- `torch.compile(mode="max-autotune")` with CUDA graphs. Untimed synthetic warmup in `build`;
  every trial, `prepare` resets parameters, buffers and selection state and creates fresh optimizers.

## Results

40 fresh seeds per run (not the organizer seeds), one container per run, official harness at
`25237e3`, 4 CPUs:

| GPU | mean accuracy | min | mean prepare + train | build |
| --- | ---: | ---: | ---: | ---: |
| A100 80GB PCIe, 300 W (official card class) | 75.13 ± 0.25% | 74.71% | 4.061 s | 236 s |
| A100 SXM4 80GB, 400 W | 75.11 ± 0.23% | 74.65% | 3.717 s | 321 s |

`build` includes max-autotune compilation (limit 600 s).
