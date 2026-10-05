"""Device-resident training data with batched augmentation (all timed work)."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from .config import RecipeConfig


def crop(images: torch.Tensor, offsets: torch.Tensor, size: int) -> torch.Tensor:
    """Crop one ``size`` window per image at per-image (row, column) offsets, by gathering."""
    window = torch.arange(size, device=images.device)
    rows = (offsets[:, 0, None] + window)[:, None, :, None]
    cols = (offsets[:, 1, None] + window)[:, None, None, :]
    n, c, _, width = images.shape
    images = torch.gather(images, 2, rows.expand(n, c, size, width))
    return torch.gather(images, 3, cols.expand(n, c, size, size))


class TrainingStream:
    """Yields shuffled, translated, flipped batches. Each image gets a fixed random flip,
    and every image is flipped on odd epochs."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        config: RecipeConfig,
        generator: torch.Generator,
    ) -> None:
        if config.batch_size > len(images):
            raise ValueError(
                f"batch_size {config.batch_size} exceeds the {len(images)} training images"
            )
        self.config = config
        self.generator = generator
        self.labels = labels
        flip = torch.rand(len(images), device=images.device, generator=generator) < 0.5
        base = torch.where(flip.view(-1, 1, 1, 1), images.flip(-1), images)
        pad = config.translate
        self.base = F.pad(base, (pad,) * 4, mode="reflect") if pad else base
        self.size = images.size(-1)
        self.steps_per_epoch = len(images) // config.batch_size

    def epoch(self, index: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        base, r = self.base, self.config.translate
        if r:
            # Random translation by up to r pixels in each direction, drawn per image per epoch.
            shifts = torch.randint(
                -r, r + 1, (len(base), 2), device=base.device, generator=self.generator
            )
        order = torch.randperm(len(base), device=base.device, generator=self.generator)
        bs = self.config.batch_size
        for step in range(self.steps_per_epoch):
            idx = order[step * bs : (step + 1) * bs]
            images = crop(base[idx], shifts[idx] + r, self.size) if r else base[idx]
            if index % 2 == 1:
                images = images.flip(-1)
            yield images, self.labels[idx]
