"""Download once before evaluation; official runs never download data."""

import argparse
from pathlib import Path

import torch
from torchvision.datasets import CIFAR100

from benchmark.api import TrainingData


def download(root: Path) -> None:
    CIFAR100(str(root), train=True, download=True)
    CIFAR100(str(root), train=False, download=True)


def load_split(root: Path, *, train: bool) -> TrainingData:
    dataset = CIFAR100(str(root), train=train, download=False)
    images = torch.from_numpy(dataset.data.transpose(0, 3, 1, 2).copy())
    labels = torch.tensor(dataset.targets, dtype=torch.int64)
    expected = 50_000 if train else 10_000
    if images.shape != (expected, 3, 32, 32) or labels.shape != (expected,):
        raise ValueError("Unexpected CIFAR-100 split shape")
    return TrainingData(images, labels)


def synthetic_split(*, train: bool) -> TrainingData:
    """Tiny deterministic fixtures for smoke tests, never calibration/scoring."""
    generator = torch.Generator().manual_seed(123 if train else 456)
    count = 64 if train else 32
    images = torch.randint(0, 256, (count, 3, 32, 32), dtype=torch.uint8, generator=generator)
    labels = torch.randint(0, 100, (count,), generator=generator)
    return TrainingData(images, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    download(args.root)
    print(f"CIFAR-100 is ready at {args.root.resolve()}")


if __name__ == "__main__":
    main()
