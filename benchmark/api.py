"""The small, local Python interface implemented by submission.py."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BuildContext:
    device: torch.device
    parameters: dict[str, Any]
    num_classes: int = 100
    eval_batch_size: int = 1024


@dataclass(frozen=True)
class TrainingData:
    """CPU tensors: uint8 RGB NCHW images, int64 fine-class labels in [0, 99].

    Treat both tensors as immutable. Transfers, normalization, and augmentation
    belong in prepare/train and are charged to the submission.
    """

    images: torch.Tensor
    labels: torch.Tensor
