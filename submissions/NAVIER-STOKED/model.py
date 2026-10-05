"""Network: an airbench-style conv net with a frozen patch-whitening stem.

Input is float RGB in [0, 1]; output is [B, 100] float logits.
"""

import torch
from torch import nn

from .fused import BNSiLU, PoolBNSiLU

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


class WhiteningStem(nn.Module):
    """2x2 conv set from training-patch statistics in prepare (timed): the top 12
    whitened patch directions and their negatives. Its weight is not trained.

    The initialisation is adapted from Keller Jordan's cifar10-airbench
    (https://github.com/KellerJordan/cifar10-airbench, MIT License,
    Copyright (c) 2024 Keller Jordan; full notice in LICENSE.airbench)."""

    def __init__(self, c_out: int = 24):
        super().__init__()
        self.conv = nn.Conv2d(3, c_out, 2, padding=0, bias=True)

    @torch.no_grad()
    def fit(self, images: torch.Tensor, eps: float = 5e-4) -> None:
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1)  # N, 3, 31, 31, 2, 2
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 12).double()
        patches = patches - patches.mean(0)
        cov = patches.T @ patches / len(patches)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        w = (eigvecs.T / torch.sqrt(eigvals + eps)[:, None]).flip(0)  # top direction first
        w = w.reshape(12, 3, 2, 2).float()
        half = self.conv.out_channels // 2
        self.conv.weight.copy_(torch.cat([w[:half], -w[:half]]))
        self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


class ResGroup(nn.Module):
    """conv(k1) -> 2x2 max pool -> BN -> SiLU, then a residual pair through `mid` channels:
    x + SiLU(BN(conv_b(SiLU(BN(conv_a(x)))))). Pool/BN/SiLU run as fused Triton kernels in
    training (see fused.py) and as plain PyTorch in eval."""

    def __init__(self, c_in: int, c_out: int, k1: int, ka: int, kb: int, mid: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k1, padding="same", bias=False)
        self.pool_bn1 = PoolBNSiLU(c_out)
        self.conv_a = nn.Conv2d(c_out, mid, ka, padding="same", bias=False)
        self.bn_a = BNSiLU(mid)
        self.conv_b = nn.Conv2d(mid, c_out, kb, padding="same", bias=False)
        self.bn_b = BNSiLU(c_out)

    def forward(self, x):
        x = self.pool_bn1(self.conv1(x))
        return x + self.bn_b(self.conv_b(self.bn_a(self.conv_a(x))))


class Net(nn.Module):
    def __init__(
        self,
        widths=(64, 256, 768),
        mids=(64, 192, 512),
        k1=(3, 3, 2),
        pair_kernels=((1, 1), (3, 1), (3, 3)),
        num_classes: int = 100,
        scale: float = 1.25 / 9,
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.stem = WhiteningStem(24)
        groups, c = [], 24
        for w, m, k, (ka, kb) in zip(widths, mids, k1, pair_kernels, strict=True):
            groups.append(ResGroup(c, w, k, ka, kb, m))
            c = w
        self.body = nn.Sequential(nn.SiLU(inplace=True), *groups)
        self.head = nn.Linear(widths[-1], num_classes, bias=False)
        self.scale = scale

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        x = self.body(self.stem(self.normalize(x)))
        # Global max + mean. flatten + max has a cheap index-scatter backward.
        x = x.flatten(2).max(2).values + x.flatten(2).mean(2)
        return (self.head(x) * self.scale).float()
