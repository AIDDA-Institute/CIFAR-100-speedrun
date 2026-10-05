"""Residual convnet with a frozen whitening stem: float32 RGB in [0, 1] in, float32 logits
out."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig

NUM_CLASSES = 100
WHITEN_KERNEL = 2


class Normalize(nn.Module):
    """Per-channel normalisation whose statistics are computed from training data in prepare."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x = (x.float() - self.mean) / self.std
        return x.to(dtype).contiguous(memory_format=torch.channels_last)


class BatchNorm(nn.BatchNorm2d):
    """BatchNorm with a frozen unit scale and a learnable bias."""

    def __init__(self, channels: int, momentum: float) -> None:
        super().__init__(channels, eps=1e-12, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    """Bias-free convolution with identity (dirac) initialisation of the first channels."""

    def __init__(self, cin: int, cout: int, kernel: int = 3) -> None:
        super().__init__(cin, cout, kernel, padding="same", bias=False)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # Identity on the first cin output channels (nn.init.dirac_ in one kernel).
        identity = self.weight.data[: self.weight.size(1)]
        identity.zero_()
        k = self.weight.size(-1) // 2
        identity.diagonal(0, 0, 1)[k, k].fill_(1)


class ConvGroup(nn.Module):
    """conv-pool-BN-GELU, then conv-BN-GELU; depth 3 adds a residual conv-BN-GELU branch."""

    def __init__(
        self,
        cin: int,
        cout: int,
        depth: int,
        bn_momentum: float,
        skip: bool = False,
        residual_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.conv1 = Conv(cin, cout)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_momentum)
        self.residual = (
            nn.Sequential(Conv(cout, cout, residual_kernel), BatchNorm(cout, bn_momentum))
            if depth == 3
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(F.max_pool2d(self.conv1(x), 2)))
        y = F.gelu(self.norm2(self.conv2(x)))
        if self.residual is None:
            return x + y if self.skip else y
        return x + F.gelu(self.residual(y))


class Net(nn.Module):
    """Whitening conv, three ConvGroups, global max-pool and a scaled bias-free head."""

    def __init__(self, config: RecipeConfig) -> None:
        super().__init__()
        w1, w2, w3 = config.widths
        whiten_width = 2 * 3 * WHITEN_KERNEL**2
        self.normalize = Normalize()
        self.whiten = nn.Conv2d(3, whiten_width, WHITEN_KERNEL, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        momentum = config.bn_momentum
        d1, d2, d3 = config.stage_depths
        s1, s2, s3 = config.stage_skips
        self.groups = nn.Sequential(
            ConvGroup(whiten_width, w1, d1, momentum, s1, config.stage1_residual_kernel),
            ConvGroup(w1, w2, d2, momentum, s2),
            ConvGroup(w2, w3, d3, momentum, s3),
        )
        # Set by ``fit``: once frozen, the stem and stage 1 run without autograd.
        self.stage1_frozen = False
        self.head = nn.Linear(w3, NUM_CLASSES, bias=False)
        self.scale = config.scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x, self.whiten.weight.dtype)
        if self.stage1_frozen and self.training:
            with torch.no_grad():
                x = self.groups[:1](F.gelu(self.whiten(x)))
            x = self.groups[1:](x)
        else:
            x = self.groups(F.gelu(self.whiten(x)))
        # Global max-pool (flattened: faster backward than adaptive_max_pool2d).
        x = x.flatten(2).max(2).values
        return (self.head(x) * self.scale).float()


def make_model(config: RecipeConfig, device: torch.device) -> Net:
    """Construct the network: fp16 channels_last with fp32 BatchNorm on CUDA, fp32 on CPU."""
    model = Net(config).to(device)
    if device.type == "cuda":
        model = model.half().to(memory_format=torch.channels_last)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d | Normalize):
                module.float()
    return model


@torch.no_grad()
def reset_model(model: Net) -> None:
    """Reinitialise every parameter and buffer in place, keeping tensor identities."""
    for module in model.modules():
        if isinstance(module, Normalize):
            module.mean.zero_()
            module.std.fill_(1.0)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()
    model.whiten.bias.zero_()
    for module in model.modules():
        if isinstance(module, Conv):
            dct_init_(module.weight)


def dct_basis(size: int) -> torch.Tensor:
    """[size*size, size, size] orthonormal 2-D DCT-II basis patterns."""
    n = torch.arange(size, dtype=torch.float64)
    c = torch.cos(math.pi * (n[None, :] + 0.5) * n[:, None] / size)
    c[0] /= math.sqrt(2)
    c *= math.sqrt(2 / size)
    return torch.einsum("ui,vj->uvij", c, c).reshape(size * size, size, size).float()


@torch.no_grad()
def dct_init_(weight: torch.Tensor) -> None:
    """Initialise the rows after the identity block as 2-D DCT patterns times random
    orthonormal channel mixing, at the default row norm (seeded per trial, not learned)."""
    cin, kh = weight.size(1), weight.size(2)
    rest = weight[cin:]
    n = rest.size(0)
    if n == 0:
        return
    device = weight.device
    target = rest.float().flatten(1).norm(dim=1).mean()
    basis = dct_basis(kh).to(device)
    mix = torch.empty(n, cin, device=device)
    nn.init.orthogonal_(mix)
    pattern = basis[torch.arange(n, device=device) % len(basis)]
    new = mix[:, :, None, None] * pattern[:, None]
    new = new / new.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1) * target
    rest.copy_(new.to(rest))


@torch.no_grad()
def init_whitening(model: Net, images: torch.Tensor, eps: float = 5e-4) -> None:
    """Set the frozen whitening conv to +/- scaled eigenpatches of normalised training images."""
    layer = model.whiten
    x = model.normalize(images, torch.float32)
    c, (h, w) = x.shape[1], layer.weight.shape[2:]
    patches = x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(len(x), -1, c * h * w)
    # Per-image partial sums: one long-K GEMM is much slower on CUDA.
    products = torch.bmm(patches.transpose(1, 2), patches)
    covariance = products.sum(0) / (patches.size(0) * patches.size(1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.copy_(torch.cat((scaled, -scaled)).to(layer.weight.dtype))
