"""CIFAR-100 speedrun recipe: an airbench-style network trained from scratch.

Adapted from Keller Jordan's airbench (https://github.com/KellerJordan/cifar10-airbench),
Copyright (c) 2024 Keller Jordan, released under the MIT License. Changes: 100-class
head, widths 64/256/768 with three convolutions per group and residual pairs through 192
(group 2) and 512 (group 3) channels, group 1's pair as two 1x1 convolutions and group 2's as a
3x3 then a 1x1, a 2x2 kernel for group 3's first convolution, max + mean global pooling, label
smoothing 0.25, an 11.5-epoch schedule with 20 px and 24 px phases, half-precision BatchNorm, a
compiled forward+loss step, the harness build/prepare/train split, and no test-time
augmentation. The optional Muon optimizer follows hiverge/cifar10-speedrun (MIT); see
LICENSE.hiverge.

Untimed build() compiles the network and warms up every kernel on synthetic data.
Timed prepare() resets all learned state, moves the images to the GPU, normalizes
them, and initializes the patch-whitening layer from training images.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

# Override any value with --params, e.g. '{"epochs": 9, "widths": [128, 384, 768]}'.
DEFAULTS = {
    "epochs": 11.5,
    "batch_size": 1024,
    "lr": 11.5,  # per 1024 examples, decoupled from momentum (airbench convention)
    "momentum": 0.85,
    "weight_decay": 0.017,  # per 1024 examples, decoupled from the learning rate
    "bias_scaler": 32.0,  # learning-rate multiplier for BatchNorm biases
    "label_smoothing": 0.25,
    "warmup": 0.23,  # fraction of steps spent ramping the learning rate up
    "final_lr": 0.07,  # learning-rate multiplier reached at the last step
    "whiten_bias_epochs": 3,
    "translate": 2,
    "cutout": 0,
    "widths": [64, 256, 768],
    "depth": 3,  # convs per group; the third adds a residual connection
    "depths": [3, 3, 3],  # per-group conv count; overrides depth
    "train_resolution": 32,  # reduced resolution for the first training stage
    "resolution_switch": 0.5,  # fraction of steps before returning to 32 pixels
    # Multi-stage schedule, e.g. [[24, 0.33], [28, 0.67]]: resolution until that
    # fraction of steps, then 32. Overrides train_resolution/resolution_switch.
    "resolution_schedule": [[20, 0.15], [24, 0.5]],
    # Batch-size schedule, e.g. [[512, 0.5]]: batch size until that fraction of the
    # training examples, then batch_size. Chosen per epoch. Weight decay per step
    # scales with the batch so the per-example decay is unchanged.
    "batch_schedule": [],
    "crop_mode": "indexed",  # one gather per epoch; "masked" is airbench's 25 masked copies
    "fused_sgd": True,  # one fused CUDA kernel for the SGD step
    "compile_loss": False,
    "hard_fraction": 1.0,  # <1 enables a freshly trained small proxy
    "proxy_widths": [32, 64, 128],
    "proxy_every": 4,  # proxy backward/update period; scores every batch
    "proxy_mode": "offline",  # airbench-style prepass, or online selection
    "gelu_approximate": "none",  # "tanh" uses a cheaper approximation
    "autotune_backends": "ATEN,TRITON",  # ATen/cuDNN and Inductor Triton candidates
    "pool_first": [False, False, False],  # move selected group pools before conv1
    "scaling_factor": 1.25 / 9,
    "bn_momentum": 0.5,
    "ema_every": 5,  # lookahead EMA period in steps; 0 disables it
    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    "compile_step": True,  # compile forward and loss as one graph
    "activation": "gelu",  # or "silu"
    "bn_dtype": "half",  # BatchNorm in the network dtype; "float" keeps fp32 BN
    "color_jitter": [0.2, 0.2],  # per-image brightness and contrast ranges
    "inductor_tuning": [],  # e.g. ["coordinate_descent_tuning", "aggressive_fusion"]
    "bn_recal_batches": 0,  # re-estimate BN statistics on center crops after training
    "stem": "patch2",  # "patch2": 2x2 whitening at 31x31; "patch4s2": 4x4 stride-2 at 15x15
    # Kernel size of conv2 / conv3 in each group: 1 or 3 for both, or [k2, k3] (each 1 or 3) to
    # mix them. Group 1's pair (64 -> 64 -> 64, on the 15x15 map) is two 1x1 convolutions and group
    # 2's (256 -> 192 -> 256) a 3x3 followed by a 1x1: the 1x1 has a ninth of the 3x3's FLOPs.
    "inner_kernels": [[1, 1], [3, 1], 3],
    # Kernel size of each group's first convolution (1, 2 or 3). Group 3's (256 -> 768 on the 7x7
    # map, the network's largest convolution) is a 2x2 kernel: 4/9 of the 3x3's FLOPs, with
    # PyTorch's asymmetric 'same' padding so the map stays 7x7.
    "conv1_kernels": [3, 3, 2],
    # Group 3's residual pair (the two convs after conv1; the residual and the group output keep
    # widths[2] channels): "full" = two 3x3 convs at widths[2]; "inner512" / "inner640" = the
    # 3x3 pair through 512 / 640 channels; "bottleneck384" = 1x1 -> 384, 3x3 at 384, 1x1 back.
    "g3_pair": "inner512",
    # Group 2's residual pair, same scheme: "full" (256 -> 256 -> 256) or "inner192" (through 192).
    "g2_pair": "inner192",
    # "max" is max(dim).values; adaptive_max_pool2d's backward uses slow atomics and
    # "amax" trains to NaN under torch.compile in torch 2.4.
    # "maxmean_sum": per-channel max plus mean over positions (same width as "max");
    # "fullpool_avgsum": the same max + mean through the pooling kernels (F.max_pool2d /
    # F.avg_pool2d over the whole map), 0.13 s per trial faster than the compiled reductions.
    "global_pool": "fullpool_avgsum",
    # "muon" follows hiverge/cifar10-speedrun: Muon on conv filters, SGD on biases and head.
    "optimizer": "sgd",
    "muon_lr": 0.205,
    "muon_momentum": 0.655,
    "muon_wd": 1.04e-6,  # per example
    "bias_lr": 0.0573,
    "head_lr": 0.5415,
}


# g3_pair variants: (kind, inner channels); None keeps the full-width pair.
G3_PAIRS = {
    "full": None,
    "inner512": ("inner", 512),
    "inner640": ("inner", 640),
    "bottleneck384": ("bottleneck", 384),
}
G2_PAIRS = {"full": None, "inner192": ("inner", 192)}


#############################################
#                  Network                  #
#############################################


def make_activation(hyp):
    if hyp["activation"] == "silu":
        return F.silu
    approximate = hyp["gelu_approximate"]
    return lambda x: F.gelu(x, approximate=approximate)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum):
        super().__init__(num_features, eps=1e-12, momentum=1 - momentum)
        self.weight.requires_grad = False

    def activate(self, x, act, residual=None):
        x = self(x)
        if residual is not None:
            x = x + residual
        return act(x)


class Conv(nn.Conv2d):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super().__init__(
            channels_in, channels_out, kernel_size=kernel_size, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        # Same result as nn.init.dirac_(w[:cin]) without its per-channel Python loop
        # (one kernel launch per channel, about 60 ms per reset across the network).
        w = self.weight.data
        cin, (kh, kw) = w.size(1), w.shape[2:]
        n = min(w.size(0), cin)
        w[:cin].zero_()
        idx = torch.arange(n, device=w.device)
        w[idx, idx, kh // 2, kw // 2] = 1


class ConvGroup(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        depth,
        bn_momentum,
        act,
        pool_first,
        kernel=3,
        pool=True,
        pair=None,
        conv1_kernel=3,
    ):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out, conv1_kernel)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.pool_first = pool_first
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        # inner_kernels: one kernel size for conv2 and conv3, or a [k2, k3] pair.
        k2, k3 = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        if pair is None:
            self.conv2 = Conv(channels_out, channels_out, k2)
            self.norm2 = BatchNorm(channels_out, bn_momentum)
            self.conv3 = Conv(channels_out, channels_out, k3) if depth == 3 else None
            self.norm3 = BatchNorm(channels_out, bn_momentum) if depth == 3 else None
        else:
            # A cheaper residual pair (g3_pair): the residual and the output keep channels_out.
            kind, inner = pair
            if kind == "bottleneck":
                self.conv2a = Conv(channels_out, inner, 1)
                self.norm2a = BatchNorm(inner, bn_momentum)
                self.conv2 = Conv(inner, inner, k2)
                self.norm2 = BatchNorm(inner, bn_momentum)
                self.conv3 = Conv(inner, channels_out, 1)
            else:
                self.conv2 = Conv(channels_out, inner, k2)
                self.norm2 = BatchNorm(inner, bn_momentum)
                self.conv3 = Conv(inner, channels_out, k3)
            self.norm3 = BatchNorm(channels_out, bn_momentum)
        self.act = act

    def forward(self, x):
        x = self.conv1(self.pool(x)) if self.pool_first else self.pool(self.conv1(x))
        x = self.norm1.activate(x, self.act)
        if self.conv3 is None:
            return self.norm2.activate(self.conv2(x), self.act)
        x0 = x
        conv2a = getattr(self, "conv2a", None)
        if conv2a is not None:  # bottleneck pair: 1x1 reduction before the 3x3
            x = self.norm2a.activate(conv2a(x), self.act)
        x = self.norm2.activate(self.conv2(x), self.act)
        return self.norm3.activate(self.conv3(x), self.act, x0)


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        depths = hyp["depths"] or [hyp["depth"]] * 3
        bn_momentum = hyp["bn_momentum"]
        self.act = make_activation(hyp)
        # The stride-2 stem whitens 4x4 patches straight to 15x15, so the first
        # group skips its pool and never runs a conv at 31x31.
        patch, self.whiten_stride = (4, 2) if hyp["stem"] == "patch4s2" else (2, 1)
        stem_width = 2 * 3 * patch * patch
        self.whiten = nn.Conv2d(3, stem_width, kernel_size=patch, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        pairs = (None, G2_PAIRS[hyp["g2_pair"]], G3_PAIRS[hyp["g3_pair"]])
        groups = []
        for index, (c_in, c_out, depth, pool_first, kernel, pool, conv1_kernel) in enumerate(
            zip(
                (stem_width, w1, w2),
                (w1, w2, w3),
                depths,
                hyp["pool_first"],
                hyp["inner_kernels"],
                (self.whiten_stride == 1, True, True),
                hyp["conv1_kernels"],
                strict=True,
            )
        ):
            groups.append(
                ConvGroup(
                    c_in,
                    c_out,
                    depth,
                    bn_momentum,
                    self.act,
                    pool_first,
                    kernel,
                    pool,
                    pair=pairs[index],
                    conv1_kernel=conv1_kernel,
                )
            )
        self.layers = nn.Sequential(*groups)
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scaling_factor = hyp["scaling_factor"]
        self.muon_head = hyp["optimizer"] == "muon"
        self.global_pool = hyp["global_pool"]

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.BatchNorm2d | nn.Linear):
                m.reset_parameters()
        self.whiten.bias.data.zero_()
        if self.muon_head:
            self.head.weight.data /= self.head.weight.data.std()

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        c, (h, w) = images.shape[1], self.whiten.weight.shape[2:]
        patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w)
        flat = patches.float().view(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance, UPLO="U")
        scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.copy_(torch.cat((scaled, -scaled)))

    def forward(self, x, whiten_bias_grad: bool = True):
        b = self.whiten.bias
        b = b if whiten_bias_grad else b.detach()
        x = self.act(F.conv2d(x, self.whiten.weight, b, stride=self.whiten_stride))
        x = self.layers(x)
        if self.global_pool == "adaptive":
            x = F.adaptive_max_pool2d(x, 1).flatten(1)
        elif self.global_pool == "amax":
            x = x.flatten(2).amax(2)
        elif self.global_pool == "maxmean_sum":
            flat = x.flatten(2)
            x = flat.max(2).values + flat.mean(2)
        elif self.global_pool == "fullpool_avgsum":
            # max and mean through the pooling kernels: 0.13 s per trial faster than the
            # reduction kernels torch.compile generates for max(dim).values and mean.
            flat = x.flatten(2)
            x4, kernel = flat.view(flat.shape[0], flat.shape[1], -1, 1), (flat.shape[2], 1)
            x = (F.max_pool2d(x4, kernel) + F.avg_pool2d(x4, kernel)).flatten(1)
        else:
            x = x.flatten(2).max(2).values
        if self.muon_head:
            return self.head(x) / x.size(-1)
        return self.head(x) * self.scaling_factor


class Classifier(nn.Module):
    """Evaluation wrapper: harness inputs are float32 RGB in [0, 1]."""

    def __init__(self, net, dtype):
        super().__init__()
        self.net = net
        self.dtype = dtype
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.ones(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = ((x - self.mean) / self.std).to(self.dtype, memory_format=torch.channels_last)
        return self.net(x).float()


#############################################
#                   Muon                    #
#############################################


def newton_schulz(G):
    """Approximately orthogonalize a stack of matrices [B, D, K] with D <= K."""
    a, b, c = (3.4576, -4.7391, 2.0843)
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-5)
    for _ in range(3):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


class Muon(torch.optim.Optimizer):
    """Nesterov momentum and orthogonalized updates for conv filters (off by default).

    Adapted from hiverge/cifar10-speedrun (MIT, see LICENSE.hiverge): Newton-Schulz
    coefficients, periodic norm reset and decoupled weight decay.

    Filters with the same shape are stacked so each shape needs one batched
    Newton-Schulz call. Filter norms are reset to sqrt(out_channels) every few
    steps, with the interval growing over training as in hiverge/cifar10-speedrun.
    """

    def __init__(self, params, lr, momentum, weight_decay, total_steps, zeropower):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        buckets = {}
        for p in self.param_groups[0]["params"]:
            buckets.setdefault(tuple(p.shape), []).append(p)
        self.buckets = list(buckets.values())
        self.buffers = [[torch.zeros_like(p) for p in ps] for ps in self.buckets]
        self.total_steps = total_steps
        self.zeropower = zeropower
        self.steps = 0
        self.last_norm = 0

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        lr, momentum = group["lr"], group["momentum"]
        self.steps += 1
        renorm = self.steps - self.last_norm >= 2 + int(15 * self.steps / self.total_steps)
        if renorm:
            self.last_norm = self.steps
        for params, buffers in zip(self.buckets, self.buffers):
            grads = [p.grad for p in params]
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)
            updates = torch._foreach_add(grads, buffers, alpha=momentum)
            G = torch.stack(updates).flatten(2)
            transposed = G.size(1) > G.size(2)
            U = self.zeropower(G.mT if transposed else G)
            U = (U.mT if transposed else U).reshape(len(params), *params[0].shape)
            if renorm:
                norms = torch._foreach_norm(params)
                torch._foreach_mul_(
                    params, [len(p) ** 0.5 / (n + 1e-7) for p, n in zip(params, norms)]
                )
            torch._foreach_add_(params, list(U.to(params[0].dtype).unbind(0)), alpha=-lr)
            if group["weight_decay"]:
                torch._foreach_mul_(params, 1 - lr * group["weight_decay"])


#############################################
#               Augmentation                #
#############################################


def color_jitter(images, brightness, contrast):
    n = len(images)
    shift = (torch.rand(n, 1, 1, 1, device=images.device, dtype=images.dtype) * 2 - 1) * brightness
    scale = (torch.rand(n, 1, 1, 1, device=images.device, dtype=images.dtype) * 2 - 1) * contrast
    return (images + shift) * (scale + 1)


def batch_flip_lr(images):
    flip = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip, images.flip(-1), images)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            out[mask] = tmp[mask, :, :, r + s : r + s + crop_size]
    return out


def batch_cutout(images, size):
    n, _, h, w = images.shape
    y = torch.randint(0, h - size + 1, size=(n, 1, 1, 1), device=images.device)
    x = torch.randint(0, w - size + 1, size=(n, 1, 1, 1), device=images.device)
    rows = torch.arange(h, device=images.device).view(1, 1, h, 1) - y
    cols = torch.arange(w, device=images.device).view(1, 1, 1, w) - x
    mask = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    return images.masked_fill(mask, 0)


def indexed_crop(images, crop_size):
    """One gather per epoch, with the same uniform crop distribution as batch_crop."""
    n, _, h, w = images.shape
    shifts = torch.randint(0, w - crop_size + 1, (n, 2), device=images.device)
    batch = torch.arange(n, device=images.device).view(n, 1, 1)
    rows = shifts[:, 0, None, None] + torch.arange(crop_size, device=images.device).view(1, -1, 1)
    cols = shifts[:, 1, None, None] + torch.arange(crop_size, device=images.device).view(1, 1, -1)
    return images.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)


#############################################
#                 Interface                 #
#############################################


def _kernel_ok(k) -> bool:
    """inner_kernels entry: 1 or 3, or a [k2, k3] list of 1 or 3."""
    if type(k) is int:
        return k in (1, 3)
    return isinstance(k, list) and len(k) == 2 and all(type(v) is int and v in (1, 3) for v in k)


def _pair_kernel_ok(k) -> bool:
    """A narrowed residual pair needs 3x3 or mixed [k2, k3] inner kernels (not plain 1x1)."""
    return k == 3 or isinstance(k, list)


def build(context: BuildContext):
    hyp = {**DEFAULTS, **context.parameters}
    unknown = set(hyp) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    if len(hyp["widths"]) != 3 or any(w <= 0 for w in hyp["widths"]):
        raise ValueError("widths must contain three positive channel counts")
    depths = hyp["depths"] or [hyp["depth"]] * 3
    if len(depths) != 3 or any(d not in (2, 3) for d in depths):
        raise ValueError("depths must contain three values of 2 or 3")
    if hyp["train_resolution"] not in (24, 28, 32) or not 0 <= hyp["resolution_switch"] <= 1:
        raise ValueError(
            "train_resolution must be 24, 28, or 32; resolution_switch must be in [0, 1]"
        )
    if hyp["crop_mode"] not in ("masked", "indexed", "triton"):
        raise ValueError("crop_mode must be masked, indexed, or triton")
    if hyp["epochs"] <= 0 or hyp["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 < hyp["hard_fraction"] <= 1 or hyp["proxy_every"] < 1:
        raise ValueError("hard_fraction must be in (0, 1]; proxy_every must be positive")
    if hyp["proxy_mode"] not in ("online", "offline"):
        raise ValueError("proxy_mode must be online or offline")
    if hyp["gelu_approximate"] not in ("none", "tanh"):
        raise ValueError("gelu_approximate must be none or tanh")
    if hyp["autotune_backends"] not in ("ATEN", "TRITON", "ATEN,TRITON"):
        raise ValueError("autotune_backends must be ATEN, TRITON, or ATEN,TRITON")
    if len(hyp["pool_first"]) != 3 or any(type(v) is not bool for v in hyp["pool_first"]):
        raise ValueError("pool_first must contain three booleans")
    if hyp["activation"] not in ("gelu", "silu") or hyp["bn_dtype"] not in ("float", "half"):
        raise ValueError("activation must be gelu or silu; bn_dtype must be float or half")
    if not hyp["resolution_schedule"] and hyp["train_resolution"] < 32:
        hyp["resolution_schedule"] = [[hyp["train_resolution"], hyp["resolution_switch"]]]
    if any(r not in (16, 20, 24, 28) or not 0 <= f <= 1 for r, f in hyp["resolution_schedule"]):
        raise ValueError("resolution_schedule entries must be [16|20|24|28, fraction]")
    if len(hyp["inner_kernels"]) != 3 or not all(_kernel_ok(k) for k in hyp["inner_kernels"]):
        raise ValueError(
            "inner_kernels must contain three values of 1 or 3, or [k2, k3] pairs of 1 or 3"
        )
    if len(hyp["conv1_kernels"]) != 3 or any(
        type(k) is not int or k not in (1, 2, 3) for k in hyp["conv1_kernels"]
    ):
        raise ValueError("conv1_kernels must contain three values of 1, 2 or 3")
    if not isinstance(hyp["g3_pair"], str) or hyp["g3_pair"] not in G3_PAIRS:
        raise ValueError(f"g3_pair must be one of {sorted(G3_PAIRS)}")
    if hyp["g3_pair"] != "full" and (
        depths[2] != 3 or not _pair_kernel_ok(hyp["inner_kernels"][2])
    ):
        raise ValueError("g3_pair variants need group 3 at depth 3 with 3x3 inner kernels")
    if not isinstance(hyp["g2_pair"], str) or hyp["g2_pair"] not in G2_PAIRS:
        raise ValueError(f"g2_pair must be one of {sorted(G2_PAIRS)}")
    if hyp["g2_pair"] != "full" and (
        depths[1] != 3 or not _pair_kernel_ok(hyp["inner_kernels"][1])
    ):
        raise ValueError("g2_pair variants need group 2 at depth 3 with 3x3 inner kernels")
    if hyp["stem"] not in ("patch2", "patch4s2"):
        raise ValueError("stem must be patch2 or patch4s2")
    if any(b <= 0 or not 0 <= f <= 1 for b, f in hyp["batch_schedule"]):
        raise ValueError("batch_schedule entries must be [positive batch, fraction]")
    if hyp["global_pool"] not in ("adaptive", "amax", "max", "maxmean_sum", "fullpool_avgsum"):
        raise ValueError("global_pool must be adaptive, amax, max, maxmean_sum, or fullpool_avgsum")
    if hyp["optimizer"] not in ("sgd", "muon") or len(hyp["color_jitter"]) != 2:
        raise ValueError("optimizer must be sgd or muon; color_jitter needs two ranges")
    device = context.device
    cuda = device.type == "cuda"
    dtype = torch.float16 if cuda else torch.float32
    crop_kernel = None
    if cuda and hyp["crop_mode"] == "triton":
        from .kernels import crop_flip

        crop_kernel = crop_flip
    torch.backends.cudnn.benchmark = True
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = hyp["autotune_backends"]
    for flag in hyp["inductor_tuning"]:
        if flag not in ("coordinate_descent_tuning", "aggressive_fusion"):
            raise ValueError(f"Unsupported inductor_tuning flag: {flag}")
        setattr(inductor_config, flag, True)

    net = Net(hyp, context.num_classes).to(device, dtype, memory_format=torch.channels_last)
    if hyp["bn_dtype"] == "float":
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.float()
    compiled = cuda and hyp["compile"]
    # Static shapes: one graph per training resolution, all compiled in the warmup
    # below. Automatic dynamic shapes would recompile mid-trial at a new size.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
    train_net = (
        torch.compile(net, mode=hyp["compile"], dynamic=False)
        if compiled and not hyp["compile_step"]
        else net
    )

    def loss_fn(outputs, labels):
        return F.cross_entropy(
            outputs.float(), labels, label_smoothing=hyp["label_smoothing"], reduction="sum"
        )

    loss_fn = torch.compile(loss_fn) if cuda and hyp["compile_loss"] else loss_fn

    def forward_loss(inputs, labels, whiten_bias_grad: bool):
        return loss_fn(train_net(inputs, whiten_bias_grad), labels)

    if compiled and hyp["compile_step"]:
        forward_loss = torch.compile(forward_loss, mode=hyp["compile"], dynamic=False)
    float_state = [t for t in net.state_dict().values() if t.is_floating_point()]
    # Bilinear 32 -> r resize as two matmuls (A x A^T); F.interpolate on 50k fp16
    # channels-last images takes about 65 ms, the matmuls about 2 ms.
    resize = {}
    for resolution, _ in hyp["resolution_schedule"]:
        eye = torch.eye(32).view(1, 1, 32, 32)
        matrix = F.interpolate(eye, size=(resolution, 32), mode="bilinear", align_corners=False)
        resize[resolution] = matrix[0, 0].to(device, dtype)
    state = SimpleNamespace(
        hyp=hyp,
        device=device,
        dtype=dtype,
        resize=resize,
        net=net,
        train_net=train_net,
        loss_fn=loss_fn,
        forward_loss=forward_loss,
        zeropower=torch.compile(newton_schulz, dynamic=False) if compiled else newton_schulz,
        classifier=Classifier(net, dtype).to(device),
        float_state=float_state,
        ema=[t.clone() for t in float_state],
        proxy=None,
        crop_kernel=crop_kernel,
    )
    if hyp["hard_fraction"] < 1:
        proxy_hyp = {
            **hyp,
            "widths": hyp["proxy_widths"],
            "depths": [2, 2, 2],
            "g3_pair": "full",
            "g2_pair": "full",
        }
        proxy = Net(proxy_hyp, context.num_classes).to(
            device, dtype, memory_format=torch.channels_last
        )
        proxy.whiten.bias.requires_grad_(False)
        for m in proxy.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.float()
        state.proxy = proxy
        state.proxy_net = (
            torch.compile(proxy, mode=hyp["compile"]) if cuda and hyp["compile"] else proxy
        )

    # Untimed warmup on random synthetic images: compiles both whitening-bias graphs,
    # autotunes cuDNN, initializes cuBLAS/cuSOLVER, and warms evaluation shapes.
    # prepare() in each trial resets everything this changes.
    if cuda:
        count = 50_000
        synthetic = TrainingData(
            torch.randint(0, 256, (count, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (count,)),
        )
        batches = sorted({b for b, _ in hyp["batch_schedule"]} | {hyp["batch_size"]})
        resolutions = sorted({r for r, _ in hyp["resolution_schedule"]} | {32})
        for batch in batches:
            for resolution in resolutions:
                state.warmup_batch, state.warmup_resolution = batch, resolution
                for _ in range(2):
                    prepare(state, synthetic, seed=0)
                    state.whiten_bias_steps = 3
                    _fit(state, total_steps=6)
        del state.warmup_batch, state.warmup_resolution
        state.classifier.eval()
        with torch.inference_mode():
            for size in (context.eval_batch_size, 10_000 % context.eval_batch_size, 1):
                state.classifier(torch.rand(size, 3, 32, 32, device=device))
        state.classifier.train()
        torch.cuda.synchronize()
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    """Timed: reset every learned value and stage the training data on the GPU."""
    hyp, net, device = state.hyp, state.net, state.device
    net.reset()
    net.zero_grad(set_to_none=True)
    net.train()

    raw = data.images.to(device, non_blocking=True).float().div_(255)
    mean = raw.mean(dim=(0, 2, 3), keepdim=True)
    std = raw.std(dim=(0, 2, 3), keepdim=True)
    state.classifier.mean.copy_(mean)
    state.classifier.std.copy_(std)
    images = ((raw - mean) / std).to(state.dtype, memory_format=torch.channels_last)
    del raw
    net.init_whiten(images[:5000])
    if state.proxy is not None:
        state.proxy.reset()
        state.proxy.zero_grad(set_to_none=True)
        state.proxy.train()
        state.proxy.whiten.weight.data.copy_(net.whiten.weight)
    torch._foreach_copy_(state.ema, state.float_state)

    # Alternating flip: flip a random half once, then mirror everything on odd epochs.
    images = batch_flip_lr(images)
    state.small_images = {}
    for resolution, _ in hyp["resolution_schedule"]:
        matrix = state.resize[resolution]
        small = torch.matmul(torch.matmul(matrix, images), matrix.T)
        if hyp["translate"]:
            small = F.pad(small, (hyp["translate"],) * 4, "reflect")
        state.small_images[resolution] = small.to(memory_format=torch.channels_last)
    if hyp["translate"]:
        images = F.pad(images, (hyp["translate"],) * 4, "reflect")
    state.images = images
    state.labels = data.labels.to(device, non_blocking=True)

    batch_size = min(hyp["batch_size"], len(data.labels))
    momentum = hyp["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    state.batch_size = batch_size
    state.masks = []
    state.steps_per_epoch = len(data.labels) // batch_size
    state.total_steps = math.ceil(hyp["epochs"] * state.steps_per_epoch)
    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)
    if hyp["optimizer"] == "muon":
        state.optimizers = _make_muon(net, batch_size, state.total_steps, hyp, state)
    else:
        sgd_wd = wd * hyp["hard_fraction"]
        state.optimizers = [_make_optimizer(net, lr, lr_biases, sgd_wd, hyp, device)]
    if state.proxy is not None:
        state.proxy_optimizer = _make_optimizer(state.proxy, lr, lr_biases, wd, hyp, device)


def _make_optimizer(net, lr, lr_biases, wd, hyp, device):
    norm_biases = [p for name, p in net.named_parameters() if "norm" in name and p.requires_grad]
    others = [p for name, p in net.named_parameters() if "norm" not in name and p.requires_grad]
    optimizer = torch.optim.SGD(
        [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=others, lr=lr, weight_decay=wd / lr),
        ],
        momentum=hyp["momentum"],
        nesterov=True,
        fused=hyp["fused_sgd"] and device.type == "cuda",
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["initial_weight_decay"] = group["weight_decay"]
    return optimizer


def _make_muon(net, batch_size, total_steps, hyp, state):
    wd = hyp["muon_wd"] * batch_size
    bias_lr, head_lr = hyp["bias_lr"], hyp["head_lr"]
    norm_biases = [p for name, p in net.named_parameters() if "norm" in name and p.requires_grad]
    filters = [p for p in net.parameters() if p.ndim == 4 and p.requires_grad]
    sgd = torch.optim.SGD(
        [
            dict(params=[net.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr, whiten=True),
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=[net.head.weight], lr=head_lr, weight_decay=wd / head_lr),
        ],
        momentum=hyp["momentum"],
        nesterov=True,
        fused=hyp["fused_sgd"] and state.device.type == "cuda",
    )
    muon = Muon(filters, hyp["muon_lr"], hyp["muon_momentum"], wd, total_steps, state.zeropower)
    for optimizer in (sgd, muon):
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return [sgd, muon]


def train(state) -> nn.Module:
    if state.proxy is not None and state.hyp["proxy_mode"] == "offline":
        # Replay the same augmentation/order stream for the main net. Only masks
        # persist within this trial; prepare discards them before the next one.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(state.device) if state.device.type == "cuda" else None
        _fit(state, state.total_steps, proxy_only=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, state.device)
    _fit(state, state.total_steps)
    return state.classifier


def _fit(state, total_steps, proxy_only=False):
    hyp, net, optimizers = state.hyp, state.net, state.optimizers
    labels, base_batch = state.labels, state.batch_size
    # Schedules run on progress = examples seen / total examples, so a batch-size
    # schedule changes the step count but not the learning-rate curve. With a
    # constant batch this equals step / total_steps exactly.
    total_examples = total_steps * base_batch
    warmup_frac = int(total_steps * hyp["warmup"]) / total_steps
    whiten_frac = state.whiten_bias_steps / total_steps
    ema_scale = 0.95**5
    step = 0
    seen = 0
    net.train()
    epoch = 0
    while seen < total_examples:
        progress = seen / total_examples
        batch_size = getattr(state, "warmup_batch", None) or next(
            (b for b, end in hyp["batch_schedule"] if progress < end), base_batch
        )
        batch_size = min(batch_size, len(labels))
        if batch_size != base_batch or hyp["batch_schedule"]:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    if "initial_weight_decay" in group:
                        group["weight_decay"] = (
                            group["initial_weight_decay"] * batch_size / base_batch
                        )
        steps_per_epoch = len(labels) // batch_size
        images = None
        current_resolution = None
        order = torch.randperm(len(labels), device=labels.device)
        for i in range(steps_per_epoch):
            if seen >= total_examples:
                break
            progress = seen / total_examples
            resolution = getattr(state, "warmup_resolution", None)
            if resolution is None:
                resolution = next(
                    (r for r, end in hyp["resolution_schedule"] if progress < end), 32
                )
            if resolution != current_resolution:
                source = state.small_images[resolution] if resolution < 32 else state.images
                if state.crop_kernel is not None:
                    images = state.crop_kernel(source, resolution, flip=epoch % 2 == 1)
                else:
                    crop = batch_crop if hyp["crop_mode"] == "masked" else indexed_crop
                    images = crop(source, resolution) if hyp["translate"] else source
                if epoch % 2 == 1 and state.crop_kernel is None:
                    images = images.flip(-1)
                if hyp["cutout"]:
                    images = batch_cutout(images, hyp["cutout"])
                if any(hyp["color_jitter"]):
                    images = color_jitter(images, *hyp["color_jitter"])
                current_resolution = resolution
            idx = order[i * batch_size : (i + 1) * batch_size]
            inputs, targets = images[idx], labels[idx]
            offline_main = (
                state.proxy is not None
                and hyp["proxy_mode"] == "offline"
                and not proxy_only
                and not hasattr(state, "warmup_resolution")
            )
            if offline_main:
                chosen = state.masks[step]
                inputs, targets = inputs[chosen], targets[chosen]
            elif state.proxy is not None:
                update_proxy = step % hyp["proxy_every"] == 0
                with torch.set_grad_enabled(update_proxy):
                    proxy_outputs = state.proxy_net(inputs, False)
                    proxy_losses = F.cross_entropy(
                        proxy_outputs.float(),
                        targets,
                        reduction="none",
                        label_smoothing=hyp["label_smoothing"],
                    )
                kept = max(1, int(batch_size * hyp["hard_fraction"]))
                chosen = proxy_losses.detach().topk(kept, sorted=False).indices
                if update_proxy:
                    state.proxy_optimizer.zero_grad(set_to_none=True)
                    proxy_losses[chosen].sum().backward()
                    proxy_frac = progress
                    proxy_scale = min(1.0, 0.2 + 8 * proxy_frac) * (1 - proxy_frac)
                    for group in state.proxy_optimizer.param_groups:
                        group["lr"] = group["initial_lr"] * proxy_scale
                    state.proxy_optimizer.step()
                inputs, targets = inputs[chosen], targets[chosen]
                if proxy_only:
                    state.masks.append(chosen)
                    step += 1
                    seen += batch_size
                    continue
            loss = state.forward_loss(inputs, targets, progress < whiten_frac)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if progress < warmup_frac:
                frac = progress / warmup_frac
                scale = 0.2 * (1 - frac) + frac
            else:
                frac = (progress - warmup_frac) / max(1e-9, 1 - warmup_frac)
                scale = (1 - frac) + hyp["final_lr"] * frac
            whiten_scale = max(0.0, 1 - progress / max(1e-9, whiten_frac))
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * (
                        whiten_scale if "whiten" in group else scale
                    )
                optimizer.step()
            step += 1
            seen += batch_size
            if hyp["ema_every"] and step % hyp["ema_every"] == 0:
                _lookahead(state, ema_scale * min(1.0, seen / total_examples) ** 3)
        epoch += 1
    if hyp["ema_every"] and not proxy_only:
        _lookahead(state, 1.0)
    if hyp["bn_recal_batches"] and not proxy_only:
        _recalibrate_bn(state)


@torch.no_grad()
def _recalibrate_bn(state):
    """Re-estimate BatchNorm running statistics on unaugmented training images.

    Training batches are translated and flipped; evaluation sees plain 32x32
    images. A few forward passes over center crops align the statistics with
    that. This runs inside train() and uses only training images.
    """
    net, t = state.net, state.hyp["translate"]
    norms = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the recalibration batches
    images = state.images[:, :, t : t + 32, t : t + 32] if t else state.images
    batch = state.batch_size
    order = torch.randperm(len(images), device=images.device)
    net.train()
    for i in range(min(state.hyp["bn_recal_batches"], len(images) // batch)):
        net(images[order[i * batch : (i + 1) * batch]], False)
    for m in norms:
        m.momentum = 1 - state.hyp["bn_momentum"]


@torch.no_grad()
def _lookahead(state, decay):
    torch._foreach_lerp_(state.ema, state.float_state, 1 - decay)
    torch._foreach_copy_(state.float_state, state.ema)
