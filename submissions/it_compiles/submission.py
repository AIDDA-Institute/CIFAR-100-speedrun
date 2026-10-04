"""CIFAR-100 speedrun: an airbench-style CNN trained with Muon and progressive resolution.

Ported from Keller Jordan's airbench94_muon.py (https://github.com/KellerJordan/cifar10-airbench),
with these changes, all measured on CIFAR-100:
- Three conv groups of depth 3 with a residual connection (airbench96-style), widths
  128-512-512.
- Output-layer (head) LR 3.0 instead of 0.67: the 100-class head needs a much larger step.
- LR schedule: 5% warmup, hold at peak for 30% of training, then linear decay to zero.
- Label smoothing 0.3, ±1 px translation, flips.
- Progressive resolution: 2 epochs at 16x16, 1 at 20x20 and 2 at 24x24 (area-downsampled
  crops, ~0.25x / 0.39x / 0.56x the compute per step), then 3 epochs at 32x32: 200 steps
  under one LR schedule.
- Global max pooling via flatten + max (same as AdaptiveMaxPool2d(1), faster backward).
- No test-time augmentation: plain single-view inference on float32 [0, 1] inputs.
- torch.compile (max-autotune) with a synthetic warmup of every graph in build(), which is
  untimed; prepare() resets weights, BatchNorm stats and optimizers and fits the whitening
  layer on real data (timed).

Development results (A100 80GB): 75.30% mean test accuracy over 80 seeds; 6.22 s mean
training time on A100 PCIe (40 seeds).
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

DEFAULTS = {
    # [resolution, epochs] phases in order, under one LR schedule.
    "schedule": [[16, 2], [20, 1], [24, 2], [32, 3]],
    "batch_size": 2000,
    "widths": [128, 512, 512],
    "depth": 3,  # convs per group; the third adds a residual connection
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    "bias_lr": 0.053,
    "head_lr": 3.0,
    "sgd_momentum": 0.85,
    "weight_decay": 2e-6,  # multiplied by batch_size, as in airbench
    "label_smoothing": 0.3,
    "translate": 1,  # random translation of ±1 px
    "whiten_bias_epochs": 3,
    # LR for conv filters / head / BN biases: linear warmup, hold at peak, then linear decay
    # to 0, as fractions of total steps.
    "lr_warmup_frac": 0.05,
    "lr_hold_frac": 0.3,
    "bn_momentum": 0.6,
}

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


#############################################
#               Muon optimizer              #
#############################################


def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Approximately orthogonalise G with a quintic Newton-Schulz iteration (see airbench)."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, nesterov=True, zeropower=zeropower_via_newtonschulz5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.zeropower = zeropower

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.mul_(len(p) ** 0.5 / p.norm())  # normalise the weight
                update = self.zeropower(g.reshape(len(g), -1)).view(g.shape)
                p.add_(update, alpha=-lr)


#############################################
#            Network Definition             #
#############################################


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        # Identity init on the first in_channels filters. Same result as nn.init.dirac_, whose
        # per-channel Python loop cost ~54 ms per trial across all convs (timed in prepare).
        w = self.weight.data
        n = min(w.size(0), w.size(1))
        w[:n].zero_()
        idx = torch.arange(n, device=w.device)
        w[idx, idx, w.size(2) // 2, w.size(3) // 2] = 1


class ConvGroup(nn.Module):
    """conv -> pool -> BN -> GELU, then depth-1 more conv/BN/GELU layers, with a residual
    connection around those when depth >= 3 (airbench96)."""

    def __init__(self, channels_in, channels_out, bn_momentum, depth):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.convs = nn.ModuleList(Conv(channels_out, channels_out) for _ in range(depth - 1))
        self.norms = nn.ModuleList(BatchNorm(channels_out, bn_momentum) for _ in range(depth - 1))
        self.residual = depth >= 3
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x0 = x
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = self.activ(norm(conv(x)))
        return x + x0 if self.residual else x


class GlobalMaxPool(nn.Module):
    """Global max over H, W. Same forward and gradient as AdaptiveMaxPool2d(1) + flatten, but
    its backward avoids that op's slow atomic-scatter kernel."""

    def forward(self, x):
        return x.flatten(2).max(dim=2).values


class CifarNet(nn.Module):
    def __init__(self, widths, num_classes, bn_momentum, dtype, depth):
        super().__init__()
        self.dtype = dtype
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0], bn_momentum, depth),
            ConvGroup(widths[0], widths[1], bn_momentum, depth),
            ConvGroup(widths[1], widths[2], bn_momentum, depth),
            GlobalMaxPool(),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.register_buffer("mean", torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR100_STD).view(1, 3, 1, 1))
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            elif mod is not self:
                mod.to(dtype)

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    @torch.no_grad()
    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3)
        patches_flat = patches.reshape(-1, c * h * w).float()
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def normalize(self, x):
        """float [0, 1] NCHW -> normalised, model dtype, channels_last."""
        x = (x - self.mean) / self.std
        return x.to(self.dtype).contiguous(memory_format=torch.channels_last)

    def forward(self, x, whiten_bias_grad=True, preprocessed=False):
        # The harness passes raw float32 [0, 1] images; training passes preprocessed ones.
        if not preprocessed:
            x = self.normalize(x)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return (self.head(x) / x.size(-1)).float()


#############################################
#                Data / aug                 #
#############################################


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    """Random crops of size crop_size from images padded by r <= 2 pixels per side."""
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            images_out[mask] = images[
                mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
            ]
    return images_out


def epoch_images(state, epoch, res):
    """Airbench loader: pre-flipped, pre-padded images; fresh crops each epoch, and all
    images flipped together every other epoch (more diverse than independent flips).
    Downsampled (area) to res x res for the low-resolution phases."""
    images = batch_crop(state.padded_images, state.images.shape[-2])
    if epoch % 2 == 1:
        images = images.flip(-1)
    if res != images.size(-1):
        images = F.interpolate(images, size=(res, res), mode="area")
    return images


#############################################
#            Harness entry points           #
#############################################


def make_optimizers(state):
    cfg, model = state.cfg, state.model
    wd = cfg["weight_decay"] * cfg["batch_size"]
    bias_lr, head_lr = cfg["bias_lr"], cfg["head_lr"]
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    fused = state.device.type == "cuda"
    sgd = torch.optim.SGD(param_configs, momentum=cfg["sgd_momentum"], nesterov=True, fused=fused)
    muon = Muon(filter_params, lr=cfg["muon_lr"], momentum=cfg["muon_momentum"],
                zeropower=state.zeropower)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, muon


def build(context: BuildContext):
    cfg = {**DEFAULTS, **context.parameters}
    device = context.device
    cuda = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if cuda else torch.float32
    model = CifarNet(cfg["widths"], context.num_classes, cfg["bn_momentum"], dtype, cfg["depth"])
    model = model.to(device, memory_format=torch.channels_last)
    zeropower = zeropower_via_newtonschulz5
    if cuda:
        # Static shapes: one specialised graph per resolution, never a dynamic-shape graph.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        model.compile(mode="max-autotune", dynamic=False)
        zeropower = torch.compile(zeropower_via_newtonschulz5)
    state = SimpleNamespace(
        model=model, context=context, cfg=cfg, device=device, zeropower=zeropower
    )
    if cuda:
        warmup(state)
    return state


def warmup(state):
    """Compile every graph on synthetic data (untimed). prepare() resets all of it.

    Runs the real prepare/train code so tensor dtypes, strides and flags match exactly;
    any mismatch (e.g. memory layout) would make torch.compile recompile inside the timed
    trial. Each resolution trains long enough to cross the whitening-bias cutoff, so both
    of its graphs compile.
    """
    cfg, model, device = state.cfg, state.model, state.device
    bs = cfg["batch_size"]
    generator = torch.Generator().manual_seed(0)
    synthetic = TrainingData(
        torch.randint(0, 256, (2 * bs, 3, 32, 32), dtype=torch.uint8, generator=generator),
        torch.randint(0, state.context.num_classes, (2 * bs,), generator=generator),
    )
    epochs = cfg["whiten_bias_epochs"] + 1
    for res in sorted({res for res, _ in cfg["schedule"]}, reverse=True):
        state.cfg = {**cfg, "schedule": [[res, epochs]]}
        prepare(state, synthetic, 0)
        train(state)
    state.cfg = cfg
    # Evaluation graphs: full batches, the final 10000 % 1024 batch, and tiny batches.
    model.eval()
    eval_bs = state.context.eval_batch_size
    with torch.inference_mode():
        for n in (eval_bs, 10_000 % eval_bs or eval_bs, 1):
            model(torch.rand(n, 3, 32, 32, device=device))
    torch.cuda.synchronize(device)


def prepare(state, data: TrainingData, seed: int) -> None:
    cfg, model, device = state.cfg, state.model, state.device
    model.reset()  # weights and BatchNorm running stats
    model.train()
    state.sgd, state.muon = make_optimizers(state)

    images = data.images.to(device, non_blocking=True).float().div_(255)
    images = model.normalize(images)
    state.labels = data.labels.to(device, non_blocking=True)
    model.init_whiten(images[:5000])
    images = batch_flip_lr(images)
    state.images = images
    pad = cfg["translate"]
    state.padded_images = F.pad(images, (pad,) * 4, "reflect")


def lr_factor(cfg, step, total_steps):
    warm = cfg["lr_warmup_frac"] * total_steps
    hold_end = warm + cfg["lr_hold_frac"] * total_steps
    if step < warm:
        return (step + 1) / warm
    if step < hold_end:
        return 1.0
    return (total_steps - step) / (total_steps - hold_end)


def train(state) -> nn.Module:
    cfg, model = state.cfg, state.model
    sgd, muon = state.sgd, state.muon
    n, bs = len(state.labels), cfg["batch_size"]
    steps_per_epoch = n // bs
    epoch_res = [res for res, epochs in cfg["schedule"] for _ in range(epochs)]
    total_steps = len(epoch_res) * steps_per_epoch
    whiten_bias_steps = cfg["whiten_bias_epochs"] * steps_per_epoch

    step = 0
    for epoch, res in enumerate(epoch_res):
        model.train()
        images = epoch_images(state, epoch, res)
        indices = torch.randperm(n, device=images.device)
        for i in range(steps_per_epoch):
            idxs = indices[i * bs : (i + 1) * bs]
            outputs = model(images[idxs], step < whiten_bias_steps, True)
            F.cross_entropy(outputs, state.labels[idxs], label_smoothing=cfg["label_smoothing"],
                            reduction="sum").backward()
            for group in sgd.param_groups[:1]:
                group["lr"] = group["initial_lr"] * max(0.0, 1 - step / whiten_bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * lr_factor(cfg, step, total_steps)
            sgd.step()
            muon.step()
            model.zero_grad(set_to_none=True)
            step += 1
    return model
