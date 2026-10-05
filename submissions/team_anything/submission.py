"""airbench96 for the AIDDA CIFAR-100 speedrun: the result of the 6-hour optimisation (notes/opt6h.md).

This is recipes/airbench96_opt with the settings the search kept as its defaults (each changed default
names the old value in a comment):
- 258 optimizer steps of 1024 images: two full epochs at low resolution (antialiased downscale; epoch 0
  at 22x22, epoch 1 at 24x24, sizes cuDNN runs fast), then from epoch 2
  every epoch trains on the 75% of examples with the highest loss when last seen (top-k selection within the
  trial, 36 steps per epoch), at 32x32 ("epochs" = 6.5 counts the full epochs plus the selected ones);
  15% of each selected epoch is drawn at random from the easier examples;
- head logits x 1/5 (was 1/9), label smoothing 0.4, no cutout, BatchNorm momentum 0.6 (PyTorch 0.4),
  whitening eigenvalue floor 1e-2, Muon lr 0.28, BatchNorm-bias lr x32 (was x64), per-image colour
  jitter (brightness +/-0.14, contrast x(1 +/- 0.13));
- from epoch 5 (the last 54 of 258 steps) group 1 is frozen: its output is detached, so no backward runs
  through the whitening conv and group 1 (~5% faster per frozen step), and before that group 1's learning
  rates decay linearly to zero at the freeze (FreezeOut-style); both ideas of the parallel moonshots session;
- QuickGELU, x * sigmoid(1.702 x), instead of GELU (2.7% faster at equal accuracy);
- torch.compile mode "max-autotune" (CUDA graphs for forward and backward), Muon's per-filter arithmetic as
  foreach kernels;
- a pinned staging buffer for the host-to-device copy, a full-size synthetic warmup, a vectorised dirac
  init and one compiled gather per epoch for translate, flip and cutout (prepare ~100 ms -> ~33 ms).

What follows describes airbench96_opt.

airbench96 for the AIDDA CIFAR-100 speedrun: the working recipe of the 6-hour optimisation (notes/opt6h.md).

Starts as recipes/airbench96_compiled with candidate 0's settings (Muon on the conv
filters, cutout 4, translate 2, label smoothing 0.3) at 7 epochs; every option added
during the search defaults to the old behaviour. What follows describes
airbench96_compiled.

Legacy airbench96 for the AIDDA CIFAR-100 speedrun, compiled, with optional progressive resizing.

Derived from recipes/airbench96_e18 (the 18-epoch port of Keller Jordan's legacy
airbench96). Changes from that recipe:
- the forward pass and loss are one torch.compile'd function (static shapes,
  `compile` sets the mode; None runs eagerly), warmed up in build on synthetic
  data for every (resolution, whitening-bias-grad) combination used in training;
- the whitening bias is frozen after `whiten_bias_epochs` by detaching it in the
  forward pass (equivalent to the original's requires_grad toggle, but without a
  recompile);
- the lookahead EMA uses foreach kernels over the float tensors of the state dict;
- the final 3x3 max-pool is a global max (identical at 32x32, and lets the net
  train at lower resolutions);
- optional progressive resizing (`res_schedule`): early epochs at lower
  resolution, by antialiased bilinear downscaling ("resize") or random crops
  ("crop") of the translated images, with cutout scaled to the resolution;
- optional class-stratified subsampling (`subsample`): train on the same fraction of
  every class, either one subset for the whole trial ("fixed", chosen in prepare) or
  a fresh subset each epoch ("epoch"). An epoch is one pass over the subset;
- a choice of orthogonalisation for Muon (`ns_coeffs`, `ns_steps`): "ns5" is the
  quintic Newton-Schulz of airbench94_muon (3.4445, -4.7750, 2.0315 every
  iteration); "polar" is the Polar Express iteration of Amsel, Persson, Musco and
  Gower, "The Polar Express: Optimal Matrix Sign Methods and Their Application to
  the Muon Algorithm" (arXiv:2505.16932), with the per-iteration coefficients and
  input normalisation of the authors' reference implementation
  (https://github.com/NoahAmsel/PolarExpress, polar_express.py at commit 71cc379,
  MIT): the output of optimal_composition(l=1e-3, num_iters=10, degree=5,
  safety_factor_eps=1e-2, cushion=0.02). `ns_steps` iterations are run (3, as
  before, by default).

Source: Keller Jordan, cifar10-airbench, `legacy/airbench96.py` (MIT; see LICENSE),
https://github.com/KellerJordan/cifar10-airbench at commit 4c1b6d1.

Kept from the original: the whitening conv and three three-layer conv groups
with residual connections (GELU, frozen-scale BatchNorm); 12-pixel cutout and
4-pixel translation; decoupled Nesterov SGD with a 64x bias learning rate;
the warmup-then-decay schedule; lookahead EMA every five steps; label
smoothing; alternating flip and random translation.

Changed for this benchmark: a 100-class head; no test-time augmentation;
normalisation statistics, whitening and augmentation are computed on the GPU in
prepare and train (timed); compilation and cuDNN autotuning are warmed up in
build on synthetic data and every parameter, buffer and optimizer is reset in
prepare; the returned model is the eager network and normalises float32 inputs
itself. CPU smoke runs use float32, no compilation and a short step cap.
"""

import time
from math import ceil, cos, pi
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

torch.backends.cudnn.benchmark = True

DEFAULTS = dict(
    epochs=6.5,  # airbench96_opt: 7.0
    batch_size=1024,
    lr=9.0,              # per 1024 examples
    momentum=0.85,
    weight_decay=0.012,  # per 1024 examples, decoupled from the learning rate
    bias_scaler=32.0,  # airbench96_opt: 64.0
    label_smoothing=0.4,  # airbench96_opt: 0.3
    whiten_bias_epochs=3,
    flip=True,
    translate=2,
    cutout=0,  # airbench96_opt: 4
    widths=(128, 384, 512),
    depth=3,             # convs per group, or one per group, e.g. [3, 2, 3]
    bn_momentum=0.6,  # airbench96_opt: None
    scaling_factor=1 / 5,  # airbench96_opt: 1 / 9
    warmup=0.1,
    final_lr=0.0,        # decays to zero
    whiten_samples=5000,
    lookahead_every=5,   # 0: no lookahead EMA at all
    compile="max-autotune",  # airbench96_opt: "default"
    res_schedule=[[0.1, 22], [0.25, 24]],  # airbench96_opt: None
    res_mode="resize",   # "resize" (antialiased bilinear) or "crop"
    optimizer="muon",    # "muon": Muon on the 3x3 conv filters, the SGD above on the rest
    muon_lr=0.28,  # airbench96_opt: 0.24
    muon_momentum=0.6,
    muon_fused=0,        # 1: one compiled function for the whole Muon step (CUDA only)
    ns_coeffs="ns5",     # "ns5": Jordan's quintic Newton-Schulz; "polar": Polar Express (see the docstring)
    ns_steps=3,          # orthogonalisation iterations per Muon update
    compile_scope="loss",  # "loss": forward + loss in one graph; "net": the network only
    subsample=1.0,       # fraction of each class trained on per epoch (1.0: all images)
    subsample_mode="fixed",  # "fixed": one subset per trial; "epoch": a fresh subset every epoch
    act="quick_gelu",  # airbench96_opt: "gelu"
    pool_first=0,        # 1: max-pool before each group's first conv (a quarter of its FLOPs), or per group [1, 0, 0]
    decay_shape="linear",  # after warmup: "linear", "cosine" or "quad" ((1 - t)^2) down to final_lr
    hold=0.0,            # fraction of the post-warmup steps held at the peak before decaying
    select_frac_end=None,  # top-k: if set, the keep fraction ramps linearly from select_frac to this
    select_frac=0.75,  # airbench96_opt: 1.0
    select_start=2,      #   examples with the highest loss when last seen (within the trial)
    select_balanced=0,   # top-k: 1 = the hardest examples by rank within their class (same share per class)
    select_random=0.15,  # airbench96_opt: 0.0
    select_mode="topk",  # "topk" (above), or "soft": InfoBatch-style, from epoch `select_start` drop each
    select_p=0.5,        #   example whose last loss is below the mean with probability `select_p` and weight
    select_anneal=0.125,  #   the kept ones by 1/(1-p); all examples, unweighted, in the last `select_anneal`
    muon_foreach=1,  # airbench96_opt: 0
    muon_rownorm=0,      # 1: rescale each row of the orthogonalised update to the same norm (Frobenius kept)
    whiten_eps=0.01,  # airbench96_opt: 5e-4
    fast_prep=1,  # airbench96_opt: 0
    fused_aug=2,  # airbench96_opt: 0
                         # 2: also the downscaled epochs of progressive resizing
    cudnn_limit=None,    # torch.backends.cudnn.benchmark_limit (PyTorch's default 10; 0 tries every algorithm)
    inductor_cdt=0,      # 1: Inductor coordinate-descent tuning of its Triton kernels (longer build)
    bn_half=1,           # 1: BatchNorm parameters and buffers in half precision (0: fp32)
    bn_recal=0,          # N > 0: after training, recompute BatchNorm statistics on N clean training batches
    flat_max=0,          # 1: global max as flatten(2).max(2) instead of max_pool2d over the whole map
    jitter=(0.14, 0.13),  # airbench96_opt: (0.0, 0.0)
    freeze_epoch=5,  # airbench96_opt: None
                         #   forward only); from the parallel "moonshots" session (recipes/airbench96_freeze)
    freeze_anneal=1,  # airbench96_opt: 0
                         #   (FreezeOut-style; from the moonshots session)
    bn_freeze_epoch=None,  # from this epoch on, BatchNorm uses its running statistics (no batch reductions)
    debug=0,             # 1/2/3: raise a diagnostic report instead of returning (development only; 3 = profile)
)
CPU_STEP_CAP = 3
TRAIN_SIZE = 50_000  # CIFAR-100's training set: the shape the pinned buffer and the full-size warmup use
torch._dynamo.config.cache_size_limit = 64  # one static graph per shape


# --------------------------------------------------------------------------- Muon (from airbench94_muon)

def zeropower_via_newtonschulz5(G, steps: int = 3, eps: float = 1e-7):
    """Quintic Newton-Schulz orthogonalisation (airbench94_muon; the same coefficients every iteration)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# Polar Express (Amsel et al., arXiv:2505.16932): coefficients of p_t(x) = a x + b x^3 + c x^5 for iterations
# t = 1..10, the output of optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2,
# cushion=0.02) in the authors' polar_express.py (github.com/NoahAmsel/PolarExpress, commit 71cc379).
POLAR_EXPRESS_COEFFS = (
    (8.237312490495555, -23.1577474145582, 16.680568411445915),
    (4.082441999064829, -2.8930477353325843, 0.5252849256975644),
    (3.9263479922546485, -2.854746803476524, 0.5318022422894979),
    (3.2982187133085197, -2.4245419810267106, 0.4863200835884415),
    (2.2970369434552587, -1.6366255812590325, 0.4002628455953631),
    (1.8763805351440381, -1.23478965777222, 0.3589188750166826),
    (1.856442348565278, -1.2132449881003775, 0.3568003487859341),
    (1.85643356156706, -1.2132336029471178, 0.3567976320793584),
    (1.8564311647224787, -1.2132289037487436, 0.3567953287942881),
    (1.8749954775631825, -1.249990955148181, 0.37499547758499846),
)


def zeropower_via_polar_express(G, steps: int = 3, eps: float = 1e-7):
    """Polar Express orthogonalisation, as the reference `PolarExpress` (bf16; the 1.01x norm margin is its
    safety factor; after the tenth iteration the last polynomial repeats)."""
    X = G.bfloat16()
    X = X / (X.norm() * 1.01 + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    coeffs = POLAR_EXPRESS_COEFFS[:steps] + (POLAR_EXPRESS_COEFFS[-1],) * max(0, steps - len(POLAR_EXPRESS_COEFFS))
    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


ORTHOGONALIZERS = {"ns5": zeropower_via_newtonschulz5, "polar": zeropower_via_polar_express}


def make_orthogonalizer(hyp):
    """The configured orthogonalisation as a one-argument function of the gradient matrix."""
    fn, steps = ORTHOGONALIZERS[hyp["ns_coeffs"]], int(hyp["ns_steps"])

    def orthogonalize(G):
        return fn(G, steps)
    return orthogonalize


def muon_update(params, grads, bufs, lr, momentum: float, orthogonalize=zeropower_via_newtonschulz5):
    """The Muon step below for a list of filters, as one function (compiled when fused)."""
    torch._foreach_mul_(bufs, momentum)
    torch._foreach_add_(bufs, grads)
    for p, g, buf in zip(params, grads, bufs):
        g = g.add(buf, alpha=momentum)
        p.mul_(len(p) ** 0.5 / p.norm())  # normalise the weight
        update = orthogonalize(g.reshape(len(g), -1)).view(g.shape)
        p.sub_(lr * update)


class Muon(torch.optim.Optimizer):
    """Muon on conv filters. `foreach`: the momentum, Nesterov and weight-normalisation arithmetic of all
    filters in a few foreach kernels instead of ~5 kernels per filter (the same arithmetic)."""

    def __init__(self, params, newton_schulz, lr=1e-3, momentum=0.0, nesterov=False, fused_update=None, rownorm=False,
                 foreach=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.newton_schulz = newton_schulz
        self.rownorm = rownorm
        self.foreach = foreach
        self.fused_update = fused_update  # nesterov only
        if fused_update is not None:
            self.lr_tensor = torch.zeros((), device=self.param_groups[0]["params"][0].device)

    @torch.no_grad()
    def step(self):
        if self.fused_update is not None:
            group = self.param_groups[0]
            params = [p for p in group["params"] if p.grad is not None]
            bufs = []
            for p in params:
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad)
                bufs.append(self.state[p]["momentum_buffer"])
            self.lr_tensor.fill_(group["lr"])
            self.fused_update(params, [p.grad for p in params], bufs, self.lr_tensor, group["momentum"])
            return
        if self.foreach and not self.rownorm:
            self._foreach_step()
            return
        self._loop_step()

    @torch.no_grad()
    def _foreach_step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            bufs = []
            for p in params:
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad)
                bufs.append(self.state[p]["momentum_buffer"])
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)
            updates = torch._foreach_add(grads, bufs, alpha=momentum) if group["nesterov"] else bufs
            norms = torch._foreach_norm(params)
            torch._foreach_div_(params, norms)
            torch._foreach_mul_(params, [len(p) ** 0.5 for p in params])  # normalise the weights
            for p, g in zip(params, updates):
                p.add_(self.newton_schulz(g.reshape(len(g), -1)).view(g.shape), alpha=-lr)

    @torch.no_grad()
    def _loop_step(self):
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
                update = self.newton_schulz(g.reshape(len(g), -1))
                if self.rownorm:
                    rows = update.float().norm(dim=1, keepdim=True)
                    update = update * (rows.square().mean().sqrt() / rows.clamp_min(1e-7))
                p.add_(update.view(g.shape), alpha=-lr)


# --------------------------------------------------------------------------- augmentation (GPU-resident)

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype)
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s:r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s:r + s + crop_size]
    return images_out


def make_random_square_masks(inputs, size):
    n, c, h, w = inputs.shape
    corner_y = torch.randint(0, h - size + 1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w - size + 1, size=(n,), device=inputs.device)
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)
    return mask_y * mask_x


def batch_cutout(inputs, size):
    return inputs.masked_fill(make_random_square_masks(inputs, size), 0)


def fused_epoch_aug(images, pad: int, cutout: int, flip_all: bool, jitter=(0.0, 0.0), inv_std=None):
    """batch_crop + (flip) + batch_cutout as one gather: random `pad`-pixel translation of reflect-padded
    images, then a horizontal flip of every image if `flip_all`, then a random `cutout` square. Optional colour
    jitter (b, c), per image: contrast x(1 +/- c) around the image mean, then brightness +/- b in [0, 1] units
    (`inv_std` converts them to the normalised scale)."""
    n, dev = images.shape[0], images.device
    ar = torch.arange(32, device=dev)
    ys = torch.randint(0, 2 * pad + 1, (n, 1), device=dev) + ar
    xs = torch.randint(0, 2 * pad + 1, (n, 1), device=dev) + (ar.flip(0) if flip_all else ar)
    out = images[torch.arange(n, device=dev).view(n, 1, 1, 1), torch.arange(3, device=dev).view(1, 3, 1, 1),
                 ys.view(n, 1, 32, 1), xs.view(n, 1, 1, 32)]
    if cutout > 0:
        cy = torch.randint(0, 32 - cutout + 1, (n, 1, 1, 1), device=dev)
        cx = torch.randint(0, 32 - cutout + 1, (n, 1, 1, 1), device=dev)
        dy, dx = ar.view(1, 1, 32, 1) - cy, ar.view(1, 1, 1, 32) - cx
        out = out.masked_fill((dy >= 0) & (dy < cutout) & (dx >= 0) & (dx < cutout), 0)
    bright, contrast = jitter
    if bright > 0 or contrast > 0:
        x = out.float()
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        c = 1 + contrast * (2 * torch.rand(n, 1, 1, 1, device=dev) - 1)
        b = bright * (2 * torch.rand(n, 1, 1, 1, device=dev) - 1)
        out = ((x - mean) * c + mean + b * inv_std).to(images.dtype)
    return out.contiguous(memory_format=torch.channels_last)


def fused_epoch_aug_resized(images, pad: int, cutout: int, flip_all: bool, size: int, jitter=(0.0, 0.0), inv_std=None):
    """fused_epoch_aug at 32x32, then an antialiased bilinear downscale to `size` (progressive resizing)."""
    x = fused_epoch_aug(images, pad, cutout, flip_all, jitter, inv_std)
    x = F.interpolate(x.float(), size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return x.to(images.dtype).contiguous(memory_format=torch.channels_last)


def stratified_subset(labels, fraction, num_classes):
    """Indices of a random `fraction` of each class (rounded per class), in ascending order."""
    n = len(labels)
    perm = torch.randperm(n, device=labels.device)
    grouped = perm[torch.argsort(labels[perm], stable=True)]  # by class, random within each class
    counts = torch.bincount(labels, minlength=num_classes)
    starts = counts.cumsum(0) - counts
    rank = torch.arange(n, device=labels.device) - starts[labels[grouped]]
    keep = (counts.float() * fraction).round().long()
    return grouped[rank < keep[labels[grouped]]].sort().values


# --------------------------------------------------------------------------- network

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=None, eps=1e-12, weight=False, bias=True):
        if momentum is None:
            super().__init__(num_features, eps=eps)
        else:
            super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        c = min(w.size(0), w.size(1))
        w[:c].zero_()  # torch.nn.init.dirac_(w[:c]) without its per-channel loop of kernel launches
        d = torch.arange(c, device=w.device)
        w[d, d, w.size(2) // 2, w.size(3) // 2] = 1


class QuickGELU(nn.Module):
    """x * sigmoid(1.702 x): GELU's shape with one exp instead of erf."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


ACTIVATIONS = {"gelu": lambda: nn.GELU(), "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
               "relu": lambda: nn.ReLU(), "silu": lambda: nn.SiLU(), "quick_gelu": QuickGELU}


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth, bn_momentum, act="gelu", pool_first=False):
        super().__init__()
        self.pool_first = pool_first
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, bn_momentum)
        if depth == 3:
            self.conv3 = Conv(channels_out, channels_out)
            self.norm3 = BatchNorm(channels_out, bn_momentum)
        self.activ = ACTIVATIONS[act]()

    def forward(self, x):
        if self.pool_first:
            x = self.activ(self.norm1(self.conv1(self.pool(x))))
        else:
            x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x0 = x
        x = self.activ(self.norm2(self.conv2(x)))
        if self.depth == 3:
            x = self.activ(self.norm3(self.conv3(x)) + x0)
        return x


class CifarNet(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        whiten_width = 2 * 3 * 2**2
        self.whiten = Conv(3, whiten_width, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.act = ACTIVATIONS[hyp["act"]]()
        d1, d2, d3 = hyp["depth"] if isinstance(hyp["depth"], (list, tuple)) else (hyp["depth"],) * 3
        pf = hyp["pool_first"]
        p1, p2, p3 = (bool(x) for x in (pf if isinstance(pf, (list, tuple)) else (pf,) * 3))
        act = hyp["act"]
        self.group1 = ConvGroup(whiten_width, w1, d1, hyp["bn_momentum"], act=act, pool_first=p1)
        self.group2 = ConvGroup(w1, w2, d2, hyp["bn_momentum"], act=act, pool_first=p2)
        self.group3 = ConvGroup(w2, w3, d3, hyp["bn_momentum"], act=act, pool_first=p3)
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scaling_factor = hyp["scaling_factor"]
        self.flat_max = bool(hyp["flat_max"])

    def reset(self):
        for m in self.modules():
            if type(m) in (Conv, BatchNorm, nn.Linear):
                m.reset_parameters()

    def forward(self, x, whiten_bias_grad: bool = True, frozen: bool = False):
        b = self.whiten.bias
        x = self.act(F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach()))
        x = self.group1(x)
        if frozen:  # group 1 frozen (freeze_epoch): no backward through group 1 or the whitening conv
            x = x.detach()
        x = self.group3(self.group2(x))
        # A max-pool over the whole map: MaxPool2d(3) at 32x32 input. (Inductor's amax backward
        # returned NaN here in PyTorch 2.4.)
        if self.flat_max:
            x = x.flatten(2).max(dim=2).values
        else:
            x = F.max_pool2d(x, kernel_size=x.shape[-1]).flatten(1)
        return self.head(x) * self.scaling_factor


def make_net(hyp, num_classes, device, dtype):
    net = CifarNet(hyp, num_classes)
    net = net.to(device=device, dtype=dtype).to(memory_format=torch.channels_last)
    if not hyp["bn_half"]:  # bn_half=1: BatchNorm stays in the network's half precision
        for mod in net.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
    return net


@torch.no_grad()
def init_whitening_conv(layer, train_set, eps=5e-4):
    c, (h, w) = train_set.shape[1], layer.weight.shape[2:]
    patches = train_set.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    n = len(patches)
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
    eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c * h * w, c, h, w).flip(0)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))


class LookaheadState:
    def __init__(self, net):
        tensors = [v for v in net.state_dict().values() if v.dtype in (torch.half, torch.float)]
        self.groups = {}
        for t in tensors:
            self.groups.setdefault(t.dtype, []).append(t)
        self.ema = {k: [t.clone() for t in v] for k, v in self.groups.items()}

    @torch.no_grad()
    def update(self, decay):
        for k, live in self.groups.items():
            torch._foreach_lerp_(self.ema[k], live, 1 - decay)
            torch._foreach_copy_(live, self.ema[k])


class Classifier(nn.Module):
    """Single-view classifier: float32 images in [0, 1] to float32 logits."""

    def __init__(self, net, dtype):
        super().__init__()
        self.net = net
        self.dtype = dtype
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))

    def forward(self, x):
        x = ((x - self.mean) / self.std).to(self.dtype).contiguous(memory_format=torch.channels_last)
        return self.net(x).float()


# --------------------------------------------------------------------------- interface

def build(context: BuildContext):
    hyp = {**DEFAULTS, **context.parameters}
    device = context.device
    cuda = device.type == "cuda"
    dtype = torch.half if cuda else torch.float32
    if hyp["cudnn_limit"] is not None:
        torch.backends.cudnn.benchmark_limit = int(hyp["cudnn_limit"])
    if hyp["inductor_cdt"]:
        torch._inductor.config.coordinate_descent_tuning = True
    net = make_net(hyp, context.num_classes, device, dtype)
    smoothing = hyp["label_smoothing"]

    compiled = cuda and hyp["compile"]
    mode = None if hyp["compile"] == "default" else hyp["compile"]
    forward = torch.compile(net, mode=mode, dynamic=False, fullgraph=True) if (
        compiled and hyp["compile_scope"] == "net") else net

    selecting = _selecting(hyp)

    def loss_fn(inputs, labels, whiten_bias_grad: bool, weights=None, frozen: bool = False):
        outputs = forward(inputs, whiten_bias_grad, frozen)
        losses = F.cross_entropy(outputs, labels, label_smoothing=smoothing, reduction="none")
        if selecting:
            total = (losses * weights).sum() if weights is not None else losses.sum()
            return total, losses.detach().float()
        return losses.sum()

    newton_schulz = make_orthogonalizer(hyp)

    def fused_muon_update(params, grads, bufs, lr, momentum: float):
        return muon_update(params, grads, bufs, lr, momentum, newton_schulz)

    if compiled:
        if hyp["compile_scope"] == "loss":
            loss_fn = torch.compile(loss_fn, mode=mode, dynamic=False, fullgraph=True)
        newton_schulz = torch.compile(newton_schulz, dynamic=False)
    fused_update = torch.compile(fused_muon_update, dynamic=False) if (compiled and hyp["muon_fused"]) else None
    epoch_aug = torch.compile(fused_epoch_aug, dynamic=False) if (compiled and hyp["fused_aug"]) else fused_epoch_aug
    epoch_aug_resized = (torch.compile(fused_epoch_aug_resized, dynamic=False) if (compiled and hyp["fused_aug"] == 2)
                         else fused_epoch_aug_resized)
    state = SimpleNamespace(hyp=hyp, device=device, cuda=cuda, dtype=dtype, net=net, loss_fn=loss_fn, epoch_aug=epoch_aug,
                            epoch_aug_resized=epoch_aug_resized,
                            num_classes=context.num_classes,
                            newton_schulz=newton_schulz, fused_update=fused_update, model=Classifier(net, dtype).to(device))
    if cuda and hyp["fast_prep"]:
        state.pinned = torch.empty((TRAIN_SIZE, 3, 32, 32), dtype=torch.uint8).pin_memory()
    if cuda:
        warmup(state, context)
    return state


def _sizes(hyp, total_epochs):
    """Training resolution of each epoch."""
    schedule = sorted(hyp["res_schedule"] or [])
    out = []
    for epoch in range(total_epochs):
        size = 32
        for frac, s in schedule:
            if epoch < frac * hyp["epochs"]:
                size = int(s)
                break
        out.append(size)
    return out


def _set_bn_frozen(net, frozen: bool):
    """BatchNorm layers normalise with their running statistics (eval mode) while the rest trains."""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(not frozen)


def warmup(state, context):
    """Compile and autotune every training and evaluation shape on synthetic data."""
    hyp = state.hyp
    bs = hyp["batch_size"]
    count = TRAIN_SIZE if (hyp["fast_prep"] or hyp["fused_aug"]) else 2 * bs  # full size primes allocator + graphs
    images = torch.randint(0, 256, (count, 3, 32, 32), dtype=torch.uint8)
    labels = torch.randint(0, context.num_classes, (count,))
    prepare(state, TrainingData(images, labels), seed=0)
    sizes = sorted(set(_sizes(hyp, ceil(hyp["epochs"])) + [32]))
    y = state.labels[:bs]
    for size in sizes:
        x = _batch(_epoch_images(state, 0, size), torch.arange(bs, device=state.device))
        for flag in (True, False):
            for _ in range(3):
                extra = [torch.ones(bs, device=state.device)] if hyp["select_mode"] == "soft" else []
                out = state.loss_fn(x, y, flag, *extra)
                (out[0] if isinstance(out, tuple) else out).backward()
                for opt in state.optimizers:
                    opt.step()
                state.net.zero_grad(set_to_none=True)
    if hyp["freeze_epoch"] is not None:  # the frozen-group-1 graph (full resolution, whitening bias frozen)
        x = _batch(_epoch_images(state, 0, 32), torch.arange(bs, device=state.device))
        for _ in range(3):
            extra = [torch.ones(bs, device=state.device)] if hyp["select_mode"] == "soft" else []
            out = state.loss_fn(x, y, False, *extra, frozen=True)
            (out[0] if isinstance(out, tuple) else out).backward()
            for opt in state.optimizers:
                opt.step()
            state.net.zero_grad(set_to_none=True)
    if hyp["bn_freeze_epoch"] is not None:  # the frozen-statistics graph (BatchNorm in eval mode, full res)
        x = _batch(_epoch_images(state, 0, 32), torch.arange(bs, device=state.device))
        _set_bn_frozen(state.net, True)
        for _ in range(3):
            out = state.loss_fn(x, y, False)
            (out[0] if isinstance(out, tuple) else out).backward()
            for opt in state.optimizers:
                opt.step()
            state.net.zero_grad(set_to_none=True)
        _set_bn_frozen(state.net, False)
    _train(state, total_steps=4)  # the eager per-epoch augmentation and lookahead path
    if hyp["select_frac"] < 1 and hyp["select_mode"] == "topk":  # the selection kernels (topk, randperm of k)
        _, sel = _epoch_steps(hyp, state.epoch_size)
        _epoch_order(state, hyp["select_start"], sel * min(bs, state.epoch_size))
    if hyp["fused_aug"]:
        for size in sizes:
            _epoch_images(state, 1, size)  # the flipped-epoch graphs
    state.model.eval()
    with torch.inference_mode():
        for n in (context.eval_batch_size, 10_000 % context.eval_batch_size or context.eval_batch_size):
            state.model(torch.rand(n, 3, 32, 32, device=state.device))
    state.model.train()
    torch.cuda.synchronize()


def _to_device(state, images, chunks=5):
    """Host-to-device copy, staged through a pinned buffer in chunks so the CPU copy overlaps the DMA."""
    buf = getattr(state, "pinned", None)
    if buf is None or buf.shape != images.shape:
        return images.to(state.device, non_blocking=True)
    out = torch.empty(images.shape, dtype=images.dtype, device=state.device)
    step = -(-len(images) // chunks)
    for a in range(0, len(images), step):
        buf[a:a + step].copy_(images[a:a + step])
        out[a:a + step].copy_(buf[a:a + step], non_blocking=True)
    return out


def _mark(state, name):
    """debug=3: a (name, CUDA event, CPU clock) mark for the profile report."""
    if state.cuda and state.hyp["debug"] == 3:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        state.marks.append((name, e, time.perf_counter()))


def prepare(state, data: TrainingData, seed: int) -> None:
    hyp, net = state.hyp, state.net
    state.marks = []
    _mark(state, "start")
    net.reset()
    net.train()
    state.lookahead = LookaheadState(net)  # as in the original: taken before whitening
    x = _to_device(state, data.images)
    state.labels = data.labels.to(state.device, non_blocking=True)
    fraction = hyp["subsample"]
    if fraction < 1 and hyp["subsample_mode"] == "fixed":
        keep = stratified_subset(state.labels, fraction, state.num_classes)
        x, state.labels = x[keep], state.labels[keep]
    if fraction < 1 and hyp["subsample_mode"] == "epoch":
        counts = torch.bincount(data.labels, minlength=state.num_classes)
        state.epoch_size = int((counts.float() * fraction).round().sum())
    else:
        state.epoch_size = len(state.labels)
    _mark(state, "reset+h2d")
    x = x.float().div_(255)
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True)
    state.model.mean.copy_(mean)
    state.model.std.copy_(std)
    x = ((x - mean) / std).to(state.dtype).contiguous(memory_format=torch.channels_last)
    _mark(state, "normalise")
    init_whitening_conv(net.whiten, x[: hyp["whiten_samples"]], eps=hyp["whiten_eps"])
    _mark(state, "whiten")
    if hyp["flip"]:
        x = batch_flip_lr(x)  # pre-flip once; whole epochs then alternate
    pad = hyp["translate"]
    state.images = F.pad(x, (pad,) * 4, "reflect") if pad > 0 else x
    if _selecting(hyp):
        state.ex_loss = torch.full((len(state.labels),), float("inf"), device=state.device)
        state.ex_weight = torch.ones(len(state.labels), device=state.device)
    _mark(state, "flip+pad")

    batch_size, momentum = hyp["batch_size"], hyp["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    muon = hyp["optimizer"] == "muon"
    filters = [p for k, p in net.named_parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for k, p in net.named_parameters() if "norm" in k and p.requires_grad]
    other_params = [p for k, p in net.named_parameters() if "norm" not in k and p.requires_grad
                    and not (muon and p.ndim == 4)]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd / lr)]
    muon_params = filters
    if hyp["freeze_anneal"]:  # group 1's parameters in their own groups (tagged g1) for their own schedule
        g1 = {id(p) for k, p in net.named_parameters() if k.startswith("group1.")}
        param_configs = [dict(c, params=[p for p in c["params"] if (id(p) in g1) == tag], g1=tag)
                         for c in param_configs for tag in (False, True)]
        param_configs = [c for c in param_configs if c["params"]]
        muon_params = [dict(params=[p for p in filters if (id(p) in g1) == tag], g1=tag) for tag in (False, True)]
    state.optimizers = [torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)]
    if muon:
        state.optimizers.append(Muon(muon_params, state.newton_schulz, lr=hyp["muon_lr"],
                                     momentum=hyp["muon_momentum"], nesterov=True,
                                     fused_update=state.fused_update, rownorm=bool(hyp["muon_rownorm"]),
                                     foreach=bool(hyp["muon_foreach"])))
    _mark(state, "optim")


def _selecting(hyp):
    return hyp["select_frac"] < 1 or hyp["select_mode"] == "soft"


def _epoch_steps(hyp, n):
    """(steps in a full epoch, steps in a selected epoch)."""
    batch_size = min(hyp["batch_size"], n)
    full = n // batch_size
    frac = hyp["select_frac"] if hyp["select_frac_end"] is None else (hyp["select_frac"] + hyp["select_frac_end"]) / 2
    sel = max(1, int(n * frac) // batch_size) if hyp["select_frac"] < 1 else full  # a ramp budgets its mean
    return full, sel


def train(state):
    hyp, n = state.hyp, state.epoch_size
    full, sel = _epoch_steps(hyp, n)
    early = min(hyp["select_start"], hyp["epochs"]) if (hyp["select_frac"] < 1 and hyp["select_mode"] == "topk") \
        else hyp["epochs"]
    total = ceil(full * early + sel * (hyp["epochs"] - early))
    if not state.cuda:
        total = min(total, CPU_STEP_CAP)
    _train(state, total)
    return state.model


def _lr_lambda(hyp, total_train_steps):
    warmup_steps = int(total_train_steps * hyp["warmup"])
    warmdown_steps = total_train_steps - warmup_steps

    hold_steps = int(warmdown_steps * hyp["hold"])
    decay_steps = max(1, warmdown_steps - hold_steps)
    shape = hyp["decay_shape"]

    def get_lr(step):
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        frac = min(1.0, max(0.0, (step - warmup_steps - hold_steps) / decay_steps))
        if shape == "cosine":
            frac = 0.5 * (1 - cos(pi * frac))
        elif shape == "quad":
            frac = 1 - (1 - frac) ** 2
        return 1.0 * (1 - frac) + hyp["final_lr"] * frac
    return get_lr


def _lr_lambda_until(hyp, total_train_steps, end_step):
    """freeze_anneal: the same warmup, then a linear decay from 1 to 0 at end_step, then 0."""
    warmup_steps = int(total_train_steps * hyp["warmup"])

    def get_lr(step):
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        return max(0.0, 1 - (step - warmup_steps) / max(1, end_step - warmup_steps))
    return get_lr


def _epoch_images(state, epoch, size):
    hyp, images = state.hyp, state.images
    pad = hyp["translate"]
    if hyp["fused_aug"] and size == 32:
        return state.epoch_aug(images, pad, hyp["cutout"], bool(hyp["flip"] and epoch % 2 == 1),
                               tuple(hyp["jitter"]), 1 / state.model.std)
    if hyp["fused_aug"] == 2 and hyp["res_mode"] == "resize":
        # cutout at 32x32 before the downscale (the eager path cuts a scaled square after it)
        return state.epoch_aug_resized(images, pad, hyp["cutout"], bool(hyp["flip"] and epoch % 2 == 1), size,
                                       tuple(hyp["jitter"]), 1 / state.model.std)
    if size != 32 and hyp["res_mode"] == "crop":
        x = batch_crop(images, size)
    else:
        x = batch_crop(images, 32) if pad > 0 else images
    if hyp["flip"] and epoch % 2 == 1:
        x = x.flip(-1)
    if size != 32 and hyp["res_mode"] == "resize":
        x = F.interpolate(x.float(), size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        x = x.to(state.dtype).contiguous(memory_format=torch.channels_last)
    cutout = round(hyp["cutout"] * size / 32)
    if cutout > 0:
        x = batch_cutout(x, cutout)
    return x


def _epoch_order(state, epoch=0, keep=0):
    """This epoch's training examples in random order: all of them, a fresh stratified subset, or
    (selection) the `keep` examples with the highest loss when last seen."""
    labels, hyp = state.labels, state.hyp
    if hyp["select_frac"] < 1 and hyp["select_mode"] == "topk" and epoch >= hyp["select_start"]:
        rand = int(keep * hyp["select_random"])
        if hyp["select_balanced"]:  # the hardest keep - rand by rank within their class, plus rand at random
            hard = _balanced_hardest(state.ex_loss, labels, state.num_classes, keep - rand)
            chosen = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
            chosen[hard] = True
            rest = (~chosen).nonzero().squeeze(1)
            idx = torch.cat([hard, rest[torch.randperm(len(rest), device=labels.device)[:rand]]])
        elif rand == 0:
            idx = torch.topk(state.ex_loss, keep, sorted=False).indices
        else:  # the hardest keep - rand, plus rand drawn uniformly from the rest
            order = torch.argsort(state.ex_loss, descending=True)
            rest = order[keep - rand:]
            idx = torch.cat([order[:keep - rand], rest[torch.randperm(len(rest), device=labels.device)[:rand]]])
        return idx[torch.randperm(keep, device=labels.device)]
    if hyp["subsample"] < 1 and hyp["subsample_mode"] == "epoch":
        idx = stratified_subset(labels, hyp["subsample"], state.num_classes)
        return idx[torch.randperm(len(idx), device=labels.device)]
    return torch.randperm(len(labels), device=labels.device)


def _balanced_hardest(loss, labels, num_classes, k):
    """The k examples with the lowest loss rank within their own class (hardest first), so every class gives
    about the same share of its examples."""
    n = len(loss)
    order = torch.argsort(loss, descending=True)
    grouped = order[torch.argsort(labels[order], stable=True)]  # by class, hardest first within each class
    counts = torch.bincount(labels, minlength=num_classes)
    starts = counts.cumsum(0) - counts
    rank = (torch.arange(n, device=loss.device) - starts[labels[grouped]]).float() / counts[labels[grouped]]
    return grouped[torch.topk(-rank, k, sorted=False).indices]


def _soft_order(state, epoch, step, total_steps):
    """InfoBatch-style order: drop low-loss examples at random and up-weight the kept ones (sets
    state.ex_weight). One host sync per epoch for the number of kept examples."""
    hyp, loss = state.hyp, state.ex_loss
    n = len(loss)
    if epoch < hyp["select_start"] or step >= (1 - hyp["select_anneal"]) * total_steps:
        state.ex_weight.fill_(1.0)
        return torch.randperm(n, device=loss.device)
    finite = torch.isfinite(loss)
    mean = torch.where(finite, loss, torch.zeros_like(loss)).sum() / finite.sum().clamp(min=1)
    low = loss < mean
    p = hyp["select_p"]
    keep = ~low | (torch.rand(n, device=loss.device) >= p)
    state.ex_weight = 1 + low.float() * (1 / (1 - p) - 1)
    idx = keep.nonzero().squeeze(1)
    return idx[torch.randperm(len(idx), device=loss.device)]


def _batch(epoch_images, idx):
    """One training batch, always channels_last (one layout, so one compiled graph per shape)."""
    return epoch_images[idx].contiguous(memory_format=torch.channels_last)


def _debug_report(state, losses, epoch_ms):
    net = state.net
    bad = [k for k, v in net.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all()]
    net.eval()
    with torch.no_grad():
        logits = state.model(torch.rand(256, 3, 32, 32, device=state.device))
    net.train()
    counters = {k: dict(v) for k, v in torch._dynamo.utils.counters.items() if k in ("stats", "frames", "recompiles")}
    raise RuntimeError(f"DEBUG nonfinite={bad} eval_finite={bool(torch.isfinite(logits).all())} "
                       f"losses={[round(x, 2) for x in losses]} epoch_ms={[round(x) for x in epoch_ms]} "
                       f"dynamo={counters}")


def _profile_report(state, epoch_marks, prof, cpu_steps):
    """debug=3: a compact report (it must survive a ~1500-character log tail)."""
    torch.cuda.synchronize()
    m = state.marks
    prep = " ".join(f"{m[i][0]}={m[i - 1][1].elapsed_time(m[i][1]):.1f}/{1e3 * (m[i][2] - m[i - 1][2]):.1f}"
                    for i in range(1, len(m)))
    aug = [a.elapsed_time(b) for a, b, _ in epoch_marks]
    steps = [b.elapsed_time(c) for _, b, c in epoch_marks]
    cats = dict(conv=0.0, triton=0.0, gemm=0.0, foreach=0.0, elem=0.0, index=0.0, other=0.0)
    kernels = []
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
        name = e.key
        low = name.lower()
        if "triton" in low:
            c = "triton"
        elif any(k in low for k in ("conv", "xmma", "implicit", "cudnn", "wgrad", "dgrad", "fprop")):
            c = "conv"
        elif any(k in low for k in ("gemm", "cutlass", "ampere_", "sm80")):
            c = "gemm"
        elif "foreach" in low or "multi_tensor" in low:
            c = "foreach"
        elif any(k in low for k in ("index", "gather", "scatter", "nonzero", "masked")):
            c = "index"
        elif "elementwise" in low or "vectorized" in low or "reduce" in low:
            c = "elem"
        else:
            c = "other"
        cats[c] += t / 1e3
        kernels.append((t / 1e3, e.count, name))
    kernels.sort(reverse=True)
    total = sum(cats.values())
    n = sum(k[1] for k in kernels)
    top = "; ".join(f"{name[:30]}:{count}:{t:.1f}" for t, count, name in kernels[:10])
    raise RuntimeError(f"P3 prep(gpu/cpu ms) {prep} | aug_ms={[round(x, 1) for x in aug]} "
                       f"steps_ms={[round(x) for x in steps]} cpu_steps_ms={[round(x) for x in cpu_steps]} | "
                       f"ep2 kernels={total:.0f}ms n={n} " + " ".join(f"{k}={v:.0f}" for k, v in cats.items())
                       + f" | top {top}")


@torch.no_grad()
def _recalibrate_bn(state, n_batches, batch_size):
    """Replace the BatchNorm running statistics by their average over `n_batches` clean (un-augmented,
    base-flipped) random training batches under the final weights."""
    norms = [m for m in state.net.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    pad = state.hyp["translate"]
    images = state.images[:, :, pad:pad + 32, pad:pad + 32] if pad > 0 else state.images
    order = torch.randperm(len(images), device=images.device)
    for i in range(n_batches):
        state.net(_batch(images, order[i * batch_size:(i + 1) * batch_size]), False)
    for m, momentum in zip(norms, saved):
        m.momentum = momentum


def _train(state, total_steps):
    hyp, net, optimizers = state.hyp, state.net, state.optimizers
    labels = state.labels
    n = state.epoch_size
    batch_size = min(hyp["batch_size"], n)
    selecting = _selecting(hyp)
    soft = hyp["select_mode"] == "soft"
    full_steps, sel_steps = _epoch_steps(hyp, n)

    f0, f1 = hyp["select_frac"], hyp["select_frac_end"]
    early_steps = full_steps * hyp["select_start"]
    n_sel_epochs = max(1, ceil((total_steps - early_steps) / sel_steps)) if selecting else 1

    def steps_in(epoch):
        if not (selecting and epoch >= hyp["select_start"]):
            return full_steps
        if f1 is None or soft:
            return sel_steps
        # the keep fraction moves linearly from select_frac to select_frac_end over the selected epochs
        k = epoch - hyp["select_start"]
        f = f0 + (f1 - f0) * min(1.0, k / max(1, n_sel_epochs - 1))
        return max(1, int(n * f) // batch_size)
    schedulers = []
    for opt in optimizers:
        for group in opt.param_groups:
            group.pop("initial_lr", None)
        if hyp["freeze_anneal"] and hyp["freeze_epoch"] is not None:  # group 1 decays to 0 by its freeze
            freeze_step = sum(steps_in(e) for e in range(int(hyp["freeze_epoch"])))
            base, g1 = _lr_lambda(hyp, total_steps), _lr_lambda_until(hyp, total_steps, freeze_step)
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(
                opt, [g1 if group.get("g1") else base for group in opt.param_groups]))
        else:
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(hyp, total_steps)))
    every = hyp["lookahead_every"]
    alpha_schedule = (0.95**every * (torch.arange(total_steps + 1) / total_steps) ** 3).tolist()
    lookahead = state.lookahead
    num_epochs, planned = 0, 0
    while planned < total_steps:
        planned += steps_in(num_epochs)
        num_epochs += 1
    if soft:
        num_epochs = 2 * num_epochs + 2  # epochs shrink to at least half; the step budget ends the loop
    sizes = _sizes(hyp, num_epochs)
    step = 0
    debug = hyp["debug"] if (state.cuda and total_steps > 8) else 0
    losses, events = [], []
    epoch_marks, cpu_steps, prof = [], [], None
    net.train()
    for epoch in range(num_epochs):
        if debug == 1:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if debug == 3:
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            marks[0].record()
            epoch_marks.append(marks)
        whiten_bias_grad = epoch < hyp["whiten_bias_epochs"]
        frozen = hyp["freeze_epoch"] is not None and epoch >= hyp["freeze_epoch"]
        if hyp["bn_freeze_epoch"] is not None and epoch == hyp["bn_freeze_epoch"]:
            _set_bn_frozen(net, True)
        steps_per_epoch = steps_in(epoch)
        epoch_images = _epoch_images(state, epoch, sizes[epoch])
        if soft:
            indices = _soft_order(state, epoch, step, total_steps)
            steps_per_epoch = len(indices) // batch_size
        else:
            indices = _epoch_order(state, epoch, steps_per_epoch * batch_size)
        if debug == 3:
            marks[1].record()
            if epoch == 2:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
            cpu0 = time.perf_counter()
        for i in range(steps_per_epoch):
            idx = indices[i * batch_size:(i + 1) * batch_size]
            extra = [state.ex_weight[idx]] if soft else []
            loss = state.loss_fn(_batch(epoch_images, idx), labels[idx], whiten_bias_grad, *extra, frozen=frozen)
            if selecting:
                loss, per_example = loss
                state.ex_loss[idx] = per_example
            if debug == 2:
                losses.append(loss.detach().float())
            elif debug and i == 0:
                losses.append(loss.item())
            loss.backward()
            if debug == 2 and step < 3:
                events.append([p.grad.float().norm() if p.grad is not None else torch.zeros((), device=labels.device)
                               for p in net.parameters()])
            if debug == 2 and step == steps_per_epoch - 1:
                losses = torch.stack(losses).tolist()
                bad = [j for j, x in enumerate(losses) if not (abs(x) < 1e30)]
                names = [k.replace("group", "g").replace(".weight", ".w").replace(".bias", ".b")
                         for k, _ in net.named_parameters()]
                grads = {nm: [f"{float(g[j]):.3g}" for g in events] for j, nm in enumerate(names)}
                raise RuntimeError(f"DEBUG2 first_bad_step={bad[:1]} losses={[round(x, 1) for x in losses[:12]]} "
                                   f"grads={grads}")
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
            net.zero_grad(set_to_none=True)
            step += 1
            if every and step % every == 0:
                lookahead.update(decay=alpha_schedule[step])
            if debug == 3 and i == steps_per_epoch - 1:
                cpu_steps.append(1e3 * (time.perf_counter() - cpu0))
                marks[2].record()
                if epoch == 2:
                    torch.cuda.synchronize()
                    prof.__exit__(None, None, None)
            if step >= total_steps:
                if every:
                    lookahead.update(decay=1.0)
                if hyp["bn_recal"]:
                    _recalibrate_bn(state, int(hyp["bn_recal"]), batch_size)
                if debug == 3:
                    _profile_report(state, epoch_marks, prof, cpu_steps)
                if debug:
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()
                    torch.cuda.synchronize()
                    _debug_report(state, losses, [a.elapsed_time(b) for a, b in zip(events, events[1:])])
                return
