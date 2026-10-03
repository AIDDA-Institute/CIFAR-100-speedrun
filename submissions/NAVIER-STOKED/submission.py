"""NAVIER-STOKED: CIFAR-100 to 75% in ~3.5 s on one A100 80GB PCIe.

A residual airbench-style conv net (64/256/768 channels, three convs per group, frozen
patch-whitening stem, dirac init, SiLU, max + mean head; pool/BatchNorm/SiLU as fused Triton
kernels) trained with Muon on the conv filters and Nesterov SGD on the rest: batch 1024,
onecycle schedule, label smoothing 0.25, alternating flips, 2-pixel random crops and
progressive resizing (20 -> 24 -> 32 px), 8.75 epochs. Each training step (augmentation,
forward, backward, both optimizers) is captured once per resolution as a CUDA graph in
build(), on synthetic data; prepare() resets everything and train() only replays graphs.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import _dynamo, nn
from torch._inductor import config as _inductor_config

from benchmark.api import BuildContext, TrainingData

from .model import Net
from .optim import GraphSGD, Muon

CONFIG = dict(
    widths=(64, 256, 768),
    mids=(64, 192, 512),  # channels inside each group's residual pair
    k1=(3, 3, 2),  # kernel of each group's first conv
    pair_kernels=((1, 1), (3, 1), (3, 3)),  # kernels of each group's residual pair
    epochs=8.75,
    batch_size=1024,
    lr=1.38,  # Nesterov SGD for the head, BN parameters and the stem bias
    momentum=0.9,
    weight_decay=5e-4,  # head weight only
    muon_lr=0.18,
    muon_momentum=0.6,
    ns_steps=3,
    label_smoothing=0.25,
    warmup=0.25,  # onecycle: linear warmup over this fraction of steps, then linear decay
    crop_pad=2,
    prog_res=((0.15, 20), (0.5, 24), (1.0, 32)),  # (end fraction of steps, image size)
    whiten_images=5000,
    train_size=50_000,
    warmup_steps=3,  # untimed synthetic steps per resolution before graph capture
    coordinate_descent_tuning=False,  # Inductor tunes its Triton kernels' block sizes in build
)


def build(context: BuildContext):
    """Untimed: model, compiled step, static buffers, warmup and graph capture, all on
    synthetic data. Parameters can override CONFIG keys (development only)."""
    unknown = set(context.parameters) - set(CONFIG)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    cfg = SimpleNamespace(**{**CONFIG, **context.parameters})
    device = context.device
    if device.type != "cuda":
        raise RuntimeError("this recipe captures CUDA graphs and needs a CUDA device")
    torch.backends.cudnn.benchmark = True
    _dynamo.config.automatic_dynamic_shapes = False  # one static graph per size
    _dynamo.config.cache_size_limit = 64
    _inductor_config.coordinate_descent_tuning = cfg.coordinate_descent_tuning

    model = Net(cfg.widths, cfg.mids, cfg.k1, cfg.pair_kernels, context.num_classes)
    model = model.to(device).to(memory_format=torch.channels_last)
    model.stem.conv.weight.requires_grad_(False)
    state = SimpleNamespace(cfg=cfg, context=context, model=model)
    state.forward_loss = torch.compile(_forward_loss)  # augmentation + forward + loss, one graph
    n, pad = cfg.train_size, cfg.crop_pad
    state.images = torch.randint(
        0, 256, (n, 3, 32 + 2 * pad, 32 + 2 * pad), dtype=torch.uint8, device=device
    )
    state.labels = torch.randint(0, context.num_classes, (n,), device=device)
    state.flip0 = torch.zeros(n, dtype=torch.bool, device=device)
    state.epoch_t = torch.zeros((), dtype=torch.long, device=device)
    sizes = sorted({size for _, size in cfg.prog_res})
    state.static_idx = {s: torch.randint(0, n, (cfg.batch_size,), device=device) for s in sizes}
    state.optimizers = _make_optimizers(model, cfg, device)

    torch.linalg.eigh(torch.eye(12, device=device, dtype=torch.float64))  # init cuSOLVER
    model.train()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # compile and autotune every shape before capture
        for size in sizes:
            for _ in range(cfg.warmup_steps):
                _zero_grad(state)
                _train_step(state, size)
    torch.cuda.current_stream().wait_stream(side)
    state.graphs, pool = {}, None
    for size in sizes:
        _zero_grad(state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            _train_step(state, size)
        pool = graph.pool()
        state.graphs[size] = graph
    model.eval()
    with torch.inference_mode():  # warm the evaluation shapes (batches of 1024 and 784)
        for b in (1024, 10_000 % 1024):
            model(torch.rand(b, 3, 32, 32, device=device))
    torch.cuda.synchronize(device)
    return state


def _make_optimizers(model, cfg, device):
    filters = [p for p in model.parameters() if p.requires_grad and p.ndim == 4]
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim in (2, 3)]
    other = [p for p in model.parameters() if p.requires_grad and p.ndim == 1]
    sgd = GraphSGD(
        [
            {"params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": other, "weight_decay": 0.0, "lr": cfg.lr},
        ],
        cfg.momentum,
        device,
    )
    return [sgd, Muon(filters, cfg.muon_lr, cfg.muon_momentum, cfg.ns_steps, device)]


def _zero_grad(state):
    for opt in state.optimizers:
        opt.zero_grad()


def _forward_loss(state, idx, size: int):
    x = _augment(state, idx, size)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        logits = state.model(x)
        return F.cross_entropy(logits, state.labels[idx], label_smoothing=state.cfg.label_smoothing)


def _train_step(state, size: int):
    """Augment a batch on the GPU, forward, backward and update. Captured as a graph."""
    loss = state.forward_loss(state, state.static_idx[size], size)
    loss.backward()
    for opt in state.optimizers:
        opt.step()


def _augment(state, idx, size: int):
    """Random crop (from reflect-padded uint8 images), alternating flip, resize."""
    batch = state.images[idx]
    n, pad, dev = batch.shape[0], state.cfg.crop_pad, batch.device
    full = batch.shape[-1]
    dy = torch.randint(0, 2 * pad + 1, (n, 1, 1, 1), device=dev)
    dx = torch.randint(0, 2 * pad + 1, (n, 1, 1, 1), device=dev)
    rows = torch.arange(32, device=dev).view(1, 1, 32, 1) + dy
    batch = batch.gather(2, rows.expand(n, 3, 32, full))
    cols = torch.arange(32, device=dev).view(1, 1, 1, 32) + dx
    batch = batch.gather(3, cols.expand(n, 3, 32, 32))
    # Alternating flips, adapted from Keller Jordan's cifar10-airbench (MIT License, notice in
    # LICENSE.airbench): random in epoch 0, then every image alternates each epoch.
    flip = state.flip0[idx] ^ (state.epoch_t % 2).bool()
    batch = torch.where(flip.view(n, 1, 1, 1), batch.flip(3), batch)
    x = batch.float().div_(255)
    if size != 32:
        x = F.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False, antialias=True
        )
    return x.contiguous(memory_format=torch.channels_last)


def prepare(state, data: TrainingData, seed: int) -> None:
    """Timed: copy the data to the GPU, reset every weight, statistic and optimizer
    buffer, and fit the whitening stem on this trial's training images."""
    cfg, device, model = state.cfg, state.context.device, state.model
    if len(data.labels) != cfg.train_size:
        raise ValueError(f"expected {cfg.train_size} training images")
    images = data.images.to(device, non_blocking=True)
    pad = cfg.crop_pad
    state.images.copy_(F.pad(images, (pad,) * 4, mode="reflect"))
    state.labels.copy_(data.labels, non_blocking=True)
    sample = images[: cfg.whiten_images].float().div_(255)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # BatchNorm also resets its running statistics
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module is not model.stem.conv:
                w = module.weight
                nn.init.dirac_(w[: min(w.size(0), w.size(1))])  # identity on the first channels
        model.stem.fit(model.normalize(sample))
    model.train()
    for opt in state.optimizers:
        opt.reset()
    state.flip0.copy_(torch.rand(cfg.train_size, device=device) < 0.5)


def _steps_per_epoch(cfg):
    full, steps, remaining = cfg.train_size // cfg.batch_size, [], float(cfg.epochs)
    while remaining > 1e-9:
        steps.append(int(round(full * min(1.0, remaining))))
        remaining -= 1.0
    return steps


def _lr_factor(cfg, step: int, total: int) -> float:
    warm = max(1, int(cfg.warmup * total))
    if step < warm:
        return step / warm
    return max(0.0, (total - step) / max(1, total - warm))


def _size_at(cfg, step: int, total: int) -> int:
    for end, size in cfg.prog_res:
        if step < end * total:
            return size
    return 32


def train(state) -> nn.Module:
    """Timed: per step, pick the batch indices and learning rates, replay the graph."""
    cfg = state.cfg
    schedule = _steps_per_epoch(cfg)
    total, step, bs = sum(schedule), 0, cfg.batch_size
    groups = [g for opt in state.optimizers for g in opt.groups()]
    for epoch, steps in enumerate(schedule):
        state.epoch_t.fill_(epoch)
        order = torch.randperm(cfg.train_size, device=state.labels.device)
        for i in range(steps):
            size = _size_at(cfg, step, total)
            state.static_idx[size].copy_(order[i * bs : (i + 1) * bs])
            f = _lr_factor(cfg, step, total)
            for base_lr, lr_t in groups:
                lr_t.fill_(base_lr * f)
            state.graphs[size].replay()
            step += 1
    return state.model
