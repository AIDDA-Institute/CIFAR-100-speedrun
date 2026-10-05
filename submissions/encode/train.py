"""Training loop: Nesterov SGD with a triangular schedule, lookahead and label smoothing."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from .config import RecipeConfig
from .data import TrainingStream
from .model import Net

Schedule = Callable[[int], float]


def triangle(steps: int, start: float, peak_frac: float, end: float) -> Schedule:
    """Piecewise-linear multiplier rising from ``start`` to 1 at ``peak_frac``, then to ``end``."""
    peak = max(1, int(peak_frac * steps))

    def schedule(i: int) -> float:
        if i < peak:
            return start + (1 - start) * i / peak
        return 1 + (end - 1) * (i - peak) / max(1, steps - peak)

    return schedule


def _nesterov_step(
    params: list[list[torch.Tensor]],
    grads: list[list[torch.Tensor]],
    buffers: list[list[torch.Tensor]],
    decays: tuple[float, ...],
    groups: tuple[int, ...],
    momentum: float,
    table: torch.Tensor,
    counter: torch.Tensor,
) -> None:
    """One Nesterov SGD step (torch.optim.SGD semantics, coupled weight decay) for every
    group, with each group's lr read on the device from ``table[counter]``."""
    lrs = table.index_select(0, counter.view(1))[0]  # stays on the device (no .item())
    for ps, gs, bs, wd, k in zip(params, grads, buffers, decays, groups, strict=True):
        for p, g, b in zip(ps, gs, bs, strict=True):
            d = g + wd * p
            b.mul_(momentum).add_(d)
            p.sub_((d + momentum * b) * lrs[k])
    counter.add_(1)


_compiled_nesterov_step = torch.compile(_nesterov_step, fullgraph=True, dynamic=False)


class SGD:
    """Nesterov SGD as one compiled update per step (faster than torch's fused SGD here).

    Learning rates come from a precomputed device table, so neither schedule changes nor a new
    trial's optimiser recompile. Parameters without a gradient (frozen) are skipped."""

    def __init__(
        self, groups: list[tuple[dict, Schedule]], momentum: float, total_steps: int
    ) -> None:
        self.param_groups = [g for g, _ in groups]
        self.momentum = momentum
        device = self.param_groups[0]["params"][0].device
        self.buffers = {p: torch.zeros_like(p) for g in self.param_groups for p in g["params"]}
        rows = [[g["lr"] * s(i) for g, s in groups] for i in range(total_steps + 1)]
        self.table = torch.tensor(rows, dtype=torch.float32, device=device)
        self.counter = torch.zeros((), dtype=torch.long, device=device)
        self.update = _compiled_nesterov_step if device.type == "cuda" else _nesterov_step

    @torch.no_grad()
    def step(self) -> None:
        params, grads, buffers, decays, groups = [], [], [], [], []
        for k, group in enumerate(self.param_groups):
            live = [p for p in group["params"] if p.grad is not None]
            if live:
                params.append(live)
                grads.append([p.grad for p in live])
                buffers.append([self.buffers[p] for p in live])
                decays.append(group["weight_decay"])
                groups.append(k)
        self.update(
            params,
            grads,
            buffers,
            tuple(decays),
            tuple(groups),
            self.momentum,
            self.table,
            self.counter,
        )

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None


def make_optimizer(model: Net, config: RecipeConfig, total_steps: int, steps_per_epoch: int) -> SGD:
    """Parameter groups and lr schedules. BatchNorm biases take ``bias_scaler`` times the lr
    at the same decay strength; the whitening bias trains for ``whiten_bias_epochs``."""
    norm_ids = {id(m.bias) for m in model.modules() if isinstance(m, nn.BatchNorm2d)}
    whiten = model.whiten.bias
    params = [p for p in model.parameters() if p.requires_grad]
    norms = [p for p in params if id(p) in norm_ids]
    others = [p for p in params if id(p) not in norm_ids and p is not whiten]
    stage1 = {id(p) for p in model.groups[0].parameters()} if config.stage1_freeze else set()
    norms1 = [p for p in norms if id(p) in stage1]
    others1 = [p for p in others if id(p) in stage1]
    norms = [p for p in norms if id(p) not in stage1]
    others = [p for p in others if id(p) not in stage1]
    kilostep = 1024 * (1 + 1 / (1 - config.momentum))
    lr = config.lr / kilostep
    wd = config.weight_decay * config.batch_size / kilostep
    lr_bias = lr * config.bias_scaler
    schedule = triangle(total_steps, config.lr_start, config.lr_peak_frac, config.lr_end)
    whiten_steps = ceil(config.whiten_bias_epochs * steps_per_epoch)

    def whiten_schedule(i: int) -> float:
        return schedule(i) if i < whiten_steps else 0.0

    cooled = schedule
    if config.stage1_freeze:
        start, end = (f * total_steps for f in config.stage1_freeze)

        def cooled(i: int) -> float:
            # Stage 1's lr ramps linearly to zero between the two fractions.
            return schedule(i) * min(1.0, max(0.0, (end - i) / max(1.0, end - start)))

    groups = [
        ({"params": norms, "lr": lr_bias, "weight_decay": wd / lr_bias}, schedule),
        ({"params": others, "lr": lr, "weight_decay": wd / lr}, schedule),
        ({"params": [whiten], "lr": lr, "weight_decay": wd / lr}, whiten_schedule),
        ({"params": norms1, "lr": lr_bias, "weight_decay": wd / lr_bias}, cooled),
        ({"params": others1, "lr": lr, "weight_decay": wd / lr}, cooled),
    ]
    return SGD([(g, s) for g, s in groups if g["params"]], config.momentum, total_steps)


class Lookahead:
    """Lookahead: every few steps, pull the weights toward a slow average and copy it back."""

    def __init__(self, model: Net) -> None:
        floating = (torch.half, torch.float)
        self.current = [v for v in model.state_dict().values() if v.dtype in floating]
        self.slow = [v.detach().clone() for v in self.current]

    @torch.no_grad()
    def update(self, decay: float) -> None:
        torch._foreach_lerp_(self.slow, self.current, 1 - decay)
        torch._foreach_copy_(self.current, self.slow)


class Objective(nn.Module):
    """Model plus the label-smoothed training loss, compiled as one graph."""

    def __init__(self, model: Net, label_smoothing: float) -> None:
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        return F.cross_entropy(
            outputs, labels, label_smoothing=self.label_smoothing, reduction="sum"
        )


def resolution(config: RecipeConfig, step: int, total_steps: int) -> int | None:
    """Training resolution in force at ``step`` under ``res_schedule``."""
    size = None
    for start, value in config.res_schedule:
        if step >= start * total_steps:
            size = value
    return size


def fit(model: Net, step_model: nn.Module, stream: TrainingStream, config: RecipeConfig) -> None:
    """Train for ``config.epochs`` epochs (a fractional final epoch stops mid-epoch)."""
    total_steps = ceil(stream.steps_per_epoch * config.epochs)
    optimizer = make_optimizer(model, config, total_steps, stream.steps_per_epoch)
    lookahead = Lookahead(model) if config.lookahead else None
    whiten_steps = ceil(config.whiten_bias_epochs * stream.steps_per_epoch)
    # Steps after the last periodic lookahead update cannot change the result: stop there.
    stop = total_steps - total_steps % 5 if lookahead is not None else total_steps
    model.train()
    step = 0
    for epoch in range(ceil(config.epochs)):
        if step >= stop:
            break
        for inputs, labels in stream.epoch(epoch):
            if config.stage1_freeze:
                # Stage 1's lr is zero from here, so skipping its backward is exact.
                model.stage1_frozen = step >= config.stage1_freeze[1] * total_steps
            size = resolution(config, step, total_steps)
            if size is not None and size != inputs.size(-1):
                inputs = F.interpolate(
                    inputs, size=(size, size), mode="bilinear", align_corners=False, antialias=True
                )
            if step == whiten_steps:
                # The whitening bias is frozen from here: drop it (and the stem's gradient).
                model.whiten.bias.requires_grad_(False)
            loss = step_model(inputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if lookahead is not None and step % 5 == 0:
                lookahead.update(decay=0.95**5 * (step / total_steps) ** 3)
            if step >= stop:
                break
    model.whiten.bias.requires_grad_(True)
    model.stage1_frozen = False
    if lookahead is not None:
        lookahead.update(decay=1.0)
