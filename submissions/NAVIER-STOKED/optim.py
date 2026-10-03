"""Optimizers that can be captured in a CUDA graph: learning rates live in 0-dim GPU
tensors, and every buffer is allocated once and zeroed in place by reset()."""

import torch

# zeropower_newton_schulz and _muon_update are adapted from Keller Jordan's Muon
# (https://github.com/KellerJordan/Muon, MIT License, Copyright (c) 2024 Keller Jordan;
# full notice in LICENSE.muon).


def zeropower_newton_schulz(g: torch.Tensor, steps: int = 3, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2D matrix (quintic Newton-Schulz iteration, as in Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return x.T if transposed else x


def _muon_update(params, grads, bufs, lr, momentum: float, ns_steps: int):
    """One Muon step for a list of conv filters: Nesterov momentum, filter
    renormalization, orthogonalized update. Compiled once, so its elementwise
    work is fused."""
    for p, g, buf in zip(params, grads, bufs, strict=True):
        buf.mul_(momentum).add_(g)
        u = g + buf * momentum
        p.mul_(len(p) ** 0.5 / p.norm())
        update = zeropower_newton_schulz(u.reshape(len(u), -1), ns_steps).view(u.shape)
        p.sub_(update.to(p.dtype) * lr)


class Muon:
    """Muon for 3x3 conv filters."""

    def __init__(self, params, lr: float, momentum: float, ns_steps: int, device):
        self.params = list(params)
        self.base_lr, self.momentum, self.ns_steps = lr, momentum, ns_steps
        self.lr_t = torch.zeros((), device=device)
        self.bufs = [torch.zeros_like(p) for p in self.params]
        self._step = torch.compile(_muon_update)

    def groups(self):
        return [(self.base_lr, self.lr_t)]

    def reset(self) -> None:
        torch._foreach_zero_(self.bufs)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self._step(
            self.params,
            [p.grad for p in self.params],
            self.bufs,
            self.lr_t,
            self.momentum,
            self.ns_steps,
        )


class GraphSGD:
    """Nesterov SGD with decoupled parameter groups (head with weight decay; BN and
    biases without)."""

    def __init__(self, groups, momentum: float, device):
        self.momentum = momentum
        self.param_groups = [
            {
                "params": list(g["params"]),
                "weight_decay": g["weight_decay"],
                "base_lr": g["lr"],
                "lr_t": torch.zeros((), device=device),
                "bufs": [torch.zeros_like(p) for p in g["params"]],
            }
            for g in groups
            if g["params"]
        ]

    def groups(self):
        return [(g["base_lr"], g["lr_t"]) for g in self.param_groups]

    def reset(self) -> None:
        for g in self.param_groups:
            torch._foreach_zero_(g["bufs"])

    def zero_grad(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for g in self.param_groups:
            params, bufs = g["params"], g["bufs"]
            grads = [p.grad for p in params]
            if g["weight_decay"]:
                grads = torch._foreach_add(grads, params, alpha=g["weight_decay"])
            torch._foreach_mul_(bufs, self.momentum)
            torch._foreach_add_(bufs, grads)
            update = torch._foreach_add(grads, bufs, alpha=self.momentum)
            update = torch._foreach_mul(update, g["lr_t"])
            torch._foreach_sub_(params, update)
