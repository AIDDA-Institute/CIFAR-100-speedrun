"""Recipe parameters: one typed, validated source of every tunable choice."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class RecipeConfig:
    """Recipe hyperparameters; the defaults are the submission. ``lr`` and ``weight_decay``
    are per 1024 examples and independent of ``momentum``."""

    widths: tuple[int, int, int] = (128, 384, 640)
    epochs: float = 8.25
    batch_size: int = 1024
    lr: float = 11.5
    momentum: float = 0.85
    weight_decay: float = 0.0153
    bias_scaler: float = 16.0
    label_smoothing: float = 0.4
    lr_start: float = 0.2
    lr_peak_frac: float = 0.23
    lr_end: float = 0.07
    lookahead: bool = True
    whiten_bias_epochs: float = 3.0
    whiten_samples: int = 5000
    bn_momentum: float = 0.6
    scaling_factor: float = 1 / 6
    translate: int = 2
    # Progressive resizing: ((start_fraction, size), ...), ending at the 32 px test resolution.
    res_schedule: tuple[tuple[float, int], ...] = ((0.0, 20), (0.5, 32))
    # Convs per stage; 3 adds the residual conv-BN-GELU branch.
    stage_depths: tuple[int, int, int] = (3, 2, 3)
    # Identity skip around depth-2 stages (x + conv-BN-GELU(x)).
    stage_skips: tuple[bool, bool, bool] = (False, True, False)
    # Kernel size of stage 1's residual conv.
    stage1_residual_kernel: int = 1
    # (start, end) fractions of training: stage 1's lr ramps to zero between them, and from
    # ``end`` stage 1 runs without autograd (exact, since its lr is already zero).
    stage1_freeze: tuple[float, float] | None = (0.6, 0.8)
    compile: bool = True
    compile_mode: str = "max-autotune"

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> RecipeConfig:
        """Build a config from harness JSON parameters, failing loudly on unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise ValueError(f"Unknown recipe parameters: {unknown}; known: {sorted(known)}")
        values = dict(parameters)
        if "widths" in values:
            values["widths"] = tuple(values["widths"])
        if "stage_depths" in values:
            values["stage_depths"] = tuple(values["stage_depths"])
        if "stage_skips" in values:
            values["stage_skips"] = tuple(values["stage_skips"])
        if values.get("stage1_freeze") is not None:
            values["stage1_freeze"] = tuple(values["stage1_freeze"])
        if "res_schedule" in values:
            values["res_schedule"] = tuple(tuple(entry) for entry in values["res_schedule"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.widths) != 3 or not all(isinstance(w, int) and w > 0 for w in self.widths):
            raise ValueError("widths must be three positive integers")
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "whiten_samples": self.whiten_samples,
            "scaling_factor": self.scaling_factor,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.momentum < 1 or not 0 < self.bn_momentum < 1:
            raise ValueError("momentum must be in [0, 1) and bn_momentum in (0, 1)")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0 < self.lr_peak_frac < 1:
            raise ValueError("lr_peak_frac must be in (0, 1)")
        if self.translate < 0 or self.weight_decay < 0:
            raise ValueError("translate and weight_decay must be non-negative")
        if self.stage1_residual_kernel not in (1, 3):
            raise ValueError("stage1_residual_kernel must be 1 or 3")
        if len(self.stage_depths) != 3 or any(d not in (2, 3) for d in self.stage_depths):
            raise ValueError("stage_depths must be three values of 2 or 3")
        freeze = self.stage1_freeze
        if freeze is not None and (len(freeze) != 2 or not 0 <= freeze[0] < freeze[1] < 1):
            raise ValueError("stage1_freeze must be (start, end) with 0 <= start < end < 1")
        modes = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
        if self.compile_mode not in modes:
            raise ValueError(f"compile_mode must be one of {modes}")
        starts = [entry[0] for entry in self.res_schedule]
        if any(len(entry) != 2 for entry in self.res_schedule) or starts != sorted(starts):
            raise ValueError("res_schedule must be ascending (start_fraction, size) pairs")
        if any(not 0 <= f < 1 or not 8 <= size <= 32 for f, size in self.res_schedule):
            raise ValueError("res_schedule fractions must be in [0, 1) and sizes in [8, 32]")
