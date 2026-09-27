"""Minimal API demonstration. This is intentionally not a speedrun baseline."""

from types import SimpleNamespace

import torch
from torch import nn

from benchmark.api import BuildContext, TrainingData


def build(context: BuildContext):
    """Set up the model once and return the objects shared by prepare/train.

    This setup is outside the score. Optional compilation on synthetic inputs
    can go here; all work using real training data belongs in prepare or train.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, context.num_classes),
    ).to(context.device)
    return SimpleNamespace(model=model, context=context)


def prepare(state, data: TrainingData, seed: int) -> None:
    """Start one trial from scratch and store its training data in state.

    This is timed. Reset everything learned in a previous trial, including the
    model weights and optimizer, even though the same state object is reused.
    """
    # The harness has already seeded Python, NumPy, and PyTorch. Seed your own
    # generators here if your recipe uses any. All fresh-run work belongs here.
    for layer in state.model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    state.model.train()
    state.optimizer = torch.optim.SGD(state.model.parameters(), lr=0.05)
    # Only a tiny subset: enough to exercise the harness, not a useful baseline.
    state.images = data.images[:64].to(state.context.device, dtype=torch.float32).div_(255)
    state.labels = data.labels[:64].to(state.context.device)


def train(state) -> nn.Module:
    """Train for one complete trial, then return the model for accuracy checking.

    This is timed. The harness handles evaluation and scoring after we return.
    """
    for _ in range(3):
        state.optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(state.model(state.images), state.labels)
        loss.backward()
        state.optimizer.step()
    return state.model
