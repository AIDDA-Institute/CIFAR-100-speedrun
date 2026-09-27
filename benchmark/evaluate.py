"""Organizer-owned inference. This module never gives test labels to a model."""

import torch
from torch import nn

from benchmark.timing import timestamp


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    # named_buffers includes nonpersistent buffers omitted by state_dict().
    return {
        **{f"parameter:{k}": v for k, v in model.named_parameters()},
        **{f"buffer:{k}": v for k, v in model.named_buffers()},
    }


def predict(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], float]:
    start = timestamp(device)
    if not isinstance(model, nn.Module):
        raise TypeError("train() must return a torch.nn.Module classifier")
    # Snapshot before eval() so a custom eval()/train(False) cannot change learned state.
    before = {name: value.detach().clone() for name, value in model_state(model).items()}
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in images.split(batch_size):
            # The input convention is fixed; recipe-specific normalization is in forward().
            inputs = batch.to(device=device, dtype=torch.float32).div_(255)
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch), 100):
                raise ValueError("Classifier must return a tensor of shape [B, 100]")
            if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
                raise ValueError("Classifier returned nonfloating or nonfinite logits")
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    after = model_state(model)
    if before.keys() != after.keys() or any(
        value.dtype != after[name].dtype
        or value.device != after[name].device
        or value.layout != after[name].layout
        or not torch.equal(value, after[name])
        for name, value in before.items()
    ):
        raise ValueError("Evaluation mutated model parameters or buffers")
    elapsed = timestamp(device) - start
    return predictions, elapsed


def accuracy(predictions: list[int], labels: torch.Tensor) -> float:
    if len(predictions) != len(labels) or not len(labels):
        raise ValueError("Prediction count does not match the full test set")
    if any(type(p) is not int or not 0 <= p < 100 for p in predictions):
        raise ValueError("Predictions must be integer class indices in [0, 99]")
    return sum(p == y for p, y in zip(predictions, labels.tolist(), strict=True)) / len(labels)
