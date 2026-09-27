import pytest
import torch
from torch import nn

from benchmark.evaluate import accuracy, predict


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    return torch.device(request.param)


class Classifier(nn.Module):
    def __init__(self, behavior="ok"):
        super().__init__()
        self.behavior = behavior
        self.register_buffer("seen", torch.tensor(0), persistent=False)

    def forward(self, inputs):
        assert inputs.dtype == torch.float32
        assert inputs.min() >= 0 and inputs.max() <= 1
        logits = inputs.new_zeros((len(inputs), 100))
        logits[:, 7] = 1
        if self.behavior == "mutate":
            self.seen.add_(1)
        if self.behavior == "cast":
            self.seen = self.seen.to(torch.float32)
        if self.behavior == "nan":
            logits[0, 0] = float("nan")
        if self.behavior == "shape":
            return logits[:, :99]
        return logits


def test_plain_inference_uses_every_example_including_partial_batch(device):
    images = torch.randint(0, 256, (17, 3, 32, 32), dtype=torch.uint8)
    predictions, elapsed = predict(Classifier().to(device), images, device, 8)
    assert predictions == [7] * 17
    assert elapsed >= 0
    assert accuracy(predictions, torch.tensor([7] * 16 + [8])) == 16 / 17


@pytest.mark.parametrize("behavior", ["mutate", "cast", "nan", "shape"])
def test_evaluation_rejects_invalid_or_mutating_classifiers(behavior, device):
    with pytest.raises(ValueError):
        predict(
            Classifier(behavior).to(device),
            torch.zeros(2, 3, 32, 32, dtype=torch.uint8),
            device,
            8,
        )


def test_supervisor_rejects_incomplete_predictions():
    with pytest.raises(ValueError):
        accuracy([1], torch.tensor([1, 2]))
