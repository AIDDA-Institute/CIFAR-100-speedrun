from pathlib import Path

import torch

from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission, seed_everything


def test_template_resets_after_an_intervening_trained_trial():
    torch.set_num_threads(2)
    module = load_submission(Path(__file__).resolve().parents[1] / "submission_template")
    state = module.build(BuildContext(torch.device("cpu"), {}))
    data = synthetic_split(train=True)
    original_images = data.images.clone()

    def train(seed):
        seed_everything(seed)
        module.prepare(state, data, seed)
        model = module.train(state)
        return {key: value.clone() for key, value in model.state_dict().items()}

    first = train(42)
    different = train(43)
    repeated = train(42)
    assert any(not torch.equal(first[key], different[key]) for key in first)
    assert all(torch.equal(first[key], repeated[key]) for key in first)
    assert torch.equal(data.images, original_images)
