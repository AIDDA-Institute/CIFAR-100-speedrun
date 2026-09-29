import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmark.config import RunConfig
from benchmark.data import synthetic_split
from benchmark.harness import run_submission

RECIPE = """
import time
import torch
from torch import nn
from .helper import PREDICTED_CLASS

class Model(nn.Module):
    def forward(self, x):
        {forward}
        result = x.new_zeros((len(x), 100))
        result[:, PREDICTED_CLASS] = 1
        return result

def build(context):
    {build}
    return Model()

def prepare(state, data, seed):
    pass

def train(state):
    {train}
    return state
"""


def make_recipe(tmp_path, name="example", **overrides):
    directory = tmp_path / name
    directory.mkdir()
    code = RECIPE.format(**({"forward": "pass", "build": "pass", "train": "pass"} | overrides))
    (directory / "submission.py").write_text(code)
    (directory / "helper.py").write_text("PREDICTED_CLASS = 7\n")
    return directory


def run(tmp_path, recipe, **kwargs):
    config = RunConfig(device="cpu", synthetic=True, n_trials=2, accuracy_target=0.01, **kwargs)
    return run_submission(
        recipe,
        data_root=tmp_path / "unused",
        results_root=tmp_path / "results",
        seeds=[123, 456],
        parameters={},
        config=config,
    )


def test_real_worker_roundtrip_and_source_freezing(tmp_path):
    recipe = make_recipe(tmp_path)
    directory, summary = run(tmp_path, recipe)
    assert summary["complete"] is True
    assert summary["qualified"] is None  # synthetic cannot qualify
    expected = (synthetic_split(train=False).labels == 7).float().mean().item()
    assert summary["mean_accuracy"] == expected
    rows = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows] == [123, 456]
    for row in rows:
        assert row["total_timed_time"] == row["prepare_time"] + row["train_time"]
        assert "predictions" not in row
    assert (directory / "source" / "helper.py").read_text() == "PREDICTED_CLASS = 7\n"
    config = json.loads((directory / "config.json").read_text())
    assert config["build_time"] >= 0
    assert "submission.py" in config["submission_sha256"]


def test_worker_releases_previous_model_before_preparing_next_trial(tmp_path):
    recipe = make_recipe(tmp_path)
    source = recipe / "submission.py"
    source.write_text(
        source.read_text()
        + """
import gc
import weakref

def build(context):
    return {"previous_model": None, "previous_weight": None}

def prepare(state, data, seed):
    state.pop("model", None)
    gc.collect()
    for name in ("previous_model", "previous_weight"):
        reference = state[name]
        assert reference is None or reference() is None, name + " is still alive"
    model = Model()
    model.register_parameter("weight", nn.Parameter(torch.ones(1)))
    state["model"] = model

def train(state):
    model = state["model"]
    state["previous_model"] = weakref.ref(model)
    state["previous_weight"] = weakref.ref(model.weight)
    return model
"""
    )
    _, summary = run(tmp_path, recipe)
    assert summary["complete"] is True, summary["run_error"]
    assert summary["successful_trials"] == 2


def test_spawned_dataloader_imports_submission_and_relative_helpers(tmp_path):
    recipe = make_recipe(tmp_path)
    (recipe / "helper.py").write_text(
        """
import torch

PREDICTED_CLASS = 7

def collate_samples(samples):
    return torch.stack(samples)
"""
    )
    source = recipe / "submission.py"
    source.write_text(
        source.read_text()
        + """
from torch.utils.data import DataLoader, Dataset
from .helper import collate_samples

class Samples(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

def build(context):
    return {}

def prepare(state, data, seed):
    state["images"] = data.images

def train(state):
    loader = DataLoader(
        Samples(state["images"]), batch_size=8, num_workers=1,
        multiprocessing_context="spawn", collate_fn=collate_samples,
    )
    count = 0
    for batch in loader:
        assert batch.shape[1:] == (3, 32, 32)
        count += len(batch)
    assert count == len(state["images"])
    return Model()
"""
    )
    directory, summary = run(tmp_path, recipe, trial_timeout=30)
    assert summary["complete"] is True, summary["run_error"]
    assert summary["successful_trials"] == 2
    assert (directory / "source" / "submission.py").read_bytes() == source.read_bytes()
    assert (directory / "source" / "helper.py").read_bytes() == (recipe / "helper.py").read_bytes()


@pytest.mark.parametrize("phase", ["eval", "train", "build"])
def test_watchdog_kills_a_hung_worker(tmp_path, phase):
    entrypoint = {"eval": "forward", "train": "train", "build": "build"}[phase]
    recipe = make_recipe(tmp_path, **{entrypoint: "time.sleep(60)"})
    start = time.monotonic()
    directory, summary = run(
        tmp_path, recipe, **{f"{phase if phase != 'train' else 'trial'}_timeout": 0.3}
    )
    assert time.monotonic() - start < 30  # cannot wait for the 60-second sleep to return
    assert summary["complete"] is False
    assert summary["qualified"] is False
    assert summary["run_error"] == f"{phase}_timeout"
    rows = (directory / "trials.jsonl").read_text().splitlines()
    if phase == "build":
        assert rows == []
    else:
        assert len(rows) == 1
        assert json.loads(rows[0])["status"] == f"{phase}_timeout"


def test_recipe_exception_preserves_failure(tmp_path):
    directory, summary = run(tmp_path, make_recipe(tmp_path, train="raise RuntimeError('broken')"))
    assert summary["qualified"] is False
    assert summary["failed_trials"] == 1
    assert "broken" in (directory / "error.txt").read_text()


def test_seed_list_requires_all_distinct_requested_trials(tmp_path):
    with pytest.raises(ValueError, match="distinct seeds"):
        run_submission(
            Path("submission_template"),
            data_root=tmp_path,
            results_root=tmp_path,
            seeds=[1, 1],
            parameters={},
            config=RunConfig(device="cpu", n_trials=2),
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups and /proc inspection")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cancellation_stops_workers_preserves_results_and_aborts_all(tmp_path, signum):
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    pid_file = tmp_path / "worker-pids.json"
    marker = tmp_path / "second-submission-ran"
    make_recipe(
        submissions,
        train=(
            "child = __import__('subprocess').Popen("
            "[__import__('sys').executable, '-c', 'import time; time.sleep(60)']); "
            f"__import__('pathlib').Path({str(pid_file)!r}).write_text("
            "__import__('json').dumps([__import__('os').getpid(), child.pid])); "
            "time.sleep(60)"
        ),
    )
    make_recipe(
        submissions,
        name="z_after",
        build=f"__import__('pathlib').Path({str(marker)!r}).touch()",
    )
    command = [
        sys.executable,
        "-m",
        "benchmark.run",
        "--all",
        "--device",
        "cpu",
        "--synthetic",
        "--n",
        "2",
    ]
    process = subprocess.Popen(
        command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    worker_pids = []
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            if pid_file.exists():
                try:
                    worker_pids = json.loads(pid_file.read_text())
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.05)
        assert worker_pids, "Training worker did not start"
        process.send_signal(signum)
        output, _ = process.communicate(timeout=15)
        assert process.returncode == 128 + signum, output
        assert not marker.exists(), output
        summaries = list((tmp_path / "results").rglob("summary.json"))
        assert len(summaries) == 1, output
        summary = json.loads(summaries[0].read_text())
        assert summary["complete"] is False
        assert summary["qualified"] is False
        assert summary["failed_trials"] == 1
        assert signal.Signals(signum).name in summary["run_error"]
        rows = (summaries[0].parent / "trials.jsonl").read_text().splitlines()
        assert len(rows) == 1
        assert json.loads(rows[0])["status"] == "interrupted"
        for pid in worker_pids:
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                assert stat.read_text().split(")", 1)[1].split()[0] == "Z", output
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        for pid in worker_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
