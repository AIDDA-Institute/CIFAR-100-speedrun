"""One disposable process per submission; build is reused across its trials."""

import importlib
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from benchmark.api import BuildContext
from benchmark.config import RunConfig
from benchmark.data import load_split, synthetic_split
from benchmark.evaluate import predict
from benchmark.hardware import gpu_telemetry, inspect_environment
from benchmark.timing import timestamp


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_submission(directory: Path):
    directory = directory.resolve()
    os.environ["CIFAR100_SUBMISSION_DIR"] = str(directory)
    name = "benchmark._submission"
    # The import bridge also exists in fresh spawn interpreters. Clear any prior
    # recipe and its relative imports when loading another recipe in this process.
    for loaded in tuple(sys.modules):
        if loaded == name or loaded.startswith(name + "."):
            del sys.modules[loaded]
    sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    for entrypoint in ("build", "prepare", "train"):
        if not callable(getattr(module, entrypoint, None)):
            raise ValueError(f"submission.py must define {entrypoint}()")
    return module


def run_worker(
    connection,
    submission: str,
    data_root: str,
    test_images: np.ndarray,
    seeds: list[int],
    parameters: dict,
    config_dict: dict,
) -> None:
    # The supervisor can kill the process group, including subprocesses launched by a recipe.
    if os.name == "posix":
        os.setsid()
    config = RunConfig(**config_dict)
    phase = "startup"
    trial_index = None
    seed = None
    timing = {}

    def begin(name: str) -> None:
        nonlocal phase
        phase = name
        connection.send(
            {
                "type": "phase",
                "phase": name,
                "started": time.monotonic(),
                "trial": trial_index,
                "seed": seed,
            }
        )
        # Let the supervisor establish the trial identity and deadline before
        # any phase work begins, including work interrupted by cancellation.
        if connection.recv() != "ready":
            raise RuntimeError("Unexpected phase acknowledgement")

    try:
        torch.set_num_threads(config.cpu_threads)
        torch.set_num_interop_threads(1)
        device = torch.device(config.device)
        environment = inspect_environment(config)
        connection.send({"type": "environment", "environment": environment})
        train_data = (
            synthetic_split(train=True)
            if config.synthetic
            else load_split(Path(data_root), train=True)
        )
        images = torch.from_numpy(test_images)
        begin("build")
        start = timestamp(device)
        module = load_submission(Path(submission))
        state = module.build(
            BuildContext(device, parameters, eval_batch_size=config.eval_batch_size)
        )
        build_time = timestamp(device) - start
        connection.send({"type": "built", "build_time": build_time})
        for trial_index, seed in enumerate(seeds):
            phase = "setup"
            timing = {}
            telemetry = gpu_telemetry() if device.type == "cuda" else []
            seed_everything(seed)
            begin("train")
            start = timestamp(device)
            module.prepare(state, train_data, seed)
            prepared = timestamp(device)
            model = module.train(state)
            trained = timestamp(device)
            timing = {
                "prepare_time": prepared - start,
                "train_time": trained - prepared,
                "total_timed_time": trained - start,
            }
            connection.send({"type": "trained", "trial": trial_index, "seed": seed, **timing})
            begin("eval")
            predictions, evaluation_time = predict(model, images, device, config.eval_batch_size)
            # The recipe owns any reusable model in state. Do not keep an extra
            # reference alive while the next trial creates and trains a new one.
            del model
            connection.send(
                {
                    "type": "trial",
                    "trial": trial_index,
                    "seed": seed,
                    "status": "ok",
                    "predictions": predictions,
                    **timing,
                    "evaluation_time": evaluation_time,
                    "telemetry": telemetry,
                }
            )
        connection.send({"type": "done"})
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "phase": phase,
                    "trial": trial_index,
                    "seed": seed,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    **timing,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            # The supervisor may already have exited during cancellation.
            pass
    finally:
        connection.close()
