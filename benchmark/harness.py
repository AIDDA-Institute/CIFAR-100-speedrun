import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from benchmark.config import RunConfig
from benchmark.data import load_split, synthetic_split
from benchmark.evaluate import accuracy
from benchmark.interrupts import RunInterrupted, handle_interrupts
from benchmark.scoring import summarize
from benchmark.worker import run_worker


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_manifest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def stop_worker(process) -> None:
    if process.pid is None:
        return
    # Also terminate children left behind by a recipe after its main process exits.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.kill()
    process.join(timeout=2)


def run_submission(
    submission: Path,
    *,
    data_root: Path,
    results_root: Path,
    seeds: list[int],
    parameters: dict,
    config: RunConfig,
) -> tuple[Path, dict]:
    config.validate()
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    if len(seeds) != config.n_trials or len(set(seeds)) != len(seeds):
        raise ValueError("Supply exactly n_trials distinct seeds")
    submission = submission.resolve()
    if results_root.resolve().is_relative_to(submission):
        raise ValueError("Results must be outside the submission folder")
    if not (submission / "submission.py").is_file():
        raise ValueError(f"Missing {submission / 'submission.py'}")
    if any(p.is_symlink() for p in submission.rglob("*")):
        raise ValueError("Submission folders must contain source files, not symlinks")
    test_data = (
        synthetic_split(train=False) if config.synthetic else load_split(data_root, train=False)
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    result_dir = results_root.resolve() / submission.name / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    # Freeze exactly the source being measured and preserve it with the result.
    source = result_dir / "source"
    shutil.copytree(
        submission, source, ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv")
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    benchmark_dir = Path(__file__).parent
    harness_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(benchmark_dir.glob("*.py"))
    }
    metadata = {
        **config.to_dict(),
        "run_id": run_id,
        "submission": str(submission),
        "parameters": parameters,
        "seeds": seeds,
        "harness_commit": commit,
        "harness_sha256": harness_hashes,
        "submission_sha256": source_manifest(source),
        "dataset": "synthetic" if config.synthetic else "CIFAR-100",
        "evaluation": "single-view",
        "accuracy_units": "fraction",
        "time_units": "seconds",
    }
    write_json(result_dir / "config.json", metadata)
    context = mp.get_context("spawn")
    # The return channel acknowledges phase boundaries before work can begin.
    receiver, sender = context.Pipe(duplex=True)
    # Only images cross into the worker. Test labels stay in this supervisor.
    process = context.Process(
        target=run_worker,
        args=(
            sender,
            str(source),
            str(data_root.resolve()),
            test_data.images.numpy(),
            seeds,
            parameters,
            config.to_dict(),
        ),
    )
    trials = []
    run_error = None
    current_trial = None
    current_seed = None
    last_timing = {}
    phase = "startup"
    deadline = time.monotonic() + config.startup_timeout
    completed = False
    interruption = None

    with handle_interrupts(), (result_dir / "trials.jsonl").open("w", buffering=1) as log:

        def record(row: dict) -> None:
            trials.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if row["status"] == "ok":
                print(
                    f"trial {row['trial'] + 1}/{config.n_trials}: "
                    f"accuracy={row['accuracy']:.2%} "
                    f"training={row['total_timed_time']:.3f}s "
                    f"inference={row['evaluation_time']:.3f}s",
                    flush=True,
                )

        try:
            process.start()
            sender.close()
            while not completed:
                if not receiver.poll(max(0, deadline - time.monotonic())):
                    run_error = f"{phase}_timeout"
                    if current_trial is not None:
                        record(
                            {
                                "trial": current_trial,
                                "seed": current_seed,
                                "status": run_error,
                                "failure_reason": run_error,
                                **last_timing,
                            }
                        )
                    break
                try:
                    message = receiver.recv()
                except EOFError:
                    run_error = f"worker_exited_during_{phase}"
                    if current_trial is not None:
                        record(
                            {
                                "trial": current_trial,
                                "seed": current_seed,
                                "status": "worker_error",
                                "failure_reason": run_error,
                                **last_timing,
                            }
                        )
                    break
                kind = message["type"]
                if kind == "environment":
                    metadata["environment"] = message["environment"]
                    write_json(result_dir / "config.json", metadata)
                elif kind == "phase":
                    phase = message["phase"]
                    current_trial, current_seed = message["trial"], message["seed"]
                    timeout = {
                        "build": config.build_timeout,
                        "train": config.trial_timeout,
                        "eval": config.eval_timeout,
                    }[phase]
                    deadline = message["started"] + timeout
                    if phase == "train":
                        last_timing = {}
                    receiver.send("ready")
                elif kind == "built":
                    metadata["build_time"] = message["build_time"]
                    write_json(result_dir / "config.json", metadata)
                    if message["build_time"] > config.build_timeout:
                        run_error = "build_timeout"
                        break
                    phase, deadline = "setup", time.monotonic() + config.startup_timeout
                elif kind == "trained":
                    last_timing = {
                        key: message[key]
                        for key in ("prepare_time", "train_time", "total_timed_time")
                    }
                    if last_timing["total_timed_time"] > config.trial_timeout:
                        run_error = "train_timeout"
                        record(
                            {
                                "trial": current_trial,
                                "seed": current_seed,
                                "status": run_error,
                                "failure_reason": run_error,
                                **last_timing,
                            }
                        )
                        break
                elif kind == "trial":
                    row = {
                        key: value
                        for key, value in message.items()
                        if key not in ("type", "predictions")
                    }
                    if row["trial"] != len(trials) or row["seed"] != seeds[len(trials)]:
                        raise ValueError("Unexpected trial order or seed")
                    if row["evaluation_time"] > config.eval_timeout:
                        row.update(status="eval_timeout", failure_reason="eval_timeout")
                        run_error = "eval_timeout"
                    else:
                        row["accuracy"] = accuracy(message["predictions"], test_data.labels)
                    record(row)
                    if run_error:
                        break
                    current_trial, current_seed = None, None
                    phase, deadline = "setup", time.monotonic() + config.startup_timeout
                elif kind == "error":
                    run_error = message["failure_reason"]
                    (result_dir / "error.txt").write_text(message["traceback"])
                    if message["trial"] is not None:
                        record(
                            {
                                key: value
                                for key, value in message.items()
                                if key not in ("type", "traceback")
                            }
                            | {"status": "error"}
                        )
                    break
                elif kind == "done":
                    completed = True
                else:
                    raise ValueError(f"Unknown worker message: {kind}")
        except BaseException as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, KeyboardInterrupt):
                interruption = (
                    exc if isinstance(exc, RunInterrupted) else RunInterrupted(signal.SIGINT)
                )
            if current_trial is not None and not any(
                row["trial"] == current_trial for row in trials
            ):
                record(
                    {
                        "trial": current_trial,
                        "seed": current_seed,
                        "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        "failure_reason": run_error,
                        **last_timing,
                    }
                )
            if isinstance(exc, KeyboardInterrupt):
                print("Interrupted; preserving partial results.", flush=True)
        finally:
            stop_worker(process)
            sender.close()
            receiver.close()
    summary = summarize(
        trials,
        requested=config.n_trials,
        target=config.accuracy_target,
        synthetic=config.synthetic,
        run_error=run_error,
    )
    summary.update(official=config.official, dataset=metadata["dataset"], run_id=run_id)
    write_json(result_dir / "summary.json", summary)
    if interruption is not None:
        interruption.result_dir = result_dir
        interruption.summary = summary
        raise interruption
    return result_dir, summary
