"""Run a PR submission using organizer-owned timing and evaluation."""

import argparse
import json
import re
from pathlib import Path

from benchmark.config import ACCURACY_TARGET, EVAL_TIMEOUT_SECONDS, OFFICIAL_TRIALS, RunConfig
from benchmark.harness import run_submission
from benchmark.interrupts import RunInterrupted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    submissions = parser.add_mutually_exclusive_group(required=True)
    submissions.add_argument("--submission", help="Folder name under submissions/")
    submissions.add_argument("--submission-path", type=Path, help="Development/calibration folder")
    submissions.add_argument(
        "--all", action="store_true", help="Run every folder under submissions/"
    )
    parser.add_argument(
        "--n", type=int, default=OFFICIAL_TRIALS, help="Number of fresh training trials"
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--accuracy-target",
        type=float,
        default=ACCURACY_TARGET,
        help="Accuracy fraction (default: 0.75); overrides are development-only",
    )
    target.add_argument(
        "--no-accuracy-target",
        dest="accuracy_target",
        action="store_const",
        const=None,
        default=argparse.SUPPRESS,
        help="Report without qualification during development",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--official", action="store_true")
    parser.add_argument(
        "--synthetic", action="store_true", help="Tiny smoke fixtures; never a score"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=EVAL_TIMEOUT_SECONDS,
        help=(
            f"Seconds for the full test pass (default: {EVAL_TIMEOUT_SECONDS:g}; "
            "fixed for official runs)"
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--build-timeout", type=float, default=600)
    parser.add_argument("--trial-timeout", type=float, default=600)
    parser.add_argument("--cpu-threads", type=int, default=4)
    rng = parser.add_mutually_exclusive_group()
    rng.add_argument("--seed", type=int, help="Development seeds start here (default: 0)")
    rng.add_argument("--seed-file", type=Path, help="Organizer JSON array of distinct trial seeds")
    parameters = parser.add_mutually_exclusive_group()
    parameters.add_argument("--params", default="{}", help="JSON object passed to build()")
    parameters.add_argument("--params-file", type=Path, help="JSON recipe parameters")
    args = parser.parse_args()
    try:
        config = RunConfig(
            n_trials=args.n,
            accuracy_target=args.accuracy_target,
            device=args.device,
            official=args.official,
            synthetic=args.synthetic,
            eval_timeout=args.eval_timeout,
            eval_batch_size=args.eval_batch_size,
            build_timeout=args.build_timeout,
            trial_timeout=args.trial_timeout,
            cpu_threads=args.cpu_threads,
        )
        config.validate()
        if args.official and (args.submission_path is not None or args.seed is not None):
            raise ValueError("Official runs use submissions/ and an organizer-owned seed file")
        if args.official and args.seed_file is None:
            raise ValueError("Official runs require --seed-file; use the same file for every team")
        if args.submission:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.submission):
                raise ValueError(
                    "Submission name must contain only letters, digits, underscores, hyphens"
                )
            paths = [Path("submissions") / args.submission]
        elif args.all:
            paths = sorted(p.parent for p in Path("submissions").glob("*/submission.py"))
            if not paths:
                raise ValueError("No submissions found")
        else:
            paths = [args.submission_path]
        params = json.loads(args.params_file.read_text() if args.params_file else args.params)
        if not isinstance(params, dict):
            raise ValueError("Recipe parameters must be a JSON object")
        if args.seed_file:
            seeds = json.loads(args.seed_file.read_text())
            if not isinstance(seeds, list):
                raise ValueError("Seed file must contain a JSON array")
        else:
            start_seed = 0 if args.seed is None else args.seed
            seeds = list(range(start_seed, start_seed + args.n))
        failed = False
        for path in paths:
            result_dir, summary = run_submission(
                path,
                data_root=args.data_root,
                results_root=args.results_root,
                seeds=seeds,
                parameters=params,
                config=config,
            )
            print(json.dumps(summary, indent=2))
            print(f"Results: {result_dir}", flush=True)
            failed |= not summary["complete"] or summary["qualified"] is False
        if failed:
            raise SystemExit(1)
    except RunInterrupted as exc:
        if exc.summary is not None:
            print(json.dumps(exc.summary, indent=2))
            print(f"Results: {exc.result_dir}", flush=True)
        parser.exit(128 + exc.signum, f"Stopped: {exc}\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
