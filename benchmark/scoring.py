import math
import statistics


def summarize(
    trials: list[dict],
    *,
    requested: int,
    target: float | None,
    synthetic: bool = False,
    run_error: str | None = None,
) -> dict:
    successful = [t for t in trials if t["status"] == "ok"]
    complete = len(trials) == requested and len(successful) == requested and run_error is None
    result = {
        "requested_trials": requested,
        "number_of_trials": len(trials),
        "successful_trials": len(successful),
        "failed_trials": len(trials) - len(successful),
        "complete": complete,
        "accuracy_target": target,
        "run_error": run_error,
    }
    for source, prefix in (
        ("accuracy", "accuracy"),
        ("total_timed_time", "training_time"),
        ("evaluation_time", "evaluation_time"),
    ):
        values = [t[source] for t in successful]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite trial values cannot be scored")
        result[f"mean_{prefix}"] = statistics.mean(values) if values else None
        result[f"{prefix}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    result["qualified"] = (
        False
        if not complete
        else None
        if target is None or synthetic
        else result["mean_accuracy"] >= target
    )
    return result
