import pytest

from benchmark.config import RunConfig
from benchmark.scoring import summarize


def trial(accuracy, time):
    return {"status": "ok", "accuracy": accuracy, "total_timed_time": time, "evaluation_time": 0.1}


def test_score_is_mean_training_time_with_mean_accuracy_gate():
    summary = summarize([trial(0.6, 2), trial(0.8, 4)], requested=2, target=0.7)
    assert summary["qualified"] is True
    assert summary["mean_training_time"] == 3
    assert summary["mean_accuracy"] == 0.7


def test_failed_or_missing_trials_never_qualify_from_survivors():
    good = trial(0.9, 1)
    failed = {"status": "eval_timeout"}
    assert summarize([good, failed], requested=2, target=0.5)["qualified"] is False
    assert summarize([good], requested=2, target=0.5)["qualified"] is False
    assert summarize([good], requested=1, target=0.5, run_error="exit")["qualified"] is False


def test_unset_target_and_synthetic_are_not_qualifying_scores():
    assert summarize([trial(0.9, 1)], requested=1, target=None)["qualified"] is None
    assert summarize([trial(0.9, 1)], requested=1, target=0.5, synthetic=True)["qualified"] is None


@pytest.mark.parametrize("target", [float("nan"), 65.0, 0, -0.1])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        RunConfig(accuracy_target=target).validate()


def test_official_configuration_is_fixed():
    config = RunConfig(official=True)
    assert config.n_trials == 50
    assert config.eval_timeout == 5
    config.validate()
    for kwargs in (
        {"device": "cpu"},
        {"n_trials": 1},
        {"n_trials": 200},
        {"eval_timeout": 6},
        {"eval_timeout": 30},
        {"synthetic": True},
        {"cpu_threads": 8},
        {"accuracy_target": 0.70},
        {"accuracy_target": None},
    ):
        with pytest.raises(ValueError):
            RunConfig(official=True, **kwargs).validate()


def test_default_threshold_and_exact_qualification_boundary():
    target = RunConfig().accuracy_target
    assert target == 0.75
    assert summarize([trial(0.7499, 1)], requested=1, target=target)["qualified"] is False
    assert summarize([trial(0.75, 1)], requested=1, target=target)["qualified"] is True
