import json
import sys

import pytest

from benchmark import run


@pytest.mark.parametrize(
    "arguments, expected_target, expected_trials",
    [
        ([], 0.75, 200),
        (["--accuracy-target", "0.8", "--n", "3"], 0.8, 3),
        (["--no-accuracy-target"], None, 200),
    ],
)
def test_cli_passes_the_selected_configuration(
    monkeypatch, tmp_path, arguments, expected_target, expected_trials
):
    observed = []

    def run_submission(*args, **kwargs):
        config = kwargs["config"]
        assert config.eval_timeout == 5
        observed.append((config.accuracy_target, config.n_trials, kwargs["seeds"]))
        return tmp_path, {"complete": True, "qualified": None}

    monkeypatch.setattr(run, "run_submission", run_submission)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.run",
            "--submission-path",
            "submission_template",
            "--device",
            "cpu",
            "--synthetic",
            *arguments,
        ],
    )
    run.main()
    assert observed == [(expected_target, expected_trials, list(range(expected_trials)))]


def test_official_cli_requires_shared_seed_file(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark.run", "--submission", "team", "--official"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert "--seed-file" in capsys.readouterr().err


def test_official_cli_preserves_the_organizer_seed_list(monkeypatch, tmp_path):
    seeds = list(range(1000, 1200))
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps(seeds))
    observed = []

    def run_submission(*args, **kwargs):
        observed.append(kwargs)
        return tmp_path, {"complete": True, "qualified": True}

    monkeypatch.setattr(run, "run_submission", run_submission)
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark.run", "--submission", "team", "--official", "--seed-file", str(seed_file)],
    )
    run.main()
    assert observed[0]["seeds"] == seeds
    assert observed[0]["config"].official is True
    assert observed[0]["config"].n_trials == 200
