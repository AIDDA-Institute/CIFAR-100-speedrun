"""Render SCOREBOARD.md from the machine-readable scoreboard.json."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "scoreboard.json"
OUTPUT_PATH = ROOT / "SCOREBOARD.md"


def render(data: dict) -> str:
    policy = data["official_policy"]
    lines = [
        "# CIFAR-100 speedrun scoreboard",
        "",
        "<!-- Generated from scoreboard.json by `python -m benchmark.render_scoreboard`. -->",
        "",
        f"Official submissions use {policy['trials']} trials on {policy['gpu']} and need at least "
        f"{policy['accuracy_target_percent']}% mean accuracy. Qualifying submissions are ordered "
        "by mean preparation + training time; lower is better.",
        "",
        "The organizer baseline is a short reference pilot, not a ranked submission. Its timing "
        "is an indication only; the run used two trials. Accuracy and timing variation are shown "
        "as mean ± sample standard deviation across trials.",
        "",
        "| Entry | Submission / attribution | Status | Trials | Mean test accuracy | "
        "Mean prepare + train | Submitted / measured |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]

    for entry in sorted(data["entries"], key=lambda row: row["display_order"]):
        run = entry["run"]
        if entry["category"] == "reference_baseline":
            name = f"[{entry['display_name']}](README.md#verified-a100-setup)"
            status = "Reference only; not ranked"
            submitted = "—"
        else:
            pull = entry["pull_request"]
            submitter = entry["submitter"].get("github_login")
            name = (
                f"[{entry['display_name']} / {entry['submission_slug']} "
                f"(PR #{pull['number']})]({pull['url']})"
            )
            if submitter:
                name += f"<br>@{submitter}"
            status = f"Official qualifier; rank {entry['rank']}"
            submitted = entry["submitted_at"][:10]

        measured = run["measured_at"]
        trials = f"{run['successful_trials']}/{run['trial_count']}"
        if run["official"] is False:
            trials += " (pilot)"
        accuracy = (
            f"{run['mean_accuracy_percent']:.4f}% ± "
            f"{run['accuracy_std_percentage_points']:.4f} pp"
        )
        timing = (
            f"{run['mean_prepare_train_seconds']:.4f} ± "
            f"{run['prepare_train_std_seconds']:.4f} s"
        )
        lines.append(
            f"| {entry['display_order']} | {name} | {status} | {trials} | {accuracy} | "
            f"{timing} | {submitted} / {measured} |"
        )

    lines.extend(
        [
            "",
            "The baseline pilot ran on an A100 80GB PCIe with four CPU threads, but it used only "
            "two trials and is not an official score. Raw seed values are withheld; the JSON "
            "stores "
            "a hash of each ordered seed list for provenance.",
            "",
            "PR #2’s source was frozen and matched to the 200-trial run by file hashes. The run "
            "artifact did not contain a Git commit for the harness, so its base commit and exact "
            "harness file hashes are recorded in `scoreboard.json`.",
            "",
            "For the complete entry metadata, source hashes, and run IDs, see "
            "[`scoreboard.json`](scoreboard.json).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if SCOREBOARD.md is stale instead of writing it",
    )
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text())
    expected = render(data)
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text() != expected:
            parser.error("SCOREBOARD.md is stale; rerun without --check to regenerate it")
        return
    OUTPUT_PATH.write_text(expected)


if __name__ == "__main__":
    main()
