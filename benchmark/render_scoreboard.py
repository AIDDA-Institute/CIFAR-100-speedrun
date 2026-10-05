"""Render SCOREBOARD.md from the machine-readable scoreboard.json."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "scoreboard.json"
OUTPUT_PATH = ROOT / "SCOREBOARD.md"
SVG_PATH = ROOT / "scoreboard.svg"


def measurement_date(entry: dict) -> str:
    return entry["run"]["measured_at"]


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
            "The graph uses each run’s measurement date. PR creation dates are listed separately "
            "in the table.",
            "",
            "![Mean preparation and training time by measurement date](scoreboard.svg)",
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


def render_svg(data: dict) -> str:
    """Render a dependency-free timeline chart of scoreboard run times."""
    from datetime import date

    entries = sorted(data["entries"], key=lambda row: row["display_order"])
    width, height = 1000, 620
    left, right, top, bottom = 150, 850, 115, 485
    min_seconds, max_seconds = 1, 100
    dates = [measurement_date(entry) for entry in entries]
    start_day = date.fromisoformat(min(dates))
    end_day = date.fromisoformat(max(dates))
    span_days = max(1, (end_day - start_day).days)

    def x_for(date_string: str) -> float:
        day = date.fromisoformat(date_string)
        return left + (day - start_day).days / span_days * (right - left)

    def y_for(seconds: float) -> float:
        clipped = min(max(seconds, min_seconds), max_seconds)
        fraction = math.log(clipped / min_seconds) / math.log(max_seconds / min_seconds)
        return bottom - fraction * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="chart-title chart-description">',
        '<title id="chart-title">CIFAR-100 speedrun scoreboard performance</title>',
        '<desc id="chart-description">Mean preparation and training seconds by measurement date. '
        "The vertical axis is logarithmic. The baseline is a two-trial reference; PR number 2 "
        'is the official 200-trial qualifier.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.grid{stroke:#d9e0ea;stroke-width:1}'
        '.axis{stroke:#596579;stroke-width:1.5}.tick{font-size:14px;fill:#526071}'
        '.title{font-size:24px;font-weight:700}.subtitle{font-size:15px;fill:#526071}'
        '.label{font-size:16px;font-weight:700}.detail{font-size:14px;fill:#526071}'
        '.baseline{fill:#64748b;stroke:#fff;stroke-width:2}.submission{fill:#1769aa;stroke:#fff;stroke-width:2}'
        '</style>',
        '<text class="title" x="150" y="42">Mean preparation + training time</text>',
        '<text class="subtitle" x="150" y="70">Lower is faster · logarithmic seconds · horizontal '
        "axis shows run measurement dates</text>",
    ]

    for seconds in (1, 2, 5, 10, 20, 50, 100):
        y = y_for(seconds)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{left - 14}" y="{y + 5:.1f}" '
            f'text-anchor="end">{seconds}</text>'
        )

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<text class="tick" transform="translate(38 {(top + bottom) / 2:.1f}) '
            'rotate(-90)" text-anchor="middle">Mean prepare + train time (seconds)</text>',
        ]
    )

    for date_string in sorted(set(dates)):
        x = x_for(date_string)
        parsed = date.fromisoformat(date_string)
        label = f"{parsed.strftime('%b')} {parsed.day}, {parsed.year}"
        parts.extend(
            [
                f'<line class="axis" x1="{x:.1f}" y1="{bottom}" '
                f'x2="{x:.1f}" y2="{bottom + 6}"/>',
                f'<text class="tick" x="{x:.1f}" y="{bottom + 30}" '
                f'text-anchor="middle">{label}</text>',
            ]
        )

    for entry in entries:
        run = entry["run"]
        x = x_for(measurement_date(entry))
        y = y_for(run["mean_prepare_train_seconds"])
        if entry["category"] == "reference_baseline":
            marker = (
                f'<rect class="baseline" x="{x - 8:.1f}" y="{y - 8:.1f}" '
                'width="16" height="16" rx="2"/>'
            )
            anchor, label_x = "start", x + 25
            title = f"Organizer baseline · {run['mean_prepare_train_seconds']:.2f} s"
            detail = f"{run['mean_accuracy_percent']:.2f}% accuracy"
        else:
            marker = (
                f'<circle class="submission" cx="{x:.1f}" cy="{y:.1f}" r="10"/>'
            )
            anchor, label_x = "end", x - 25
            title = f"PR #2 · {run['mean_prepare_train_seconds']:.4f} s"
            detail = f"{run['mean_accuracy_percent']:.4f}% accuracy"
        parts.extend(
            [
                marker,
                f'<text class="label" x="{label_x:.1f}" y="{y - 6:.1f}" '
                f'text-anchor="{anchor}">{title}</text>',
                f'<text class="detail" x="{label_x:.1f}" y="{y + 17:.1f}" '
                f'text-anchor="{anchor}">{detail}</text>',
            ]
        )

    parts.extend(
        [
            '<text class="subtitle" x="150" y="570">Run measurement dates are listed '
            "in the table.</text>",
            "</svg>",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated scoreboard files are stale instead of writing them",
    )
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text())
    expected = render(data)
    expected_svg = render_svg(data)
    if args.check:
        if (
            not OUTPUT_PATH.exists()
            or OUTPUT_PATH.read_text() != expected
            or not SVG_PATH.exists()
            or SVG_PATH.read_text() != expected_svg
        ):
            parser.error("scoreboard output is stale; rerun without --check to regenerate it")
        return
    OUTPUT_PATH.write_text(expected)
    SVG_PATH.write_text(expected_svg)


if __name__ == "__main__":
    main()
