# Organizer guide

This guide covers official evaluation and maintenance of the benchmark. For the
contestant workflow, start with [README.md](README.md). Launch checks
are listed in the README's [TO DO section](README.md#to-do).

## Official container

The Dockerfile targets Linux x86-64, Ubuntu 22.04, CUDA 12.4.1, Python 3.12.
Official judging requires one full NVIDIA A100 80GB PCIe with MIG disabled.
Build on that GPU host (or with an amd64 builder) and download data with networking
enabled before the competition run:

```bash
docker build --platform linux/amd64 -t cifar100-speedrun .
docker run --rm -v "$PWD/data:/data" cifar100-speedrun python -m benchmark.data --root /data
```

Generate the organizer's seed file once, keep it private until submissions are
frozen, and reuse that exact file for every team. This refuses to overwrite an
existing file:

```bash
uv run python - <<'PY'
import json
import secrets
from benchmark.config import OFFICIAL_TRIALS

seeds = secrets.SystemRandom().sample(range(2**32), OFFICIAL_TRIALS)
with open("seeds.json", "x") as output:
    json.dump(seeds, output)
PY
```

Run the frozen submission with that seed file:

```bash
docker run --rm --gpus '"device=0"' --cpus 4 --network none --ipc=host \
  -v "$PWD/data:/data:ro" -v "$PWD/results:/results" \
  -v "$PWD/seeds.json:/seeds.json:ro" \
  cifar100-speedrun python -m benchmark.run --submission my_team \
  --official --seed-file /seeds.json --data-root /data --results-root /results
```

The image must contain the frozen submission. Official mode requires PyTorch and
`nvidia-smi` to report exactly one `NVIDIA A100 80GB PCIe`, and `nvidia-smi` to report
`mig.mode.current` as `Disabled`. It checks software versions and OS, and rejects
non-loopback network interfaces. Docker's `--network none` enforces network isolation; `--cpus 4`
enforces the CPU quota across all submission processes. The harness sets
PyTorch's thread count to four but does not verify the container CPU quota.

Use the same host/provider, CPU allocation, driver, and GPU power/clock policy
for all entries; record the container image digest. Compare close results under
matching host and thermal conditions. Official mode requires `--seed-file` for
individual submissions and `--all`; organizers must reuse the same file across
separate runs.

Official runs require 40 successful trials and enforce the 75% target. Development
results are never labeled official. The harness records software and hardware
details, telemetry, seeds, parameters, the exact submitted source, and source hashes.

## Review and final results

Freeze the environment, inference convention, and limits before accepting official
submissions. Import only each team's frozen submission folder into the trusted
repository; do not apply participant changes to benchmark files.

Review finalists against [the prohibited conduct rules](RULES.md#3-prohibited-conduct-requiring-review).
Inspect module import, `build`, `prepare`, `train`, inference, and supporting
Python/kernel source. Check the origin of weights and constants, dataset access,
and state held in ordinary attributes, global variables, files, or subprocesses.

Check trial independence with repeat runs, reordered seeds, and fresh containers.
Investigate accuracy or timing changes tied to trial order or earlier runs.
Bitwise-identical results are not required.

Keep failed trials and all raw results. Only an independently verified
infrastructure failure permits a rerun: restart the entire frozen submission
with the same seeds and retain both attempts' logs. Never retry or select trials
based on their accuracy or training time.

## Calibration recipes

Organizer calibration recipes may live in `.local/`, which is ignored by Git and
excluded from the public image. The public template intentionally trains a trivial
model for only three steps; use a competitive recipe for accuracy and timing
calibration.

## Benchmark maintenance checks

Run these checks when changing the benchmark itself. They test the runner and its
rules; contestants measure their recipes with `python -m benchmark.run` as shown
in the README.

```bash
uv run ruff check .
uv run pytest
```

Tests cover scoring, shared official seeds, invalid outputs, evaluation mutation,
repeat-seed resets, cancellation, and process termination on timeouts. Evaluation integrity tests
run on both CPU and CUDA when a GPU is available; CUDA cases are skipped otherwise.
Calibrate training time and the 5-second inference limit on the official A100 80GB
PCIe. CPU timings and previous L40/L40S results are not A100 estimates.
