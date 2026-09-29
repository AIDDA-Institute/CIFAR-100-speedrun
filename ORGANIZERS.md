# Organizer guide

This guide covers official evaluation and maintenance of the benchmark. For the
contestant workflow, start with [README.md](README.md). Outstanding launch tasks
are listed in the README's [TO DO section](README.md#to-do).

## Official container

The Dockerfile targets Linux x86-64, Ubuntu 22.04, CUDA 12.4.1, Python 3.12.
Build on the GPU host (or with an amd64 builder) and download data with networking
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

The image must contain the frozen submission. Official mode verifies the GPU,
software versions, OS, and network isolation. Pin the host/provider, CPU allocation,
driver and power settings for all official measurements; record the container image
digest. Official mode requires `--seed-file` for both individual submissions and
`--all`, so separate invocations also use the same organizer-owned seeds.
The Docker CPU quota covers all submission processes; the harness also fixes
PyTorch's thread count to four. Keep that quota when changing launch commands.

Official runs require 50 successful trials and enforce the 75% target. Development
results are never labeled official. The harness records software and hardware
details, telemetry, seeds, parameters, the exact submitted source, and source hashes.

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
Calibrate training time on the official GPU; CPU timings are not L40S estimates.
