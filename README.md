# CIFAR-100 training speedrun

Train a classifier from scratch on the 50,000 CIFAR-100 training images using one
NVIDIA L40S 48GB. Qualifying submissions minimize **mean preparation + training
time**, subject to **mean test accuracy >= 75%**, over 50 seeds.
The default accuracy target is `0.75`. Accuracy values are fractions between 0 and 1;
times are seconds. Inference does not contribute to the score.

## Install and smoke test

Python 3.12, PyTorch 2.4.0, torchvision 0.19.0. Install [uv](https://docs.astral.sh/uv/),
then use the committed lockfile:

```bash
uv sync --frozen
uv run python -m benchmark.run --submission-path submission_template --device cpu --synthetic --n 2
uv run pytest
```

Linux x86-64 installs the CUDA 12.4 wheels; ARM/macOS use their available PyPI
wheels for development. CPU and synthetic runs are explicitly nonofficial.
Synthetic runs cannot qualify, even if a target is supplied.

Download the dataset once, before running any submission:

```bash
uv run python -m benchmark.data --root data
```

## Submit a PR

Copy the template, implement your recipe, and open a PR adding only your team folder:

```bash
cp -r submission_template submissions/my_team
uv run python -m benchmark.run --submission my_team --n 1
```

Your `submission.py` provides three Python functions that the benchmark runner
(the harness) calls for you:

```python
def build(context): ...
def prepare(state, train_data, seed): ...
def train(state): ...
```

| Function | What you do | When it runs | Counts toward training time? |
| --- | --- | --- | --- |
| `build` | Create the model structure and reusable resources. Optional compilation can go here. Return an object holding what the next functions need. | Once, before the trials | No |
| `prepare` | Start a fresh training run: reset the model's weights and training state, and get the training images ready. | Before each trial | Yes |
| `train` | Train the model, then return it so the harness can check its predictions. | Once per trial | Yes |

A **trial** is one complete training run from scratch followed by an accuracy check.
The harness calls `build` once, then repeats `prepare → train → accuracy check` for
each seed. A **seed** controls random choices such as initial weights and shuffled
training examples. The object returned by `build` is called **state** and is passed
to both `prepare` and `train`; it can be a simple dictionary or a class instance.
Reuse the model structure between trials, but reset everything it learned.

You implement the training recipe. The harness supplies the data and seeds,
measures time, runs the test images through your returned model, and computes
accuracy. You do not need to write the scoring or test loop. The template shows a
complete example; no custom GPU kernels or compilation are required.

The template and [submission contract](submission_template/README.md) explain the
input tensors, reset requirements, and classifier outputs. Relative imports such
as `from .model import Classifier` work within your folder. Include custom kernel
source there. Organizers freeze the PR's source and run it using the official
harness; modifications outside your submission folder are ignored.

You may change architecture, optimizer, precision, augmentations, schedule,
training resolution, compilation, and kernels. Every trial starts fresh: no
pretrained weights, external datasets, or learned state carried across trials.
The submission runtime is the pinned PyTorch environment. Custom CUDA/Triton/C++
kernels are allowed; alternative training frameworks and per-submission dependency
installs are not supported in this version. Development automation is unrestricted.
See [RULES.md](RULES.md) for the complete timing and evaluation rules.

## Calibration and development

Use `--n 1`, `--n 10`, or `--n 20` for development. Recipe parameters are an optional
JSON object supplied to `build()`; a submitted recipe should have working defaults.

```bash
uv run python -m benchmark.run --submission-path /path/to/recipe --n 3 --params '{"epochs": 10}'
```

Development runs use the same 75% target by default. Use `--no-accuracy-target` to
report timing and accuracy with `qualified: null`, or `--accuracy-target` to explore
another target. Official runs enforce 75%. The convention is plain inference, with
a **5-second deadline for the entire 10,000-image test pass per trial**. Record inference time separately.
Exceeding the deadline stops that submission and marks it nonqualifying.

Organizer calibration recipes may live in `.local/`, which is ignored by Git and
excluded from the public image. The public template intentionally trains a trivial
model for only three steps; its accuracy is not useful for choosing a threshold.

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

## TO DO

Organizer tasks to complete before the first official evaluation:

- [ ] **Verify the standard Docker GPU launch on the chosen L40S host.** Confirm
  that the documented `docker run --gpus ...` command passes the environment checks
  and runs a real-data trial. The earlier test host needed a GPU container startup
  workaround; the standard launch still needs verification on the official host.
- [ ] **Complete a 50-trial GPU calibration with the baseline recipe.** Run the
  frozen recipe from scratch for 50 different seeds under the official conditions.
  Record mean accuracy, mean preparation + training time, and their variability;
  check that all trials succeed, mean accuracy reaches the fixed 75% target, and
  each full test pass finishes within 5 seconds. Earlier GPU calibration covered
  five trials; the complete 50-trial run remains outstanding.

## Results

Each run creates `results/<team>/<timestamp>-<id>/` containing:

- `config.json`: settings, seeds, source hashes, environment and untimed build time;
- `trials.jsonl`: raw accuracy, preparation/training/inference times and status;
- `summary.json`: completion, means, sample standard deviations and qualification;
- `source/`: the frozen source that was actually run;
- `error.txt` when a worker raises an exception.

A failed or interrupted run never qualifies on its successful subset. No failed seed
is silently replaced. Logs from an incomplete run are retained. The supervisor kills
a worker that exceeds its deadline, including when it hangs inside a classifier.
Ctrl-C and SIGTERM stop the current worker and its subprocesses, preserve a
nonqualifying partial result, and stop `--all` without launching another team.

The CLI exits with 0 for completed qualifying or diagnostic runs, 1 for
nonqualifying/incomplete runs, and 2 for invalid configuration. Interruptions exit
with 130 (Ctrl-C) or 143 (SIGTERM). The minimal template normally exits with 1 on
real data because it falls below 75%; use `--no-accuracy-target` for an API-only check.

## Checks

```bash
uv run ruff check .
uv run pytest
```

Tests cover scoring, shared official seeds, invalid outputs, evaluation mutation,
repeat-seed resets, cancellation, and process termination on timeouts. Evaluation integrity tests
run on both CPU and CUDA when a GPU is available; CUDA cases are skipped otherwise.
Calibrate training time on the official GPU; CPU timings are not L40S estimates.
