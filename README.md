# CIFAR-100 training speedrun

Build a training recipe that reaches **at least 75% average test accuracy** on
CIFAR-100 in as little time as possible. Official judging uses one NVIDIA A100
80GB PCIe and 200 fresh training trials. Your score is the average **preparation +
training time** across those trials; inference time is excluded.

See the [official scoreboard](SCOREBOARD.md) and its
[machine-readable source](scoreboard.json).

To enter, fork this repository, develop your recipe in `submissions/<your_team>/`,
and open a pull request. Start with the steps below, then read the rules:
[competition format](RULES.md#1-competition-format),
[automatic harness checks](RULES.md#2-automatic-harness-checks), and
[prohibited conduct requiring review](RULES.md#3-prohibited-conduct-requiring-review).

## 1. Set up your development environment

The instructions below assume a **remote Linux x86-64 machine with an NVIDIA
GPU**, where your training experiments will run. Connect using SSH or your
provider's terminal, then **run the setup and experiment commands on that
machine**. Your personal laptop can provide the browser and editor. A local
Linux workstation with a suitable NVIDIA GPU also works.

Fork this repository on GitHub using your browser, then clone your fork on the
GPU machine. Replace `YOUR_GITHUB_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CIFAR-100-speedrun.git
cd CIFAR-100-speedrun
```

Install [uv](https://docs.astral.sh/uv/) on that machine. It manages Python and
project dependencies. This command creates or updates the project's `.venv`
environment using the versions recorded in `uv.lock`, without changing that
lock file:

```bash
uv sync --frozen
```

Run the remaining commands from the repository directory on the GPU machine;
`uv run` runs a command inside the project's environment. The Linux x86-64 setup
uses Python 3.12, PyTorch 2.4.0, torchvision 0.19.0, and CUDA 12.4 packages.
Windows and macOS laptops can connect to this remote environment; native Windows
GPU execution is not configured or tested here.

### Optional: check that your setup works

Run this on the **same machine where you will run experiments** to check its
installation. This quick check, sometimes called a **smoke test**, deliberately
uses that machine's CPU, even on a GPU host. It runs the tiny example recipe twice
using generated images, checking that the software can load a recipe, run it,
and save results. The test itself needs no GPU or dataset download.

```bash
uv run python -m benchmark.run --submission-path submission_template --device cpu --synthetic --n 2
```

A successful run ends with `"complete": true` and `"qualified": null`. The accuracy
from these generated images is not meaningful. This is a setup check; measure
CIFAR-100 accuracy and GPU training speed in step 3.

## 2. Create your submission

Copy the starter example into your team's folder. Replace `my_team` with your
chosen team name in this and subsequent commands:

```bash
cp -r submission_template submissions/my_team
```

Edit `submissions/my_team/submission.py` to implement your training recipe. The
example uses only 64 images and three learning steps to demonstrate the interface;
you will need to replace that tiny demonstration to pursue 75% accuracy.

Your `submission.py` provides three Python functions that the benchmark runner
(the harness) calls for you. Implement them as standalone functions with these
arguments:

```python
def build(context): ...
def prepare(state, train_data, seed): ...
def train(state): ...
```

| Function | What you do | When it runs | Counts toward training time? |
| --- | --- | --- | --- |
| `build` | Create the model structure and reusable resources. Optional compilation/warmup on synthetic inputs can go here. Return an object holding what the next functions need. | Once, before the trials | No |
| `prepare` | Start a fresh training run: reset the model's weights and training state, and get the training images ready. | Before every trial, including the first | Yes |
| `train` | Train the model, then return it so the harness can check its predictions. | Once per trial | Yes |

`build` receives a `BuildContext` supplied by the harness: `context.device` is the
target GPU (or CPU for a smoke test), `context.parameters` holds your optional
recipe settings, `context.num_classes` is 100, and `context.eval_batch_size` is
1024. Real training data and the trial seed are supplied later to `prepare`.
`train_data.images` and `train_data.labels` contain the training images and their
correct classes, initially in CPU memory.

**State** is the ordinary Python object you return from `build`; there is no
required state class. The starter uses Python's `types.SimpleNamespace`, an object
with named attributes such as `state.model` and `state.context`. A dictionary or
an instance of your own class also works. Both later functions receive this
**same object**. `prepare` updates its contents for a fresh run and returns nothing.
In the starter, it resets the weights and adds `state.optimizer`, `state.images`,
and `state.labels`; `train` uses them to learn and returns `state.model`, the
trained PyTorch model.

A **trial** is one complete training run from scratch followed by an accuracy check.
The harness calls `build` once, then repeats `prepare → train → accuracy check` for
each seed. A **seed** controls random choices such as initial weights and shuffled
training examples. Reuse the model structure between trials, but reset everything
it learned.

Preparation is deliberately timed: resetting training state, moving images to the
GPU, and preprocessing are work needed for each fresh run. This keeps work using
real training data inside the score. For example, 1 second preparing plus
20 seconds training gives a trial time of 21 seconds.

You implement the training recipe. The harness supplies the data and seeds,
measures time, runs the test images through your returned model, and computes
accuracy. You do not need to write the scoring or test loop. Custom GPU kernels
and compilation are optional.

The [submission guide](submission_template/README.md) explains the input tensors,
reset requirements, and classifier outputs. Relative imports such as
`from .model import Classifier` work within your folder. Supporting Python and
custom kernel source belong in that folder too.

You may change architecture, optimizer, precision, augmentations, schedule,
training resolution, compilation, and kernels. Every trial starts fresh: no
pretrained weights, external datasets, or learned state carried across trials.
The submission runtime is the pinned PyTorch environment. Custom CUDA/Triton/C++
kernels are allowed; alternative training frameworks and per-submission dependency
installs are not supported in this version. You can develop manually or use
automation tools of your choice. See [RULES.md](RULES.md) for the complete rules.

## 3. Test and improve your recipe

Run **all commands in this section on the GPU machine**, using your latest recipe
code there. If you edit files on your laptop, copy or commit/push and pull those
changes onto the GPU machine before running them. An NVIDIA A100 80GB PCIe gives
representative timings for official judging; CPU setup checks do not estimate
A100 performance.

Download CIFAR-100 once before your first real-data run:

```bash
uv run python -m benchmark.data --root data
```

Run one fresh training trial to see your recipe's accuracy and training time:

```bash
uv run python -m benchmark.run --submission my_team --n 1
```

Use a small number of trials while iterating, then test promising recipes across
more seeds to see how consistent they are:

```bash
uv run python -m benchmark.run --submission my_team --n 10
```

Development runs use the same 75% accuracy target as official judging. A completed
run below that target reports `"qualified": false` and exits with code 1. The
unchanged starter example normally produces this result on real data.

For a diagnostic run that reports measurements without applying the accuracy
target, add `--no-accuracy-target`. You can also pass optional JSON recipe settings
to `build()` while experimenting:

```bash
uv run python -m benchmark.run --submission my_team --n 3 --params '{"epochs": 10}'
```

Your recipe decides which settings to support. Before submitting, make sure its
defaults run the final recipe without extra command-line settings.

### Read your results

The runner prints each trial's accuracy and timing, then an overall summary.
Each run also creates `results/<team>/<timestamp>-<id>/` containing:

- `summary.json`: completion, mean accuracy, mean preparation + training time,
  variability, and qualification;
- `trials.jsonl`: each trial's accuracy, timings, and status;
- `config.json`: settings, seeds, source hashes, environment, and untimed build time;
- `source/`: a copy of the exact submission that was run;
- `error.txt` when a worker raises an exception.

In JSON results, accuracy is a fraction (`0.75` means 75%) and times are seconds.
`mean_training_time` includes both preparation and training. Development results
help you compare recipes; official scores come from the organizers' evaluation.

A failed or interrupted run cannot qualify using only its successful trials.
Ctrl-C stops the run and preserves partial results. Exit code 0 means a completed
qualifying or diagnostic run, 1 means nonqualifying or incomplete, and 2 means
invalid configuration. Interruptions use 130 (Ctrl-C) or 143 (SIGTERM).

## 4. Open a pull request

Commit and push your final recipe to your fork, then open a pull request to this
repository adding only `submissions/my_team/` and its contents.

Include the source for the model and training algorithm, with any supporting
source files and configuration. The recipe must work with its default settings.
Do not include trained weights, checkpoints, downloaded datasets, or local results.

Organizers review and freeze your submission folder, then run it with the official
harness. Changes outside your team folder are not part of the submitted recipe.

## 5. How judging works

- Every submission runs on one NVIDIA A100 80GB PCIe, with MIG disabled, in the
  fixed software environment.
- Each recipe trains from scratch for the same 200 organizer-selected seeds.
- All 200 trials must succeed, and average test accuracy must reach **at least 75%**.
  There is no additional accuracy requirement for each individual trial.
- Accuracy is **top-1**: the percentage of the 10,000 test images for which the
  model's highest-scoring class matches the correct label among the 100 classes.
  Correctly classifying 7,500 images gives 75% accuracy for that trial.
- Qualifying submissions are ranked by **mean preparation + training time**;
  the lowest time wins.
- The complete evaluation on all 10,000 test images must finish within **5 seconds
  per trial**. Evaluation time is excluded from the score.

See [RULES.md](RULES.md) for the full timing boundaries, resource limits, and
allowed training methods. The organizers handle the official seed file and final
200-trial evaluation.

## License

This repository's code and documentation are licensed under the [MIT License](LICENSE).
Submissions are contributed under MIT; see [submission licensing](RULES.md#submission-licensing).
Dependencies and the CIFAR-100 dataset retain their own terms.

## Organizer information

Instructions for the official Docker environment, calibration recipes, and
benchmark maintenance checks are in [ORGANIZERS.md](ORGANIZERS.md).

## Verified A100 setup

On 2 October 2026, a ResNet9-style baseline (40 epochs, width 64) completed two
real CIFAR-100 trials on an NVIDIA A100 80GB PCIe using the repository's Dockerfile:

- Mean accuracy: **75.36%** (individual trials: 75.56% and 75.16%).
- Mean preparation + training time: **59.30 seconds**.
- Mean inference time: **0.117 seconds**; slowest test pass: **0.132 seconds**,
  within the 5-second limit.

Standard Docker GPU launch, disabled MIG mode, the four-CPU quota, and network
isolation all passed. This two-trial pilot completes the launch check; no further
baseline calibration is required. It is a development result, not an official
score. Official judging now uses **200 trials per submission**.

### Previous hardware calibration

On 1 October 2026, the baseline completed all 50 trials under the previous L40
format. Mean accuracy was 75.4844% (standard deviation 0.2516 percentage points);
mean preparation + training time was 62.3165 seconds (standard deviation 0.1395
seconds). The slowest test pass took 0.1813 seconds. Docker GPU access, the CPU
quota, and network isolation passed on that host. These are L40 measurements;
the A100 pilot results are recorded above.
