# CIFAR-100 training speedrun

Build a training recipe that reaches **at least 75% average test accuracy** on
CIFAR-100 in as little time as possible. Official judging uses one NVIDIA L40S
48GB and 50 fresh training trials. Your score is the average **preparation +
training time** across those trials; inference time is excluded.

To enter, fork this repository, develop your recipe in `submissions/<your_team>/`,
and open a pull request. Start with the steps below and read the full
[competition rules](RULES.md) before developing your recipe.

## 1. Set up your development environment

Fork this repository on GitHub, then clone your fork. Replace
`YOUR_GITHUB_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CIFAR-100-speedrun.git
cd CIFAR-100-speedrun
```

Install [uv](https://docs.astral.sh/uv/), then install the project's pinned Python
environment and dependencies:

```bash
uv sync --frozen
```

Run the remaining commands from the repository directory. The environment uses
Python 3.12, PyTorch 2.4.0, and torchvision 0.19.0. Linux x86-64 installs the CUDA
12.4 wheels; ARM/macOS use their available PyPI wheels for development.

### Optional: check that your setup works

This quick check, sometimes called a **smoke test**, runs the tiny example recipe
twice on your CPU using generated images. It checks that the installed software
can load a recipe, run it, and save results. It needs no GPU or dataset download.

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

Use an NVIDIA GPU with CUDA support for training experiments. An L40S gives
representative timings for official judging; CPU setup checks do not estimate
L40S performance.

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

- Every submission runs on one NVIDIA L40S 48GB in the fixed software environment.
- Each recipe trains from scratch for the same 50 organizer-selected seeds.
- All 50 trials must succeed, and average test accuracy must reach **at least 75%**.
  There is no additional accuracy requirement for each individual trial.
- Qualifying submissions are ranked by **mean preparation + training time**;
  the lowest time wins.
- The complete evaluation on all 10,000 test images must finish within **5 seconds
  per trial**. Evaluation time is excluded from the score.

See [RULES.md](RULES.md) for the full timing boundaries, resource limits, and
allowed training methods. The organizers handle the official seed file and final
50-trial evaluation.

## Organizer information

Instructions for the official Docker environment, calibration recipes, and
benchmark maintenance checks are in [ORGANIZERS.md](ORGANIZERS.md).

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
