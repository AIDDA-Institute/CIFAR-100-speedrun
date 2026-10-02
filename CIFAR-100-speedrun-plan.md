# CIFAR-100 Speedrun Repository

The initial implementation is now in this repository. See [README.md](README.md)
for setup and commands, [RULES.md](RULES.md) for concrete benchmark rules, and the
[submission template](submission_template/README.md) for the implemented Python
interface. The A100 80GB PCIe launch check is complete; see the
[two-trial pilot results](README.md#verified-a100-setup). No further baseline
calibration is required. The selected accuracy target is **75% mean test accuracy**,
and official judging uses **40 trials per submission**.

## Goal

Create the official repository and evaluation harness for a CIFAR-100 training speedrun.

Contestants may use any autoresearch framework during development.

Their final submission is simply the resulting **model + training algorithm**, added as a folder under `submissions/`.

The benchmark measures:

\[
\text{minimum mean training time}
\]

subject to

\[
\text{mean test accuracy} \ge A^*.
\]

\(A^* = 0.75\). Official runs enforce this target; development runs may override it.

---

# 1. Dataset

Use the standard CIFAR-100 split:

- 50,000 training images
- 10,000 test images
- RGB
- \(32\times32\)
- 100 classes

Provide a script/function that automatically downloads and prepares CIFAR-100.

The training algorithm receives only the training data during an official trial.

The organizer-controlled evaluator owns the test data and evaluation.

---

# 2. Official Hardware and Environment

Official hardware:

- **1× NVIDIA A100 80GB PCIe**
- Single GPU only
- MIG disabled; no 40GB or SXM variants
- Same provider / machine class for all official measurements where possible

Start with a software environment closely matching the Fulcrum CIFAR-10 harness:

- Ubuntu 22.04
- CUDA 12.4.1
- Python 3.12
- PyTorch 2.4.0
- torchvision 0.19.0

Pin all dependencies.

Provide either a Dockerfile or another reproducible container specification.

The evaluator should verify at startup that the expected GPU is present.

Record basic telemetry such as:

- GPU name
- temperature
- power draw
- SM clock

with each official run.

---

# 3. Number of Trials and Scoring

Default:

```text
N_TRIALS = 40
ACCURACY_TARGET = 0.75
```

Each submission is independently trained from scratch for 40 organizer-controlled seeds.

For trial \(i\), record:

\[
A_i = \text{test accuracy}
\]

and

\[
T_i = \text{training time}.
\]

Calculate:

\[
\bar A = \frac{1}{40}\sum_i A_i
\]

and

\[
\bar T = \frac{1}{40}\sum_i T_i.
\]

A submission qualifies when:

\[
\bar A \ge A^*.
\]

Among qualifying submissions, rank by:

\[
\boxed{\text{lowest }\bar T}.
\]

Save the raw result of every trial as well as the summary.

For development, allow:

```bash
--n 1
--n 10
--n 20
```

so participants do not need to run 40 trials every time.

Official evaluation uses 40.

---

# 4. Submission Contract

Participants submit a PR adding `submissions/<team_name>/submission.py` and any supporting source files. The organizer's harness imports that module and calls the functions below.

The submission template must provide the exact Python function signatures, return values, and input tensor formats, so teams can copy it and implement their recipe. This local calling convention is what the plan means by the submission API.

Closely follow Fulcrum's three-stage model:

```python
build(...)
prepare(...)
train(...)
```

### `build()`

Called once before the trials.

May perform genuinely one-time work such as:

- construct model objects;
- allocate buffers;
- `torch.compile`;
- compile CUDA/Triton kernels;
- CUDA graph setup/warm-up using synthetic data.

Synthetic warm-up may exercise forward, backward, and optimizer code for compilation
or capture. All state it changes must be reset in `prepare()` before a trial.

It is **untimed**.

It must not:

- train on real data or retain learned state from warm-up;
- access test data;
- perform per-trial data-dependent work;
- use the official trial seed;
- construct learned state that should properly belong to a trial.

### `prepare()`

Called fresh for every trial.

It should perform things such as:

- resetting/reinitializing model parameters;
- resetting optimizer state;
- training-data-dependent preprocessing;
- whitening;
- construction of augmented batches;
- other per-run preparation.

The entire call is **timed**.

### `train()`

Runs the participant's actual training procedure.

The entire call is **timed**.

The participant chooses:

- number of steps;
- number/fraction of epochs;
- batch sizes;
- stopping point;
- everything else about training.

There is no organizer-prescribed number of epochs.

---

# 5. Model / Evaluation Interface

The implemented interface uses PyTorch classifiers and plain single-view inference.

At minimum, after training the submission must expose a classifier taking:

```text
[B, 3, 32, 32]
```

and producing:

```text
[B, 100]
```

class logits.

The organizer computes predictions from those logits.

Do **not** let submissions calculate or return their own accuracy.

The organizer-owned evaluator controls batching, validates logits and model state,
and computes accuracy in the supervisor.

### Selected evaluation convention

Official accuracy uses plain single-view inference. Internal test-time augmentation,
test-set statistics and fitting during evaluation are prohibited.

Evaluation is **not included in training time**.

### Evaluation time limit

Use an organizer-controlled limit of **5 seconds per trial** to evaluate all 10,000 test images on the official A100 80GB PCIe. This is a limit on the complete test pass, not on each batch.

Measure evaluation wall time separately, including evaluation-specific preprocessing, device transfers, any lazy compilation, inference, and the final CUDA synchronization. Dataset download and initial loading happen before the trials. Evaluation time is recorded for diagnostics and enforcement only; it does not affect the training-time score.

Enforce the deadline with an organizer-controlled watchdog. If it is exceeded, stop that submission's official run, record the trial as `eval_timeout`, preserve the results collected so far, and mark the submission as not qualified. Do not drop the failed trial from the results or replace its seed to obtain a qualifying score.

The two-trial A100 80GB PCIe pilot completed each full test pass within 0.132 seconds,
below the 5-second limit. Submissions cannot override the official limit.

---

# 6. What Participants May Change

Anything forming part of the training recipe may be changed, including:

- model architecture;
- optimizer;
- loss function;
- initialization;
- learning-rate schedule;
- batch size;
- number of training steps;
- data augmentation;
- data ordering;
- training resolution;
- precision;
- PyTorch compilation;
- CUDA graphs;
- custom CUDA/Triton kernels;
- kernel fusion;
- systems-level optimizations.

The intention is to optimize the **complete training system**, rather than one fixed architecture or optimizer.

---

# 7. Prohibited Submissions

Every official trial must train from scratch.

Do not allow:

- pretrained models;
- pretrained weights;
- checkpoints;
- weights discovered during autoresearch;
- learned state persisted between official trials;
- external training datasets;
- embeddings/features generated by pretrained models;
- constants that effectively encode pretrained weights;
- hard-coded test predictions or labels;
- internet/network access during official evaluation;
- use of test labels during training;
- using test accuracy to determine when training should stop.

All submitted training code must be available as source code for organizer inspection.

Custom CUDA/Triton/C++ is allowed if its source is included.

The submitted recipe must run in the pinned PyTorch environment, with its existing
dependencies. JAX, TensorFlow and additional package installs are not supported in
this version. Development automation may use any framework.

Opaque model binaries are not allowed.

---

# 8. Timing

Use Fulcrum's measurement philosophy.

The official score should be the **compute-honest** time:

```text
build()                      UNTIMED, once

for each trial:

    seed everything

    START
    prepare()                TIMED
    train()                  TIMED
    STOP

    evaluate()               UNTIMED
```

Use:

```python
torch.cuda.synchronize()
time.perf_counter()
```

at the timing boundaries.

Anything repeated for every fresh training run must therefore be charged.

This prevents contestants from obtaining artificial speed improvements by moving:

- augmentation generation;
- whitening;
- model reset;
- optimizer reset;
- other per-trial work

outside the timer.

The official harness owns the timing boundary; submissions do not control their own timer.

---

# 9. Repository Structure

Suggested structure:

```text
cifar100-speedrun/
│
├── README.md
├── RULES.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
│
├── benchmark/
│   ├── data.py
│   ├── harness.py
│   ├── evaluate.py
│   ├── timing.py
│   ├── config.py
│   └── run.py
│
├── submission_template/
│   ├── submission.py
│   └── README.md
│
├── submissions/
│   └── .gitkeep
│
└── results/
    └── .gitkeep
```

The benchmark code is organizer-controlled.

Contestants only modify/add:

```text
submissions/<team_name>/
```

---

# 10. Submission Workflow

Each team starts from the official repository.

They create:

```text
submissions/<team_name>/
```

containing their final solution.

Their folder may contain:

```text
submission.py
custom kernels
additional Python modules
configuration
etc.
```

but must obey the required submission interface.

At the deadline, each team submits an exact Git commit / pull request containing its submission folder.

The organizers merge/freeze the submissions.

Official evaluation should then be as simple as:

```bash
python -m benchmark.run --submission team_name --official --seed-file seeds.json
```

Optionally support:

```bash
python -m benchmark.run --all --official --seed-file seeds.json
```

to evaluate all submitted teams.

Changes teams make outside their own submission folder must be ignored during official evaluation.

---

# 11. Results

For every submission save:

```text
results/<team_name>/<timestamp>-<run_id>/
├── config.json
├── trials.jsonl
├── summary.json
└── source/
```

Each trial should include at least:

```text
seed
accuracy
prepare_time
train_time
total_timed_time
evaluation_time
status
failure_reason (if applicable)
GPU information
telemetry
```

`summary.json` should include:

```text
number of trials
mean accuracy
accuracy std
mean training time
training-time std
accuracy target
qualified: true/false
```

The leaderboard uses:

```text
mean training time
```

provided:

```text
mean accuracy >= A*
```

---

# 12. Seeding

The organizer selects and freezes the official seed file. The harness requires
exactly 40 distinct unsigned 32-bit seeds and records their order for each team.

Before every `prepare()` call reset:

- Python RNG
- NumPy RNG
- PyTorch CPU RNG
- PyTorch CUDA RNG

from the official seed.

Seeding supplies reproducible RNG starting points. Recipes must also reset all
learned state and their custom generators between trials. Nondeterministic CUDA
kernels are permitted, so bitwise-identical results are not guaranteed.

---

# 13. Safety / Benchmark Guards

Port the useful ideas from Fulcrum:

- verify correct GPU;
- verify exactly one GPU;
- synchronize CUDA around timing;
- prohibit long deliberate sleeps/cooldowns;
- ensure evaluation does not mutate model parameters;
- prevent state learned during one trial from carrying into another;
- keep all per-trial preparation inside the charged region;
- record raw trial results rather than only summary values.

The harness is not expected to be a perfect security sandbox.

The main protections are:

1. fixed organizer-owned harness;
2. no network during official runs;
3. open-source submissions;
4. manual inspection of finalists;
5. fresh model state for every trial.

---

# 14. README

The README should explain:

1. What the CIFAR-100 speedrun is.
2. How scoring works.
3. How to install the environment.
4. How to download CIFAR-100.
5. How to copy the submission template.
6. The required submission interface.
7. What is timed and untimed.
8. What participants may change.
9. What is prohibited.
10. How to locally evaluate a submission.
11. How to submit at the hackathon deadline.

Keep detailed edge-case rules in `RULES.md`.

---

# 15. Do Not Include a Competitive Baseline

The repository does not need to provide a good solution.

`submission_template/` should contain only enough code to demonstrate the required API and prove the harness works.

For example, it may construct a trivial network and perform a tiny amount of training.

It should not establish a meaningful speedrun baseline.

---

# 16. Benchmark Validation

The selected target is \(A^* = 0.75\).

The two-trial A100 pilot completes launch validation. The following checks guide
future changes to the harness or official environment.

### A. Measure time to the selected accuracy threshold

Run one or more sensible CIFAR-100 from-scratch training recipes on the official A100 80GB PCIe.

Measure the time/accuracy frontier.

Verify that reaching 75%:

- is non-trivial;
- permits runs short enough for autoresearch;
- leaves substantial room for optimization.

### B. Validate evaluation

Plain single-view inference is selected for official evaluation.

Confirm that ordinary classifiers can evaluate the full test set comfortably within the 5-second limit. Check that evaluation time stays outside the training score and that exceeding the limit produces a recorded failure and a non-qualifying result.

### C. Check number of trials

Use repeated runs to measure accuracy and timing variance.

Official evaluation uses 40 trials. Report the accuracy and timing variance alongside the mean.

### D. Red-team the timing boundary

Try deliberately to:

- move whitening into `build()`;
- precompute augmented batches in `build()`;
- cache trained state between trials;
- manipulate the RNG;
- use long cooldown sleeps;
- modify evaluation;
- access test labels;
- smuggle pretrained weights into a submission.

Verify that these either fail or are clearly prohibited/detectable.

### E. Reproducibility

Run the same submission:

- repeatedly;
- in different trial orders;
- in fresh containers.

Check that the reported means are stable.

---

# Initial Constants

For implementation, use:

```text
DATASET             = CIFAR-100
TRAIN_IMAGES        = 50,000
TEST_IMAGES         = 10,000
NUM_CLASSES         = 100

OFFICIAL_GPU        = NVIDIA A100 80GB PCIe
N_TRIALS            = 40
EVAL_TIMEOUT_SECONDS = 5

ACCURACY_TARGET     = 0.75

PYTHON              = 3.12
CUDA                 = 12.4.1
PYTORCH              = 2.4.0
TORCHVISION          = 0.19.0
```

The selected accuracy target is 75%. The completed A100 pilot measured the time
needed to reach it using plain inference and checked the configured resource limits.
