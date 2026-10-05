# Submission contract

Copy this directory to `submissions/<team_name>/` and submit a PR. The harness
imports `submission.py` from a frozen copy of your folder. Supporting Python
modules, configuration, and kernel source may be included in that folder.
Use the pinned PyTorch runtime and dependencies. `torch.compile` and custom
CUDA/Triton/C++ kernels are supported; extra package installation and other
training frameworks are not supported in this version.

## Start here

You write the model and the code that teaches it to classify images. The benchmark
runner (the **harness**) supplies the dataset, calls your functions, times the
training, and checks the trained model's accuracy. You do not need to implement
the timer or a test loop.

The three functions divide your recipe into setup, a fresh start, and learning:

| Function | Purpose | Return value | Timing |
| --- | --- | --- | --- |
| `build(context)` | Set up the model structure and reusable resources once. | An object holding your resources, called `state` | Outside the training score |
| `prepare(state, data, seed)` | Reset the model and training state, and get the training data ready for a fresh run. | Nothing (`None`); update `state` | Included in the training score |
| `train(state)` | Run the learning steps and finish training. | The trained PyTorch model (`torch.nn.Module`) | Included in the training score |

One **trial** means training from scratch and then checking accuracy. Conceptually,
the harness does this for an official submission:

```python
state = build(context)                 # once for the submission
for seed in organizer_seeds:           # 200 separate trials
    # The harness seeds the standard random generators first.
    prepare(state, training_data, seed)  # timed: start fresh
    model = train(state)                # timed: learn from training images
    # The harness checks this model on all 10,000 test images, outside the score.
```

`state` is simply your way to pass objects between functions. The template uses a
`SimpleNamespace`, so `state.model` stores the model, `state.optimizer` stores the
rule for updating its weights, and `state.images` stores the prepared images.
A dictionary or your own class works too. Keeping an object in `state` does not
permit keeping learned values from the previous trial.

`context` contains the device (GPU for official runs) and settings provided by the
harness. `data` contains the training images and their correct class labels. A
`seed` is a number that controls random choices, including the starting weights
and ordering of training examples. Every trial must start without knowledge
learned in earlier trials.

For a first submission, follow the ordinary PyTorch code in `submission.py`.
Compilation, custom kernels, and other GPU optimizations are optional. The example
uses only 64 training images and three learning steps to demonstrate the interface;
replace that tiny demonstration with your training recipe to pursue 75% accuracy.

## Function details

```python
def build(context: BuildContext) -> object: ...
def prepare(state: object, data: TrainingData, seed: int) -> None: ...
def train(state: object) -> torch.nn.Module: ...
```

### `build`: set up once

`build()` runs once, outside the training score. `context.device` is the selected
device, `context.parameters` contains optional JSON recipe settings,
`context.num_classes` is 100, and
`context.eval_batch_size` is 1024 for official evaluation. Its return value is yours:
a class instance, dictionary, or other object. You can allocate buffers, compile
kernels, and warm up on synthetic inputs. No real training data or trial seed is
provided to build. All state changed by synthetic warmup must be reset in prepare.
For a simple recipe, just construct the model and return it inside your state
object. Work that reads or learns from the real dataset belongs in `prepare` or
`train`, where it is timed.

### `prepare`: start a fresh trial

`prepare()` runs for every seed and is timed. The harness seeds Python, NumPy's
global RNG, and PyTorch first. Reset model parameters, buffers (including BatchNorm
statistics), optimizer, scheduler, gradient scaler, EMA, and any custom generators
or other learned state here. Reusing memory and compiled code is allowed; reusing
learned state is not. The `seed` argument lets you seed your own generators.

`data.images` is a contiguous CPU tensor of shape `[50000, 3, 32, 32]`, dtype
`torch.uint8`, RGB values 0 through 255. `data.labels` is a CPU `torch.int64` tensor
of shape `[50000]` with fine-class IDs 0 through 99 in torchvision CIFAR-100 order.
Treat both as immutable. Copying to the GPU, casting, normalization, whitening,
augmentation and any data-derived initialization must happen inside prepare/train.
The harness loads the raw dataset once before the timed trials.

### `train`: teach the model and return it

`train()` runs the complete training algorithm and returns an `nn.Module`. The
number of steps, batch sizes and stopping rule are up to you. Validation using a
split of the training data is allowed and is charged to training time.

If you use a DataLoader with spawned worker processes, define custom dataset
classes and collate functions at module level in `submission.py` or a supporting
module. Avoid lambdas and functions defined inside `train`, which Python cannot
send to spawned workers. Relative imports work in those workers too. Finish all
training work in subprocesses before returning the model.

## Classifier

The evaluator supplies batches of RGB images shaped `[B, 3, 32, 32]`, float32,
scaled to `[0, 1]`, on `context.device`. The final batch can be smaller than 1024.
Here `B` is the number of images in the batch, `3` is the red/green/blue color
channels, and each image is 32 by 32 pixels. Your classifier must handle any batch
size from 1 through 1024 and return finite floating-point logits shaped `[B, 100]`.
**Logits** are the model's scores for the 100 classes, one row per image; they do
not have to be probabilities. The harness chooses the highest-scoring class as
the prediction and compares it with the correct label. Put any further input
normalization in the model itself; use the same convention when training it.

Evaluation calls `model.eval()` and uses `torch.inference_mode()`. Model parameters
and buffers must keep their values, dtypes, devices and layouts unchanged.
Predictions must not depend on other test images or batch order. All fitting must
be finished before train returns. No test labels
are supplied to the worker; the organizer's supervisor computes accuracy.

Plain inference means one view of each image, without internally adding test-time
augmentation. A classifier may contain multiple jointly trained components; all
their training must be charged. Evaluation must finish within 5 seconds for the
whole test set, including model state checks, transfers, normalization and lazy
compilation. Synthetic inference warmup in build is allowed. The deadline is
excluded from the score.

## Local checks

```bash
uv run python -m benchmark.run --submission my_team --n 1
uv run python -m benchmark.run --submission my_team --n 10
```

For a CPU check without downloading data:

```bash
uv run python -m benchmark.run --submission my_team --device cpu --synthetic --n 2
```

Synthetic fixtures have fewer images and are solely an API smoke test. They cannot
establish accuracy or speed on CIFAR-100. Use training tensors' actual lengths.
