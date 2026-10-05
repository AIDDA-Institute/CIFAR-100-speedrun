# Competition rules

All rules apply whether or not the harness detects a violation. Passing its
checks does not replace source review. A failed check can also be an
implementation error; it is not by itself evidence of cheating.

## 1. Competition format

### Task and score

Train from scratch on CIFAR-100's 50,000 training images. Test accuracy is top-1
on all 10,000 test images: the percentage whose highest-scoring class matches the
correct label among the 100 fine classes.

Every submission uses the same ordered list of 200 distinct organizer-selected
seeds. All 200 trials must succeed, and their mean test accuracy must be **at least
75%**. There is no per-trial accuracy threshold. Qualifying submissions are ranked
by mean `prepare_time + train_time`; the lowest time wins. No seeds, failed
trials, or outliers may be dropped.

### Environment and permitted methods

Official runs use one NVIDIA A100 80GB PCIe, with MIG disabled, the
[pinned container](Dockerfile), a four-CPU container quota, four PyTorch CPU
threads, and networking disabled.

The A100 40GB and A100 SXM variants are not accepted for official judging.

Submit model, training, and supporting source in `submissions/<team>/`. The recipe
must run in the pinned PyTorch 2.4.0 environment without installing packages or
changing dependencies. Other training runtimes, including JAX and TensorFlow,
are not allowed. `torch.compile` and custom CUDA/Triton/C++ source are allowed.

You may choose the architecture, optimizer, loss, schedule, augmentations,
precision, resolution, and kernels. Any development tools, including autoresearch
frameworks, are allowed. Nondeterministic CUDA kernels are allowed; identical seeds
need not produce bitwise-identical results.

### Submission licensing

By submitting a pull request, you agree to license your original contribution under
the [MIT License](LICENSE). Contributors retain copyright in their work. Only
include code you have the right to contribute under these terms, and preserve
required third-party notices.

### Timing and inference

Dataset download and raw CPU loading are outside the score. Each trial starts
with training data in CPU memory. The score includes resets, GPU transfers,
casting, preprocessing (including whitening), augmentation, and all fitting.
The harness synchronizes CUDA between `prepare` and `train`, and after `train`;
these waits count toward the score.

| Phase | Limit | Included in score? |
| --- | ---: | --- |
| Module import + `build`, once | 600 seconds | No |
| `prepare` + `train`, each trial | 600 seconds | Yes |
| Full test-set inference, each trial | 5 seconds | No |

Untimed setup may allocate memory, compile, autotune, capture CUDA graphs, and
warm up on synthetic inputs, including synthetic forward/backward/optimizer
steps. Reset all state changed by warmup before each trial. Compilation deferred
to preparation, training, or inference counts against that phase's limit.

Evaluation uses one view per image, batches of 1024 (with a smaller final batch),
and the [classifier interface](submission_template/README.md#classifier).
Its deadline includes model state checks, preprocessing, transfers, lazy
compilation, and synchronization.

## 2. Automatic harness checks

Official runs apply the following checks. Development runs may use different
trial counts, accuracy targets, devices, and limits.

| Check | What the harness checks or rejects |
| --- | --- |
| Official settings | Requires CUDA, real-data mode, 200 trials, the fixed accuracy target and limits, evaluation batch size 1024, and four PyTorch threads. Requires `--seed-file` with 200 distinct unsigned 32-bit seeds and checks reported trial order. |
| Environment | Requires exactly one NVIDIA A100 80GB PCIe reported by PyTorch and `nvidia-smi`, with MIG reported as disabled. Checks Ubuntu, Python, PyTorch, torchvision, and CUDA versions. Rejects non-loopback network interfaces. |
| Submission interface | Requires callable `build`, `prepare`, and `train`; rejects symlinks inside the submission folder. `train` must return a `torch.nn.Module`. |
| Predictions | Requires a finite floating-point tensor of shape `[B, 100]` for each batch of `B` images, and one prediction per test image. |
| Evaluation state | Compares registered parameters and buffers before `model.eval()` and after inference. Rejects changes in their names, values, shapes, dtypes, devices, or layouts. |
| Completion and score | Stops the worker on timeout. Exceptions, out-of-memory errors, invalid outputs, detected state changes, or incomplete trials prevent qualification. A complete run below the accuracy target does not qualify. |

The state comparison does not inspect ordinary Python attributes, global
variables, or files, and cannot detect temporary changes restored before the
final comparison. The harness does not set the container CPU quota or disable
networking. Organizers must apply those limits when launching the container.

## 3. Prohibited conduct requiring review

The harness cannot reliably detect all of the following. Finalists require source
inspection under the [organizer review procedure](ORGANIZERS.md#review-and-final-results).

- **Prior learning or external data:** no pretrained models, weights, features,
  checkpoints, external training datasets, or constants encoding learned model
  tensors. Architectures and scalar hyperparameters found during development
  are allowed.
- **Learning carried between trials:** reset parameters, buffers, optimizer,
  scheduler, gradient scaler, moving averages, and custom random generators.
  Do not preserve learned values or use previous trial results to change the
  next trial. Reusing allocated memory and compiled code is allowed.
- **Work outside the timer:** no real dataset access, learned weights, or trial
  seed use during module import or `build`. All work using real training data
  belongs in `prepare` or `train`. Training threads and subprocesses must finish
  before `train` returns.
- **Test-set use:** do not fit on test images or labels, encode test answers, or
  use test accuracy to choose a stopping point within an official trial. During
  judging, access test images only through the harness's inference calls; do not
  read test files or retain test images for later trials. The test set is public,
  and using reported test accuracy to compare recipes during development is allowed.
- **Extra computation during evaluation:** no additional augmented views
  (test-time augmentation), fitting, adaptation, test-set statistics, training-data
  lookup, or model state changes, including state outside registered tensors.
  Predictions must not depend on other test images or their order.
- **Measurement interference:** no changes to the harness, clocks, GPU power
  settings, or GPU clock settings; no intentional sleeps, cooldowns, or host
  selection to influence scores.

Test labels are excluded from the submission API, but the harness does not block
access to dataset files or isolate malicious Python code. Organizers run only
the frozen submission folder in their trusted repository; participant changes
to benchmark files are not used.
