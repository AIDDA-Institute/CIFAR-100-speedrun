# Competition rules

## Score

For 50 organizer-selected distinct seeds, train independently from scratch on
the standard CIFAR-100 training split. For each of the 10,000 test images, the
model scores the 100 possible classes. Its highest-scoring class is its one
prediction ("top-1"). Accuracy is the percentage of those predictions that match
the correct labels: 7,500 correct predictions out of 10,000 means 75% accuracy.
We use the 100 specific categories ("fine classes"), such as apple or tiger,
rather than CIFAR-100's 20 broader groups ("coarse classes").

A complete submission qualifies when its mean accuracy across the 50 trials
reaches **75%**. Qualifiers are ranked by mean
`prepare_time + train_time`. Accuracy is averaged across trials; there is no
additional per-trial accuracy requirement. No best-seed selection or outlier removal.
Official CLI runs require an organizer-owned seed file, reused unchanged for every
team. The saved configuration records the exact ordered list.

The selected threshold is **0.75** and official runs enforce it. Development runs
may explore another target or disable qualification reporting. Freeze the environment,
inference convention, and limits before accepting official submissions.

## Timing

The harness owns synchronized wall-clock boundaries. Build and module import are
untimed once per submission process. Each trial charges parameter/buffer reset,
optimizer reset, input transfer, casting, preprocessing, whitening, augmentation,
training, and any fitting or data-derived state construction. Raw dataset download
and CPU loading are organizer setup outside the score. The official input location
is CPU memory for every fresh training run.

Compilation, allocation, autotuning and CUDA graph capture on synthetic inputs
may happen during build. Synthetic warmup may exercise forward, backward, and
optimizer code, but its changed state must be discarded by prepare. No real
training/test data, learned weights or trial seed may be used in build. Deferred
compilation occurring in prepare/train is charged there. Compilation happening
during inference is subject to the evaluation deadline.

There is a synchronization between prepare and train to record both durations;
its overhead is included in the total. The final synchronization waits for all
CUDA work, including other streams, before stopping the training timer. All CPU
threads/subprocesses doing training work must also finish before train returns.

## Evaluation and limits

Use plain single-view inference with the input convention in the
[submission contract](submission_template/README.md). No internal TTA, fitting,
test-time adaptation, test-set statistics, training-data lookup, or state changes
during evaluation. Registered parameters and buffers are checked before and after;
other state and custom code remain subject to source review.

Limits per submission/trial:

| Phase | Limit | Included in score? |
| --- | ---: | --- |
| Build, once | 600 seconds | No |
| Prepare + train, each trial | 600 seconds | Yes |
| Full test-set inference, each trial | 5 seconds | No |

These are organizer-owned resource limits.
Official submissions cannot raise them. The inference watchdog also covers model
state checks, preprocessing, transfers and synchronization. Record evaluation time
separately. The worker is terminated on timeout; sleeping inside forward cannot
evade the deadline. Intentional sleeps/cooldowns, host selection, and manipulation
of clocks, power settings or the harness are prohibited.

An exception, OOM, timeout, invalid/nonfinite output, detected evaluation mutation,
or incomplete run makes the submission nonqualifying. Preserve its failed trial
and prior raw results. Never qualify from surviving trials alone. An independently
verified infrastructure failure may be rerun only by an organizer restarting the
entire frozen submission with the same seeds and retaining both attempts' logs;
no selective retries chosen using accuracy or training time.

## Allowed development, prohibited learned state

Any autoresearch framework is allowed in development. Submit source for the final
model and training recipe. Models, optimizers, losses, schedules, augmentation,
precision, resolution, custom CUDA/Triton/C++ and systems optimizations may change.
The submitted training and inference code must run in the pinned PyTorch 2.4.0
environment. `torch.compile` is allowed. JAX, TensorFlow and other training runtimes
are not supported in this version. Submissions cannot change pinned dependencies
or install additional packages; include supporting Python and kernel source in
the team folder. These runtime restrictions do not limit development orchestration.

No pretrained weights/models, prior checkpoints, external training datasets,
pretrained features, constants encoding learned weights, hard-coded test
predictions, or learned state across official trials. Architecture and scalar
hyperparameters found during development are allowed; learned model tensors are not.

Training may use only the training split. Official test labels stay in the
organizer's supervisor. The test images are supplied only for frozen inference;
do not retain them for later trials. The standard test set is public, and local
development reports its accuracy, as is conventional for this speedrun. This is
not a claim of a new hidden test set. Do not encode test labels, fit on the test set,
or use test accuracy to choose a stopping point within an official trial.

## Environment and audit

Use one L40S, no MIG, the pinned container, a four-CPU container quota, four PyTorch
CPU threads, and networking disabled. Keep host/provider, CPU allocation, driver,
GPU power and clock policy consistent between teams. Record telemetry and the container digest. Use the same
organizer seed list across entries; keep it private until submissions are frozen.
Compare close results under matching host and thermal conditions.

Reset all recipe state per trial, including custom RNGs. Seeding does not imply
bitwise determinism: nondeterministic CUDA kernels are permitted for speed.
Check that repeat runs, reordered trials, and fresh containers produce stable
distributions. A submission must not depend on results or learned state from
previous trials.

The harness is a measurement tool with process timeouts and integrity checks, not
a security sandbox. Keeping test labels out of the submission API does not make
dataset files inaccessible to arbitrary Python code. Direct access to test data
outside frozen inference is prohibited and remains subject to source inspection.
Network isolation is enforced by the execution environment.
Finalists require source inspection, including state stored outside registered
model tensors. Only the frozen submission folder is imported into the organizer's
trusted repository; participant changes to benchmark files have no effect.
