import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from . import kernels

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def _zeroth_power(gradient, steps=3, epsilon=1e-7):
    matrix = gradient.reshape(len(gradient), -1).bfloat16()
    matrix = matrix / (matrix.norm() + epsilon)
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        matrix = matrix.T
    for _ in range(steps):
        square = matrix @ matrix.T
        matrix = 3.4445 * matrix + (-4.7750 * square + 2.0315 * square @ square) @ matrix
    if transposed:
        matrix = matrix.T
    return matrix.reshape_as(gradient)


def _batched_zeroth_power(gradients, steps=3, epsilon=1e-7):
    """_zeroth_power applied to a stack of same-shaped gradients in one set of matmuls."""
    matrix = gradients.reshape(len(gradients), gradients.size(1), -1).bfloat16()
    matrix = matrix / (matrix.norm(dim=(1, 2), keepdim=True) + epsilon)
    transposed = matrix.size(1) > matrix.size(2)
    if transposed:
        matrix = matrix.mT
    for _ in range(steps):
        square = matrix @ matrix.mT
        matrix = 3.4445 * matrix + (-4.7750 * square + 2.0315 * square @ square) @ matrix
    if transposed:
        matrix = matrix.mT
    return matrix.reshape_as(gradients)


class Muon(torch.optim.Optimizer):
    def __init__(self, parameters, learning_rate, momentum, batched=False):
        super().__init__(
            parameters,
            {"lr": learning_rate, "initial_lr": learning_rate, "momentum": momentum},
        )
        self.batched = batched

    @torch.no_grad()
    def step(self):
        if self.batched:
            for group in self.param_groups:
                self._batched_group_step(group)
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(group["momentum"]).add_(parameter.grad)
                update = parameter.grad.add(buffer, alpha=group["momentum"])
                parameter.mul_(len(parameter) ** 0.5 / parameter.norm())
                parameter.add_(_zeroth_power(update), alpha=-group["lr"])

    def _batched_group_step(self, group):
        buckets = {}
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            buffer = state["momentum_buffer"]
            buffer.mul_(group["momentum"]).add_(parameter.grad)
            update = parameter.grad.add(buffer, alpha=group["momentum"])
            parameter.mul_(len(parameter) ** 0.5 / parameter.norm())
            buckets.setdefault(tuple(parameter.shape), []).append((parameter, update))
        for items in buckets.values():
            updates = _batched_zeroth_power(torch.stack([update for _, update in items]))
            for (parameter, _), update in zip(items, updates):
                parameter.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, channels_in, channels_out):
        super().__init__(channels_in, channels_out, 3, padding=1, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth=2):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activation = nn.GELU()
        self.extra = nn.ModuleList(
            nn.Sequential(Conv(channels_out, channels_out), BatchNorm(channels_out))
            for _ in range(depth - 2)
        )

    def forward(self, inputs):
        outputs = self.activation(self.norm1(self.pool(self.conv1(inputs))))
        outputs = self.activation(self.norm2(self.conv2(outputs)))
        for block in self.extra:
            outputs = outputs + self.activation(block(outputs))
        return outputs


class Classifier(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        widths = parameters["widths"]
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.whiten = nn.Conv2d(3, 24, 2, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(24, widths[0], parameters["group_depth"]),
            ConvGroup(widths[0], widths[1], parameters["group_depth"]),
            ConvGroup(widths[1], widths[2], parameters["group_depth"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths[2], 100, bias=False)
        self.use_half = parameters["half"]

    def reset_parameters(self):
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.head.weight.data.mul_(self.head.weight.data.std().reciprocal())

    def initialize_whitening(self, images, epsilon):
        patches = images[:5000].unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flattened = patches.flatten(1)
        covariance = flattened.T @ flattened / len(flattened)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance, UPLO="U")
        filters = eigenvectors.T.reshape(-1, 3, 2, 2)
        filters = filters / torch.sqrt(eigenvalues[:, None, None, None] + epsilon)
        self.whiten.weight.data.copy_(torch.cat((filters, -filters)).to(self.whiten.weight))

    def features(self, inputs, *, train_whiten_bias=True):
        bias = self.whiten.bias if train_whiten_bias else self.whiten.bias.detach()
        outputs = F.conv2d(inputs, self.whiten.weight, bias)
        return self.layers(outputs).flatten(1)

    def normalized_forward(self, inputs, *, train_whiten_bias=True):
        outputs = self.features(inputs, train_whiten_bias=train_whiten_bias)
        return self.head(outputs) / outputs.size(-1)

    def averaged_tensors(self):
        """Trainable parameters plus BatchNorm running statistics, for weight averaging."""
        tensors = [parameter for parameter in self.parameters() if parameter.requires_grad]
        for module in self.modules():
            if isinstance(module, BatchNorm):
                tensors.extend((module.running_mean, module.running_var))
        return tensors

    def forward(self, inputs):
        dtype = torch.float16 if self.use_half and inputs.device.type == "cuda" else torch.float32
        inputs = ((inputs - self.mean) / self.std).to(dtype)
        return self.normalized_forward(inputs).float()


COMPILE_MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")


def _validate(parameters):
    for name in ("epochs", "batch_size"):
        if type(parameters[name]) is not int or parameters[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        type(parameters["widths"]) is not list
        or len(parameters["widths"]) != 3
        or any(type(width) is not int or width < 1 for width in parameters["widths"])
    ):
        raise ValueError("widths must be a list of three positive integers")
    for name in (
        "bias_lr",
        "head_lr",
        "conv_lr",
        "sgd_conv_lr",
        "momentum",
        "conv_momentum",
        "weight_decay",
        "label_smoothing",
        "whitening_epsilon",
    ):
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    for name in ("bias_lr", "head_lr", "conv_lr", "sgd_conv_lr", "whitening_epsilon"):
        if parameters[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("momentum", "conv_momentum"):
        if not 0 < parameters[name] < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if parameters["weight_decay"] < 0 or not 0 <= parameters["label_smoothing"] <= 1:
        raise ValueError("weight_decay must be nonnegative and label_smoothing in [0, 1]")
    if parameters["optimizer"] not in ("muon", "sgd"):
        raise ValueError("optimizer must be muon or sgd")
    for name in ("half", "compile", "whitening", "batched_muon", "triton_crop", "profile"):
        if type(parameters[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(parameters["group_depth"]) is not int or not 2 <= parameters["group_depth"] <= 4:
        raise ValueError("group_depth must be an integer from 2 to 4")
    if parameters["compile_mode"] not in COMPILE_MODES:
        raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
    if parameters["compile_mode"] != "default" and not parameters["compile"]:
        raise ValueError("compile_mode requires compile=true")
    for name in ("ema_epochs", "small_epochs"):
        if type(parameters[name]) is not int or not 0 <= parameters[name] <= parameters["epochs"]:
            raise ValueError(f"{name} must be an integer from 0 to epochs")
    decay = parameters["ema_decay"]
    if type(decay) not in (int, float) or not 0 < decay < 1:
        raise ValueError("ema_decay must be in (0, 1)")
    resolution = parameters["small_resolution"]
    if type(resolution) is not int or not 25 <= resolution <= 32:
        raise ValueError("small_resolution must be an integer from 25 to 32")
    if type(parameters["ridge_head"]) is not bool:
        raise ValueError("ridge_head must be boolean")
    ridge = parameters["ridge_lambda"]
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge_lambda must be positive and finite")


def build(context):
    parameters = {
        "epochs": 14,
        "batch_size": 2000,
        "widths": [128, 384, 512],
        "bias_lr": 0.053,
        "head_lr": 0.67,
        "conv_lr": 0.24,
        "sgd_conv_lr": 0.0015,
        "momentum": 0.85,
        "conv_momentum": 0.6,
        "weight_decay": 0.004,
        "label_smoothing": 0.2,
        "whitening_epsilon": 5e-4,
        "optimizer": "muon",
        "half": True,
        "compile": True,
        "whitening": True,
        "compile_mode": "default",
        "batched_muon": True,
        "triton_crop": False,
        "profile": False,
        "group_depth": 3,
        "ema_epochs": 4,
        "ema_decay": 0.98,
        "small_resolution": 32,
        "small_epochs": 0,
        "ridge_head": False,
        "ridge_lambda": 1e-3,
    }
    unknown = context.parameters.keys() - parameters.keys()
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    parameters.update(context.parameters)
    _validate(parameters)
    model = Classifier(parameters).to(context.device, memory_format=torch.channels_last)
    if parameters["half"] and context.device.type == "cuda":
        model.half()
        model.mean = model.mean.float()
        model.std = model.std.float()
        for module in model.modules():
            if isinstance(module, BatchNorm):
                module.float()
    state = SimpleNamespace(model=model, context=context, parameters=parameters)
    state.train_forward = model.normalized_forward
    if parameters["compile"]:
        options = {"dynamic": False}
        if parameters["compile_mode"] != "default":
            options["mode"] = parameters["compile_mode"]
        state.train_forward = torch.compile(model.normalized_forward, **options)
        resolutions = [32]
        if _small_epochs(parameters):
            resolutions.insert(0, parameters["small_resolution"])
        for resolution in resolutions:
            synthetic = torch.randn(
                parameters["batch_size"],
                3,
                resolution,
                resolution,
                device=context.device,
                dtype=model.whiten.weight.dtype,
            ).contiguous(memory_format=torch.channels_last)
            for train_whiten_bias in (True, False):
                _mark_step(parameters)
                outputs = state.train_forward(synthetic, train_whiten_bias=train_whiten_bias)
                outputs.float().square().mean().backward()
                model.zero_grad(set_to_none=True)
        model.reset_parameters()
    return state


def prepare(state, data, seed):
    del seed
    model = state.model
    parameters = state.parameters
    model.reset_parameters()
    model.zero_grad(set_to_none=True)
    model.train()
    dtype = (
        torch.float16
        if parameters["half"] and state.context.device.type == "cuda"
        else torch.float32
    )
    images = data.images.to(state.context.device, dtype=torch.float32, non_blocking=True).div_(255)
    images = ((images - model.mean) / model.std).to(dtype, memory_format=torch.channels_last)
    labels = data.labels.to(state.context.device, non_blocking=True)
    if not len(images):
        raise ValueError("Training data must not be empty")
    if parameters["whitening"]:
        model.initialize_whitening(images, parameters["whitening_epsilon"])
    flip_mask = torch.rand(len(images), device=images.device) < 0.5
    flipped = torch.where(flip_mask[:, None, None, None], images.flip(-1), images)
    state.images = F.pad(flipped, (2, 2, 2, 2), mode="reflect")
    state.labels = labels
    decay = parameters["weight_decay"]
    state.primary_optimizer = torch.optim.SGD(
        [
            {
                "params": [model.whiten.bias],
                "lr": parameters["bias_lr"],
                "initial_lr": parameters["bias_lr"],
                "weight_decay": decay / parameters["bias_lr"],
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "norm" in name and parameter.requires_grad
                ],
                "lr": parameters["bias_lr"],
                "initial_lr": parameters["bias_lr"],
                "weight_decay": decay / parameters["bias_lr"],
            },
            {
                "params": [model.head.weight],
                "lr": parameters["head_lr"],
                "initial_lr": parameters["head_lr"],
                "weight_decay": decay / parameters["head_lr"],
            },
        ],
        momentum=parameters["momentum"],
        nesterov=True,
        fused=True if state.context.device.type == "cuda" else None,
    )
    filters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim == 4
    ]
    if parameters["optimizer"] == "muon":
        state.filter_optimizer = Muon(
            filters,
            parameters["conv_lr"],
            parameters["conv_momentum"],
            batched=parameters["batched_muon"],
        )
    else:
        state.filter_optimizer = torch.optim.SGD(
            filters,
            lr=parameters["sgd_conv_lr"],
            momentum=parameters["conv_momentum"],
            nesterov=True,
            fused=True if state.context.device.type == "cuda" else None,
        )
        state.filter_optimizer.param_groups[0]["initial_lr"] = parameters["sgd_conv_lr"]


def _crop(images, flip, use_triton):
    shifts = torch.randint(0, 5, (len(images), 2), device=images.device)
    if use_triton and images.is_cuda:
        return kernels.triton_crop_flip(images, shifts, flip)
    return kernels.reference_crop_flip(images, shifts, flip)


def _mark_step(parameters):
    if parameters["compile_mode"] == "reduce-overhead":
        torch.compiler.cudagraph_mark_step_begin()


def _small_epochs(parameters):
    if parameters["small_resolution"] == 32:
        return 0
    return parameters["small_epochs"]


def _downscale(images, resolution):
    outputs = F.interpolate(images, size=(resolution, resolution), mode="bilinear")
    return outputs.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def _fit_ridge_head(state):
    """Refit the linear head by ridge regression on eval-mode features of the training images."""
    model = state.model
    parameters = state.parameters
    model.eval()
    images = state.images[:, :, 2:-2, 2:-2]
    width = model.head.weight.size(1)
    gram = torch.zeros(width, width, device=images.device)
    cross = torch.zeros(width, model.head.weight.size(0), device=images.device)
    for start in range(0, len(images), parameters["batch_size"]):
        batch = images[start : start + parameters["batch_size"]]
        features = model.features(batch.contiguous(memory_format=torch.channels_last)).float()
        targets = F.one_hot(state.labels[start : start + len(batch)], cross.size(1)).float()
        gram.addmm_(features.T, features)
        cross.addmm_(features.T, targets)
    gram.diagonal().add_(parameters["ridge_lambda"] * gram.trace() / width)
    solution = torch.linalg.solve(gram, cross)
    model.head.weight.copy_(solution.T)


def train(state):
    if not state.parameters["profile"]:
        return _train(state)
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = state.context.device.type == "cuda"
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        model = _train(state)
        if cuda:
            torch.cuda.synchronize()
    order = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    print("turbo-profile")
    print(profiler.key_averages().table(sort_by=order, row_limit=25), flush=True)
    return model


def _train(state):
    model = state.model
    parameters = state.parameters
    optimizers = (state.primary_optimizer, state.filter_optimizer)
    batch_size = parameters["batch_size"]
    batches_per_epoch = max(1, len(state.images) // batch_size)
    total_steps = parameters["epochs"] * batches_per_epoch
    bias_steps = min(3 * batches_per_epoch, total_steps)
    small_epochs = _small_epochs(parameters)
    ema_start = parameters["epochs"] - parameters["ema_epochs"]
    averaged = model.averaged_tensors() if parameters["ema_epochs"] else []
    ema = None
    step = 0
    for epoch in range(parameters["epochs"]):
        images = _crop(state.images, epoch % 2 == 1, parameters["triton_crop"])
        if epoch < small_epochs:
            images = _downscale(images, parameters["small_resolution"])
        if averaged and epoch == ema_start:
            ema = [tensor.detach().float().clone() for tensor in averaged]
        order = torch.randperm(len(images), device=images.device)
        for indices in order[: batches_per_epoch * batch_size].split(batch_size):
            _mark_step(parameters)
            logits = state.train_forward(images[indices], train_whiten_bias=step < bias_steps)
            loss = F.cross_entropy(
                logits,
                state.labels[indices],
                label_smoothing=parameters["label_smoothing"],
                reduction="sum",
            )
            loss.backward()
            for group in state.primary_optimizer.param_groups:
                duration = (
                    bias_steps if group is state.primary_optimizer.param_groups[0] else total_steps
                )
                group["lr"] = group["initial_lr"] * max(0, 1 - step / duration)
            for group in state.filter_optimizer.param_groups:
                group["lr"] = group["initial_lr"] * max(0, 1 - step / total_steps)
            for optimizer in optimizers:
                optimizer.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                with torch.no_grad():
                    current = [tensor.detach().float() for tensor in averaged]
                    torch._foreach_lerp_(ema, current, 1 - parameters["ema_decay"])
    if ema is not None:
        with torch.no_grad():
            for tensor, average in zip(averaged, ema):
                tensor.copy_(average)
    if parameters["ridge_head"]:
        _fit_ridge_head(state)
    return model
