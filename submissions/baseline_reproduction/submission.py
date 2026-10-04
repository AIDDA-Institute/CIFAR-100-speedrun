"""Fast CIFAR-100 ResNet9 with progressive resolution and graph warmup."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, pool: bool = False):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class ResNet9(nn.Module):
    def __init__(self, num_classes: int, width: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.conv1 = ConvBlock(3, width)
        self.conv2 = ConvBlock(width, width * 2, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(width * 2, width * 2),
            ConvBlock(width * 2, width * 2),
        )
        self.conv3 = ConvBlock(width * 2, width * 4, pool=True)
        self.conv4 = ConvBlock(width * 4, width * 8, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(width * 8, width * 8),
            ConvBlock(width * 8, width * 8),
        )
        self.classifier = nn.Linear(width * 8, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        out = self.conv1(images)
        out = self.conv2(out)
        out = out + self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out + self.res2(out)
        if out.shape[-2:] == (4, 4):
            out = F.max_pool2d(out, 4)
        else:
            out = F.adaptive_max_pool2d(out, 1)
        return self.classifier(out.flatten(1))


def _reset_model(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _warm_compiled_training_graphs(
    model: nn.Module,
    base_model: nn.Module,
    device: torch.device,
    shapes: tuple[tuple[int, int], ...],
) -> None:
    parameters = tuple(base_model.parameters())
    parameter_values = tuple(parameter.detach().clone() for parameter in parameters)
    buffers = tuple(base_model.named_buffers())
    buffer_values = {name: buffer.detach().clone() for name, buffer in buffers}
    gradients = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    training_modes = {name: module.training for name, module in base_model.named_modules()}
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    base_model.train()
    for resolution, batch_size in shapes:
        images = torch.zeros(
            batch_size,
            3,
            resolution,
            resolution,
            device=device,
        ).contiguous(memory_format=torch.channels_last)
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        for _ in range(2):
            for parameter in parameters:
                parameter.grad = None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                F.cross_entropy(model(images), labels).backward()
    torch.cuda.synchronize(device)

    with torch.no_grad():
        for parameter, value in zip(parameters, parameter_values, strict=True):
            parameter.copy_(value)
        for name, buffer in buffers:
            buffer.copy_(buffer_values[name])
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    for name, module in base_model.named_modules():
        module.train(training_modes[name])
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)

    exact = all(
        torch.equal(parameter, value)
        for parameter, value in zip(parameters, parameter_values, strict=True)
    )
    exact &= all(torch.equal(buffer, buffer_values[name]) for name, buffer in buffers)
    exact &= all(
        (parameter.grad is None and gradient is None)
        or (
            parameter.grad is not None
            and gradient is not None
            and torch.equal(parameter.grad, gradient)
        )
        for parameter, gradient in zip(parameters, gradients, strict=True)
    )
    exact &= all(
        module.training == training_modes[name]
        for name, module in base_model.named_modules()
    )
    exact &= torch.equal(torch.random.get_rng_state(), cpu_rng)
    exact &= torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    if not exact:
        raise RuntimeError("Synthetic graph warmup did not restore state exactly")


def build(context: BuildContext):
    parameters = context.parameters
    width = int(parameters.get("width", 64))
    low_resolution = int(parameters.get("low_resolution", 24))
    low_epochs = int(parameters.get("low_epochs", 10))
    base_model = ResNet9(context.num_classes, width=width)
    base_model = base_model.to(context.device, memory_format=torch.channels_last)
    compile_mode = parameters.get("compile_mode", "reduce-overhead")
    model = (
        torch.compile(base_model, mode=str(compile_mode))
        if compile_mode
        else base_model
    )
    if compile_mode and context.device.type == "cuda":
        resolutions = (
            (32,)
            if low_epochs == 0 or low_resolution == 32
            else (low_resolution, 32)
        )
        shapes = tuple(
            (resolution, batch_size)
            for resolution in resolutions
            for batch_size in (512, 336)
        )
        _warm_compiled_training_graphs(model, base_model, context.device, shapes)
    return SimpleNamespace(
        context=context,
        model=model,
        base_model=base_model,
        epochs=int(parameters.get("epochs", 40)),
        batch_size=int(parameters.get("batch_size", 512)),
        learning_rate=float(parameters.get("learning_rate", 0.2)),
        weight_decay=float(parameters.get("weight_decay", 5e-4)),
        label_smoothing=float(parameters.get("label_smoothing", 0.1)),
        cutout=int(parameters.get("cutout", 10)),
        mixup_alpha=float(parameters.get("mixup_alpha", 0.1)),
        pct_start=float(parameters.get("pct_start", 0.3)),
        div_factor=float(parameters.get("div_factor", 25)),
        final_div_factor=float(parameters.get("final_div_factor", 10000)),
        low_resolution=low_resolution,
        low_epochs=low_epochs,
    )


def prepare(state, data: TrainingData, seed: int) -> None:
    _reset_model(state.base_model)
    state.model.train()
    state.images = (
        data.images.to(state.context.device, dtype=torch.float32)
        .div_(255)
        .contiguous(memory_format=torch.channels_last)
    )
    state.labels = data.labels.to(state.context.device)
    state.generator = torch.Generator(device=state.context.device).manual_seed(seed)
    state.optimizer = torch.optim.SGD(
        state.model.parameters(),
        lr=state.learning_rate,
        momentum=0.9,
        weight_decay=state.weight_decay,
        nesterov=True,
    )
    steps_per_epoch = (len(state.images) + state.batch_size - 1) // state.batch_size
    state.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        state.optimizer,
        max_lr=state.learning_rate,
        epochs=state.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=state.pct_start,
        div_factor=state.div_factor,
        final_div_factor=state.final_div_factor,
    )


def _augment(
    images: torch.Tensor,
    generator: torch.Generator,
    cutout: int,
) -> torch.Tensor:
    count = len(images)
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    offsets_y = torch.randint(9, (count,), device=images.device, generator=generator)
    offsets_x = torch.randint(9, (count,), device=images.device, generator=generator)
    rows = offsets_y[:, None] + torch.arange(32, device=images.device)
    columns = offsets_x[:, None] + torch.arange(32, device=images.device)
    batch = torch.arange(count, device=images.device)[:, None, None]
    cropped = padded.permute(0, 2, 3, 1)[batch, rows[:, :, None], columns[:, None, :]]
    cropped = cropped.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    flipped = torch.rand(count, device=images.device, generator=generator) < 0.5
    cropped[flipped] = cropped[flipped].flip(-1)
    if cutout:
        centers_y = torch.randint(32, (count,), device=images.device, generator=generator)
        centers_x = torch.randint(32, (count,), device=images.device, generator=generator)
        coordinates = torch.arange(32, device=images.device)
        mask_y = (coordinates[None, :] - centers_y[:, None]).abs() < cutout / 2
        mask_x = (coordinates[None, :] - centers_x[:, None]).abs() < cutout / 2
        mask = mask_y[:, None, :, None] & mask_x[:, None, None, :]
        mean = cropped.new_tensor(MEAN).view(1, 3, 1, 1)
        cropped = torch.where(mask, mean, cropped)
    return cropped


def train(state) -> nn.Module:
    sample_count = len(state.images)
    for epoch in range(state.epochs):
        order = torch.randperm(sample_count, device=state.images.device, generator=state.generator)
        for start in range(0, sample_count, state.batch_size):
            indices = order[start : start + state.batch_size]
            images = _augment(state.images[indices], state.generator, state.cutout)
            if epoch < state.low_epochs and state.low_resolution != 32:
                images = F.interpolate(
                    images,
                    size=(state.low_resolution, state.low_resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).contiguous(memory_format=torch.channels_last)
            labels = state.labels[indices]
            mixup_weight = 1.0
            if state.mixup_alpha:
                concentration = images.new_full((2,), state.mixup_alpha)
                mixup_weight = torch._sample_dirichlet(concentration)[0]
                paired = torch.randperm(len(images), device=images.device)
                images = images.lerp(images[paired], 1 - mixup_weight)
                paired_labels = labels[paired]
            state.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=state.images.device.type, dtype=torch.bfloat16):
                logits = state.model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=state.label_smoothing,
                )
                if state.mixup_alpha:
                    paired_loss = F.cross_entropy(
                        logits,
                        paired_labels,
                        label_smoothing=state.label_smoothing,
                    )
                    loss = loss * mixup_weight + paired_loss * (1 - mixup_weight)
            loss.backward()
            state.optimizer.step()
            state.scheduler.step()
    return state.base_model
