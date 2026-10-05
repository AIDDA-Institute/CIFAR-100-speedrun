"""Fused random-crop and horizontal-flip augmentation for padded image batches."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the pinned environment ships Triton
    triton = None

SIZE = 32


def reference_crop_flip(images, shifts, flip):
    """Crop 32x32 windows at per-image (row, column) offsets, optionally mirrored."""
    device = images.device
    rows = shifts[:, :1] + torch.arange(SIZE, device=device)
    columns = shifts[:, 1:] + torch.arange(SIZE, device=device)
    batches = torch.arange(len(images), device=device)[:, None, None]
    outputs = images[batches, :, rows[:, :, None], columns[:, None, :]].permute(0, 3, 1, 2)
    return outputs.flip(-1) if flip else outputs


if triton is not None:

    @triton.jit
    def _crop_flip_kernel(
        source,
        destination,
        shifts,
        elements,
        source_n,
        source_c,
        source_h,
        source_w,
        destination_n,
        destination_c,
        destination_h,
        destination_w,
        FLIP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        image = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements
        channel = offsets // 1024
        row = (offsets // 32) % 32
        column = offsets % 32
        if FLIP:
            source_column = 31 - column
        else:
            source_column = column
        row_shift = tl.load(shifts + image * 2)
        column_shift = tl.load(shifts + image * 2 + 1)
        value = tl.load(
            source
            + image * source_n
            + channel * source_c
            + (row + row_shift) * source_h
            + (source_column + column_shift) * source_w,
            mask=mask,
        )
        tl.store(
            destination
            + image * destination_n
            + channel * destination_c
            + row * destination_h
            + column * destination_w,
            value,
            mask=mask,
        )


def triton_crop_flip(images, shifts, flip):
    """One-launch equivalent of reference_crop_flip; output is channels-last."""
    if triton is None:
        raise RuntimeError("Triton is not available")
    batch, channels = images.shape[:2]
    if images.shape[2] < SIZE + 4 or images.shape[3] < SIZE + 4:
        raise ValueError("images must be padded by at least two pixels on each side")
    destination = torch.empty(
        (batch, channels, SIZE, SIZE), device=images.device, dtype=images.dtype
    ).contiguous(memory_format=torch.channels_last)
    shifts = shifts.to(torch.int64).contiguous()
    elements = channels * SIZE * SIZE
    _crop_flip_kernel[(batch,)](
        images,
        destination,
        shifts,
        elements,
        *images.stride(),
        *destination.stride(),
        FLIP=bool(flip),
        BLOCK=triton.next_power_of_2(elements),
    )
    return destination
