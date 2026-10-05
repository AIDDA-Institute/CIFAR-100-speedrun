"""Optional CUDA crop kernel; imported only when the Triton path is selected."""

import torch
import triton
import triton.language as tl


@triton.jit
def _crop_flip(
    source,
    shifts,
    output,
    TOTAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    SIZE: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SY: tl.constexpr,
    SX: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = linear < TOTAL
    channel = linear % CHANNELS
    pixel = linear // CHANNELS
    col = pixel % SIZE
    row = (pixel // SIZE) % SIZE
    batch = pixel // (SIZE * SIZE)
    shift_y = tl.load(shifts + 2 * batch, valid, other=0)
    shift_x = tl.load(shifts + 2 * batch + 1, valid, other=0)
    if FLIP:
        col = SIZE - 1 - col
    offset = batch * SN + channel * SC + (row + shift_y) * SY + (col + shift_x) * SX
    value = tl.load(source + offset, valid, other=0)
    tl.store(output + linear, value, valid)


def crop_flip(images, size, flip=False, shifts=None):
    n, c, _, w = images.shape
    if shifts is None:
        shifts = torch.randint(0, w - size + 1, (n, 2), device=images.device)
    out = torch.empty(
        (n, c, size, size),
        device=images.device,
        dtype=images.dtype,
        memory_format=torch.channels_last,
    )
    _crop_flip[(triton.cdiv(out.numel(), 512),)](
        images, shifts, out, out.numel(), c, size, *images.stride(), flip, 512
    )
    return out
