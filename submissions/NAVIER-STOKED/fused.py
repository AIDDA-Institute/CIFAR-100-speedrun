"""Fused Triton kernels for the memory-bound work around each convolution (training mode):

    bn_silu:       y = SiLU(BatchNorm(x))
    pool_bn_silu:  y = SiLU(BatchNorm(MaxPool2x2(x)))

Activations are channels-last bf16, viewed as [rows, C]. Forward is a statistics pass
(per-channel fp32 sums via atomics; for pool_bn_silu it also writes the pooled tensor and
a 2-bit argmax) plus one normalize+affine+SiLU pass. Backward is a reduction pass
(sum g, sum g*xhat) plus one pass that writes dx; for pool_bn_silu that pass scatters the
gradient straight into the pool input. BatchNorm running statistics are updated like
nn.BatchNorm2d (momentum, unbiased variance). Exposed as torch.library custom ops so
torch.compile and CUDA-graph capture treat them as opaque kernels.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor, nn

BLOCK_C = 64
ROWS = 64  # rows per tile (best of a PCIe/SXM sweep)
TILES = 4  # tiles per program in reductions (fewer atomics)
NUM_WARPS = 4


# ------------------------------------------------------------------ kernels
@triton.jit
def _stats_kernel(X, SUM, SQ, M, C, ROWS: tl.constexpr, TILES: tl.constexpr, BC: tl.constexpr):
    pid, cid = tl.program_id(0), tl.program_id(1)
    cols = cid * BC + tl.arange(0, BC)
    cmask = cols < C
    acc = tl.zeros((BC,), tl.float32)
    acc2 = tl.zeros((BC,), tl.float32)
    for t in range(TILES):
        rows = (pid * TILES + t) * ROWS + tl.arange(0, ROWS)
        mask = (rows[:, None] < M) & cmask[None, :]
        x = tl.load(X + rows[:, None] * C + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, 0)
        acc2 += tl.sum(x * x, 0)
    tl.atomic_add(SUM + cols, acc, mask=cmask)
    tl.atomic_add(SQ + cols, acc2, mask=cmask)


@triton.jit
def _pool_stats_kernel(
    X,
    P,
    IDX,
    SUM,
    SQ,
    M,
    C,
    H,
    W,
    H2,
    W2,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
    BC: tl.constexpr,
):
    """X: [N, H, W, C] -> P: [N, H2, W2, C] (2x2 max, floor), IDX: argmax 0..3, plus stats."""
    pid, cid = tl.program_id(0), tl.program_id(1)
    cols = cid * BC + tl.arange(0, BC)
    cmask = cols < C
    acc = tl.zeros((BC,), tl.float32)
    acc2 = tl.zeros((BC,), tl.float32)
    for t in range(TILES):
        rows = (pid * TILES + t) * ROWS + tl.arange(0, ROWS)  # pooled rows
        rmask = rows < M
        n = rows // (H2 * W2)
        rem = rows % (H2 * W2)
        h2 = rem // W2
        w2 = rem % W2
        src = (n * H + 2 * h2) * W + 2 * w2  # top-left source row
        mask = rmask[:, None] & cmask[None, :]
        base = X + src[:, None] * C + cols[None, :]
        v0 = tl.load(base, mask=mask, other=float("-inf")).to(tl.float32)
        v1 = tl.load(base + C, mask=mask, other=float("-inf")).to(tl.float32)
        v2 = tl.load(base + W * C, mask=mask, other=float("-inf")).to(tl.float32)
        v3 = tl.load(base + W * C + C, mask=mask, other=float("-inf")).to(tl.float32)
        best = v0
        idx = tl.zeros((ROWS, BC), tl.int32)
        idx = tl.where(v1 > best, 1, idx)
        best = tl.maximum(best, v1)
        idx = tl.where(v2 > best, 2, idx)
        best = tl.maximum(best, v2)
        idx = tl.where(v3 > best, 3, idx)
        best = tl.maximum(best, v3)
        offs = rows[:, None] * C + cols[None, :]
        tl.store(P + offs, best.to(P.dtype.element_ty), mask=mask)
        tl.store(IDX + offs, idx.to(tl.int8), mask=mask)
        # statistics of the stored (bf16-rounded) pooled values, as BatchNorm would see them
        bq = tl.where(mask, best.to(P.dtype.element_ty).to(tl.float32), 0.0)
        acc += tl.sum(bq, 0)
        acc2 += tl.sum(bq * bq, 0)
    tl.atomic_add(SUM + cols, acc, mask=cmask)
    tl.atomic_add(SQ + cols, acc2, mask=cmask)


@triton.jit
def _apply_kernel(X, Y, MEAN, RSTD, Wt, B, M, C, ROWS: tl.constexpr, BC: tl.constexpr):
    pid, cid = tl.program_id(0), tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = cid * BC + tl.arange(0, BC)
    cmask = cols < C
    mask = (rows[:, None] < M) & cmask[None, :]
    offs = rows[:, None] * C + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(MEAN + cols, mask=cmask, other=0.0)
    rstd = tl.load(RSTD + cols, mask=cmask, other=0.0)
    w = tl.load(Wt + cols, mask=cmask, other=0.0)
    b = tl.load(B + cols, mask=cmask, other=0.0)
    a = (x - mean[None, :]) * (rstd * w)[None, :] + b[None, :]
    y = a * tl.sigmoid(a)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_reduce_kernel(
    X,
    DY,
    MEAN,
    RSTD,
    Wt,
    B,
    SG,
    SGZ,
    M,
    C,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
    BC: tl.constexpr,
):
    pid, cid = tl.program_id(0), tl.program_id(1)
    cols = cid * BC + tl.arange(0, BC)
    cmask = cols < C
    mean = tl.load(MEAN + cols, mask=cmask, other=0.0)
    rstd = tl.load(RSTD + cols, mask=cmask, other=0.0)
    w = tl.load(Wt + cols, mask=cmask, other=0.0)
    b = tl.load(B + cols, mask=cmask, other=0.0)
    acc = tl.zeros((BC,), tl.float32)
    acc2 = tl.zeros((BC,), tl.float32)
    for t in range(TILES):
        rows = (pid * TILES + t) * ROWS + tl.arange(0, ROWS)
        mask = (rows[:, None] < M) & cmask[None, :]
        offs = rows[:, None] * C + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        z = (x - mean[None, :]) * rstd[None, :]
        a = z * w[None, :] + b[None, :]
        s = tl.sigmoid(a)
        g = dy * (s * (1.0 + a * (1.0 - s)))
        g = tl.where(mask, g, 0.0)
        acc += tl.sum(g, 0)
        acc2 += tl.sum(g * z, 0)
    tl.atomic_add(SG + cols, acc, mask=cmask)
    tl.atomic_add(SGZ + cols, acc2, mask=cmask)


@triton.jit
def _bwd_apply_kernel(
    X, DY, DX, MEAN, RSTD, Wt, B, SG, SGZ, M, C, ROWS: tl.constexpr, BC: tl.constexpr
):
    pid, cid = tl.program_id(0), tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = cid * BC + tl.arange(0, BC)
    cmask = cols < C
    mask = (rows[:, None] < M) & cmask[None, :]
    offs = rows[:, None] * C + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(MEAN + cols, mask=cmask, other=0.0)
    rstd = tl.load(RSTD + cols, mask=cmask, other=0.0)
    w = tl.load(Wt + cols, mask=cmask, other=0.0)
    b = tl.load(B + cols, mask=cmask, other=0.0)
    sg = tl.load(SG + cols, mask=cmask, other=0.0) / M
    sgz = tl.load(SGZ + cols, mask=cmask, other=0.0) / M
    z = (x - mean[None, :]) * rstd[None, :]
    a = z * w[None, :] + b[None, :]
    s = tl.sigmoid(a)
    g = dy * (s * (1.0 + a * (1.0 - s)))
    dx = (w * rstd)[None, :] * (g - sg[None, :] - z * sgz[None, :])
    tl.store(DX + offs, dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _pool_bwd_apply_kernel(
    P,
    IDX,
    DY,
    DX,
    MEAN,
    RSTD,
    Wt,
    B,
    SG,
    SGZ,
    M,
    C,
    H,
    W,
    H2,
    W2,
    ROWS: tl.constexpr,
    BC: tl.constexpr,
):
    """Gradient w.r.t. the pooled tensor, scattered to the 2x2 window of the pool input
    (the argmax position gets it, the other three get 0)."""
    pid, cid = tl.program_id(0), tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = cid * BC + tl.arange(0, BC)
    cmask = cols < C
    rmask = rows < M
    mask = rmask[:, None] & cmask[None, :]
    offs = rows[:, None] * C + cols[None, :]
    x = tl.load(P + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    idx = tl.load(IDX + offs, mask=mask, other=0).to(tl.int32)
    mean = tl.load(MEAN + cols, mask=cmask, other=0.0)
    rstd = tl.load(RSTD + cols, mask=cmask, other=0.0)
    w = tl.load(Wt + cols, mask=cmask, other=0.0)
    b = tl.load(B + cols, mask=cmask, other=0.0)
    sg = tl.load(SG + cols, mask=cmask, other=0.0) / M
    sgz = tl.load(SGZ + cols, mask=cmask, other=0.0) / M
    z = (x - mean[None, :]) * rstd[None, :]
    a = z * w[None, :] + b[None, :]
    s = tl.sigmoid(a)
    g = dy * (s * (1.0 + a * (1.0 - s)))
    dp = (w * rstd)[None, :] * (g - sg[None, :] - z * sgz[None, :])
    n = rows // (H2 * W2)
    rem = rows % (H2 * W2)
    h2 = rem // W2
    w2 = rem % W2
    src = (n * H + 2 * h2) * W + 2 * w2
    base = DX + src[:, None] * C + cols[None, :]
    zero = tl.zeros((ROWS, BC), tl.float32)
    dt = DX.dtype.element_ty
    tl.store(base, tl.where(idx == 0, dp, zero).to(dt), mask=mask)
    tl.store(base + C, tl.where(idx == 1, dp, zero).to(dt), mask=mask)
    tl.store(base + W * C, tl.where(idx == 2, dp, zero).to(dt), mask=mask)
    tl.store(base + W * C + C, tl.where(idx == 3, dp, zero).to(dt), mask=mask)


# ------------------------------------------------------------------ host helpers
def _rows(x: Tensor) -> Tensor:
    """Channels-last NCHW tensor -> [N*H*W, C] view (no copy)."""
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def _empty_cl(shape, like: Tensor, dtype=None) -> Tensor:
    """Uninitialised channels-last tensor (allocated in that layout: no copy)."""
    return torch.empty(
        shape, dtype=dtype or like.dtype, device=like.device, memory_format=torch.channels_last
    )


def _grid_reduce(m, c):
    return (triton.cdiv(m, ROWS * TILES), triton.cdiv(c, BLOCK_C))


def _grid_apply(m, c):
    return (triton.cdiv(m, ROWS), triton.cdiv(c, BLOCK_C))


def _finish_stats(s, sq, m, eps):
    mean = s / m
    var = (sq / m - mean * mean).clamp_min(0.0)
    return mean, torch.rsqrt(var + eps), var


def update_running(module: nn.BatchNorm2d, mean: Tensor, var: Tensor, m: int) -> None:
    """Running statistics exactly like nn.BatchNorm2d (momentum, unbiased variance)."""
    with torch.no_grad():
        module.num_batches_tracked.add_(1)
        mom = module.momentum
        module.running_mean.mul_(1 - mom).add_(mean, alpha=mom)
        module.running_var.mul_(1 - mom).add_(var * (m / max(m - 1, 1)), alpha=mom)


# ------------------------------------------------------------------ custom ops
@torch.library.custom_op("navier::bn_silu_fwd", mutates_args=())
def bn_silu_fwd(
    x: Tensor, weight: Tensor, bias: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n, c, h, w = x.shape
    x2 = _rows(x)
    m = x2.shape[0]
    s = torch.zeros(c, device=x.device, dtype=torch.float32)
    sq = torch.zeros(c, device=x.device, dtype=torch.float32)
    _stats_kernel[_grid_reduce(m, c)](
        x2, s, sq, m, c, ROWS=ROWS, TILES=TILES, BC=BLOCK_C, num_warps=NUM_WARPS
    )
    mean, rstd, var = _finish_stats(s, sq, m, eps)
    y = _empty_cl((n, c, h, w), x)
    _apply_kernel[_grid_apply(m, c)](
        x2,
        _rows(y),
        mean,
        rstd,
        weight.float(),
        bias.float(),
        m,
        c,
        ROWS=ROWS,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    return y, mean, rstd, var


@bn_silu_fwd.register_fake
def _(x, weight, bias, eps):
    c = x.shape[1]
    stat = x.new_empty(c, dtype=torch.float32)
    return _empty_cl(x.shape, x), stat, torch.empty_like(stat), torch.empty_like(stat)


@torch.library.custom_op("navier::bn_silu_bwd", mutates_args=())
def bn_silu_bwd(
    dy: Tensor, x: Tensor, weight: Tensor, bias: Tensor, mean: Tensor, rstd: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    c = x.shape[1]
    x2, dy2 = _rows(x), _rows(dy)
    m = x2.shape[0]
    sg = torch.zeros(c, device=x.device, dtype=torch.float32)
    sgz = torch.zeros(c, device=x.device, dtype=torch.float32)
    w, b = weight.float(), bias.float()
    _bwd_reduce_kernel[_grid_reduce(m, c)](
        x2,
        dy2,
        mean,
        rstd,
        w,
        b,
        sg,
        sgz,
        m,
        c,
        ROWS=ROWS,
        TILES=TILES,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    dx = _empty_cl(x.shape, x)
    _bwd_apply_kernel[_grid_apply(m, c)](
        x2,
        dy2,
        _rows(dx),
        mean,
        rstd,
        w,
        b,
        sg,
        sgz,
        m,
        c,
        ROWS=ROWS,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    return dx, sgz.to(weight.dtype), sg.to(bias.dtype)  # d weight = sum g*z, d bias = sum g


@bn_silu_bwd.register_fake
def _(dy, x, weight, bias, mean, rstd):
    return _empty_cl(x.shape, x), torch.empty_like(weight), torch.empty_like(bias)


def _bn_silu_setup(ctx, inputs, output):
    x, weight, bias = inputs[0], inputs[1], inputs[2]
    _, mean, rstd, _ = output
    ctx.save_for_backward(x, weight, bias, mean, rstd)


def _bn_silu_backward(ctx, dy, _dmean, _drstd, _dvar):
    x, weight, bias, mean, rstd = ctx.saved_tensors
    dx, dw, db = bn_silu_bwd(
        dy.contiguous(memory_format=torch.channels_last), x, weight, bias, mean, rstd
    )
    return dx, dw, db, None


bn_silu_fwd.register_autograd(_bn_silu_backward, setup_context=_bn_silu_setup)


@torch.library.custom_op("navier::pool_bn_silu_fwd", mutates_args=())
def pool_bn_silu_fwd(
    x: Tensor, weight: Tensor, bias: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    x2 = _rows(x)
    p = _empty_cl((n, c, h2, w2), x)
    idx = _empty_cl((n, c, h2, w2), x, torch.int8)
    m = n * h2 * w2
    s = torch.zeros(c, device=x.device, dtype=torch.float32)
    sq = torch.zeros(c, device=x.device, dtype=torch.float32)
    _pool_stats_kernel[_grid_reduce(m, c)](
        x2,
        _rows(p),
        _rows(idx),
        s,
        sq,
        m,
        c,
        h,
        w,
        h2,
        w2,
        ROWS=ROWS,
        TILES=TILES,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    mean, rstd, var = _finish_stats(s, sq, m, eps)
    y = _empty_cl((n, c, h2, w2), x)
    _apply_kernel[_grid_apply(m, c)](
        _rows(p),
        _rows(y),
        mean,
        rstd,
        weight.float(),
        bias.float(),
        m,
        c,
        ROWS=ROWS,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    return y, p, idx, mean, rstd, var


@pool_bn_silu_fwd.register_fake
def _(x, weight, bias, eps):
    n, c, h, w = x.shape
    shape = (n, c, h // 2, w // 2)
    idx = _empty_cl(shape, x, torch.int8)
    stat = x.new_empty(c, dtype=torch.float32)
    return (
        _empty_cl(shape, x),
        _empty_cl(shape, x),
        idx,
        stat,
        torch.empty_like(stat),
        torch.empty_like(stat),
    )


@torch.library.custom_op("navier::pool_bn_silu_bwd", mutates_args=())
def pool_bn_silu_bwd(
    dy: Tensor,
    p: Tensor,
    idx: Tensor,
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    rstd: Tensor,
    h: int,
    w: int,
) -> tuple[Tensor, Tensor, Tensor]:
    n, c, h2, w2 = p.shape
    p2, dy2 = _rows(p), _rows(dy)
    m = p2.shape[0]
    sg = torch.zeros(c, device=p.device, dtype=torch.float32)
    sgz = torch.zeros(c, device=p.device, dtype=torch.float32)
    wf, bf = weight.float(), bias.float()
    _bwd_reduce_kernel[_grid_reduce(m, c)](
        p2,
        dy2,
        mean,
        rstd,
        wf,
        bf,
        sg,
        sgz,
        m,
        c,
        ROWS=ROWS,
        TILES=TILES,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    dx = _empty_cl((n, c, h, w), p)
    if h % 2:  # rows / columns the floor-mode pool never reads get zero gradient
        dx[:, :, h - 1, :] = 0
    if w % 2:
        dx[:, :, :, w - 1] = 0
    _pool_bwd_apply_kernel[_grid_apply(m, c)](
        p2,
        _rows(idx),
        dy2,
        _rows(dx),
        mean,
        rstd,
        wf,
        bf,
        sg,
        sgz,
        m,
        c,
        h,
        w,
        h2,
        w2,
        ROWS=ROWS,
        BC=BLOCK_C,
        num_warps=NUM_WARPS,
    )
    return dx, sgz.to(weight.dtype), sg.to(bias.dtype)


@pool_bn_silu_bwd.register_fake
def _(dy, p, idx, weight, bias, mean, rstd, h, w):
    n, c = p.shape[:2]
    return _empty_cl((n, c, h, w), p), torch.empty_like(weight), torch.empty_like(bias)


def _pool_setup(ctx, inputs, output):
    x, weight, bias = inputs[0], inputs[1], inputs[2]
    _, p, idx, mean, rstd, _ = output
    ctx.hw = (x.shape[2], x.shape[3])
    ctx.save_for_backward(p, idx, weight, bias, mean, rstd)


def _pool_backward(ctx, dy, _dp, _didx, _dmean, _drstd, _dvar):
    p, idx, weight, bias, mean, rstd = ctx.saved_tensors
    dx, dw, db = pool_bn_silu_bwd(
        dy.contiguous(memory_format=torch.channels_last),
        p,
        idx,
        weight,
        bias,
        mean,
        rstd,
        ctx.hw[0],
        ctx.hw[1],
    )
    return dx, dw, db, None


pool_bn_silu_fwd.register_autograd(_pool_backward, setup_context=_pool_setup)


# ------------------------------------------------------------------ modules
class BNSiLU(nn.BatchNorm2d):
    """BatchNorm2d followed by SiLU; fused Triton kernels in training on CUDA. Same
    parameters, buffers and state-dict keys as nn.BatchNorm2d."""

    def forward(self, x):
        if self.training and x.is_cuda:
            y, mean, _, var = bn_silu_fwd(x, self.weight, self.bias, self.eps)
            update_running(self, mean, var, x.shape[0] * x.shape[2] * x.shape[3])
            return y
        return nn.functional.silu(super().forward(x))


class PoolBNSiLU(nn.BatchNorm2d):
    """MaxPool2d(2) -> BatchNorm2d -> SiLU; fused in training on CUDA."""

    def forward(self, x):
        if self.training and x.is_cuda:
            y, _, _, mean, _, var = pool_bn_silu_fwd(x, self.weight, self.bias, self.eps)
            update_running(self, mean, var, x.shape[0] * (x.shape[2] // 2) * (x.shape[3] // 2))
            return y
        return nn.functional.silu(super().forward(nn.functional.max_pool2d(x, 2)))
