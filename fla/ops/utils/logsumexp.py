# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import autotune_cache_kwargs

_CUTE_LOGSUMEXP_MIN_ROWS = 128
_CUTE_LOGSUMEXP_MIN_D = 32
_CUTE_LOGSUMEXP_MAX_D = 256
_CUTE_LOGSUMEXP_MAX_ELEMENTS = 2**31 - 1
_USE_FAST_OPS = os.environ.get("FLA_USE_FAST_OPS", "0") == "1"


def _can_use_cute_logsumexp(x: torch.Tensor, scale: float | None, dtype: torch.dtype | None) -> bool:
    if torch.compiler.is_compiling() or x.ndim == 0 or not x.is_cuda or not x.is_contiguous():
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    output_dtype = dtype or torch.float32
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if scale is not None and not isinstance(scale, int | float):
        return False
    D = x.shape[-1]
    if not (_CUTE_LOGSUMEXP_MIN_D <= D <= _CUTE_LOGSUMEXP_MAX_D):
        return False
    # Triton scales masked -inf padding, which yields NaN for non-power-of-two
    # rows when scale <= 0. Preserve that observable fallback behavior.
    if scale is not None and scale <= 0 and D & (D - 1):
        return False
    if x.numel() > _CUTE_LOGSUMEXP_MAX_ELEMENTS or x.numel() // D < _CUTE_LOGSUMEXP_MIN_ROWS:
        return False
    from fla.ops.backends.cute.runtime import is_cute_dsl_available

    return is_cute_dsl_available()


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float("inf"))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = log(tl.sum(exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


def logsumexp_fwd(
    x,
    scale: float | None = None,
    dtype: torch.dtype | None = None,
):
    r"""
    Compute the logsumexp of the input tensor over the last dimension.

    Args:
        x (Tensor):
            The input tensor of any shape.
        scale (Optional[float]):
            The scale applied to the input tensor. Default: `None`.
        dtype (Optional[torch.dtype]):
            The data type of the output tensor. Default: `None`.
    Returns:
        Tensor: The logsumexp of the input tensor.
    """

    if _can_use_cute_logsumexp(x, scale, dtype):
        from fla.ops.backends.cute.logsumexp import logsumexp_fwd_cute

        return logsumexp_fwd_cute(
            x,
            scale=scale,
            dtype=dtype,
            use_fast_ops=_USE_FAST_OPS,
        )

    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](
        x=x,
        z=z,
        scale=scale,
        D=D,
        B=B,
    )
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z
