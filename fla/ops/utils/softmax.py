# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import IS_AMD, autotune_cache_kwargs

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def softmax_fwd_kernel(
    x,
    p,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float("inf"))
    b_m = tl.max(b_x, 0)
    b_x = exp(b_x - b_m)
    b_p = b_x / tl.sum(b_x, 0)

    tl.store(p + i_n * D + o_d, b_p.to(p.dtype.element_ty), mask=m_d)


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def softmax_bwd_kernel(
    p,
    dp,
    ds,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D

    b_p = tl.load(p + i_n * D + o_d, mask=m_d, other=0.0)
    b_dp = tl.load(dp + i_n * D + o_d, mask=m_d, other=0.0)
    b_pp = tl.sum(b_p * b_dp, 0)
    b_ds = b_p * b_dp - b_p * b_pp
    tl.store(ds + i_n * D + o_d, b_ds.to(ds.dtype.element_ty), mask=m_d)


def softmax_fwd(
    x: torch.Tensor,
    dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    shape = x.shape
    output_dtype = dtype or x.dtype
    if (
        not torch.compiler.is_compiling()
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and output_dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < x.shape[-1] <= 1024
    ):
        from fla.ops.backends.cute import is_cute_dsl_available

        if is_cute_dsl_available():
            from fla.ops.backends.cute.softmax import softmax_fwd_cute

            return softmax_fwd_cute(x, output_dtype)
    x = x.view(-1, x.shape[-1])

    N, D = x.shape
    B = triton.next_power_of_2(D)

    p = torch.empty_like(x, dtype=dtype)
    softmax_fwd_kernel[(N,)](
        x=x,
        p=p,
        D=D,
        B=B,
    )
    return p.view(*shape)


def softmax_bwd(
    p: torch.Tensor,
    dp: torch.Tensor,
    dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    shape = p.shape
    output_dtype = dtype or p.dtype
    if (
        not torch.compiler.is_compiling()
        and p.is_cuda
        and dp.is_cuda
        and p.is_contiguous()
        and dp.is_contiguous()
        and p.shape == dp.shape
        and p.device == dp.device
        and p.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and dp.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and output_dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < p.shape[-1] <= 1024
    ):
        from fla.ops.backends.cute import is_cute_dsl_available

        if is_cute_dsl_available():
            from fla.ops.backends.cute.softmax import softmax_bwd_cute

            return softmax_bwd_cute(p, dp, output_dtype)
    p = p.view(-1, p.shape[-1])
    ds = torch.empty_like(p, dtype=dtype)

    N, D = p.shape
    B = triton.next_power_of_2(D)
    softmax_bwd_kernel[(N,)](
        p=p,
        dp=dp,
        ds=ds,
        D=D,
        B=B,
    )
    return ds.view(*shape)
