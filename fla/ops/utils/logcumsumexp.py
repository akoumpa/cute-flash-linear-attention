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

_CUTE_LOGCUMSUMEXP_MIN_T = 64
_CUTE_LOGCUMSUMEXP_MAX_T = 512
_CUTE_LOGCUMSUMEXP_MIN_S = 128
_CUTE_LOGCUMSUMEXP_MAX_S = 256
_CUTE_LOGCUMSUMEXP_MIN_LANES = 256
_CUTE_LOGCUMSUMEXP_MAX_LANES = 8192
_USE_FAST_OPS = os.environ.get("FLA_USE_FAST_OPS", "0") == "1"


def _can_use_cute_logcumsumexp(s: torch.Tensor) -> bool:
    if torch.compiler.is_compiling() or s.ndim < 2 or not s.is_cuda or not s.is_contiguous():
        return False
    if s.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    T, S = s.shape[-2:]
    if not (
        _CUTE_LOGCUMSUMEXP_MIN_T <= T <= _CUTE_LOGCUMSUMEXP_MAX_T and _CUTE_LOGCUMSUMEXP_MIN_S <= S <= _CUTE_LOGCUMSUMEXP_MAX_S
    ):
        return False
    lanes = s.numel() // T
    if not (_CUTE_LOGCUMSUMEXP_MIN_LANES <= lanes <= _CUTE_LOGCUMSUMEXP_MAX_LANES):
        return False
    from fla.ops.backends.cute.runtime import is_cute_dsl_available

    return is_cute_dsl_available()


@triton.autotune(
    configs=[triton.Config({"BT": BT}, num_warps=num_warps) for BT in [16, 32, 64] for num_warps in [2, 4, 8]],
    key=["S"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def logcumsumexp_fwd_kernel(
    s,
    z,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    b_mp = tl.full([S], float("-inf"), dtype=tl.float32)
    b_zp = tl.zeros([S], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * T * S, (T, S), (S, 1), (i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * T * S, (T, S), (S, 1), (i_t * BT, 0), (BT, S), (1, 0))

        # [BT, S]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        # [S,]
        b_mc = tl.max(b_s, 0)
        b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * exp(b_mp - b_mc)
        # [BT, S]
        b_s = exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        # [S,]
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        # [BT, BS]
        # small eps to prevent underflows
        b_z = log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))


def logcumsumexp_fwd(s: torch.Tensor) -> torch.Tensor:
    """Compute a cumulative logsumexp over the second-to-last dimension."""
    if _can_use_cute_logcumsumexp(s):
        from fla.ops.backends.cute.logcumsumexp import logcumsumexp_fwd_cute

        return logcumsumexp_fwd_cute(s, use_fast_ops=_USE_FAST_OPS)

    shape = s.shape
    T, S = shape[-2:]
    N = s.numel() // (T * S)
    z = torch.empty_like(s, dtype=torch.float32)
    logcumsumexp_fwd_kernel[(N,)](
        s,
        z,
        T=T,
        S=S,
    )
    return z
