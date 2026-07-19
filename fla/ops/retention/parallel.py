# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.simple_gla.parallel import parallel_simple_gla, parallel_simple_gla_bwd
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _can_use_cute_parallel_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    output_attentions: bool,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    if (
        output_attentions
        or cu_seqlens is not None
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.device != k.device
        or q.device != v.device
        or not isinstance(scale, float | int)
        or q.ndim != 4
        or q.shape != k.shape
        or q.shape[:3] != v.shape[:3]
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
    ):
        return False
    B, T, H, K = q.shape
    if B < 1 or H < 1 or B * H < 8 or T != 16 or K != 16 or v.shape[-1] != 16:
        return False
    from fla.ops.backends.cute.runtime import is_cute_dsl_available

    return is_cute_dsl_available()


class ParallelRetentionCuteFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, decay_log, scale):
        from fla.ops.backends.cute.parallel_retention import parallel_retention_fwd_cute

        o = parallel_retention_fwd_cute(q, k, v, decay_log, scale)
        B, T, H, _ = q.shape
        raw_gate = decay_log[None, None, :].expand(B, T, H)
        local_gate = chunk_local_cumsum(raw_gate, chunk_size=128, scale=RCP_LN2)
        ctx.save_for_backward(q, k, v, local_gate)
        ctx.scale = scale
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, local_gate = ctx.saved_tensors
        dq, dk, dv, _ = parallel_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=local_gate,
            do=do,
            scale=ctx.scale,
            chunk_size=128,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None


def parallel_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    output_attentions: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape `[B, H, T, T]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    s = (1 - q.new_tensor(2.0, dtype=torch.float).pow(-5.0 - q.new_tensor(range(q.shape[2]), dtype=torch.float))).log()
    if _can_use_cute_parallel_retention(q, k, v, scale, output_attentions, cu_seqlens):
        return ParallelRetentionCuteFunction.apply(q, k, v, s, scale), None
    g = s[None, None, :].expand(q.shape[0], q.shape[1], q.shape[2])

    o, attn = parallel_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        g=g,
        output_attentions=output_attentions,
        cu_seqlens=cu_seqlens,
    )
    return o, attn
