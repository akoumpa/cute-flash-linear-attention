# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.attn.parallel import parallel_attn


def _can_use_cute_forgetting_attn(q, k, v, g, window_size, cu_seqlens):
    if (
        torch.is_grad_enabled()
        or window_size is not None
        or cu_seqlens is not None
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
    ):
        return False
    B, T, HQ, K = q.shape
    H, V = k.shape[2], v.shape[-1]
    if (
        B < 1
        or not 1 <= T <= 16
        or H < 1
        or HQ < H
        or HQ % H != 0
        or B * HQ < 4
        or K != 64
        or V != 64
        or k.shape != (B, T, H, K)
        or v.shape != (B, T, H, V)
        or g.shape != (B, T, HQ)
        or g.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or q.device != k.device
        or q.device != v.device
        or q.device != g.device
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or not g.is_contiguous()
    ):
        return False
    from fla.ops.backends.cute.runtime import is_cute_dsl_available

    return is_cute_dsl_available()


def parallel_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            log decay factors of shape `[B, T, HQ]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        window_size (Optional[int]):
            Sliding window size. If provided, each query at position i only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )

    if _can_use_cute_forgetting_attn(q, k, v, g, window_size, cu_seqlens):
        from fla.ops.backends.cute.forgetting_attn import forgetting_attn_fwd_cute

        return forgetting_attn_fwd_cute(q, k, v, g, scale)
    o = parallel_attn(q, k, v, g, scale, window_size=window_size, cu_seqlens=cu_seqlens)
    return o
