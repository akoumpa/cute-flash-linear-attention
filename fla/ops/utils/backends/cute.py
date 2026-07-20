# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend
from fla.ops.backends.cute.runtime import is_cute_dsl_available


class CuteUtilsBackend(BaseBackend):
    backend_type = "cute"
    priority = 1

    @classmethod
    def is_available(cls) -> bool:
        return is_cute_dsl_available()

    @staticmethod
    def _verify_tensor(s: torch.Tensor, output_dtype) -> tuple[bool, str | None]:
        if s.ndim not in (3, 4):
            return False, "CuTe cumsum supports rank-3 scalar or rank-4 vector tensors"
        if not s.is_cuda or not s.is_contiguous():
            return False, "CuTe cumsum requires a contiguous CUDA tensor"
        if s.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"unsupported input dtype {s.dtype}"
        out_dtype = output_dtype or s.dtype
        if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"unsupported output dtype {out_dtype}"
        return True, None

    def chunk_global_cumsum_verifier(
        self,
        s,
        reverse=False,
        cu_seqlens=None,
        scale=None,
        head_first=False,
        output_dtype=torch.float,
    ):
        if cu_seqlens is not None:
            return False, "the CuTe global cumsum path does not yet support ragged tensors"
        can_use, reason = self._verify_tensor(s, output_dtype)
        if not can_use:
            return can_use, reason
        time_dim = 2 if head_first else 1
        if s.ndim == 4 and s.shape[time_dim] > 64:
            return False, "the serial CuTe global scan is benchmark-gated to sequence lengths <= 64"
        return True, None

    def chunk_global_cumsum(
        self,
        s,
        reverse=False,
        cu_seqlens=None,
        scale=None,
        head_first=False,
        output_dtype=torch.float,
    ):
        from fla.ops.backends.cute.cumsum import dense_scalar_global_cumsum_cute, dense_vector_cumsum_cute

        if s.ndim == 3:
            return dense_scalar_global_cumsum_cute(
                s,
                reverse=reverse,
                scale=scale,
                head_first=head_first,
                output_dtype=output_dtype,
            )

        return dense_vector_cumsum_cute(
            s,
            local=False,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )

    def chunk_local_cumsum_verifier(
        self,
        g,
        chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=False,
        output_dtype=torch.float,
        chunk_indices=None,
        **kwargs,
    ):
        can_use, reason = self._verify_tensor(g, output_dtype)
        if not can_use:
            return can_use, reason
        if cu_seqlens is not None and cu_seqlens.dtype not in (torch.int32, torch.int64):
            return False, f"unsupported cu_seqlens dtype {cu_seqlens.dtype}"
        if chunk_indices is not None and chunk_indices.dtype not in (torch.int32, torch.int64):
            return False, f"unsupported chunk_indices dtype {chunk_indices.dtype}"
        if cu_seqlens is not None and (not cu_seqlens.is_cuda or not cu_seqlens.is_contiguous()):
            return False, "CuTe cumsum requires contiguous CUDA cu_seqlens"
        if chunk_indices is not None:
            if not chunk_indices.is_cuda or not chunk_indices.is_contiguous():
                return False, "CuTe cumsum requires contiguous CUDA chunk_indices"
            if cu_seqlens is not None and chunk_indices.dtype != cu_seqlens.dtype:
                return False, "cu_seqlens and chunk_indices must have the same dtype"
        return True, None

    def chunk_local_cumsum(
        self,
        g,
        chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=False,
        output_dtype=torch.float,
        chunk_indices=None,
        **kwargs,
    ):
        from fla.ops.backends.cute.cumsum import dense_vector_cumsum_cute, varlen_local_vector_cumsum_cute

        if cu_seqlens is not None:
            if chunk_indices is None:
                from fla.ops.utils.index import prepare_chunk_indices

                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            vector = g.unsqueeze(-1) if g.ndim == 3 else g
            out = varlen_local_vector_cumsum_cute(
                vector,
                cu_seqlens,
                chunk_indices,
                chunk_size=chunk_size,
                reverse=reverse,
                scale=scale,
                head_first=head_first,
                output_dtype=output_dtype,
            )
            return out.squeeze(-1) if g.ndim == 3 else out

        vector = g.unsqueeze(-1) if g.ndim == 3 else g
        out = dense_vector_cumsum_cute(
            vector,
            local=True,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
        return out.squeeze(-1) if g.ndim == 3 else out


__all__ = ["CuteUtilsBackend"]
