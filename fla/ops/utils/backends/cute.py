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
    def _verify_vector(s: torch.Tensor, cu_seqlens, output_dtype) -> tuple[bool, str | None]:
        if cu_seqlens is not None:
            return False, "the first CuTe cumsum slice supports dense tensors only"
        if s.ndim != 4:
            return False, "the first CuTe cumsum slice supports vector tensors only"
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
        can_use, reason = self._verify_vector(s, cu_seqlens, output_dtype)
        if not can_use:
            return can_use, reason
        time_dim = 2 if head_first else 1
        if s.shape[time_dim] > 64:
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
        from fla.ops.backends.cute.cumsum import dense_vector_cumsum_cute

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
        return self._verify_vector(g, cu_seqlens, output_dtype)

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
        from fla.ops.backends.cute.cumsum import dense_vector_cumsum_cute

        return dense_vector_cumsum_cute(
            g,
            local=True,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )


__all__ = ["CuteUtilsBackend"]
