# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from fla.ops.backends.cute.pack import _torch_to_cute_dtype


@cache
def _compile_prepare_position_ids(index_dtype):
    class PreparePositionIds:
        @cute.jit
        def __call__(
            self,
            mCuSeqLens: cute.Tensor,
            mPositionIds: cute.Tensor,
            num_sequences: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mCuSeqLens, mPositionIds).launch(
                grid=[num_sequences, 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, mCuSeqLens: cute.Tensor, mPositionIds: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            sequence, _, _ = cute.arch.block_idx()
            bos = Int64(mCuSeqLens[sequence])
            eos = Int64(mCuSeqLens[sequence + 1])
            length = Int32(eos - bos)
            for position in range(tidx, length, 256):
                mPositionIds[bos + position] = position

    cu_fake = cute.runtime.make_fake_tensor(index_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    out_fake = cute.runtime.make_fake_tensor(index_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        PreparePositionIds(),
        cu_fake,
        out_fake,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def prepare_position_ids_cute(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Materialize zero-based positions for every packed sequence."""
    total_tokens = int(cu_seqlens[-1].item())
    position_ids = torch.empty(total_tokens, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    if total_tokens == 0:
        return position_ids
    compiled = _compile_prepare_position_ids(_torch_to_cute_dtype(cu_seqlens.dtype))
    compiled(cu_seqlens, position_ids, cu_seqlens.numel() - 1)
    return position_ids


__all__ = ["prepare_position_ids_cute"]
