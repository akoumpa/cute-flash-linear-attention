# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr


@cache
def _compile_prepare_dense_block_csr(*, use_block_counts: bool):
    class PrepareDenseBlockCsr:
        @cute.jit
        def __call__(
            self,
            mBlockIndices: cute.Tensor,
            mBlockCounts: cute.Tensor | None,
            mCsrIndices: cute.Tensor,
            mCsrOffsets: cute.Tensor,
            block_count: Int32,
            B: Int32,
            T: Int32,
            H: Int32,
            S: Int32,
            num_blocks: Int32,
            block_size: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mBlockIndices,
                mBlockCounts,
                mCsrIndices,
                mCsrOffsets,
                block_count,
                B,
                T,
                H,
                S,
                num_blocks,
                block_size,
            ).launch(
                grid=[1, 1, 1],
                block=[32, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mBlockIndices: cute.Tensor,
            mBlockCounts: cute.Tensor | None,
            mCsrIndices: cute.Tensor,
            mCsrOffsets: cute.Tensor,
            block_count: Int32,
            B: Int32,
            T: Int32,
            H: Int32,
            S: Int32,
            num_blocks: Int32,
            block_size: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                cursor = Int32(0)
                row = Int32(0)
                for batch in cutlass.range(B, unroll=1):
                    for head in cutlass.range(H, unroll=1):
                        for block in cutlass.range(num_blocks, unroll=1):
                            mCsrOffsets[row] = cursor
                            for token in cutlass.range(T, unroll=1):
                                query_head = (Int64(batch) * T + token) * H + head
                                count = Int64(block_count)
                                if const_expr(use_block_counts):
                                    count = Int64(mBlockCounts[query_head])
                                for slot in cutlass.range(S, unroll=1):
                                    selected = Int64(mBlockIndices[query_head * S + slot])
                                    if slot < count and selected == Int64(block) and selected * block_size <= token:
                                        mCsrIndices[cursor] = Int32(batch * T + token)
                                        cursor += 1
                            row += 1
                mCsrOffsets[row] = cursor

    indices_fake = cute.runtime.make_fake_tensor(cutlass.Int64, (cute.sym_int(),), stride=(1,), assumed_align=16)
    counts_fake = (
        cute.runtime.make_fake_tensor(cutlass.Int64, (cute.sym_int(),), stride=(1,), assumed_align=16)
        if use_block_counts
        else None
    )
    csr_indices_fake = cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    csr_offsets_fake = cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        PrepareDenseBlockCsr(),
        indices_fake,
        counts_fake,
        csr_indices_fake,
        csr_offsets_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def prepare_dense_block_csr_cute(
    block_indices: torch.Tensor,
    block_counts: torch.Tensor | int,
    num_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense block-selection CSR with a bounded single-pass kernel."""
    B, T, H, S = block_indices.shape
    csr_indices = block_indices.new_empty(B * T * H * S, dtype=torch.int32)
    csr_offsets = block_indices.new_empty(B * H * num_blocks + 1, dtype=torch.int32)
    use_block_counts = isinstance(block_counts, torch.Tensor)
    compiled = _compile_prepare_dense_block_csr(use_block_counts=use_block_counts)
    compiled(
        block_indices.view(-1),
        block_counts.view(-1) if use_block_counts else None,
        csr_indices,
        csr_offsets,
        0 if use_block_counts else block_counts,
        B,
        T,
        H,
        S,
        num_blocks,
        block_size,
    )
    return csr_indices, csr_offsets


__all__ = ["prepare_dense_block_csr_cute"]
