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
from cutlass import Float32, Int32, Int64, const_expr

from fla.ops.backends.cute.pack import _torch_to_cute_dtype


@cache
def _compile_dense_scalar_global_cumsum(
    input_dtype,
    output_dtype,
    *,
    reverse: bool,
    head_first: bool,
    has_scale: bool,
):
    class DenseScalarGlobalCumsum:
        @cute.jit
        def __call__(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(mS, mO, B, T, H, scale).launch(
                grid=[B * H, 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            batch = block // H
            head = block - batch * H
            smem = cutlass.utils.SmemAllocator()
            sValues = smem.allocate_tensor(
                Float32,
                cute.make_layout(256),
                byte_alignment=16,
            )

            carry = Float32(0.0)
            num_tiles = cute.ceil_div(T, 256)
            for tile in cutlass.range(0, num_tiles, 1, unroll=1):
                logical_token = tile * 256 + tidx
                valid = logical_token < T
                token = T - 1 - logical_token if const_expr(reverse) else logical_token
                value = Float32(0.0)
                offset = Int64(0)
                if valid:
                    if const_expr(head_first):
                        offset = Int64(batch * H + head) * T + token
                    else:
                        offset = Int64(batch * T + token) * H + head
                    value = Float32(mS[offset])
                sValues[tidx] = value
                cute.arch.sync_threads()

                for scan_offset in (1, 2, 4, 8, 16, 32, 64, 128):
                    addend = Float32(0.0)
                    if tidx >= scan_offset:
                        addend = sValues[tidx - scan_offset]
                    cute.arch.sync_threads()
                    if tidx >= scan_offset:
                        sValues[tidx] = sValues[tidx] + addend
                    cute.arch.sync_threads()

                tile_total = sValues[255]
                if valid:
                    result = sValues[tidx] + carry
                    if const_expr(has_scale):
                        result *= scale
                    mO[offset] = result
                carry += tile_total
                cute.arch.sync_threads()

    s_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    o_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        DenseScalarGlobalCumsum(),
        s_fake,
        o_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_dense_vector_cumsum(
    input_dtype,
    output_dtype,
    *,
    local: bool,
    reverse: bool,
    head_first: bool,
    has_scale: bool,
    chunk_size: int,
):
    """Compile a feature-parallel dense cumsum.

    Each CUDA thread owns one ``(batch, head, feature)`` lane and retains the
    FP32 carry for its time scan in a register. This is the bandwidth-friendly
    path for the rank-4 tensors used by the recurrent operators.
    """

    class DenseVectorCumsum:
        @cute.jit
        def __call__(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            S: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            num_threads = 256
            num_chunks = cute.ceil_div(T, chunk_size) if const_expr(local) else 1
            work = B * H * S * num_chunks
            self.kernel(mS, mO, B, T, H, S, scale).launch(
                grid=[cute.ceil_div(work, num_threads), 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            S: Int32,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            linear = bid * 256 + tidx
            lanes = B * H * S
            num_chunks = cute.ceil_div(T, chunk_size) if const_expr(local) else 1
            if linear < lanes * num_chunks:
                if const_expr(local):
                    chunk = linear // lanes
                    lane = linear - chunk * lanes
                    begin = chunk * chunk_size
                    end = min(begin + chunk_size, T)
                else:
                    lane = linear
                    begin = Int32(0)
                    end = T

                feature = lane % S
                lane = lane // S
                head = lane % H
                batch = lane // H
                length = end - begin
                carry = Float32(0.0)
                for pos in cutlass.range(0, length, 1, unroll=1):
                    token = end - 1 - pos if const_expr(reverse) else begin + pos
                    if const_expr(head_first):
                        offset = (Int64(batch * H + head) * T + token) * S + feature
                    else:
                        offset = (Int64(batch) * T * H + token * H + head) * S + feature
                    carry += Float32(mS[offset])
                    value = carry * scale if const_expr(has_scale) else carry
                    mO[offset] = value

    s_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    o_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        DenseVectorCumsum(),
        s_fake,
        o_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_varlen_local_vector_cumsum(
    input_dtype,
    output_dtype,
    index_dtype,
    *,
    reverse: bool,
    head_first: bool,
    has_scale: bool,
    chunk_size: int,
):
    class VarlenLocalVectorCumsum:
        @cute.jit
        def __call__(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            mCuSeqlens: cute.Tensor,
            mChunkIndices: cute.Tensor,
            T: Int32,
            H: Int32,
            S: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            num_threads = 256
            work = mChunkIndices.shape[0] * H * S
            self.kernel(mS, mO, mCuSeqlens, mChunkIndices, T, H, S, scale).launch(
                grid=[cute.ceil_div(work, num_threads), 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            mCuSeqlens: cute.Tensor,
            mChunkIndices: cute.Tensor,
            T: Int32,
            H: Int32,
            S: Int32,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            linear = bid * 256 + tidx
            lanes = H * S
            if linear < mChunkIndices.shape[0] * lanes:
                chunk = linear // lanes
                lane = linear - chunk * lanes
                feature = lane % S
                head = lane // S

                sequence = Int32(mChunkIndices[(chunk, 0)])
                sequence_chunk = Int32(mChunkIndices[(chunk, 1)])
                bos = Int32(mCuSeqlens[sequence])
                eos = Int32(mCuSeqlens[sequence + 1])
                begin = bos + sequence_chunk * chunk_size
                end = min(begin + chunk_size, eos)
                length = end - begin
                carry = Float32(0.0)
                for pos in cutlass.range(0, length, 1, unroll=1):
                    token = end - 1 - pos if const_expr(reverse) else begin + pos
                    if const_expr(head_first):
                        offset = (Int64(head) * T + token) * S + feature
                    else:
                        offset = (Int64(token) * H + head) * S + feature
                    carry += Float32(mS[offset])
                    value = carry * scale if const_expr(has_scale) else carry
                    mO[offset] = value

    s_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    o_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    cu_fake = cute.runtime.make_fake_tensor(index_dtype, (cute.sym_int(),), stride=(1,), assumed_align=4)
    chunks_fake = cute.runtime.make_fake_tensor(
        index_dtype,
        (cute.sym_int(), 2),
        stride=(2, 1),
        assumed_align=4,
    )
    return cute.compile(
        VarlenLocalVectorCumsum(),
        s_fake,
        o_fake,
        cu_fake,
        chunks_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def dense_vector_cumsum_cute(
    s: torch.Tensor,
    *,
    local: bool,
    chunk_size: int = 0,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    out_dtype = output_dtype or s.dtype
    out = torch.empty_like(s, dtype=out_dtype)
    compiled = _compile_dense_vector_cumsum(
        _torch_to_cute_dtype(s.dtype),
        _torch_to_cute_dtype(out_dtype),
        local=local,
        reverse=reverse,
        head_first=head_first,
        has_scale=scale is not None,
        chunk_size=chunk_size,
    )
    compiled(s.view(-1), out.view(-1), B, T, H, S, 1.0 if scale is None else scale)
    return out


def dense_scalar_global_cumsum_cute(
    s: torch.Tensor,
    *,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    out_dtype = output_dtype or s.dtype
    out = torch.empty_like(s, dtype=out_dtype)
    compiled = _compile_dense_scalar_global_cumsum(
        _torch_to_cute_dtype(s.dtype),
        _torch_to_cute_dtype(out_dtype),
        reverse=reverse,
        head_first=head_first,
        has_scale=scale is not None,
    )
    compiled(s.view(-1), out.view(-1), B, T, H, 1.0 if scale is None else scale)
    return out


def varlen_local_vector_cumsum_cute(
    s: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    *,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        _, H, T, S = s.shape
    else:
        _, T, H, S = s.shape
    out_dtype = output_dtype or s.dtype
    out = torch.empty_like(s, dtype=out_dtype)
    compiled = _compile_varlen_local_vector_cumsum(
        _torch_to_cute_dtype(s.dtype),
        _torch_to_cute_dtype(out_dtype),
        _torch_to_cute_dtype(cu_seqlens.dtype),
        reverse=reverse,
        head_first=head_first,
        has_scale=scale is not None,
        chunk_size=chunk_size,
    )
    compiled(
        s.view(-1),
        out.view(-1),
        cu_seqlens,
        chunk_indices,
        T,
        H,
        S,
        1.0 if scale is None else scale,
    )
    return out


__all__ = [
    "dense_scalar_global_cumsum_cute",
    "dense_vector_cumsum_cute",
    "varlen_local_vector_cumsum_cute",
]
