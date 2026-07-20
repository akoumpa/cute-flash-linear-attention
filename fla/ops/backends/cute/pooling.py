# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_NUM_THREADS = 128


@cache
def _compile_dense_mean_pooling_fwd(input_dtype, chunk_size: int):
    class DenseMeanPoolingForward:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            D: Int32,
            stream: cuda.CUstream,
        ):
            num_chunks = cute.ceil_div(T, chunk_size)
            num_d_tiles = cute.ceil_div(D, _NUM_THREADS)
            self.kernel(mX, mO, T, H, D, num_chunks, num_d_tiles).launch(
                grid=[B * num_chunks * H * num_d_tiles, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mO: cute.Tensor,
            T: Int32,
            H: Int32,
            D: Int32,
            num_chunks: Int32,
            num_d_tiles: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            d_tile = block % num_d_tiles
            work = block // num_d_tiles
            head = work % H
            work = work // H
            chunk = work % num_chunks
            batch = work // num_chunks
            feature = d_tile * _NUM_THREADS + tidx
            if feature < D:
                start = chunk * chunk_size
                remaining = T - start
                valid = chunk_size if remaining >= chunk_size else remaining
                total = Float32(0.0)
                for offset in cutlass.range(0, valid, 1, unroll=1):
                    token = start + offset
                    x_offset = ((Int64(batch) * T + token) * H + head) * D + feature
                    total += Float32(mX[x_offset])
                out_offset = ((Int64(batch) * num_chunks + chunk) * H + head) * D + feature
                mO[out_offset] = input_dtype(total / Float32(valid))

    x_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    o_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        DenseMeanPoolingForward(),
        x_fake,
        o_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_dense_mean_pooling_bwd(grad_dtype, chunk_size: int):
    class DenseMeanPoolingBackward:
        @cute.jit
        def __call__(
            self,
            mDO: cute.Tensor,
            mDX: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            D: Int32,
            stream: cuda.CUstream,
        ):
            num_chunks = cute.ceil_div(T, chunk_size)
            num_d_tiles = cute.ceil_div(D, _NUM_THREADS)
            self.kernel(mDO, mDX, T, H, D, num_chunks, num_d_tiles).launch(
                grid=[B * num_chunks * H * num_d_tiles, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDO: cute.Tensor,
            mDX: cute.Tensor,
            T: Int32,
            H: Int32,
            D: Int32,
            num_chunks: Int32,
            num_d_tiles: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            d_tile = block % num_d_tiles
            work = block // num_d_tiles
            head = work % H
            work = work // H
            chunk = work % num_chunks
            batch = work // num_chunks
            feature = d_tile * _NUM_THREADS + tidx
            if feature < D:
                start = chunk * chunk_size
                remaining = T - start
                valid = chunk_size if remaining >= chunk_size else remaining
                out_offset = ((Int64(batch) * num_chunks + chunk) * H + head) * D + feature
                grad = Float32(mDO[out_offset]) / Float32(valid)
                for offset in cutlass.range(0, valid, 1, unroll=1):
                    token = start + offset
                    dx_offset = ((Int64(batch) * T + token) * H + head) * D + feature
                    mDX[dx_offset] = grad_dtype(grad)

    do_fake = cute.runtime.make_fake_tensor(grad_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    dx_fake = cute.runtime.make_fake_tensor(grad_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        DenseMeanPoolingBackward(),
        do_fake,
        dx_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def dense_mean_pooling_fwd_cute(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H, D = x.shape
    num_chunks = (T + chunk_size - 1) // chunk_size
    out = torch.empty(B, num_chunks, H, D, dtype=x.dtype, device=x.device)
    compiled = _compile_dense_mean_pooling_fwd(_torch_to_cute_dtype(x.dtype), chunk_size)
    compiled(x.view(-1), out.view(-1), B, T, H, D)
    return out


def dense_mean_pooling_bwd_cute(
    do: torch.Tensor,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    H, D = do.shape[-2:]
    dx = torch.empty(batch_size, seq_len, H, D, dtype=do.dtype, device=do.device)
    compiled = _compile_dense_mean_pooling_bwd(_torch_to_cute_dtype(do.dtype), chunk_size)
    compiled(do.view(-1), dx.view(-1), batch_size, seq_len, H, D)
    return dx


__all__ = ["dense_mean_pooling_bwd_cute", "dense_mean_pooling_fwd_cute"]
