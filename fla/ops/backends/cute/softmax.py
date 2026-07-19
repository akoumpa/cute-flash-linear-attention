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

_NUM_THREADS = 256
_LOG2E = 1.4426950408889634


@cute.jit
def _reduce_max(values: cute.Tensor, tidx: Int32) -> Float32:
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        if tidx < stride:
            other = values[tidx + stride]
            if other > values[tidx]:
                values[tidx] = other
        cute.arch.sync_threads()
    return values[0]


@cute.jit
def _reduce_sum(values: cute.Tensor, tidx: Int32) -> Float32:
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        if tidx < stride:
            values[tidx] = values[tidx] + values[tidx + stride]
        cute.arch.sync_threads()
    return values[0]


@cache
def _compile_softmax_fwd(input_dtype, output_dtype):
    class SoftmaxForward:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mP: cute.Tensor,
            N: Int32,
            D: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mX, mP, D).launch(grid=[N, 1, 1], block=[_NUM_THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mP: cute.Tensor, D: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            smem = cutlass.utils.SmemAllocator()
            reduction = smem.allocate_tensor(Float32, cute.make_layout(_NUM_THREADS), byte_alignment=16)
            row_base = Int64(row) * D

            local_max = Float32(-float("inf"))
            for feature in cutlass.range(tidx, D, _NUM_THREADS, unroll=1):
                value = Float32(mX[row_base + feature])
                if value > local_max:
                    local_max = value
            reduction[tidx] = local_max
            cute.arch.sync_threads()
            row_max = _reduce_max(reduction, tidx)

            local_sum = Float32(0.0)
            for feature in cutlass.range(tidx, D, _NUM_THREADS, unroll=1):
                value = Float32(mX[row_base + feature])
                local_sum += cute.math.exp2((value - row_max) * Float32(_LOG2E), fastmath=True)
            reduction[tidx] = local_sum
            cute.arch.sync_threads()
            denominator = _reduce_sum(reduction, tidx)

            for feature in cutlass.range(tidx, D, _NUM_THREADS, unroll=1):
                value = Float32(mX[row_base + feature])
                probability = cute.math.exp2((value - row_max) * Float32(_LOG2E), fastmath=True) / denominator
                mP[row_base + feature] = output_dtype(probability)

    x_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    p_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        SoftmaxForward(),
        x_fake,
        p_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_softmax_bwd(prob_dtype, grad_dtype, output_dtype):
    class SoftmaxBackward:
        @cute.jit
        def __call__(
            self,
            mP: cute.Tensor,
            mDP: cute.Tensor,
            mDS: cute.Tensor,
            N: Int32,
            D: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mP, mDP, mDS, D).launch(grid=[N, 1, 1], block=[_NUM_THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(self, mP: cute.Tensor, mDP: cute.Tensor, mDS: cute.Tensor, D: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            smem = cutlass.utils.SmemAllocator()
            reduction = smem.allocate_tensor(Float32, cute.make_layout(_NUM_THREADS), byte_alignment=16)
            row_base = Int64(row) * D

            local_dot = Float32(0.0)
            for feature in cutlass.range(tidx, D, _NUM_THREADS, unroll=1):
                local_dot += Float32(mP[row_base + feature]) * Float32(mDP[row_base + feature])
            reduction[tidx] = local_dot
            cute.arch.sync_threads()
            dot = _reduce_sum(reduction, tidx)

            for feature in cutlass.range(tidx, D, _NUM_THREADS, unroll=1):
                probability = Float32(mP[row_base + feature])
                grad = probability * (Float32(mDP[row_base + feature]) - dot)
                mDS[row_base + feature] = output_dtype(grad)

    p_fake = cute.runtime.make_fake_tensor(prob_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    dp_fake = cute.runtime.make_fake_tensor(grad_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    ds_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        SoftmaxBackward(),
        p_fake,
        dp_fake,
        ds_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def softmax_fwd_cute(x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    shape = x.shape
    D = shape[-1]
    N = x.numel() // D
    out = torch.empty_like(x, dtype=output_dtype)
    compiled = _compile_softmax_fwd(_torch_to_cute_dtype(x.dtype), _torch_to_cute_dtype(output_dtype))
    compiled(x.view(-1), out.view(-1), N, D)
    return out.view(*shape)


def softmax_bwd_cute(
    p: torch.Tensor,
    dp: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    shape = p.shape
    D = shape[-1]
    N = p.numel() // D
    out = torch.empty_like(p, dtype=output_dtype)
    compiled = _compile_softmax_bwd(
        _torch_to_cute_dtype(p.dtype),
        _torch_to_cute_dtype(dp.dtype),
        _torch_to_cute_dtype(output_dtype),
    )
    compiled(p.view(-1), dp.view(-1), out.view(-1), N, D)
    return out.view(*shape)


__all__ = ["softmax_bwd_cute", "softmax_fwd_cute"]
