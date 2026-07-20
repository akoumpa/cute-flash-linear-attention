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


@cute.jit
def _warp_max(value: Float32) -> Float32:
    for offset in (16, 8, 4, 2, 1):
        other = cute.arch.shuffle_sync_down(value, offset)
        if other > value:
            value = other
        if other != other:
            value = other
    return value


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for offset in (16, 8, 4, 2, 1):
        value += cute.arch.shuffle_sync_down(value, offset)
    return value


@cache
def _compile_logsumexp(input_dtype, output_dtype, D: int, *, has_scale: bool, use_fast_ops: bool):
    num_threads = max(32, 1 << (D - 1).bit_length())
    num_warps = num_threads // 32

    class LogSumExpForward:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mZ: cute.Tensor,
            N: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(mX, mZ, scale).launch(
                grid=[N, 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mZ: cute.Tensor,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            smem = cutlass.utils.SmemAllocator()
            sWarp = smem.allocate_tensor(Float32, cute.make_layout(num_warps), byte_alignment=16)

            value = Float32(float("-inf"))
            if tidx < D:
                value = Float32(mX[Int64(row) * D + tidx])
                if const_expr(has_scale):
                    value *= scale

            warp_max = _warp_max(value)
            if lane == 0:
                sWarp[warp] = warp_max
            cute.arch.sync_threads()

            block_max = Float32(float("-inf"))
            if warp == 0:
                if lane < num_warps:
                    block_max = sWarp[lane]
            block_max = _warp_max(block_max)
            if tidx == 0:
                sWarp[0] = block_max
            cute.arch.sync_threads()
            block_max = sWarp[0]

            value = Float32(0.0)
            if tidx < D:
                x = Float32(mX[Int64(row) * D + tidx])
                if const_expr(has_scale):
                    x *= scale
                value = cute.math.exp(x - block_max, fastmath=use_fast_ops)

            warp_sum = _warp_sum(value)
            if lane == 0:
                sWarp[warp] = warp_sum
            cute.arch.sync_threads()

            block_sum = Float32(0.0)
            if warp == 0:
                if lane < num_warps:
                    block_sum = sWarp[lane]
            block_sum = _warp_sum(block_sum)
            if tidx == 0:
                mZ[row] = output_dtype(cute.math.log(block_sum, fastmath=use_fast_ops) + block_max)

    x_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    z_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        LogSumExpForward(),
        x_fake,
        z_fake,
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def logsumexp_fwd_cute(
    x: torch.Tensor,
    scale: float | None = None,
    dtype: torch.dtype | None = None,
    *,
    use_fast_ops: bool = False,
) -> torch.Tensor:
    shape = x.shape
    D = shape[-1]
    N = x.numel() // D
    output_dtype = dtype or torch.float32
    z = x.new_empty(N, dtype=output_dtype)
    compiled = _compile_logsumexp(
        _torch_to_cute_dtype(x.dtype),
        _torch_to_cute_dtype(output_dtype),
        D,
        has_scale=scale is not None,
        use_fast_ops=use_fast_ops,
    )
    compiled(x.view(-1), z, N, scale if scale is not None else 1.0)
    return z.view(*shape[:-1])


__all__ = ["logsumexp_fwd_cute"]
