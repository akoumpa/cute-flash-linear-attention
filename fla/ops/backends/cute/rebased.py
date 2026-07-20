# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_TILE = 16
_THREADS = 32


@cache
def _compile_rebased_forward(dtype):
    class RebasedForward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mZ: cute.Tensor,
            B: Int32,
            H: Int32,
            T: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            layout = cute.make_layout((_TILE, _TILE), stride=(_TILE, 1))
            mma = cute.make_tiled_mma(warp.MmaF16BF16Op(dtype, Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16))
            self.kernel(mQ, mK, mV, mO, mZ, T, scale, layout, mma).launch(
                grid=[B * H, 1, 1], block=[_THREADS, 1, 1], smem=6 * _TILE * _TILE * 2, stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mZ: cute.Tensor,
            T: Int32,
            scale: Float32,
            layout: cute.Layout,
            mma: cute.TiledMma,
        ):
            lane, _, _ = cute.arch.thread_idx()
            bh, _, _ = cute.arch.block_idx()
            allocator = cutlass.utils.SmemAllocator()
            sQ = allocator.allocate_tensor(dtype, layout, byte_alignment=16)
            sK = allocator.allocate_tensor(dtype, layout, byte_alignment=16)
            sVt = allocator.allocate_tensor(dtype, layout, byte_alignment=16)
            sP = allocator.allocate_tensor(dtype, layout, byte_alignment=16)
            sPfp32 = allocator.allocate_tensor(Float32, layout, byte_alignment=16)
            base = Int64(bh) * T * _TILE
            for element in cutlass.range(lane, _TILE * _TILE, _THREADS, unroll=1):
                row = element // _TILE
                col = element % _TILE
                sQ[row, col] = dtype(Float32(mQ[base + element]) * scale) if row < T else dtype(0.0)
                sK[row, col] = mK[base + element] if row < T else dtype(0.0)
                sVt[col, row] = mV[base + element] if row < T else dtype(0.0)
            cute.arch.sync_warp()
            thr = mma.get_slice(lane)
            rQ = thr.make_fragment_A(thr.partition_A(sQ))
            rK = thr.make_fragment_B(thr.partition_B(sK))
            cute.autovec_copy(thr.partition_A(sQ), rQ)
            cute.autovec_copy(thr.partition_B(sK), rK)
            scores = cute.make_rmem_tensor(thr.partition_shape_C((_TILE, _TILE)), Float32)
            scores.fill(0.0)
            cute.gemm(mma, scores, rQ, rK, scores)
            cute.autovec_copy(scores, thr.partition_C(sPfp32))
            cute.arch.sync_warp()
            for element in cutlass.range(lane, _TILE * _TILE, _THREADS, unroll=1):
                row = element // _TILE
                col = element % _TILE
                score = sPfp32[row, col]
                score_squared = score * score if row < T and col <= row else Float32(0.0)
                sPfp32[row, col] = score_squared
                sP[row, col] = dtype(score_squared)
            cute.arch.sync_warp()
            rP = thr.make_fragment_A(thr.partition_A(sP))
            rV = thr.make_fragment_B(thr.partition_B(sVt))
            cute.autovec_copy(thr.partition_A(sP), rP)
            cute.autovec_copy(thr.partition_B(sVt), rV)
            output = cute.make_rmem_tensor(thr.partition_shape_C((_TILE, _TILE)), Float32)
            output.fill(0.0)
            cute.gemm(mma, output, rP, rV, output)
            output_low = cute.make_fragment_like(output, dtype)
            output_low.store(output.load().to(dtype))
            cute.autovec_copy(output_low, thr.partition_C(sQ))
            cute.arch.sync_warp()
            for row in cutlass.range(lane, T, _THREADS, unroll=1):
                total = Float32(0.0)
                for col in cutlass.range_constexpr(_TILE):
                    total += sPfp32[row, col]
                mZ[Int64(bh) * T + row] = dtype(total)
            for element in cutlass.range(lane, T * _TILE, _THREADS, unroll=1):
                mO[base + element] = sQ[element // _TILE, element % _TILE]

    fake = cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    z_fake = cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        RebasedForward(),
        fake,
        fake,
        fake,
        fake,
        z_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rebased_forward_cute(q, k, v, scale):
    B, H, T, _ = q.shape
    output, denominator = torch.empty_like(v), q.new_empty(B, H, T)
    compiled = _compile_rebased_forward(_torch_to_cute_dtype(q.dtype))
    compiled(q.view(-1), k.view(-1), v.view(-1), output.view(-1), denominator.view(-1), B, H, T, scale)
    return output, denominator


__all__ = ["rebased_forward_cute"]
