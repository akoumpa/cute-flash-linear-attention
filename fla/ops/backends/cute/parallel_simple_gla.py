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
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.op import exp2
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_TILE = 16
_NUM_THREADS = 32


@cache
def _compile_parallel_simple_gla_fwd(input_dtype, H: int, *, use_gate: bool):
    class ParallelSimpleGLAForward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor | None,
            mO: cute.Tensor,
            batches: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            layout = cute.make_layout((_TILE, _TILE), stride=(_TILE, 1))
            mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(input_dtype, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            self.kernel(mQ, mK, mV, mG, mO, scale, layout, mma).launch(
                grid=[batches * H, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=6 * _TILE * _TILE * 2,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor | None,
            mO: cute.Tensor,
            scale: Float32,
            layout: cute.Layout,
            mma: cute.TiledMma,
        ):
            lane, _, _ = cute.arch.thread_idx()
            batch_head, _, _ = cute.arch.block_idx()
            batch = batch_head // H
            head = batch_head % H
            gQ = mQ[batch, None, head, None]
            gK = mK[batch, None, head, None]
            gV = mV[batch, None, head, None]
            gO = mO[batch, None, head, None]
            gG = mG[batch, None, head] if const_expr(use_gate) else None

            allocator = cutlass.utils.SmemAllocator()
            sQ = allocator.allocate_tensor(input_dtype, layout, byte_alignment=16)
            sK = allocator.allocate_tensor(input_dtype, layout, byte_alignment=16)
            sVt = allocator.allocate_tensor(input_dtype, layout, byte_alignment=16)
            sP = allocator.allocate_tensor(input_dtype, layout, byte_alignment=16)
            sPfp32 = allocator.allocate_tensor(Float32, layout, byte_alignment=16)

            for element in cutlass.range(lane, _TILE * _TILE, _NUM_THREADS, unroll=1):
                row = element // _TILE
                column = element % _TILE
                sQ[row, column] = input_dtype(Float32(gQ[row, column]) * scale)
                sK[row, column] = gK[row, column]
                sVt[column, row] = gV[row, column]
            cute.arch.sync_warp()

            thread_mma = mma.get_slice(lane)
            rQ = thread_mma.make_fragment_A(thread_mma.partition_A(sQ))
            rK = thread_mma.make_fragment_B(thread_mma.partition_B(sK))
            cute.autovec_copy(thread_mma.partition_A(sQ), rQ)
            cute.autovec_copy(thread_mma.partition_B(sK), rK)
            scores = cute.make_rmem_tensor(thread_mma.partition_shape_C((_TILE, _TILE)), Float32)
            scores.fill(0.0)
            cute.gemm(mma, scores, rQ, rK, scores)
            cute.autovec_copy(scores, thread_mma.partition_C(sPfp32))
            cute.arch.sync_warp()

            for element in cutlass.range(lane, _TILE * _TILE, _NUM_THREADS, unroll=1):
                row = element // _TILE
                column = element % _TILE
                score = Float32(0.0)
                if column <= row:
                    score = sPfp32[row, column]
                    if const_expr(use_gate):
                        score *= exp2(Float32(gG[row]) - Float32(gG[column]))
                sP[row, column] = input_dtype(score)
            cute.arch.sync_warp()

            rP = thread_mma.make_fragment_A(thread_mma.partition_A(sP))
            rV = thread_mma.make_fragment_B(thread_mma.partition_B(sVt))
            cute.autovec_copy(thread_mma.partition_A(sP), rP)
            cute.autovec_copy(thread_mma.partition_B(sVt), rV)
            output = cute.make_rmem_tensor(thread_mma.partition_shape_C((_TILE, _TILE)), Float32)
            output.fill(0.0)
            cute.gemm(mma, output, rP, rV, output)
            output_low = cute.make_fragment_like(output, input_dtype)
            output_low.store(output.load().to(input_dtype))
            cute.autovec_copy(output_low, thread_mma.partition_C(sQ))
            cute.arch.sync_warp()

            for element in cutlass.range(lane, _TILE * _TILE, _NUM_THREADS, unroll=1):
                row = element // _TILE
                column = element % _TILE
                gO[row, column] = sQ[row, column]

    tensor_fake = cute.runtime.make_fake_tensor(
        input_dtype,
        (cute.sym_int(), _TILE, H, _TILE),
        stride=(_TILE * H * _TILE, H * _TILE, _TILE, 1),
        assumed_align=16,
    )
    gate_fake = (
        cute.runtime.make_fake_tensor(
            Float32,
            (cute.sym_int(), _TILE, H),
            stride=(_TILE * H, H, 1),
            assumed_align=16,
        )
        if use_gate
        else None
    )
    return cute.compile(
        ParallelSimpleGLAForward(),
        tensor_fake,
        tensor_fake,
        tensor_fake,
        gate_fake,
        tensor_fake,
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def parallel_simple_gla_fwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Run the fixed 16-token, 16-wide native-MMA simple-GLA forward pass."""
    batches, _, heads, _ = q.shape
    output = torch.empty_like(v)
    compiled = _compile_parallel_simple_gla_fwd(_torch_to_cute_dtype(q.dtype), heads, use_gate=g is not None)
    compiled(q, k, v, g, output, batches, scale)
    return output


__all__ = ["parallel_simple_gla_fwd_cute"]
