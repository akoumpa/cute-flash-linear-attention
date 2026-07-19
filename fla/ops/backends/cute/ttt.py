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
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.chunk_o import _gemm, _get_smem_layout_atom, _load_mma_a, _load_mma_b
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_CHUNK_SIZE = 16
_HEAD_DIM = 64
_NUM_THREADS = 32


@cache
def _compile_ttt_linear_fwd_o(input_dtype, T: int, H: int):
    NT = (T + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    class TTTLinearForwardOutput:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mEta: cute.Tensor,
            mH: cute.Tensor,
            mHB: cute.Tensor,
            mO: cute.Tensor,
            batches: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(input_dtype, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                input_dtype,
            )
            smem_tiled_copy_a = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
            smem_tiled_copy_b = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
            qk_layout_atom = _get_smem_layout_atom(input_dtype, _HEAD_DIM)
            v_layout_atom = _get_smem_layout_atom(input_dtype, _CHUNK_SIZE)
            sQ_layout = cute.tile_to_shape(qk_layout_atom, (_CHUNK_SIZE, _HEAD_DIM), (0, 1))
            sK_layout = cute.tile_to_shape(qk_layout_atom, (_CHUNK_SIZE, _HEAD_DIM), (0, 1))
            sH_layout = cute.tile_to_shape(qk_layout_atom, (_HEAD_DIM, _HEAD_DIM), (0, 1))
            sV_layout = cute.tile_to_shape(v_layout_atom, (_HEAD_DIM, _CHUNK_SIZE), (0, 1))
            matrix_layout = cute.make_layout((_CHUNK_SIZE, _CHUNK_SIZE), stride=(_CHUNK_SIZE, 1))
            output_layout = cute.make_layout((_CHUNK_SIZE, _HEAD_DIM), stride=(_HEAD_DIM, 1))
            smem_bytes = (
                2 * _CHUNK_SIZE * _HEAD_DIM + _HEAD_DIM * _HEAD_DIM + _HEAD_DIM * _CHUNK_SIZE + _CHUNK_SIZE * _CHUNK_SIZE
            ) * (input_dtype.width // 8) + _CHUNK_SIZE * _HEAD_DIM * 4
            self.kernel(
                mQ,
                mK,
                mV,
                mEta,
                mH,
                mHB,
                mO,
                scale,
                tiled_mma,
                smem_tiled_copy_a,
                smem_tiled_copy_b,
                sQ_layout,
                sK_layout,
                sH_layout,
                sV_layout,
                matrix_layout,
                output_layout,
            ).launch(
                grid=[batches * H * NT, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mEta: cute.Tensor,
            mH: cute.Tensor,
            mHB: cute.Tensor,
            mO: cute.Tensor,
            scale: Float32,
            tiled_mma: cute.TiledMma,
            smem_tiled_copy_a: cute.TiledCopy,
            smem_tiled_copy_b: cute.TiledCopy,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sH_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            matrix_layout: cute.Layout,
            output_layout: cute.Layout,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            chunk = block % NT
            batch_head = block // NT
            head = batch_head % H
            batch = batch_head // H

            allocator = cutlass.utils.SmemAllocator()
            sQ = allocator.allocate_tensor(input_dtype, sQ_layout, byte_alignment=128)
            sK = allocator.allocate_tensor(input_dtype, sK_layout, byte_alignment=128)
            sH = allocator.allocate_tensor(input_dtype, sH_layout, byte_alignment=128)
            sV = allocator.allocate_tensor(input_dtype, sV_layout, byte_alignment=128)
            sP = allocator.allocate_tensor(input_dtype, matrix_layout, byte_alignment=16)
            sO = allocator.allocate_tensor(Float32, output_layout, byte_alignment=16)

            linear = tidx
            for _ in cutlass.range_constexpr((_CHUNK_SIZE * _HEAD_DIM) // _NUM_THREADS):
                row = linear // _HEAD_DIM
                column = linear % _HEAD_DIM
                token = chunk * _CHUNK_SIZE + row
                value = input_dtype(0.0)
                key = input_dtype(0.0)
                if token < T:
                    token_base = (Int64(batch) * T + token) * H * _HEAD_DIM + Int64(head) * _HEAD_DIM
                    value = mQ[token_base + column]
                    key = mK[token_base + column]
                sQ[row, column] = value
                sK[row, column] = key
                linear += _NUM_THREADS

            state = (Int64(batch) * NT + chunk) * H + head
            state_base = state * _HEAD_DIM * _HEAD_DIM
            linear = tidx
            for _ in cutlass.range_constexpr((_HEAD_DIM * _HEAD_DIM) // _NUM_THREADS):
                value = linear // _HEAD_DIM
                key = linear % _HEAD_DIM
                sH[value, key] = mH[state_base + key * _HEAD_DIM + value]
                linear += _NUM_THREADS

            linear = tidx
            for _ in cutlass.range_constexpr((_HEAD_DIM * _CHUNK_SIZE) // _NUM_THREADS):
                value = linear // _CHUNK_SIZE
                row = linear % _CHUNK_SIZE
                token = chunk * _CHUNK_SIZE + row
                element = input_dtype(0.0)
                if token < T:
                    token_base = (Int64(batch) * T + token) * H * _HEAD_DIM + Int64(head) * _HEAD_DIM
                    element = mV[token_base + value]
                sV[value, row] = element
                linear += _NUM_THREADS
            cute.arch.sync_threads()

            thread_mma = tiled_mma.get_slice(tidx)
            rQ = _load_mma_a(thread_mma, smem_tiled_copy_a, tidx, sQ)
            rH = _load_mma_b(thread_mma, smem_tiled_copy_b, tidx, sH)
            output = cute.make_rmem_tensor(thread_mma.partition_shape_C((_CHUNK_SIZE, _HEAD_DIM)), Float32)
            output.fill(0.0)
            _gemm(tiled_mma, output, rQ, rH)

            rK = _load_mma_b(thread_mma, smem_tiled_copy_b, tidx, sK)
            scores = cute.make_rmem_tensor(thread_mma.partition_shape_C((_CHUNK_SIZE, _CHUNK_SIZE)), Float32)
            scores.fill(0.0)
            _gemm(tiled_mma, scores, rQ, rK)
            scores_low = cute.make_fragment_like(scores, input_dtype)
            scores_low.store(scores.load().to(input_dtype))
            cute.autovec_copy(scores_low, thread_mma.partition_C(sP))
            cute.arch.sync_threads()

            linear = tidx
            for _ in cutlass.range_constexpr((_CHUNK_SIZE * _CHUNK_SIZE) // _NUM_THREADS):
                row = linear // _CHUNK_SIZE
                column = linear % _CHUNK_SIZE
                token = chunk * _CHUNK_SIZE + row
                eta = Float32(0.0)
                if token < T:
                    eta = Float32(mEta[(Int64(batch) * T + token) * H + head])
                score = Float32(0.0)
                if column <= row:
                    score = -eta * Float32(sP[row, column])
                sP[row, column] = input_dtype(score)
                linear += _NUM_THREADS
            cute.arch.sync_threads()

            rP = _load_mma_a(thread_mma, smem_tiled_copy_a, tidx, sP)
            rV = _load_mma_b(thread_mma, smem_tiled_copy_b, tidx, sV)
            _gemm(tiled_mma, output, rP, rV)
            output.store(output.load() * scale)

            linear = tidx
            for _ in cutlass.range_constexpr((_CHUNK_SIZE * _CHUNK_SIZE) // _NUM_THREADS):
                row = linear // _CHUNK_SIZE
                column = linear % _CHUNK_SIZE
                token = chunk * _CHUNK_SIZE + row
                eta = Float32(0.0)
                if token < T:
                    eta = Float32(mEta[(Int64(batch) * T + token) * H + head])
                sP[row, column] = input_dtype(-eta if column <= row else 0.0)
                linear += _NUM_THREADS
            cute.arch.sync_threads()

            rE = _load_mma_a(thread_mma, smem_tiled_copy_a, tidx, sP)
            _gemm(tiled_mma, output, rE, rV)
            cute.autovec_copy(output, thread_mma.partition_C(sO))
            cute.arch.sync_threads()

            bias_base = state * _HEAD_DIM
            linear = tidx
            for _ in cutlass.range_constexpr((_CHUNK_SIZE * _HEAD_DIM) // _NUM_THREADS):
                row = linear // _HEAD_DIM
                value = linear % _HEAD_DIM
                token = chunk * _CHUNK_SIZE + row
                if token < T:
                    token_base = (Int64(batch) * T + token) * H * _HEAD_DIM + Int64(head) * _HEAD_DIM
                    mO[token_base + value] = input_dtype(sO[row, value] + Float32(mHB[bias_base + value]))
                linear += _NUM_THREADS

    tensor_fakes = tuple(
        cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16) for _ in range(7)
    )
    return cute.compile(
        TTTLinearForwardOutput(),
        *tensor_fakes,
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def ttt_linear_fwd_o_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    h: torch.Tensor,
    hb: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run the fixed 16-token, 64-wide TTT-linear output stage with native MMA."""
    batches, sequence_length, heads, _ = q.shape
    output = torch.empty_like(v)
    compiled = _compile_ttt_linear_fwd_o(_torch_to_cute_dtype(q.dtype), sequence_length, heads)
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        eta.view(-1),
        h.view(-1),
        hb.view(-1),
        output.view(-1),
        batches,
        scale,
    )
    return output


__all__ = ["ttt_linear_fwd_o_cute"]
