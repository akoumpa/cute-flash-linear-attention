# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_CHUNK, _DIM, _T, _THREADS = 16, 32, 64, 32


@cute.jit
def _mma_product(input_dtype, mma, thread_mma, sA, sB, sC):
    rA = thread_mma.make_fragment_A(thread_mma.partition_A(sA))
    rB = thread_mma.make_fragment_B(thread_mma.partition_B(sB))
    cute.autovec_copy(thread_mma.partition_A(sA), rA)
    cute.autovec_copy(thread_mma.partition_B(sB), rB)
    accumulator = cute.make_rmem_tensor(thread_mma.partition_shape_C((_CHUNK, _DIM)), Float32)
    accumulator.fill(0.0)
    cute.gemm(mma, accumulator, rA, rB, accumulator)
    output = cute.make_fragment_like(accumulator, input_dtype)
    output.store(accumulator.load().to(input_dtype))
    cute.autovec_copy(output, thread_mma.partition_C(sC))


@cache
def _compile_delta_rule_wy_recompute(input_dtype, H: int):
    class DeltaRuleWyRecompute:
        @cute.jit
        def __call__(self, mK, mV, mBeta, mA, mW, mU, B: Int32, stream: cuda.CUstream):
            a_layout = cute.make_layout((_CHUNK, _CHUNK), stride=(_CHUNK, 1))
            b_layout = cute.make_layout((_DIM, _CHUNK), stride=(_CHUNK, 1))
            c_layout = cute.make_layout((_CHUNK, _DIM), stride=(_DIM, 1))
            mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(input_dtype, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            self.kernel(mK, mV, mBeta, mA, mW, mU, a_layout, b_layout, c_layout, mma).launch(
                grid=[B * H * 4, 1, 1],
                block=[_THREADS, 1, 1],
                smem=(_CHUNK * _CHUNK + 2 * _CHUNK * _DIM) * (input_dtype.width // 8),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, mK, mV, mBeta, mA, mW, mU, a_layout, b_layout, c_layout, mma):
            lane, _, _ = cute.arch.thread_idx()
            work, _, _ = cute.arch.block_idx()
            chunk, bh = work % 4, work // 4
            batch, head = bh // H, bh % H
            token_start = chunk * _CHUNK
            gK, gV = mK[batch, None, head, None], mV[batch, None, head, None]
            gBeta, gA = mBeta[batch, None, head], mA[batch, None, head, None]
            gW, gU = mW[batch, None, head, None], mU[batch, None, head, None]

            allocator = cutlass.utils.SmemAllocator()
            sA = allocator.allocate_tensor(input_dtype, a_layout, byte_alignment=16)
            sB = allocator.allocate_tensor(input_dtype, b_layout, byte_alignment=16)
            sC = allocator.allocate_tensor(input_dtype, c_layout, byte_alignment=16)
            for element in cutlass.range(lane, _CHUNK * _CHUNK, _THREADS, unroll=1):
                row, column = element // _CHUNK, element % _CHUNK
                sA[row, column] = gA[token_start + row, column]
            for element in cutlass.range(lane, _CHUNK * _DIM, _THREADS, unroll=1):
                feature, token = element // _CHUNK, element % _CHUNK
                beta = Float32(gBeta[token_start + token])
                sB[feature, token] = input_dtype(Float32(gK[token_start + token, feature]) * beta)
            cute.arch.sync_warp()

            thread_mma = mma.get_slice(lane)
            _mma_product(input_dtype, mma, thread_mma, sA, sB, sC)
            cute.arch.sync_warp()
            for element in cutlass.range(lane, _CHUNK * _DIM, _THREADS, unroll=1):
                token, feature = element // _DIM, element % _DIM
                gW[token_start + token, feature] = sC[token, feature]
            for element in cutlass.range(lane, _CHUNK * _DIM, _THREADS, unroll=1):
                feature, token = element // _CHUNK, element % _CHUNK
                beta = Float32(gBeta[token_start + token])
                sB[feature, token] = input_dtype(Float32(gV[token_start + token, feature]) * beta)
            cute.arch.sync_warp()
            _mma_product(input_dtype, mma, thread_mma, sA, sB, sC)
            cute.arch.sync_warp()
            for element in cutlass.range(lane, _CHUNK * _DIM, _THREADS, unroll=1):
                token, feature = element // _DIM, element % _DIM
                gU[token_start + token, feature] = sC[token, feature]

    tensor_fake = cute.runtime.make_fake_tensor(
        input_dtype,
        (cute.sym_int(), _T, H, _DIM),
        stride=(_T * H * _DIM, H * _DIM, _DIM, 1),
        assumed_align=16,
    )
    beta_fake = cute.runtime.make_fake_tensor(
        input_dtype,
        (cute.sym_int(), _T, H),
        stride=(_T * H, H, 1),
        assumed_align=16,
    )
    a_fake = cute.runtime.make_fake_tensor(
        input_dtype,
        (cute.sym_int(), _T, H, _CHUNK),
        stride=(_T * H * _CHUNK, H * _CHUNK, _CHUNK, 1),
        assumed_align=16,
    )
    return cute.compile(
        DeltaRuleWyRecompute(),
        tensor_fake,
        tensor_fake,
        beta_fake,
        a_fake,
        tensor_fake,
        tensor_fake,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def delta_rule_wy_recompute_cute(k, v, beta, A):
    """Recompute the bounded 16-token WY products with native warp MMA."""
    B, _, H, _ = k.shape
    w, u = torch.empty_like(k), torch.empty_like(v)
    compiled = _compile_delta_rule_wy_recompute(_torch_to_cute_dtype(k.dtype), H)
    compiled(k, v, beta, A, w, u, B)
    return w, u


__all__ = ["delta_rule_wy_recompute_cute"]
