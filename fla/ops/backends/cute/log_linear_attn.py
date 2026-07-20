# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from fla.ops.backends.cute.op import exp

_MAX_T = 16
_D = 64
_NUM_THREADS = 256


@cache
def _compile_log_linear_attn_fwd():
    class LogLinearAttention:
        def _get_shared_storage_cls(self):
            weights_struct = cute.struct.Align[cute.struct.MemRange[Float32, _MAX_T * _MAX_T], 128]
            gate_struct = cute.struct.Align[cute.struct.MemRange[Float32, _MAX_T], 128]

            @cute.struct
            class SharedStorage:
                gate_cumsum: gate_struct
                weights: weights_struct

            return SharedStorage

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mLevelScales: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            L: Int32,
            stream: cuda.CUstream,
        ):
            SharedStorage = self._get_shared_storage_cls()
            self.kernel(mQ, mK, mV, mG, mLevelScales, mO, T, H, L, SharedStorage).launch(
                grid=[B * H, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=SharedStorage.size_in_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mLevelScales: cute.Tensor,
            mO: cute.Tensor,
            T: Int32,
            H: Int32,
            L: Int32,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bh, _, _ = cute.arch.block_idx()
            batch, head = bh // H, bh % H
            row, column = tidx // _MAX_T, tidx % _MAX_T

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            matrix_layout = cute.make_layout((_MAX_T * _MAX_T,), stride=(1,))
            gate_layout = cute.make_layout((_MAX_T,), stride=(1,))
            gate_cumsum = storage.gate_cumsum.get_tensor(gate_layout)
            weights = storage.weights.get_tensor(matrix_layout)

            if tidx < T:
                cumulative = Float32(0.0)
                for pos in cutlass.range(0, tidx + 1, 1, unroll=1):
                    cumulative += Float32(mG[(Int64(batch) * T + pos) * H + head])
                gate_cumsum[tidx] = cumulative
            cute.arch.barrier()

            valid = row < T and column <= row
            dot = Float32(0.0)
            if valid:
                query_base = (Int64(batch) * T + row) * _D
                key_base = (Int64(batch) * T + column) * _D
                for dim in cutlass.range_constexpr(_D):
                    dot += Float32(mQ[query_base + dim]) * Float32(mK[key_base + dim])

            weight = Float32(0.0)
            if valid:
                level = Int32(0)
                if row > column:
                    for candidate in cutlass.range_constexpr(1, 5):
                        width = 1 << candidate
                        half = width >> 1
                        in_right_half = row % width >= half
                        boundary = row - row % half
                        if in_right_half and column + half >= boundary and column < boundary:
                            level = candidate
                scale_offset = ((Int64(batch) * T + row) * H + head) * L + level
                weight = dot * exp(gate_cumsum[row] - gate_cumsum[column]) * Float32(mLevelScales[scale_offset])
            weights[tidx] = weight
            cute.arch.barrier()

            for output_idx in cutlass.range_constexpr(cute.ceil_div(_MAX_T * _D, _NUM_THREADS)):
                flat_output = tidx + output_idx * _NUM_THREADS
                output_row, value_dim = flat_output // _D, flat_output % _D
                if output_row < T:
                    output = Float32(0.0)
                    for source in cutlass.range(0, output_row + 1, 1, unroll=1):
                        value_offset = (Int64(batch) * T + source) * H * _D + head * _D + value_dim
                        output += weights[output_row * _MAX_T + source] * Float32(mV[value_offset])
                    output_offset = (Int64(batch) * T + output_row) * H * _D + head * _D + value_dim
                    mO[output_offset] = output

    q_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    gate_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    scale_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        LogLinearAttention(),
        q_fake,
        k_fake,
        v_fake,
        gate_fake,
        scale_fake,
        output_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def log_linear_attn_fwd_cute(q, k, v, g, level_scales):
    """Run bounded dense first-chunk log-linear attention."""
    B, T, H, _ = v.shape
    L = level_scales.shape[-1]
    output = torch.empty_like(v)
    compiled = _compile_log_linear_attn_fwd()
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        g.view(-1),
        level_scales.view(-1),
        output.view(-1),
        B,
        T,
        H,
        L,
    )
    return output, None


__all__ = ["log_linear_attn_fwd_cute"]
