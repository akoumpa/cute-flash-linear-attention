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
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_MAX_T = 16
_D = 64
_NUM_THREADS = 256


@cache
def _compile_forgetting_attn(input_dtype, gate_dtype):
    class ForgettingAttention:
        def _get_shared_storage_cls(self):
            matrix_struct = cute.struct.Align[cute.struct.MemRange[Float32, _MAX_T * _MAX_T], 128]
            gate_struct = cute.struct.Align[cute.struct.MemRange[Float32, _MAX_T], 128]

            @cute.struct
            class SharedStorage:
                gate_cumsum: gate_struct
                scores: matrix_struct
                probabilities: matrix_struct

            return SharedStorage

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            HQ: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            SharedStorage = self._get_shared_storage_cls()
            self.kernel(mQ, mK, mV, mG, mO, T, H, HQ, scale, SharedStorage).launch(
                grid=[B * HQ, 1, 1],
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
            mO: cute.Tensor,
            T: Int32,
            H: Int32,
            HQ: Int32,
            scale: Float32,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bhq, _, _ = cute.arch.block_idx()
            batch, query_head = bhq // HQ, bhq % HQ
            kv_head = query_head // (HQ // H)
            row, column = tidx // _MAX_T, tidx % _MAX_T

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            matrix_layout = cute.make_layout((_MAX_T * _MAX_T,), stride=(1,))
            gate_layout = cute.make_layout((_MAX_T,), stride=(1,))
            gate_cumsum = storage.gate_cumsum.get_tensor(gate_layout)
            scores = storage.scores.get_tensor(matrix_layout)
            probabilities = storage.probabilities.get_tensor(matrix_layout)

            if tidx < T:
                cumulative = Float32(0.0)
                for pos in cutlass.range(0, tidx + 1, 1, unroll=1):
                    cumulative += Float32(mG[(Int64(batch) * T + pos) * HQ + query_head])
                gate_cumsum[tidx] = cumulative
            cute.arch.barrier()

            valid = row < T and column < T
            dot = Float32(0.0)
            if valid:
                query_base = (Int64(batch) * T + row) * HQ * _D + query_head * _D
                key_base = (Int64(batch) * T + column) * H * _D + kv_head * _D
                for dim in cutlass.range_constexpr(_D):
                    dot += Float32(mQ[query_base + dim]) * Float32(mK[key_base + dim])
            scores[tidx] = (
                dot * scale + gate_cumsum[row] - gate_cumsum[column] if valid and row >= column else Float32(float("-inf"))
            )
            cute.arch.barrier()

            if tidx < T:
                row_max = Float32(float("-inf"))
                for source in cutlass.range(0, tidx + 1, 1, unroll=1):
                    value = scores[tidx * _MAX_T + source]
                    row_max = value if value > row_max else row_max
                denominator = Float32(0.0)
                for source in cutlass.range(0, tidx + 1, 1, unroll=1):
                    probability = exp(scores[tidx * _MAX_T + source] - row_max)
                    probabilities[tidx * _MAX_T + source] = probability
                    denominator += probability
                for source in cutlass.range(0, tidx + 1, 1, unroll=1):
                    probabilities[tidx * _MAX_T + source] /= denominator
            cute.arch.barrier()

            for output_idx in cutlass.range_constexpr(cute.ceil_div(_MAX_T * _D, _NUM_THREADS)):
                flat_output = tidx + output_idx * _NUM_THREADS
                output_row, value_dim = flat_output // _D, flat_output % _D
                if output_row < T:
                    output = Float32(0.0)
                    for source in cutlass.range(0, output_row + 1, 1, unroll=1):
                        value_offset = (Int64(batch) * T + source) * H * _D + kv_head * _D + value_dim
                        weight = Float32(input_dtype(probabilities[output_row * _MAX_T + source]))
                        output += weight * Float32(mV[value_offset])
                    output_offset = (Int64(batch) * T + output_row) * HQ * _D + query_head * _D + value_dim
                    mO[output_offset] = input_dtype(output)

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    g_fake = cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        ForgettingAttention(),
        q_fake,
        k_fake,
        v_fake,
        g_fake,
        output_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def forgetting_attn_fwd_cute(q, k, v, g, scale):
    """Run bounded dense causal forgetting attention."""
    B, T, HQ, _ = q.shape
    H = k.shape[2]
    output = torch.empty(B, T, HQ, _D, device=q.device, dtype=q.dtype)
    compiled = _compile_forgetting_attn(_torch_to_cute_dtype(q.dtype), _torch_to_cute_dtype(g.dtype))
    compiled(q.view(-1), k.view(-1), v.view(-1), g.view(-1), output.view(-1), B, T, H, HQ, scale)
    return output


__all__ = ["forgetting_attn_fwd_cute"]
