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
from cutlass import Float32, Int32, Int64, const_expr

from fla.ops.backends.cute.op import exp
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_MAX_T = 16
_D = 64
_NUM_THREADS = 256


@cache
def _compile_path_attn(input_dtype, *, use_gate: bool):
    class PathAttention:
        def _get_shared_storage_cls(self):
            matrix_struct = cute.struct.Align[cute.struct.MemRange[Float32, _MAX_T * _MAX_T], 128]

            @cute.struct
            class SharedStorage:
                transform: matrix_struct
                wb_k: matrix_struct
                transformed_wb_k: matrix_struct
                q_w: matrix_struct
                scores: matrix_struct
                probabilities: matrix_struct

            return SharedStorage

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mW: cute.Tensor,
            mBeta: cute.Tensor,
            mG: cute.Tensor | None,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            HQ: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            SharedStorage = self._get_shared_storage_cls()
            self.kernel(mQ, mK, mV, mW, mBeta, mG, mO, T, H, HQ, scale, SharedStorage).launch(
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
            mW: cute.Tensor,
            mBeta: cute.Tensor,
            mG: cute.Tensor | None,
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
            layout = cute.make_layout((_MAX_T * _MAX_T,), stride=(1,))
            transform = storage.transform.get_tensor(layout)
            wb_k = storage.wb_k.get_tensor(layout)
            transformed_wb_k = storage.transformed_wb_k.get_tensor(layout)
            q_w = storage.q_w.get_tensor(layout)
            scores = storage.scores.get_tensor(layout)
            probabilities = storage.probabilities.get_tensor(layout)

            valid = row < T and column < T
            q_base = (Int64(batch) * T + row) * HQ * _D + query_head * _D
            key_base = (Int64(batch) * T + column) * H * _D + kv_head * _D
            w_row_base = (Int64(batch) * T + row) * H * _D + kv_head * _D
            w_col_base = (Int64(batch) * T + column) * H * _D + kv_head * _D
            qk_dot, ww_dot, wk_dot, qw_dot = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            if valid:
                for dim in cutlass.range_constexpr(_D):
                    query = Float32(mQ[q_base + dim])
                    key = Float32(mK[key_base + dim])
                    w_row = Float32(mW[w_row_base + dim])
                    w_col = Float32(mW[w_col_base + dim])
                    qk_dot += query * key
                    ww_dot += w_row * w_col
                    wk_dot += w_row * key
                    qw_dot += query * w_col
            beta = Float32(mBeta[(Int64(batch) * T + row) * H + kv_head]) if row < T else Float32(0.0)
            transform[tidx] = -beta * ww_dot if valid and row > column else Float32(0.0)
            wb_k[tidx] = beta * wk_dot if valid and row > column else Float32(0.0)
            q_w[tidx] = qw_dot if valid and row >= column else Float32(0.0)
            scores[tidx] = qk_dot
            cute.arch.barrier()

            if tidx == 0:
                # Forward substitution used by the frozen reference to build
                # the inverse lower-triangular Householder product.
                for i in cutlass.range(1, T, 1, unroll=1):
                    for j in cutlass.range(0, i, 1, unroll=1):
                        value = transform[i * _MAX_T + j]
                        for middle in cutlass.range(j + 1, i, 1, unroll=1):
                            value += transform[i * _MAX_T + middle] * transform[middle * _MAX_T + j]
                        transform[i * _MAX_T + j] = value
                for i in cutlass.range(0, T, 1, unroll=1):
                    transform[i * _MAX_T + i] = Float32(1.0)
            cute.arch.barrier()

            if valid and row > column:
                value = Float32(0.0)
                for middle in cutlass.range(0, T, 1, unroll=1):
                    value += transform[row * _MAX_T + middle] * wb_k[middle * _MAX_T + column]
                transformed_wb_k[tidx] = value
            else:
                transformed_wb_k[tidx] = Float32(0.0)
            cute.arch.barrier()

            if valid and row >= column:
                value = scores[tidx]
                for middle in cutlass.range(0, T, 1, unroll=1):
                    value -= q_w[row * _MAX_T + middle] * transformed_wb_k[middle * _MAX_T + column]
                if const_expr(use_gate):
                    gate_row, gate_col = Float32(0.0), Float32(0.0)
                    for pos in cutlass.range(0, row + 1, 1, unroll=1):
                        gate_row += Float32(mG[(Int64(batch) * T + pos) * HQ + query_head])
                    for pos in cutlass.range(0, column + 1, 1, unroll=1):
                        gate_col += Float32(mG[(Int64(batch) * T + pos) * HQ + query_head])
                    value += gate_row - gate_col
                scores[tidx] = value * scale
            else:
                scores[tidx] = Float32(float("-inf"))
            cute.arch.barrier()

            if tidx < T:
                row_max = Float32(float("-inf"))
                for j in cutlass.range(0, tidx + 1, 1, unroll=1):
                    value = scores[tidx * _MAX_T + j]
                    row_max = value if value > row_max else row_max
                denominator = Float32(0.0)
                for j in cutlass.range(0, tidx + 1, 1, unroll=1):
                    probability = exp(scores[tidx * _MAX_T + j] - row_max)
                    probabilities[tidx * _MAX_T + j] = probability
                    denominator += probability
                for j in cutlass.range(0, tidx + 1, 1, unroll=1):
                    probabilities[tidx * _MAX_T + j] /= denominator
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
    w_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    beta_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    gate_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_gate else None
    output_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        PathAttention(),
        q_fake,
        k_fake,
        v_fake,
        w_fake,
        beta_fake,
        gate_fake,
        output_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def path_attn_fwd_cute(q, k, v, w, beta, g, scale, use_cache):
    """Run the one-token, D=64 PATH inference kernel."""
    B, T, HQ, _ = q.shape
    if T != 1:
        raise ValueError(f"CuTe PATH decode requires T=1, got T={T}")
    H = k.shape[2]
    output = torch.empty(B, T, HQ, _D, device=q.device, dtype=q.dtype)
    compiled = _compile_path_attn(_torch_to_cute_dtype(q.dtype), use_gate=g is not None)
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        w.view(-1),
        beta.view(-1),
        g.view(-1) if g is not None else None,
        output.view(-1),
        B,
        T,
        H,
        HQ,
        scale,
    )
    return output, k.clone() if use_cache else None


__all__ = ["path_attn_fwd_cute"]
