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

from fla.ops.backends.cute.op import exp2
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_NUM_THREADS = 256


@cache
def _compile_cp_chunk_delta_h_pre_process(input_dtype, gate_dtype, K: int, V: int, *, use_gate: bool):
    class CPChunkDeltaHPreProcess:
        @cute.jit
        def __call__(
            self,
            mK: cute.Tensor,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mG: cute.Tensor | None,
            mHM: cute.Tensor,
            T: Int32,
            H: Int32,
            HV: Int32,
            stream: cuda.CUstream,
        ):
            output_count = HV * K * (V + K)
            self.kernel(mK, mW, mU, mG, mHM, T, H, HV, output_count).launch(
                grid=[cute.ceil_div(output_count, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mK: cute.Tensor,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mG: cute.Tensor | None,
            mHM: cute.Tensor,
            T: Int32,
            H: Int32,
            HV: Int32,
            output_count: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            output_idx = bid * _NUM_THREADS + tidx

            if output_idx < output_count:
                column = output_idx % (V + K)
                key_idx = (output_idx // (V + K)) % K
                value_head = output_idx // (K * (V + K))
                key_head = value_head // (HV // H)
                total = Float32(0.0)
                gate_last = Float32(mG[(T - 1) * HV + value_head]) if const_expr(use_gate) else Float32(0.0)

                if column < V:
                    # For the sole local chunk, h = k^T @ u.
                    for pos in cutlass.range(0, T, 1, unroll=1):
                        key_offset = (Int64(pos) * H + key_head) * K + key_idx
                        value_offset = (Int64(pos) * HV + value_head) * V + column
                        gate_scale = (
                            exp2(gate_last - Float32(mG[Int64(pos) * HV + value_head]))
                            if const_expr(use_gate)
                            else Float32(1.0)
                        )
                        scaled_value = Float32(input_dtype(Float32(mU[value_offset]) * gate_scale))
                        total += Float32(mK[key_offset]) * scaled_value
                else:
                    # Its state transition is M = I - k^T @ w.
                    transition_col = column - V
                    diagonal = exp2(gate_last) if const_expr(use_gate) else Float32(1.0)
                    total = diagonal if key_idx == transition_col else Float32(0.0)
                    for pos in cutlass.range(0, T, 1, unroll=1):
                        key_offset = (Int64(pos) * H + key_head) * K + key_idx
                        weight_offset = (Int64(pos) * HV + value_head) * K + transition_col
                        gate_scale = (
                            exp2(gate_last - Float32(mG[Int64(pos) * HV + value_head]))
                            if const_expr(use_gate)
                            else Float32(1.0)
                        )
                        scaled_key = Float32(input_dtype(Float32(mK[key_offset]) * gate_scale))
                        total -= scaled_key * Float32(mW[weight_offset])

                mHM[output_idx] = total

    def input_fake():
        return cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)

    gate_fake = (
        cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_gate else None
    )
    output_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        CPChunkDeltaHPreProcess(),
        input_fake(),
        input_fake(),
        input_fake(),
        gate_fake,
        output_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def cp_chunk_delta_h_pre_process_cute(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
) -> torch.Tensor:
    """Build a dense scalar-gated single-chunk CP transfer pair ``[h, M]``."""
    _, T, H, K = k.shape
    HV, V = u.shape[2:]
    hm = k.new_empty(HV, K, V + K, dtype=torch.float32)
    compiled = _compile_cp_chunk_delta_h_pre_process(
        _torch_to_cute_dtype(k.dtype),
        _torch_to_cute_dtype(g.dtype) if g is not None else None,
        K,
        V,
        use_gate=g is not None,
    )
    compiled(
        k.view(-1),
        w.view(-1),
        u.view(-1),
        g.view(-1) if g is not None else None,
        hm.view(-1),
        T,
        H,
        HV,
    )
    return hm


__all__ = ["cp_chunk_delta_h_pre_process_cute"]
