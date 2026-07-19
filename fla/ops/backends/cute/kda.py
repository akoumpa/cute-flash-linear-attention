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
from cutlass import Float32, Int32, Int64, const_expr

from fla.ops.backends.cute.op import exp
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_K = 32
_V = 32


@cache
def _compile_kda_fwd(input_dtype, gate_dtype, beta_dtype, *, use_initial_state: bool, store_final_state: bool):
    class KDAForward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mBeta: cute.Tensor,
            mO: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            B: Int32,
            T: Int32,
            H: Int32,
            HV: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(mQ, mK, mV, mG, mBeta, mO, mH0, mHT, T, H, HV, scale).launch(
                grid=[B * HV, 1, 1],
                block=[_V, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mBeta: cute.Tensor,
            mO: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            T: Int32,
            H: Int32,
            HV: Int32,
            scale: Float32,
        ):
            value_idx, _, _ = cute.arch.thread_idx()
            nhv, _, _ = cute.arch.block_idx()
            batch = nhv // HV
            value_head = nhv % HV
            key_head = value_head // (HV // H)
            state = cute.make_rmem_tensor((_K,), Float32)

            for key_idx in cutlass.range_constexpr(_K):
                state_idx = Int64(nhv) * _K * _V + key_idx * _V + value_idx
                state[key_idx] = Float32(mH0[state_idx]) if const_expr(use_initial_state) else Float32(0.0)

            for pos in cutlass.range(0, T, 1, unroll=1):
                value_token = (Int64(batch) * T + pos) * HV + value_head
                key_token = (Int64(batch) * T + pos) * H + key_head
                erase = Float32(0.0)
                for key_idx in cutlass.range_constexpr(_K):
                    state[key_idx] *= exp(Float32(mG[value_token * _K + key_idx]))
                    erase += state[key_idx] * Float32(mK[key_token * _K + key_idx])

                value_new = (Float32(mV[value_token * _V + value_idx]) - erase) * Float32(mBeta[value_token])
                output = Float32(0.0)
                for key_idx in cutlass.range_constexpr(_K):
                    key = Float32(mK[key_token * _K + key_idx])
                    state[key_idx] += key * value_new
                    output += state[key_idx] * Float32(mQ[key_token * _K + key_idx]) * scale
                mO[value_token * _V + value_idx] = input_dtype(output)

            if const_expr(store_final_state):
                for key_idx in cutlass.range_constexpr(_K):
                    state_idx = Int64(nhv) * _K * _V + key_idx * _V + value_idx
                    mHT[state_idx] = state[key_idx]

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    g_fake = cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    beta_fake = cute.runtime.make_fake_tensor(beta_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    final_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        KDAForward(),
        q_fake,
        k_fake,
        v_fake,
        g_fake,
        beta_fake,
        output_fake,
        state_fake,
        final_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def kda_fwd_cute(q, k, v, g, beta, initial_state, output_final_state, scale):
    """Run the bounded dense pre-activated KDA recurrence."""
    B, T, H, _ = k.shape
    HV = v.shape[2]
    output = v.new_empty(v.shape)
    final_state = q.new_empty((B, HV, _K, _V), dtype=torch.float32) if output_final_state else None
    compiled = _compile_kda_fwd(
        _torch_to_cute_dtype(q.dtype),
        _torch_to_cute_dtype(g.dtype),
        _torch_to_cute_dtype(beta.dtype),
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
    )
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        g.view(-1),
        beta.view(-1),
        output.view(-1),
        initial_state.view(-1) if initial_state is not None else None,
        final_state.view(-1) if final_state is not None else None,
        B,
        T,
        H,
        HV,
        scale,
    )
    return output, final_state


__all__ = ["kda_fwd_cute"]
