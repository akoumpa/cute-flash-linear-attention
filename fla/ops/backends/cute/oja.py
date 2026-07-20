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

from fla.ops.backends.cute.op import exp
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_WARP_SIZE = 32


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for offset in (1, 2, 4, 8, 16):
        value += cute.arch.shuffle_sync_bfly(value, offset)
    return value


@cache
def _compile_oja_fwd(
    input_dtype,
    gate_dtype,
    beta_dtype,
    K: int,
    V: int,
    *,
    use_gate: bool,
    use_initial_state: bool,
    store_final_state: bool,
):
    class OjaForward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mGV: cute.Tensor | None,
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
            self.kernel(mQ, mK, mV, mGV, mBeta, mO, mH0, mHT, T, H, HV, scale).launch(
                grid=[B * HV, 1, 1],
                block=[_WARP_SIZE, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mGV: cute.Tensor | None,
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
            valid_value = value_idx < V
            state = cute.make_rmem_tensor((K,), Float32)

            for key_idx in cutlass.range_constexpr(K):
                state_idx = Int64(nhv) * K * V + key_idx * V + value_idx
                state[key_idx] = Float32(mH0[state_idx]) if const_expr(use_initial_state) and valid_value else Float32(0.0)

            for pos in cutlass.range(0, T, 1, unroll=1):
                value_token = (Int64(batch) * T + pos) * HV + value_head
                key_token = (Int64(batch) * T + pos) * H + key_head
                value = Float32(mV[value_token * V + value_idx]) if valid_value else Float32(0.0)
                decay = (
                    exp(Float32(mGV[value_token * V + value_idx])) if const_expr(use_gate) and valid_value else Float32(1.0)
                )
                beta = Float32(mBeta[value_token])
                output = Float32(0.0)

                # Oja is a recurrent GEMV/rank-1 update. A single warp owns one
                # state matrix column per lane and reduces h@v in registers;
                # there is no matrix-matrix product for tensor cores to replace.
                for key_idx in cutlass.range_constexpr(K):
                    state[key_idx] *= decay
                    prediction = _warp_sum(state[key_idx] * value)
                    key = Float32(mK[key_token * K + key_idx])
                    if valid_value:
                        state[key_idx] += beta * (key - prediction) * value
                        query = Float32(mQ[key_token * K + key_idx]) * scale
                        output += state[key_idx] * query

                if valid_value:
                    mO[value_token * V + value_idx] = input_dtype(output)

            if const_expr(store_final_state):
                for key_idx in cutlass.range_constexpr(K):
                    if valid_value:
                        state_idx = Int64(nhv) * K * V + key_idx * V + value_idx
                        mHT[state_idx] = state[key_idx]

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    gate_fake = (
        cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_gate else None
    )
    beta_fake = cute.runtime.make_fake_tensor(beta_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    final_state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        OjaForward(),
        q_fake,
        k_fake,
        v_fake,
        gate_fake,
        beta_fake,
        output_fake,
        state_fake,
        final_state_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def oja_fwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the bounded dense one-warp-per-value-head Oja forward."""
    B, T, H, K = k.shape
    HV, V = v.shape[2:]
    output = torch.empty_like(v)
    final_state = q.new_empty(B, HV, K, V, dtype=torch.float32) if output_final_state else None
    compiled = _compile_oja_fwd(
        _torch_to_cute_dtype(k.dtype),
        _torch_to_cute_dtype(gv.dtype) if gv is not None else None,
        _torch_to_cute_dtype(beta.dtype),
        K,
        V,
        use_gate=gv is not None,
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
    )
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        gv.view(-1) if gv is not None else None,
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


__all__ = ["oja_fwd_cute"]
