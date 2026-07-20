# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyu Li
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


@cache
def _compile_rwkv6_fwd(
    input_dtype,
    gate_dtype,
    bonus_dtype,
    *,
    use_initial_state: bool,
    store_final_state: bool,
):
    K = 32
    V = 32

    class RWKV6Forward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mO: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            B: Int32,
            T: Int32,
            H: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(mQ, mK, mV, mW, mU, mO, mH0, mHT, T, H, scale).launch(
                grid=[B * H, 1, 1],
                block=[V, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mO: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            T: Int32,
            H: Int32,
            scale: Float32,
        ):
            value_idx, _, _ = cute.arch.thread_idx()
            nh, _, _ = cute.arch.block_idx()
            batch = nh // H
            head = nh % H
            state = cute.make_rmem_tensor((K,), Float32)

            for key_idx in cutlass.range_constexpr(K):
                state_idx = Int64(nh) * K * V + key_idx * V + value_idx
                state[key_idx] = Float32(mH0[state_idx]) if const_expr(use_initial_state) else Float32(0.0)

            for pos in cutlass.range(0, T, 1, unroll=1):
                token = (Int64(batch) * T + pos) * H + head
                value = Float32(mV[token * V + value_idx])
                output = Float32(0.0)

                for key_idx in cutlass.range_constexpr(K):
                    key_offset = token * K + key_idx
                    key = Float32(mK[key_offset])
                    query = Float32(mQ[key_offset]) * scale
                    key_value = key * value
                    bonus = Float32(mU[head * K + key_idx])
                    output += (state[key_idx] + key_value * bonus) * query
                    state[key_idx] = state[key_idx] * exp(Float32(mW[key_offset])) + key_value

                mO[token * V + value_idx] = input_dtype(output)

            if const_expr(store_final_state):
                for key_idx in cutlass.range_constexpr(K):
                    state_idx = Int64(nh) * K * V + key_idx * V + value_idx
                    mHT[state_idx] = state[key_idx]

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    w_fake = cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    u_fake = cute.runtime.make_fake_tensor(bonus_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    out_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    final_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        RWKV6Forward(),
        q_fake,
        k_fake,
        v_fake,
        w_fake,
        u_fake,
        out_fake,
        state_fake,
        final_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rwkv6_fwd_cute(q, k, v, w, u, initial_state, output_final_state, scale):
    """Run the bounded dense forward RWKV6 recurrence."""
    B, T, H, _ = k.shape
    out = v.new_empty(v.shape)
    final_state = q.new_empty((B, H, 32, 32), dtype=torch.float32) if output_final_state else None
    compiled = _compile_rwkv6_fwd(
        _torch_to_cute_dtype(q.dtype),
        _torch_to_cute_dtype(w.dtype),
        _torch_to_cute_dtype(u.dtype),
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
    )
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        w.view(-1),
        u.view(-1),
        out.view(-1),
        initial_state.view(-1) if initial_state is not None else None,
        final_state.view(-1) if final_state is not None else None,
        B,
        T,
        H,
        scale,
    )
    return out, final_state


__all__ = ["rwkv6_fwd_cute"]
