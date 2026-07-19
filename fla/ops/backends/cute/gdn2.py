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
from cutlass import Float32, Int32, Int64, const_expr

from fla.ops.backends.cute.op import exp
from fla.ops.backends.cute.pack import _torch_to_cute_dtype


@cache
def _compile_gdn2_fwd(
    input_dtype,
    gate_dtype,
    erase_dtype,
    write_dtype,
    K: int,
    V: int,
    *,
    use_initial_state: bool,
    store_final_state: bool,
    state_v_first: bool,
):
    num_threads = max(32, 1 << (V - 1).bit_length())

    class GDN2Forward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mB: cute.Tensor,
            mW: cute.Tensor,
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
            self.kernel(mQ, mK, mV, mG, mB, mW, mO, mH0, mHT, T, H, HV, scale).launch(
                grid=[B * HV, 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor,
            mB: cute.Tensor,
            mW: cute.Tensor,
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
                if const_expr(state_v_first):
                    state_idx = Int64(nhv) * K * V + value_idx * K + key_idx
                else:
                    state_idx = Int64(nhv) * K * V + key_idx * V + value_idx
                state[key_idx] = Float32(mH0[state_idx]) if const_expr(use_initial_state) and valid_value else Float32(0.0)

            for pos in cutlass.range(0, T, 1, unroll=1):
                value_token = (Int64(batch) * T + pos) * HV + value_head
                key_token = (Int64(batch) * T + pos) * H + key_head
                value = Float32(mV[value_token * V + value_idx]) if valid_value else Float32(0.0)
                write = Float32(mW[value_token * V + value_idx]) if valid_value else Float32(0.0)
                erase = Float32(0.0)

                for key_idx in cutlass.range_constexpr(K):
                    gate_offset = value_token * K + key_idx
                    state[key_idx] *= exp(Float32(mG[gate_offset]))
                    key = Float32(mK[key_token * K + key_idx])
                    erase += state[key_idx] * Float32(mB[gate_offset]) * key

                value_new = write * value - erase
                output = Float32(0.0)
                for key_idx in cutlass.range_constexpr(K):
                    key = Float32(mK[key_token * K + key_idx])
                    state[key_idx] += key * value_new
                    query = Float32(mQ[key_token * K + key_idx]) * scale
                    output += state[key_idx] * query

                if valid_value:
                    mO[value_token * V + value_idx] = input_dtype(output)

            if const_expr(store_final_state):
                for key_idx in cutlass.range_constexpr(K):
                    if valid_value:
                        if const_expr(state_v_first):
                            state_idx = Int64(nhv) * K * V + value_idx * K + key_idx
                        else:
                            state_idx = Int64(nhv) * K * V + key_idx * V + value_idx
                        mHT[state_idx] = state[key_idx]

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    out_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    gate_fake = cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    erase_fake = cute.runtime.make_fake_tensor(erase_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    write_fake = cute.runtime.make_fake_tensor(write_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    final_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        GDN2Forward(),
        q_fake,
        k_fake,
        v_fake,
        gate_fake,
        erase_fake,
        write_fake,
        out_fake,
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


def gdn2_fwd_cute(q, k, v, g, b, w, out, initial_state, final_state, scale, state_v_first):
    """Run the bounded dense pre-activated-gate GDN-2 recurrence."""
    B, T, H, K = k.shape
    HV, V = v.shape[2:]
    compiled = _compile_gdn2_fwd(
        _torch_to_cute_dtype(k.dtype),
        _torch_to_cute_dtype(g.dtype),
        _torch_to_cute_dtype(b.dtype),
        _torch_to_cute_dtype(w.dtype),
        K,
        V,
        use_initial_state=initial_state is not None,
        store_final_state=final_state is not None,
        state_v_first=state_v_first,
    )
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        g.view(-1),
        b.view(-1),
        w.view(-1),
        out.view(-1),
        initial_state.view(-1) if initial_state is not None else None,
        final_state.view(-1) if final_state is not None else None,
        B,
        T,
        H,
        HV,
        scale,
    )
    return out, final_state


__all__ = ["gdn2_fwd_cute"]
