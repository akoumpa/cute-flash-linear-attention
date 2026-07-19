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

from fla.ops.backends.cute.pack import _torch_to_cute_dtype


@cache
def _compile_fused_recurrent_fwd(
    input_dtype,
    gate_dtype,
    K: int,
    V: int,
    *,
    use_gate: bool,
    use_initial_state: bool,
    store_final_state: bool,
):
    num_threads = max(32, 1 << (V - 1).bit_length())

    class FusedRecurrentForward:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor | None,
            mO: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            B: Int32,
            T: Int32,
            H: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(mQ, mK, mV, mG, mO, mH0, mHT, T, H, scale).launch(
                grid=[B * H, 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mG: cute.Tensor | None,
            mO: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            T: Int32,
            H: Int32,
            scale: Float32,
        ):
            v_idx, _, _ = cute.arch.thread_idx()
            nh, _, _ = cute.arch.block_idx()

            if v_idx < V:
                head = nh % H
                batch = nh // H
                state = cute.make_rmem_tensor((K,), Float32)
                state_base = Int64(nh) * K * V + v_idx

                for k_idx in cutlass.range_constexpr(K):
                    state[k_idx] = Float32(mH0[state_base + k_idx * V]) if const_expr(use_initial_state) else Float32(0.0)

                for pos in cutlass.range(0, T, 1, unroll=1):
                    token = (Int64(batch) * T + pos) * H + head
                    qk_base = token * K
                    v_base = token * V
                    decay = Float32(mG[token]) if const_expr(use_gate) else Float32(0.0)
                    decay = cute.math.exp(decay, fastmath=False) if const_expr(use_gate) else Float32(1.0)
                    value = Float32(mV[v_base + v_idx])
                    output = Float32(0.0)

                    for k_idx in cutlass.range_constexpr(K):
                        key = Float32(mK[qk_base + k_idx])
                        query = Float32(mQ[qk_base + k_idx]) * scale
                        state[k_idx] = state[k_idx] * decay + key * value
                        output += state[k_idx] * query

                    mO[v_base + v_idx] = output

                if const_expr(store_final_state):
                    for k_idx in cutlass.range_constexpr(K):
                        mHT[state_base + k_idx * V] = state[k_idx]

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    gate_fake = (
        cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_gate else None
    )
    output_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    final_state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        FusedRecurrentForward(),
        q_fake,
        k_fake,
        v_fake,
        gate_fake,
        output_fake,
        state_fake,
        final_state_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fused_recurrent_fwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the bounded dense scalar-gate CuTe recurrent forward kernel."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    o_partial = q.new_empty(1, *v.shape, dtype=torch.float32)
    ht = q.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None
    compiled = _compile_fused_recurrent_fwd(
        _torch_to_cute_dtype(q.dtype),
        _torch_to_cute_dtype(g.dtype) if g is not None else None,
        K,
        V,
        use_gate=g is not None,
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
    )
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        g.view(-1) if g is not None else None,
        o_partial.view(-1),
        initial_state.view(-1) if initial_state is not None else None,
        ht.view(-1) if ht is not None else None,
        B,
        T,
        H,
        scale,
    )
    return o_partial.sum(0), ht


__all__ = ["fused_recurrent_fwd_cute"]
