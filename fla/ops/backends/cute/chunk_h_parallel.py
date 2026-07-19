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

_NUM_THREADS = 256


@cache
def _compile_chunk_h_parallel_fwd(
    input_dtype,
    K: int,
    V: int,
    *,
    use_initial_state: bool,
    store_final_state: bool,
):
    class ChunkHParallelForward:
        @cute.jit
        def __call__(
            self,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mH: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            B: Int32,
            T: Int32,
            H: Int32,
            stream: cuda.CUstream,
        ):
            state_count = B * H * K * V
            self.kernel(mK, mV, mH, mH0, mHT, T, H, state_count).launch(
                grid=[cute.ceil_div(state_count, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mH: cute.Tensor,
            mH0: cute.Tensor | None,
            mHT: cute.Tensor | None,
            T: Int32,
            H: Int32,
            state_count: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            state_idx = bid * _NUM_THREADS + tidx

            if state_idx < state_count:
                value_idx = state_idx % V
                key_idx = (state_idx // V) % K
                nh = state_idx // (K * V)
                head = nh % H
                batch = nh // H
                state = Float32(mH0[state_idx]) if const_expr(use_initial_state) else Float32(0.0)

                mH[state_idx] = state
                if const_expr(store_final_state):
                    for pos in cutlass.range(0, T, 1, unroll=1):
                        token = (Int64(batch) * T + pos) * H + head
                        state += Float32(mK[token * K + key_idx]) * Float32(mV[token * V + value_idx])
                    mHT[state_idx] = state

    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    h_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    h0_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    ht_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        ChunkHParallelForward(),
        k_fake,
        v_fake,
        h_fake,
        h0_fake,
        ht_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_h_parallel_fwd_cute(
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the dense single-chunk CuTe state-passing forward kernel."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    h = k.new_empty(B, 1, H, K, V, dtype=torch.float32)
    ht = k.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None
    compiled = _compile_chunk_h_parallel_fwd(
        _torch_to_cute_dtype(k.dtype),
        K,
        V,
        use_initial_state=h0 is not None,
        store_final_state=output_final_state,
    )
    compiled(
        k.view(-1),
        v.view(-1),
        h.view(-1),
        h0.view(-1) if h0 is not None else None,
        ht.view(-1) if ht is not None else None,
        B,
        T,
        H,
    )
    return h, ht


__all__ = ["chunk_h_parallel_fwd_cute"]
