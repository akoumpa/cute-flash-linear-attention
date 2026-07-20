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
def _compile_hgrn_fwd(input_dtype, *, use_initial_state: bool, store_final_state: bool):
    class HgrnForward:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mG: cute.Tensor,
            mH0: cute.Tensor | None,
            mO: cute.Tensor,
            mHT: cute.Tensor | None,
            B: Int32,
            T: Int32,
            D: Int32,
            stream: cuda.CUstream,
        ):
            work = B * D
            self.kernel(mX, mG, mH0, mO, mHT, T, D, work).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mG: cute.Tensor,
            mH0: cute.Tensor | None,
            mO: cute.Tensor,
            mHT: cute.Tensor | None,
            T: Int32,
            D: Int32,
            work: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            lane = bid * _NUM_THREADS + tidx
            if lane < work:
                feature = lane % D
                batch = lane // D
                state = Float32(mH0[lane]) if const_expr(use_initial_state) else Float32(0.0)
                for token in cutlass.range(0, T, 1, unroll=1):
                    offset = (Int64(batch) * T + token) * D + feature
                    decay = cute.math.exp(Float32(mG[offset]), fastmath=False)
                    state = decay * state + Float32(mX[offset])
                    mO[offset] = state
                if const_expr(store_final_state):
                    mHT[lane] = state

    x_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    g_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    h0_fake = (
        cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
        if use_initial_state
        else None
    )
    ht_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        HgrnForward(),
        x_fake,
        g_fake,
        h0_fake,
        output_fake,
        ht_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_hgrn_bwd(input_dtype, *, use_initial_state: bool):
    class HgrnBackward:
        @cute.jit
        def __call__(
            self,
            mG: cute.Tensor,
            mO: cute.Tensor,
            mDO: cute.Tensor,
            mH0: cute.Tensor | None,
            mDX: cute.Tensor,
            mDG: cute.Tensor,
            B: Int32,
            T: Int32,
            D: Int32,
            stream: cuda.CUstream,
        ):
            work = B * D
            self.kernel(mG, mO, mDO, mH0, mDX, mDG, T, D, work).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mG: cute.Tensor,
            mO: cute.Tensor,
            mDO: cute.Tensor,
            mH0: cute.Tensor | None,
            mDX: cute.Tensor,
            mDG: cute.Tensor,
            T: Int32,
            D: Int32,
            work: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            lane = bid * _NUM_THREADS + tidx
            if lane < work:
                feature = lane % D
                batch = lane // D
                gradient = Float32(0.0)
                for reverse_token in cutlass.range(0, T, 1, unroll=1):
                    token = T - 1 - reverse_token
                    offset = (Int64(batch) * T + token) * D + feature
                    gradient += Float32(mDO[offset])
                    mDX[offset] = gradient
                    decay = cute.math.exp(Float32(mG[offset]), fastmath=False)
                    previous = Float32(0.0)
                    if token > 0:
                        previous = Float32(mO[offset - D])
                    elif const_expr(use_initial_state):
                        previous = Float32(mH0[lane])
                    gradient *= decay
                    mDG[offset] = previous * gradient

    g_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    o_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    do_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    h0_fake = (
        cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
        if use_initial_state
        else None
    )
    dx_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    dg_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        HgrnBackward(),
        g_fake,
        o_fake,
        do_fake,
        h0_fake,
        dx_fake,
        dg_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def hgrn_fwd_cute(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the bounded dense HGRN forward recurrence."""
    B, T, D = x.shape
    o = torch.empty_like(x, dtype=torch.float32)
    ht = torch.empty(B, D, dtype=torch.float32, device=x.device) if output_final_state else None
    compiled = _compile_hgrn_fwd(
        _torch_to_cute_dtype(x.dtype),
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
    )
    compiled(
        x.view(-1),
        g.view(-1),
        initial_state.view(-1) if initial_state is not None else None,
        o.view(-1),
        ht.view(-1) if ht is not None else None,
        B,
        T,
        D,
    )
    return o, ht


def hgrn_bwd_cute(
    g: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the bounded dense HGRN reverse recurrence."""
    B, T, D = do.shape
    dx = torch.empty_like(o, dtype=torch.float32)
    dg = torch.empty_like(g, dtype=torch.float32)
    compiled = _compile_hgrn_bwd(_torch_to_cute_dtype(g.dtype), use_initial_state=initial_state is not None)
    compiled(
        g.view(-1),
        o.view(-1),
        do.view(-1),
        initial_state.view(-1) if initial_state is not None else None,
        dx.view(-1),
        dg.view(-1),
        B,
        T,
        D,
    )
    return dx, dg


__all__ = ["hgrn_bwd_cute", "hgrn_fwd_cute"]
