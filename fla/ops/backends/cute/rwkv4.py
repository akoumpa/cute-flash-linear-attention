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
from cutlass import Float32, Int32, Int64

from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_NUM_THREADS = 256


@cache
def _compile_rwkv4_forward(input_dtype, state_dtype, state_output_dtype):
    class RWKV4Forward:
        @cute.jit
        def __call__(
            self,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mState: cute.Tensor,
            mWkv: cute.Tensor,
            mStateOut: cute.Tensor,
            T: Int32,
            C: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mW, mU, mK, mV, mState, mWkv, mStateOut, T, C).launch(
                grid=[cute.ceil_div(C, _NUM_THREADS), 1, 1], block=[_NUM_THREADS, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mState: cute.Tensor,
            mWkv: cute.Tensor,
            mStateOut: cute.Tensor,
            T: Int32,
            C: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            channel = bid * _NUM_THREADS + tidx
            if channel < C:
                alpha = Float32(mState[channel])
                beta = Float32(mState[C + channel])
                epsilon = Float32(mState[2 * C + channel])
                w = Float32(mW[channel])
                u = Float32(mU[channel])
                state_stride = (T + 1) * C
                mStateOut[channel] = state_output_dtype(alpha)
                mStateOut[state_stride + channel] = state_output_dtype(beta)
                mStateOut[2 * state_stride + channel] = state_output_dtype(epsilon)
                for position in cutlass.range(0, T, 1, unroll=1):
                    offset = Int64(position) * C + channel
                    key = Float32(mK[offset])
                    value = Float32(mV[offset])
                    u_key = u + key
                    tau = cutlass.max(u_key, epsilon)
                    e1 = cute.math.exp(epsilon - tau, fastmath=False)
                    e2 = cute.math.exp(u_key - tau, fastmath=False)
                    mWkv[offset] = input_dtype((e1 * alpha + e2 * value) / (e1 * beta + e2))
                    w_epsilon = w + epsilon
                    next_epsilon = cutlass.max(w_epsilon, key)
                    e1 = cute.math.exp(w_epsilon - next_epsilon, fastmath=False)
                    e2 = cute.math.exp(key - next_epsilon, fastmath=False)
                    alpha = e1 * alpha + e2 * value
                    beta = e1 * beta + e2
                    epsilon = next_epsilon
                    state_offset = Int64(position + 1) * C + channel
                    mStateOut[state_offset] = state_output_dtype(input_dtype(alpha))
                    mStateOut[state_stride + state_offset] = state_output_dtype(input_dtype(beta))
                    mStateOut[2 * state_stride + state_offset] = state_output_dtype(input_dtype(epsilon))

    w_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    u_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = cute.runtime.make_fake_tensor(state_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    wkv_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_out_fake = cute.runtime.make_fake_tensor(state_output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        RWKV4Forward(),
        w_fake,
        u_fake,
        k_fake,
        v_fake,
        state_fake,
        wkv_fake,
        state_out_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_rwkv4_backward(input_dtype, state_dtype, grad_wkv_dtype, grad_state_dtype):
    class RWKV4Backward:
        @cute.jit
        def __call__(
            self,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mState: cute.Tensor,
            mGradWkv: cute.Tensor,
            mGradStateOut: cute.Tensor,
            mGradW: cute.Tensor,
            mGradU: cute.Tensor,
            mGradK: cute.Tensor,
            mGradV: cute.Tensor,
            mGradState: cute.Tensor,
            T: Int32,
            C: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mW, mU, mK, mV, mState, mGradWkv, mGradStateOut, mGradW, mGradU, mGradK, mGradV, mGradState, T, C
            ).launch(grid=[cute.ceil_div(C, _NUM_THREADS), 1, 1], block=[_NUM_THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mW: cute.Tensor,
            mU: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mState: cute.Tensor,
            mGradWkv: cute.Tensor,
            mGradStateOut: cute.Tensor,
            mGradW: cute.Tensor,
            mGradU: cute.Tensor,
            mGradK: cute.Tensor,
            mGradV: cute.Tensor,
            mGradState: cute.Tensor,
            T: Int32,
            C: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            channel = bid * _NUM_THREADS + tidx
            if channel < C:
                grad_alpha = Float32(mGradStateOut[channel])
                grad_beta = Float32(mGradStateOut[C + channel])
                grad_epsilon = Float32(mGradStateOut[2 * C + channel])
                w = Float32(mW[channel])
                u = Float32(mU[channel])
                grad_w = Float32(0.0)
                grad_u = Float32(0.0)
                state_stride = (T + 1) * C
                alpha_prev = Float32(mState[T * C + channel])
                beta_prev = Float32(mState[state_stride + T * C + channel])
                epsilon_prev = Float32(mState[2 * state_stride + T * C + channel])
                for reverse_position in cutlass.range(0, T, 1, unroll=1):
                    position = T - reverse_position - 1
                    offset = Int64(position) * C + channel
                    key = Float32(mK[offset])
                    value = Float32(mV[offset])
                    alpha_curr = alpha_prev
                    beta_curr = beta_prev
                    epsilon_curr = epsilon_prev
                    alpha_prev = Float32(mState[offset])
                    beta_prev = Float32(mState[state_stride + offset])
                    epsilon_prev = Float32(mState[2 * state_stride + offset])
                    u_key = u + key
                    tau = cutlass.max(u_key, epsilon_prev)
                    e1 = cute.math.exp(epsilon_prev - tau, fastmath=False)
                    e2 = cute.math.exp(u_key - tau, fastmath=False)
                    euke = cute.math.exp(u_key + epsilon_prev - Float32(2.0) * tau, fastmath=False)
                    denominator = e1 * beta_prev + e2
                    denominator_sq = denominator * denominator
                    grad_wkv = Float32(mGradWkv[offset])
                    grad_u_key = grad_wkv * e2 * (e1 * beta_prev * value - e1 * alpha_prev) / denominator_sq
                    grad_u += grad_u_key
                    grad_key = grad_u_key
                    grad_value = grad_wkv * e2 / denominator
                    grad_alpha_wkv = grad_wkv * e1 / denominator
                    grad_beta_wkv = -grad_wkv * e1 * (e2 * value + e1 * alpha_prev) / denominator_sq
                    grad_epsilon_wkv = grad_wkv * euke * (alpha_prev - value * beta_prev) / denominator_sq
                    e1 = cute.math.exp(w + epsilon_prev - epsilon_curr, fastmath=False)
                    e2 = cute.math.exp(key - epsilon_curr, fastmath=False)
                    grad_alpha_we = grad_alpha * e1 * alpha_prev
                    grad_w += grad_alpha_we
                    grad_key += grad_alpha * e2 * value
                    grad_value += grad_alpha * e2
                    grad_epsilon += grad_alpha * -alpha_curr
                    grad_beta_we = grad_beta * e1 * beta_prev
                    grad_w += grad_beta_we
                    grad_key += grad_beta * e2
                    grad_epsilon += grad_beta * -beta_curr
                    epsilon_from_w = w + epsilon_prev > key
                    grad_epsilon_we = grad_epsilon if epsilon_from_w else Float32(0.0)
                    grad_w += grad_epsilon_we
                    grad_key += Float32(0.0) if epsilon_from_w else grad_epsilon
                    mGradK[offset] = input_dtype(grad_key)
                    mGradV[offset] = input_dtype(grad_value)
                    grad_alpha = grad_alpha * e1 + grad_alpha_wkv
                    grad_beta = grad_beta * e1 + grad_beta_wkv
                    grad_epsilon = grad_alpha_we + grad_beta_we + grad_epsilon_we + grad_epsilon_wkv
                mGradState[channel] = input_dtype(grad_alpha)
                mGradState[C + channel] = input_dtype(grad_beta)
                mGradState[2 * C + channel] = input_dtype(grad_epsilon)
                mGradW[channel] = grad_w * w
                mGradU[channel] = grad_u

    w_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    u_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    state_fake = cute.runtime.make_fake_tensor(state_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_wkv_fake = cute.runtime.make_fake_tensor(grad_wkv_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_state_fake = cute.runtime.make_fake_tensor(grad_state_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_w_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_u_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    grad_initial_state_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        RWKV4Backward(),
        w_fake,
        u_fake,
        k_fake,
        v_fake,
        state_fake,
        grad_wkv_fake,
        grad_state_fake,
        grad_w_fake,
        grad_u_fake,
        grad_k_fake,
        grad_v_fake,
        grad_initial_state_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rwkv4_forward_cute(w, u, k, v, state):
    _, T, C = k.shape
    state_output_dtype = torch.promote_types(state.dtype, k.dtype)
    wkv = torch.empty_like(k)
    state_out = k.new_empty(1, 3, T + 1, C, dtype=state_output_dtype)
    compiled = _compile_rwkv4_forward(
        _torch_to_cute_dtype(k.dtype), _torch_to_cute_dtype(state.dtype), _torch_to_cute_dtype(state_output_dtype)
    )
    compiled(w.view(-1), u.view(-1), k.view(-1), v.view(-1), state.view(-1), wkv.view(-1), state_out.view(-1), T, C)
    return wkv, state_out


def rwkv4_backward_cute(w, u, k, v, state, grad_wkv, grad_state):
    _, T, C = k.shape
    grad_w, grad_u = w.new_empty(C), u.new_empty(C, dtype=torch.float32)
    grad_k, grad_v = torch.empty_like(k), torch.empty_like(v)
    grad_initial_state = k.new_empty(1, 3, 1, C)
    compiled = _compile_rwkv4_backward(
        _torch_to_cute_dtype(k.dtype),
        _torch_to_cute_dtype(state.dtype),
        _torch_to_cute_dtype(grad_wkv.dtype),
        _torch_to_cute_dtype(grad_state.dtype),
    )
    compiled(
        w.view(-1),
        u.view(-1),
        k.view(-1),
        v.view(-1),
        state.view(-1),
        grad_wkv.view(-1),
        grad_state.view(-1),
        grad_w.view(-1),
        grad_u.view(-1),
        grad_k.view(-1),
        grad_v.view(-1),
        grad_initial_state.view(-1),
        T,
        C,
    )
    return grad_w, grad_u, grad_k, grad_v, grad_initial_state


__all__ = ["rwkv4_backward_cute", "rwkv4_forward_cute"]
