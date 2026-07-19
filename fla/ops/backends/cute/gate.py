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

_BLOCK_SIZE = 2048
_NUM_THREADS = 256
_ELEMENTS_PER_THREAD = _BLOCK_SIZE // _NUM_THREADS


@cache
def _compile_beta_sigmoid_fwd(input_dtype):
    class BetaSigmoidForward:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mY: cute.Tensor,
            scale: Float32,
            n_elements: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mX, mY, scale, n_elements).launch(
                grid=[cute.ceil_div(n_elements, _BLOCK_SIZE), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mY: cute.Tensor,
            scale: Float32,
            n_elements: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            block_base = Int64(bid) * _BLOCK_SIZE

            for item in cutlass.range_constexpr(_ELEMENTS_PER_THREAD):
                offset = block_base + tidx + item * _NUM_THREADS
                if offset < n_elements:
                    x = Float32(mX[offset])
                    sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=False))
                    mY[offset] = scale * sigmoid

    x_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    y_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        BetaSigmoidForward(),
        x_fake,
        y_fake,
        Float32(1.0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_beta_sigmoid_bwd(input_dtype, grad_dtype):
    class BetaSigmoidBackward:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mDY: cute.Tensor,
            mDX: cute.Tensor,
            scale: Float32,
            n_elements: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mX, mDY, mDX, scale, n_elements).launch(
                grid=[cute.ceil_div(n_elements, _BLOCK_SIZE), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mDY: cute.Tensor,
            mDX: cute.Tensor,
            scale: Float32,
            n_elements: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            block_base = Int64(bid) * _BLOCK_SIZE

            for item in cutlass.range_constexpr(_ELEMENTS_PER_THREAD):
                offset = block_base + tidx + item * _NUM_THREADS
                if offset < n_elements:
                    x = Float32(mX[offset])
                    dy = Float32(mDY[offset])
                    sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=False))
                    mDX[offset] = input_dtype(dy * scale * sigmoid * (Float32(1.0) - sigmoid))

    x_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    dy_fake = cute.runtime.make_fake_tensor(grad_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    dx_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        BetaSigmoidBackward(),
        x_fake,
        dy_fake,
        dx_fake,
        Float32(1.0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fused_beta_sigmoid_fwd_cute(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    y = torch.empty_like(x, dtype=torch.float32)
    compiled = _compile_beta_sigmoid_fwd(_torch_to_cute_dtype(x.dtype))
    compiled(x.view(-1), y.view(-1), scale, x.numel())
    return y


def fused_beta_sigmoid_bwd_cute(x: torch.Tensor, dy: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    dx = torch.empty_like(x)
    compiled = _compile_beta_sigmoid_bwd(_torch_to_cute_dtype(x.dtype), _torch_to_cute_dtype(dy.dtype))
    compiled(x.view(-1), dy.view(-1), dx.view(-1), scale, x.numel())
    return dx


__all__ = ["fused_beta_sigmoid_bwd_cute", "fused_beta_sigmoid_fwd_cute"]
