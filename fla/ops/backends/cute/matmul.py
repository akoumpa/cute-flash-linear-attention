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
_WARPS_PER_CTA = _NUM_THREADS // 32


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for offset in (16, 8, 4, 2, 1):
        value += cute.arch.shuffle_sync_down(value, offset)
    return value


@cache
def _compile_narrow_matmul(
    input_dtype,
    output_dtype,
    activation: str,
    *,
    has_input: bool,
    input_ndim: int,
    has_alpha: bool,
    has_beta: bool,
):
    class NarrowMatmul:
        @cute.jit
        def __call__(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mX: cute.Tensor | None,
            B: Int32,
            M: Int32,
            N: Int32,
            K: Int32,
            alpha: Float32,
            beta: Float32,
            stream: cuda.CUstream,
        ):
            output_count = B * M * N
            self.kernel(mA, mB, mC, mX, M, N, K, alpha, beta, output_count).launch(
                grid=[cute.ceil_div(output_count, _WARPS_PER_CTA), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mX: cute.Tensor | None,
            M: Int32,
            N: Int32,
            K: Int32,
            alpha: Float32,
            beta: Float32,
            output_count: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            output_idx = bid * _WARPS_PER_CTA + warp

            if output_idx < output_count:
                n = output_idx % N
                batch_row = output_idx // N
                m = batch_row % M
                a_base = Int64(batch_row) * K

                value = Float32(0.0)
                for k in cutlass.range(lane, K, 32, unroll=1):
                    value += Float32(mA[a_base + k]) * Float32(mB[Int64(k) * N + n])
                value = _warp_sum(value)

                if lane == 0:
                    if const_expr(activation == "leaky_relu"):
                        if value < Float32(0.0):
                            value *= Float32(0.01)
                    elif const_expr(activation == "relu"):
                        if value < Float32(0.0):
                            value = Float32(0.0)

                    if const_expr(has_alpha):
                        value *= alpha

                    if const_expr(has_input):
                        x_idx = n if const_expr(input_ndim == 1) else Int64(m) * N + n
                        x = Float32(mX[x_idx])
                        if const_expr(has_beta):
                            x *= beta
                        value += x

                    mC[output_idx] = output_dtype(value)

    a_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    b_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    c_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    x_fake = (
        cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16) if has_input else None
    )
    return cute.compile(
        NarrowMatmul(),
        a_fake,
        b_fake,
        c_fake,
        x_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def narrow_matmul_cute(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    activation: str = "",
    x: torch.Tensor | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> torch.Tensor:
    """Run the CuTe warp-reduction path for narrow-output dense matrix multiplication."""
    a_dim = a.dim()
    if a_dim == 2:
        a = a.unsqueeze(0)
    B, M, K = a.shape
    N = b.shape[1]
    output = a.new_empty(B, M, N)
    compiled = _compile_narrow_matmul(
        _torch_to_cute_dtype(a.dtype),
        _torch_to_cute_dtype(output.dtype),
        activation,
        has_input=x is not None,
        input_ndim=x.dim() if x is not None else 0,
        has_alpha=alpha is not None,
        has_beta=beta is not None,
    )
    compiled(
        a.view(-1),
        b.view(-1),
        output.view(-1),
        x.view(-1) if x is not None else None,
        B,
        M,
        N,
        K,
        alpha if alpha is not None else 1.0,
        beta if beta is not None else 1.0,
    )
    return output.squeeze(0) if a_dim == 2 else output


__all__ = ["narrow_matmul_cute"]
