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
def _compile_dense_vector_cumsum(
    input_dtype,
    output_dtype,
    *,
    local: bool,
    reverse: bool,
    head_first: bool,
    has_scale: bool,
    chunk_size: int,
):
    """Compile a feature-parallel dense cumsum.

    Each CUDA thread owns one ``(batch, head, feature)`` lane and retains the
    FP32 carry for its time scan in a register. This is the bandwidth-friendly
    path for the rank-4 tensors used by the recurrent operators.
    """

    class DenseVectorCumsum:
        @cute.jit
        def __call__(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            S: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            num_threads = 256
            num_chunks = cute.ceil_div(T, chunk_size) if const_expr(local) else 1
            work = B * H * S * num_chunks
            self.kernel(mS, mO, B, T, H, S, scale).launch(
                grid=[cute.ceil_div(work, num_threads), 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mS: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            S: Int32,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            linear = bid * 256 + tidx
            lanes = B * H * S
            num_chunks = cute.ceil_div(T, chunk_size) if const_expr(local) else 1
            if linear < lanes * num_chunks:
                if const_expr(local):
                    chunk = linear // lanes
                    lane = linear - chunk * lanes
                    begin = chunk * chunk_size
                    end = min(begin + chunk_size, T)
                else:
                    lane = linear
                    begin = Int32(0)
                    end = T

                feature = lane % S
                lane = lane // S
                head = lane % H
                batch = lane // H
                length = end - begin
                carry = Float32(0.0)
                for pos in cutlass.range(0, length, 1, unroll=1):
                    token = end - 1 - pos if const_expr(reverse) else begin + pos
                    if const_expr(head_first):
                        offset = (Int64(batch * H + head) * T + token) * S + feature
                    else:
                        offset = (Int64(batch) * T * H + token * H + head) * S + feature
                    carry += Float32(mS[offset])
                    value = carry * scale if const_expr(has_scale) else carry
                    mO[offset] = value

    s_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    o_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        DenseVectorCumsum(),
        s_fake,
        o_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def dense_vector_cumsum_cute(
    s: torch.Tensor,
    *,
    local: bool,
    chunk_size: int = 0,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    out_dtype = output_dtype or s.dtype
    out = torch.empty_like(s, dtype=out_dtype)
    compiled = _compile_dense_vector_cumsum(
        _torch_to_cute_dtype(s.dtype),
        _torch_to_cute_dtype(out_dtype),
        local=local,
        reverse=reverse,
        head_first=head_first,
        has_scale=scale is not None,
        chunk_size=chunk_size,
    )
    compiled(s.view(-1), out.view(-1), B, T, H, S, 1.0 if scale is None else scale)
    return out


__all__ = ["dense_vector_cumsum_cute"]
