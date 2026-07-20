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
from cutlass import Int32, Int64, const_expr


@cache
def _compile_packunpack(dtype, index_dtype, padding_left: bool, pack: bool, wide: bool):
    class PackUnpack:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mY: cute.Tensor,
            mCuSeqlens: cute.Tensor,
            padded_seqlen: Int32,
            width: Int32,
            stream: cuda.CUstream,
        ):
            if const_expr(wide):
                vecsize = 128 // dtype.width
                num_threads = 512
                copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
                tiled_copy = cute.make_tiled_copy_tv(
                    copy_atom,
                    cute.make_layout(num_threads),
                    cute.make_layout(vecsize),
                )
            else:
                num_threads = 256
                tiled_copy = None
            self.kernel(mX, mY, mCuSeqlens, padded_seqlen, width, tiled_copy).launch(
                grid=[padded_seqlen, mCuSeqlens.shape[0] - 1, cute.ceil_div(width, 4096)],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mY: cute.Tensor,
            mCuSeqlens: cute.Tensor,
            padded_seqlen: Int32,
            width: Int32,
            tiled_copy: cute.TiledCopy | None,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            padded_idx, batch_idx, width_tile = cute.arch.block_idx()
            bos = Int32(mCuSeqlens[batch_idx])
            eos = Int32(mCuSeqlens[batch_idx + 1])
            seqlen = eos - bos

            if const_expr(padding_left):
                valid = padded_idx >= padded_seqlen - seqlen
                token_idx = bos + padded_idx - (padded_seqlen - seqlen)
            else:
                valid = padded_idx < seqlen
                token_idx = bos + padded_idx

            if valid:
                padded_base = Int64(batch_idx * padded_seqlen + padded_idx) * width
                packed_base = Int64(token_idx) * width
                tile_start = width_tile * 4096
                if const_expr(wide):
                    vecsize = 128 // dtype.width
                    padded_base = cute.assume(padded_base, divby=vecsize)
                    packed_base = cute.assume(packed_base, divby=vecsize)
                    if const_expr(pack):
                        gX = cute.local_tile(cute.domain_offset((padded_base,), mX), (4096,), (width_tile,))
                        gY = cute.local_tile(cute.domain_offset((packed_base,), mY), (4096,), (width_tile,))
                    else:
                        gX = cute.local_tile(cute.domain_offset((packed_base,), mX), (4096,), (width_tile,))
                        gY = cute.local_tile(cute.domain_offset((padded_base,), mY), (4096,), (width_tile,))
                    thr_copy = tiled_copy.get_slice(tidx)
                    tXgX = thr_copy.partition_S(gX)
                    tXgY = thr_copy.partition_D(gY)
                    tXrX = cute.make_rmem_tensor_like(tXgX)
                    cute.copy(thr_copy, tXgX, tXrX)
                    cute.copy(thr_copy, tXrX, tXgY)
                else:
                    tile_end = min(tile_start + 4096, width)
                    for d in cutlass.range(tile_start + tidx, tile_end, 256, unroll=1):
                        if const_expr(pack):
                            mY[packed_base + d] = mX[padded_base + d]
                        else:
                            mY[padded_base + d] = mX[packed_base + d]

    x_fake = cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    y_fake = cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    cu_fake = cute.runtime.make_fake_tensor(index_dtype, (cute.sym_int(),), stride=(1,), assumed_align=4)
    return cute.compile(
        PackUnpack(),
        x_fake,
        y_fake,
        cu_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _torch_to_cute_dtype(dtype):
    return {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }[dtype]


def pack_sequence_fwdbwd_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
) -> torch.Tensor:
    batch, padded_seqlen = x.shape[:2]
    width = x.numel() // (batch * padded_seqlen)
    y = torch.empty(cu_seqlens[-1].item(), *x.shape[2:], device=x.device, dtype=x.dtype)
    compiled = _compile_packunpack(
        _torch_to_cute_dtype(x.dtype),
        _torch_to_cute_dtype(cu_seqlens.dtype),
        padding_side == "left",
        True,
        width >= 4096 and width % 4096 == 0,
    )
    compiled(x.view(-1), y.view(-1), cu_seqlens, padded_seqlen, width)
    return y


def unpack_sequence_fwdbwd_cute(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    desired_shape: torch.Size,
) -> torch.Tensor:
    y = torch.zeros(desired_shape, device=x.device, dtype=x.dtype)
    batch, padded_seqlen = y.shape[:2]
    width = y.numel() // (batch * padded_seqlen)
    compiled = _compile_packunpack(
        _torch_to_cute_dtype(x.dtype),
        _torch_to_cute_dtype(cu_seqlens.dtype),
        padding_side == "left",
        False,
        width >= 4096 and width % 4096 == 0,
    )
    compiled(x.view(-1), y.view(-1), cu_seqlens, padded_seqlen, width)
    return y


__all__ = ["pack_sequence_fwdbwd_cute", "unpack_sequence_fwdbwd_cute"]
