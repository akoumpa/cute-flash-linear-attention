# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# This source code is licensed under the MIT license found in the LICENSE file.

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.op import exp2, log2
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_TILE, _D, _NT, _THREADS = 16, 64, 4, 32
_RCP_LN2 = 1.4426950216


@cache
def _compile_parallel_attn_fwd(dtype):
    class Attention:
        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            lse: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            tile = cute.make_layout((_TILE, _TILE), stride=(_TILE, 1))
            scores = cute.make_layout((_TILE, _D), stride=(_D, 1))
            rows = cute.make_layout((_TILE,), stride=(1,))
            mma = cute.make_tiled_mma(warp.MmaF16BF16Op(dtype, Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16))
            self.kernel(q, k, v, o, lse, T, H, scale, tile, scores, rows, mma).launch(
                grid=[B * H * _NT, 1, 1], block=[_THREADS, 1, 1], smem=8768, stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            lse: cute.Tensor,
            T: Int32,
            H: Int32,
            scale: Float32,
            tile: cute.Layout,
            scores: cute.Layout,
            rows: cute.Layout,
            mma: cute.TiledMma,
        ):
            lane, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            qt, bh = block % _NT, block // _NT
            batch, head = bh // H, bh % H
            alloc = cutlass.utils.SmemAllocator()
            sQ = alloc.allocate_tensor(dtype, tile, byte_alignment=16)
            sK = alloc.allocate_tensor(dtype, tile, byte_alignment=16)
            sVt = alloc.allocate_tensor(dtype, tile, byte_alignment=16)
            sP = alloc.allocate_tensor(dtype, scores, byte_alignment=16)
            sS = alloc.allocate_tensor(Float32, scores, byte_alignment=16)
            sTile = alloc.allocate_tensor(Float32, tile, byte_alignment=16)
            sDenom = alloc.allocate_tensor(Float32, rows, byte_alignment=16)
            thr = mma.get_slice(lane)

            for nt in cutlass.range_constexpr(_NT):
                acc = cute.make_rmem_tensor(thr.partition_shape_C((_TILE, _TILE)), Float32)
                acc.fill(0.0)
                for dt in cutlass.range_constexpr(_NT):
                    for e in cutlass.range(lane, _TILE * _TILE, _THREADS, unroll=1):
                        row, col = e // _TILE, e % _TILE
                        qi, ki, d = qt * _TILE + row, nt * _TILE + row, dt * _TILE + col
                        qo = ((Int64(batch) * T + qi) * H + head) * _D + d
                        ko = ((Int64(batch) * T + ki) * H + head) * _D + d
                        sQ[row, col] = q[qo] if qi < T else dtype(0.0)
                        sK[row, col] = k[ko] if ki < T else dtype(0.0)
                    cute.arch.sync_warp()
                    rQ, rK = thr.make_fragment_A(thr.partition_A(sQ)), thr.make_fragment_B(thr.partition_B(sK))
                    cute.autovec_copy(thr.partition_A(sQ), rQ)
                    cute.autovec_copy(thr.partition_B(sK), rK)
                    cute.gemm(mma, acc, rQ, rK, acc)
                    cute.arch.sync_warp()
                cute.autovec_copy(acc, thr.partition_C(sTile))
                cute.arch.sync_warp()
                for e in cutlass.range(lane, _TILE * _TILE, _THREADS, unroll=1):
                    row, col = e // _TILE, e % _TILE
                    sS[row, nt * _TILE + col] = sTile[row, col] * scale * Float32(_RCP_LN2)
                cute.arch.sync_warp()

            for row in cutlass.range(lane, _TILE, _THREADS, unroll=1):
                qi = qt * _TILE + row
                if qi < T:
                    maximum = Float32(-1.0e30)
                    for key in cutlass.range_constexpr(_D):
                        score = sS[row, key] if key <= qi and key < T else Float32(-1.0e30)
                        maximum = score if score > maximum else maximum
                    denominator = Float32(0.0)
                    for key in cutlass.range_constexpr(_D):
                        p = exp2(sS[row, key] - maximum) if key <= qi and key < T else Float32(0.0)
                        sP[row, key], denominator = dtype(p), denominator + p
                    sDenom[row] = denominator
                    lse[(Int64(batch) * T + qi) * H + head] = maximum + log2(denominator)
            cute.arch.sync_warp()

            for vt in cutlass.range_constexpr(_NT):
                acc = cute.make_rmem_tensor(thr.partition_shape_C((_TILE, _TILE)), Float32)
                acc.fill(0.0)
                for nt in cutlass.range_constexpr(_NT):
                    for e in cutlass.range(lane, _TILE * _TILE, _THREADS, unroll=1):
                        row, col = e // _TILE, e % _TILE
                        key, value = nt * _TILE + col, vt * _TILE + row
                        vo = ((Int64(batch) * T + key) * H + head) * _D + value
                        sQ[row, col] = sP[row, key]
                        sVt[row, col] = v[vo] if key < T else dtype(0.0)
                    cute.arch.sync_warp()
                    rP, rV = thr.make_fragment_A(thr.partition_A(sQ)), thr.make_fragment_B(thr.partition_B(sVt))
                    cute.autovec_copy(thr.partition_A(sQ), rP)
                    cute.autovec_copy(thr.partition_B(sVt), rV)
                    cute.gemm(mma, acc, rP, rV, acc)
                    cute.arch.sync_warp()
                cute.autovec_copy(acc, thr.partition_C(sTile))
                cute.arch.sync_warp()
                for e in cutlass.range(lane, _TILE * _TILE, _THREADS, unroll=1):
                    row, col = e // _TILE, e % _TILE
                    qi, value = qt * _TILE + row, vt * _TILE + col
                    if qi < T:
                        oo = ((Int64(batch) * T + qi) * H + head) * _D + value
                        o[oo] = dtype(sTile[row, col] / sDenom[row])

    x = cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    z = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        Attention(),
        x,
        x,
        x,
        x,
        z,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def parallel_attn_fwd_cute(q, k, v, scale):
    B, T, H, _ = q.shape
    o, lse = torch.empty_like(v), torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    fn = _compile_parallel_attn_fwd(_torch_to_cute_dtype(q.dtype))
    fn(q.view(-1), k.view(-1), v.view(-1), o.view(-1), lse.view(-1), B, T, H, scale)
    return o, lse


__all__ = ["parallel_attn_fwd_cute"]
