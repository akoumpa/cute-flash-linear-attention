# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_TILE_T = 16
_NUM_THREADS = 32


@cute.jit
def _get_smem_layout_atom(dtype, k_dim: int):
    """FlashAttention SM80 swizzled row-major layout for LdMatrix operands."""
    dtype_bytes = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_bytes)
    block_bytes = cutlass.const_expr(
        128 if bytes_per_row % 128 == 0 else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
    )
    block_elems = cutlass.const_expr(block_bytes // dtype_bytes)
    swizzle_bits = 4 if block_elems == 128 else (3 if block_elems == 64 else (2 if block_elems == 32 else 1))
    swizzle_base = 2 if dtype_bytes == 4 else (3 if dtype_bytes == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout((8 if cutlass.const_expr(k_dim % 32 == 0) else 16, block_elems), order=(1, 0)),
    )


@cute.jit
def _load_mma_a(thr_mma, tiled_copy, tidx: Int32, smem_tensor: cute.Tensor):
    thr_copy = tiled_copy.get_slice(tidx)
    source = thr_copy.partition_S(smem_tensor)
    fragment = thr_mma.make_fragment_A(thr_mma.partition_A(smem_tensor))
    destination = thr_copy.retile(fragment)
    for k_tile in cutlass.range_constexpr(cute.size(source.shape[2])):
        cute.copy(thr_copy, source[None, None, k_tile], destination[None, None, k_tile])
    return fragment


@cute.jit
def _load_mma_b(thr_mma, tiled_copy, tidx: Int32, smem_tensor: cute.Tensor):
    thr_copy = tiled_copy.get_slice(tidx)
    source = thr_copy.partition_S(smem_tensor)
    fragment = thr_mma.make_fragment_B(thr_mma.partition_B(smem_tensor))
    destination = thr_copy.retile(fragment)
    for k_tile in cutlass.range_constexpr(cute.size(source.shape[2])):
        cute.copy(thr_copy, source[None, None, k_tile], destination[None, None, k_tile])
    return fragment


@cute.jit
def _gemm(tiled_mma: cute.TiledMma, accumulator: cute.Tensor, fragment_a: cute.Tensor, fragment_b: cute.Tensor):
    for k_tile in cutlass.range_constexpr(cute.size(fragment_a.shape[2])):
        cute.gemm(
            tiled_mma,
            accumulator,
            fragment_a[None, None, k_tile],
            fragment_b[None, None, k_tile],
            accumulator,
        )


@cache
def _compile_chunk_o_fwd_mma(input_dtype, H: int, HV: int, K: int, V: int, *, state_v_first: bool):
    class ChunkOForwardMma:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mH: cute.Tensor,
            mO: cute.Tensor,
            B: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            # CuTe layout and MMA objects own MLIR values and must be created
            # while the JIT IR context is active, not in the cached Python
            # compile wrapper.
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(input_dtype, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                input_dtype,
            )
            smem_tiled_copy_a = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
            smem_tiled_copy_b = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
            qk_layout_atom = _get_smem_layout_atom(input_dtype, K)
            v_layout_atom = _get_smem_layout_atom(input_dtype, _TILE_T)
            sQ_layout = cute.tile_to_shape(qk_layout_atom, (_TILE_T, K), (0, 1))
            sK_layout = cute.tile_to_shape(qk_layout_atom, (_TILE_T, K), (0, 1))
            sH_layout = cute.tile_to_shape(qk_layout_atom, (V, K), (0, 1))
            sV_layout = cute.tile_to_shape(v_layout_atom, (V, _TILE_T), (0, 1))
            score_layout = cute.make_layout((_TILE_T, _TILE_T), stride=(_TILE_T, 1))
            smem_bytes = (_TILE_T * K * 2 + V * K + V * _TILE_T + _TILE_T * _TILE_T) * (input_dtype.width // 8)
            self.kernel(
                mQ,
                mK,
                mV,
                mH,
                mO,
                scale,
                tiled_mma,
                smem_tiled_copy_a,
                smem_tiled_copy_b,
                sQ_layout,
                sK_layout,
                sH_layout,
                sV_layout,
                score_layout,
            ).launch(
                grid=[B * HV, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mH: cute.Tensor,
            mO: cute.Tensor,
            scale: Float32,
            tiled_mma: cute.TiledMma,
            smem_tiled_copy_a: cute.TiledCopy,
            smem_tiled_copy_b: cute.TiledCopy,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sH_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            score_layout: cute.Layout,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            nhv, _, _ = cute.arch.block_idx()
            batch = nhv // HV
            value_head = nhv % HV
            query_head = value_head // (HV // H)

            gQ = mQ[batch, None, query_head, None]
            gK = mK[batch, None, query_head, None]
            gV_native = mV[batch, None, value_head, None]
            gH_native = mH[batch, 0, value_head, None, None]
            gO = mO[batch, None, value_head, None]

            smem = cutlass.utils.SmemAllocator()
            sQ = smem.allocate_tensor(input_dtype, sQ_layout, byte_alignment=128)
            sK = smem.allocate_tensor(input_dtype, sK_layout, byte_alignment=128)
            sH = smem.allocate_tensor(input_dtype, sH_layout, byte_alignment=128)
            sV = smem.allocate_tensor(input_dtype, sV_layout, byte_alignment=128)
            sP = smem.allocate_tensor(input_dtype, score_layout, byte_alignment=16)

            linear = tidx
            for _ in cutlass.range_constexpr((_TILE_T * K) // _NUM_THREADS):
                row = linear // K
                column = linear % K
                sQ[row, column] = gQ[row, column]
                sK[row, column] = gK[row, column]
                linear += _NUM_THREADS
            linear = tidx
            for _ in cutlass.range_constexpr((V * K) // _NUM_THREADS):
                value = linear // K
                key = linear % K
                if cutlass.const_expr(state_v_first):
                    sH[value, key] = gH_native[value, key]
                else:
                    sH[value, key] = gH_native[key, value]
                linear += _NUM_THREADS
            linear = tidx
            for _ in cutlass.range_constexpr((V * _TILE_T) // _NUM_THREADS):
                value = linear // _TILE_T
                token = linear % _TILE_T
                sV[value, token] = gV_native[token, value]
                linear += _NUM_THREADS
            cute.arch.sync_threads()

            thr_mma = tiled_mma.get_slice(tidx)
            rQ = _load_mma_a(thr_mma, smem_tiled_copy_a, tidx, sQ)

            rH = _load_mma_b(thr_mma, smem_tiled_copy_b, tidx, sH)
            output = cute.make_rmem_tensor(thr_mma.partition_shape_C((_TILE_T, V)), Float32)
            output.fill(0.0)
            _gemm(tiled_mma, output, rQ, rH)

            rK = _load_mma_b(thr_mma, smem_tiled_copy_b, tidx, sK)
            scores = cute.make_rmem_tensor(thr_mma.partition_shape_C((_TILE_T, _TILE_T)), Float32)
            scores.fill(0.0)
            _gemm(tiled_mma, scores, rQ, rK)

            rP = cute.make_fragment_like(scores, input_dtype)
            rP.store(scores.load().to(input_dtype))
            tPsP = thr_mma.partition_C(sP)
            cute.autovec_copy(rP, tPsP)
            cute.arch.sync_threads()
            linear = tidx
            for _ in cutlass.range_constexpr((_TILE_T * _TILE_T) // _NUM_THREADS):
                row = linear // _TILE_T
                column = linear % _TILE_T
                if column > row:
                    sP[row, column] = input_dtype(0.0)
                linear += _NUM_THREADS
            cute.arch.sync_threads()

            rP_mma = _load_mma_a(thr_mma, smem_tiled_copy_a, tidx, sP)
            rV = _load_mma_b(thr_mma, smem_tiled_copy_b, tidx, sV)
            _gemm(tiled_mma, output, rP_mma, rV)

            output.store(output.load() * scale)
            rO = cute.make_fragment_like(output, input_dtype)
            rO.store(output.load().to(input_dtype))
            tOgO = thr_mma.partition_C(gO)
            cute.autovec_copy(rO, tOgO)

    q_fake = cute.runtime.make_fake_tensor(
        input_dtype,
        (cute.sym_int(), _TILE_T, H, K),
        stride=(_TILE_T * H * K, H * K, K, 1),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        input_dtype,
        (cute.sym_int(), _TILE_T, HV, V),
        stride=(_TILE_T * HV * V, HV * V, V, 1),
        assumed_align=16,
    )
    h_shape = (cute.sym_int(), 1, HV, V, K) if state_v_first else (cute.sym_int(), 1, HV, K, V)
    h_stride = (HV * K * V, HV * K * V, K * V, K, 1) if state_v_first else (HV * K * V, HV * K * V, K * V, V, 1)
    h_fake = cute.runtime.make_fake_tensor(input_dtype, h_shape, stride=h_stride, assumed_align=16)
    return cute.compile(
        ChunkOForwardMma(),
        q_fake,
        q_fake,
        v_fake,
        h_fake,
        v_fake,
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_o_fwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool,
) -> torch.Tensor:
    """Run the dense 16-token native-MMA CuTe output kernel."""
    B, _, H, K = q.shape
    HV, V = v.shape[2:]
    o = torch.empty_like(v)
    compiled = _compile_chunk_o_fwd_mma(_torch_to_cute_dtype(q.dtype), H, HV, K, V, state_v_first=state_v_first)
    compiled(q, k, v, h, o, B, scale)
    return o


__all__ = ["chunk_o_fwd_cute"]
