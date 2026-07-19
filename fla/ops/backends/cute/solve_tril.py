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
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import warp

from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_BLOCK_SIZE = 32
_DIAG_SIZE = 16
_NUM_THREADS = 32


@cache
def _compile_solve_tril_32(input_dtype, output_dtype):
    class SolveTril32:
        @cute.jit
        def __call__(
            self,
            mA: cute.Tensor,
            mOut: cute.Tensor,
            B: Int32,
            T: Int32,
            H: Int32,
            stream: cuda.CUstream,
        ):
            mma_layout = cute.make_layout((_DIAG_SIZE, _DIAG_SIZE), stride=(_DIAG_SIZE, 1))
            fp32_layout = cute.make_layout((_DIAG_SIZE, _DIAG_SIZE), stride=(_DIAG_SIZE, 1))
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(output_dtype, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            mma_storage = cute.struct.Align[cute.struct.MemRange[output_dtype, cute.cosize(mma_layout)], 128]
            fp32_storage = cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(fp32_layout)], 128]

            @cute.struct
            class SharedStorage:
                inv11_fp32: fp32_storage
                inv22_fp32: fp32_storage
                inv11_transposed: mma_storage
                inv22: mma_storage
                a21: mma_storage
                product: mma_storage
                product_transposed: mma_storage
                result: mma_storage

            chunks = T // _BLOCK_SIZE
            self.kernel(mA, mOut, T, H, chunks, mma_layout, fp32_layout, tiled_mma, SharedStorage).launch(
                grid=[B * H * chunks, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=SharedStorage.size_in_bytes(),
                stream=stream,
            )

        @cute.jit
        def mma_to_smem(
            self,
            sA: cute.Tensor,
            sB: cute.Tensor,
            sC: cute.Tensor,
            tidx: Int32,
            tiled_mma: cute.TiledMma,
        ):
            thr_mma = tiled_mma.get_slice(tidx)
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
            accumulator = cute.make_rmem_tensor(thr_mma.partition_shape_C((_DIAG_SIZE, _DIAG_SIZE)), Float32)
            accumulator.fill(0.0)

            copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                output_dtype,
            )
            copy_a = cute.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tidx)
            tCsA = copy_a.partition_S(sA)
            tCsB = copy_b.partition_S(sB)
            tCrA_copy = copy_a.retile(tCrA)
            tCrB_copy = copy_b.retile(tCrB)

            for k_block in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                cute.copy(copy_a, tCsA[None, None, k_block], tCrA_copy[None, None, k_block])
                cute.copy(copy_b, tCsB[None, None, k_block], tCrB_copy[None, None, k_block])
                cute.gemm(
                    tiled_mma,
                    accumulator,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    accumulator,
                )

            output_fragment = cute.make_fragment_like(accumulator, output_dtype)
            output_fragment.store(accumulator.load().to(output_dtype))
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                output_dtype,
                num_bits_per_copy=2 * output_dtype.width,
            )
            store_copy = cute.make_tiled_copy_C(store_atom, tiled_mma).get_slice(tidx)
            cute.copy(store_atom, store_copy.retile(output_fragment), store_copy.partition_D(sC))

        @cute.kernel
        def kernel(
            self,
            mA: cute.Tensor,
            mOut: cute.Tensor,
            T: Int32,
            H: Int32,
            chunks: Int32,
            mma_layout: cute.Layout,
            fp32_layout: cute.Layout,
            tiled_mma: cute.TiledMma,
            SharedStorage: cutlass.Constexpr,
        ):
            lane, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            chunk = block % chunks
            bh = block // chunks
            head = bh % H
            batch = bh // H
            row_base = Int64(batch) * T + chunk * _BLOCK_SIZE
            input_base = (row_base * H + head) * _BLOCK_SIZE

            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(SharedStorage)
            s_inv11_fp32 = storage.inv11_fp32.get_tensor(fp32_layout)
            s_inv22_fp32 = storage.inv22_fp32.get_tensor(fp32_layout)
            s_inv11_transposed = storage.inv11_transposed.get_tensor(mma_layout)
            s_inv22 = storage.inv22.get_tensor(mma_layout)
            s_a21 = storage.a21.get_tensor(mma_layout)
            s_product = storage.product.get_tensor(mma_layout)
            s_product_transposed = storage.product_transposed.get_tensor(mma_layout)
            s_result = storage.result.get_tensor(mma_layout)

            for element in cutlass.range(lane, _DIAG_SIZE * _DIAG_SIZE, _NUM_THREADS, unroll=1):
                row = element // _DIAG_SIZE
                col = element % _DIAG_SIZE
                a21_offset = (Int64(row + _DIAG_SIZE) * H) * _BLOCK_SIZE + col
                s_inv11_fp32[row, col] = Float32(0.0)
                s_inv22_fp32[row, col] = Float32(0.0)
                s_inv11_transposed[row, col] = output_dtype(0.0)
                s_inv22[row, col] = output_dtype(0.0)
                s_a21[row, col] = output_dtype(mA[input_base + a21_offset])
            cute.arch.sync_warp()

            for row in cutlass.range_constexpr(_DIAG_SIZE):
                value11 = Float32(0.0)
                value22 = Float32(0.0)
                if lane == row:
                    value11 = Float32(1.0)
                    value22 = Float32(1.0)
                elif lane < row:
                    offset11 = (Int64(row) * H) * _BLOCK_SIZE + lane
                    offset22 = (Int64(row + _DIAG_SIZE) * H) * _BLOCK_SIZE + lane + _DIAG_SIZE
                    value11 = -Float32(mA[input_base + offset11])
                    value22 = -Float32(mA[input_base + offset22])
                    for inner in cutlass.range_constexpr(_DIAG_SIZE):
                        if lane < inner and inner < row:
                            a11 = Float32(mA[input_base + (Int64(row) * H) * _BLOCK_SIZE + inner])
                            a22 = Float32(mA[input_base + (Int64(row + _DIAG_SIZE) * H) * _BLOCK_SIZE + inner + _DIAG_SIZE])
                            value11 -= a11 * s_inv11_fp32[inner, lane]
                            value22 -= a22 * s_inv22_fp32[inner, lane]
                if lane <= row:
                    s_inv11_fp32[row, lane] = value11
                    s_inv22_fp32[row, lane] = value22
                    s_inv11_transposed[lane, row] = output_dtype(value11)
                    s_inv22[row, lane] = output_dtype(value22)
                cute.arch.sync_warp()

            self.mma_to_smem(s_a21, s_inv11_transposed, s_product, lane, tiled_mma)
            cute.arch.sync_warp()

            for element in cutlass.range(lane, _DIAG_SIZE * _DIAG_SIZE, _NUM_THREADS, unroll=1):
                row = element // _DIAG_SIZE
                col = element % _DIAG_SIZE
                s_product_transposed[col, row] = s_product[row, col]
            cute.arch.sync_warp()

            self.mma_to_smem(s_inv22, s_product_transposed, s_result, lane, tiled_mma)
            cute.arch.sync_warp()

            output_base = input_base
            for element in cutlass.range(lane, _BLOCK_SIZE * _BLOCK_SIZE, _NUM_THREADS, unroll=1):
                row = element // _BLOCK_SIZE
                col = element % _BLOCK_SIZE
                value = Float32(0.0)
                if row < _DIAG_SIZE and col < _DIAG_SIZE:
                    value = s_inv11_fp32[row, col]
                elif row >= _DIAG_SIZE and col >= _DIAG_SIZE:
                    value = s_inv22_fp32[row - _DIAG_SIZE, col - _DIAG_SIZE]
                elif row >= _DIAG_SIZE and col < _DIAG_SIZE:
                    value = -Float32(s_result[row - _DIAG_SIZE, col])
                output_offset = (Int64(row) * H) * _BLOCK_SIZE + col
                mOut[output_base + output_offset] = output_dtype(value)

    input_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(output_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        SolveTril32(),
        input_fake,
        output_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def solve_tril_32_cute(A: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Invert dense 32-by-32 unit-lower blocks with CuTe warp MMA."""
    B, T, H, _ = A.shape
    output = torch.empty_like(A, dtype=output_dtype)
    compiled = _compile_solve_tril_32(_torch_to_cute_dtype(A.dtype), _torch_to_cute_dtype(output_dtype))
    compiled(A.view(-1), output.view(-1), B, T, H)
    return output


__all__ = ["solve_tril_32_cute"]
