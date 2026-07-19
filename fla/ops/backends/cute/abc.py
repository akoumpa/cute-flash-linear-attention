# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from fla.ops.backends.cute.op import exp
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_THREADS = 32


@cache
def _compile_abc_decode(input_dtype, K: int, V: int, M: int):
    class ABCDecode:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mS: cute.Tensor,
            mHK0: cute.Tensor,
            mHV0: cute.Tensor,
            mO: cute.Tensor,
            mHKT: cute.Tensor,
            mHVT: cute.Tensor,
            BH: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            slot_layout = cute.make_layout((M,), stride=(1,))
            self.kernel(mQ, mK, mV, mS, mHK0, mHV0, mO, mHKT, mHVT, scale, slot_layout).launch(
                grid=[BH, 1, 1], block=[_THREADS, 1, 1], smem=M * 6, stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mS: cute.Tensor,
            mHK0: cute.Tensor,
            mHV0: cute.Tensor,
            mO: cute.Tensor,
            mHKT: cute.Tensor,
            mHVT: cute.Tensor,
            scale: Float32,
            slot_layout: cute.Layout,
        ):
            lane, _, _ = cute.arch.thread_idx()
            bh, _, _ = cute.arch.block_idx()
            allocator = cutlass.utils.SmemAllocator()
            sOK = allocator.allocate_tensor(input_dtype, slot_layout, byte_alignment=16)
            sP = allocator.allocate_tensor(Float32, slot_layout, byte_alignment=16)
            qk_base = Int64(bh) * K
            hk_base = Int64(bh) * K * M
            hv_base = Int64(bh) * M * V
            value_base = Int64(bh) * V

            # For one token logcumsumexp(s) == s, so every normalized slot
            # weight is one. Preserve the chunk path's low-precision q scale
            # and separate inter/intra accumulations.
            intra_score = Float32(0.0)
            for key in cutlass.range_constexpr(K):
                q_scaled = input_dtype(Float32(mQ[qk_base + key]) * scale)
                intra_score += Float32(q_scaled) * Float32(mK[qk_base + key])
            for slot in cutlass.range(lane, M, _THREADS, unroll=1):
                inter_score = Float32(0.0)
                for key in cutlass.range_constexpr(K):
                    q_scaled = input_dtype(Float32(mQ[qk_base + key]) * scale)
                    state_offset = hk_base + key * M + slot
                    state = Float32(mHK0[state_offset])
                    # The chunk path materializes its pre-chunk state in the
                    # input dtype before this projection.
                    inter_score += Float32(q_scaled) * Float32(input_dtype(state))
                    mHKT[state_offset] = state + Float32(mK[qk_base + key])
                sOK[slot] = input_dtype(Float32(input_dtype(inter_score)) + Float32(input_dtype(intra_score)))
            cute.arch.sync_warp()

            if lane == 0:
                maximum = Float32(sOK[0])
                for slot in cutlass.range_constexpr(1, M):
                    value = Float32(sOK[slot])
                    maximum = value if value > maximum else maximum
                denominator = Float32(0.0)
                for slot in cutlass.range_constexpr(M):
                    probability = exp(Float32(sOK[slot]) - maximum)
                    sP[slot] = probability
                    denominator += probability
                for slot in cutlass.range_constexpr(M):
                    sP[slot] = sP[slot] / denominator
            cute.arch.sync_warp()

            for element in cutlass.range(lane, M * V, _THREADS, unroll=1):
                state_offset = hv_base + element
                value = element % V
                mHVT[state_offset] = Float32(mHV0[state_offset]) + Float32(mV[value_base + value])
            for value in cutlass.range(lane, V, _THREADS, unroll=1):
                inter_output = Float32(0.0)
                intra_output = Float32(0.0)
                token_value = Float32(mV[value_base + value])
                for slot in cutlass.range_constexpr(M):
                    probability = Float32(input_dtype(sP[slot]))
                    inter_output += probability * Float32(input_dtype(mHV0[hv_base + slot * V + value]))
                    intra_output += probability * token_value
                mO[value_base + value] = input_dtype(inter_output + intra_output)

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    s_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    hk0_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    hv0_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    hkt_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    hvt_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        ABCDecode(),
        q_fake,
        k_fake,
        v_fake,
        s_fake,
        hk0_fake,
        hv0_fake,
        output_fake,
        hkt_fake,
        hvt_fake,
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def abc_decode_cute(q, k, v, s, initial_state):
    B, _, H, K = q.shape
    M, V = s.shape[-1], v.shape[-1]
    output = torch.empty_like(v)
    final_state = (torch.empty_like(initial_state[0]), torch.empty_like(initial_state[1]))
    compiled = _compile_abc_decode(_torch_to_cute_dtype(q.dtype), K, V, M)
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        s.view(-1),
        initial_state[0].view(-1),
        initial_state[1].view(-1),
        output.view(-1),
        final_state[0].view(-1),
        final_state[1].view(-1),
        B * H,
        K**-0.5,
    )
    return output, final_state


__all__ = ["abc_decode_cute"]
