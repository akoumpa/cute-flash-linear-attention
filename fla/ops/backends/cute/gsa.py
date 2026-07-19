# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr

from fla.ops.backends.cute.op import exp
from fla.ops.backends.cute.pack import _torch_to_cute_dtype

_K = 64
_V = 64
_M = 32
_NUM_THREADS = 64


@cache
def _compile_gsa_decode(input_dtype, gate_dtype, *, use_initial_state: bool, store_final_state: bool):
    class GSADecode:
        def _get_shared_storage_cls(self):
            vector_struct = cute.struct.Align[cute.struct.MemRange[Float32, _M], 128]

            @cute.struct
            class SharedStorage:
                logits: vector_struct
                probabilities: vector_struct

            return SharedStorage

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mS: cute.Tensor,
            mG: cute.Tensor,
            mO: cute.Tensor,
            mHK0: cute.Tensor | None,
            mHV0: cute.Tensor | None,
            mHKT: cute.Tensor | None,
            mHVT: cute.Tensor | None,
            B: Int32,
            H: Int32,
            HQ: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            SharedStorage = self._get_shared_storage_cls()
            self.kernel(mQ, mK, mV, mS, mG, mO, mHK0, mHV0, mHKT, mHVT, H, HQ, scale, SharedStorage).launch(
                grid=[B * HQ, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=SharedStorage.size_in_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mS: cute.Tensor,
            mG: cute.Tensor,
            mO: cute.Tensor,
            mHK0: cute.Tensor | None,
            mHV0: cute.Tensor | None,
            mHKT: cute.Tensor | None,
            mHVT: cute.Tensor | None,
            H: Int32,
            HQ: Int32,
            scale: Float32,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bhq, _, _ = cute.arch.block_idx()
            batch, query_head = bhq // HQ, bhq % HQ
            group_size = HQ // H
            head = query_head // group_size
            bh = batch * H + head

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            layout = cute.make_layout((_M,), stride=(1,))
            logits = storage.logits.get_tensor(layout)
            probabilities = storage.probabilities.get_tensor(layout)

            if tidx < _M:
                slot = Float32(mS[bh * _M + tidx])
                decay = exp(Float32(mG[bh * _M + tidx]))
                logit = Float32(0.0)
                for key_idx in cutlass.range_constexpr(_K):
                    state_idx = (Int64(bh) * _K + key_idx) * _M + tidx
                    state = Float32(mHK0[state_idx]) if const_expr(use_initial_state) else Float32(0.0)
                    state = state * decay + Float32(mK[bh * _K + key_idx]) * slot
                    logit += state * Float32(mQ[bhq * _K + key_idx]) * scale
                    if const_expr(store_final_state):
                        if query_head % group_size == 0:
                            mHKT[state_idx] = state
                logits[tidx] = logit
            cute.arch.barrier()

            if tidx == 0:
                maximum = Float32(float("-inf"))
                for slot_idx in cutlass.range_constexpr(_M):
                    value = logits[slot_idx]
                    maximum = value if value > maximum else maximum
                denominator = Float32(0.0)
                for slot_idx in cutlass.range_constexpr(_M):
                    probability = exp(logits[slot_idx] - maximum)
                    probabilities[slot_idx] = probability
                    denominator += probability
                for slot_idx in cutlass.range_constexpr(_M):
                    probabilities[slot_idx] /= denominator
            cute.arch.barrier()

            value = Float32(mV[bh * _V + tidx])
            output = Float32(0.0)
            for slot_idx in cutlass.range_constexpr(_M):
                state_idx = (Int64(bh) * _M + slot_idx) * _V + tidx
                state = Float32(mHV0[state_idx]) if const_expr(use_initial_state) else Float32(0.0)
                decay = exp(Float32(mG[bh * _M + slot_idx]))
                state = state * decay + Float32(mS[bh * _M + slot_idx]) * value
                output += probabilities[slot_idx] * state
                if const_expr(store_final_state):
                    if query_head % group_size == 0:
                        mHVT[state_idx] = state
            mO[bhq * _V + tidx] = input_dtype(output)

    q_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    k_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    v_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    slot_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    gate_fake = cute.runtime.make_fake_tensor(gate_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    output_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    hk_state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    hv_state_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if use_initial_state else None
    )
    hkt_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    hvt_fake = (
        cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16) if store_final_state else None
    )
    return cute.compile(
        GSADecode(),
        q_fake,
        k_fake,
        v_fake,
        slot_fake,
        gate_fake,
        output_fake,
        hk_state_fake,
        hv_state_fake,
        hkt_fake,
        hvt_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def gsa_decode_cute(q, k, v, s, g, initial_state, output_final_state, scale):
    """Run the bounded one-token GSA inference recurrence."""
    B, _, H, _ = k.shape
    HQ = q.shape[2]
    hk0, hv0 = initial_state
    output = v.new_empty(B, 1, HQ, _V)
    hkt = q.new_empty((B, H, _K, _M), dtype=torch.float32) if output_final_state else None
    hvt = q.new_empty((B, H, _M, _V), dtype=torch.float32) if output_final_state else None
    compiled = _compile_gsa_decode(
        _torch_to_cute_dtype(q.dtype),
        _torch_to_cute_dtype(g.dtype),
        use_initial_state=hk0 is not None,
        store_final_state=output_final_state,
    )
    compiled(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        s.view(-1),
        g.view(-1),
        output.view(-1),
        hk0.view(-1) if hk0 is not None else None,
        hv0.view(-1) if hv0 is not None else None,
        hkt.view(-1) if hkt is not None else None,
        hvt.view(-1) if hvt is not None else None,
        B,
        H,
        HQ,
        scale,
    )
    return output, [hkt, hvt]


__all__ = ["gsa_decode_cute"]
