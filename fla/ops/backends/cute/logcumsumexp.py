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


@cute.jit
def _logaddexp_pair(
    lhs_max: Float32,
    lhs_sum: Float32,
    rhs_max: Float32,
    rhs_sum: Float32,
    *,
    use_fast_ops: cutlass.Constexpr,
):
    """Compose two adjacent normalized log-sum-exp segments."""
    out_max = lhs_max
    out_sum = lhs_sum
    positive_inf = Float32(float("inf"))
    negative_inf = Float32(float("-inf"))

    # Spell out the non-finite cases.  The usual normalized composition has
    # inf - inf and -inf - -inf indeterminacies, even though the corresponding
    # log-sum-exp is well defined.  NaNs deliberately remain sticky.
    if lhs_max != lhs_max:
        out_max = lhs_max
        out_sum = lhs_sum
    elif rhs_max != rhs_max:
        out_max = rhs_max
        out_sum = rhs_sum
    elif lhs_max == positive_inf:
        out_max = lhs_max
        out_sum = lhs_sum
    elif rhs_max == positive_inf:
        out_max = rhs_max
        out_sum = rhs_sum
    elif lhs_max == negative_inf:
        out_max = rhs_max
        out_sum = rhs_sum
        if rhs_max == negative_inf:
            out_sum = lhs_sum + rhs_sum
    elif rhs_max == negative_inf:
        out_max = lhs_max
        out_sum = lhs_sum
    elif rhs_max > lhs_max:
        out_max = rhs_max
        out_sum = lhs_sum * cute.math.exp(lhs_max - rhs_max, fastmath=use_fast_ops) + rhs_sum
    else:
        out_max = lhs_max
        out_sum = lhs_sum + rhs_sum * cute.math.exp(rhs_max - lhs_max, fastmath=use_fast_ops)
    return out_max, out_sum


@cache
def _compile_logcumsumexp(input_dtype, *, use_fast_ops: bool):
    class LogCumSumExpForward:
        @cute.jit
        def __call__(
            self,
            mS: cute.Tensor,
            mZ: cute.Tensor,
            N: Int32,
            T: Int32,
            S: Int32,
            stream: cuda.CUstream,
        ):
            work = N * S
            # Eight independent feature lanes (one warp each) per CTA.
            self.cooperative_kernel(mS, mZ, T, S, work).launch(
                grid=[cute.ceil_div(work, 8), 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def cooperative_kernel(
            self,
            mS: cute.Tensor,
            mZ: cute.Tensor,
            T: Int32,
            S: Int32,
            work: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            feature_lane = bid * 8 + warp

            if feature_lane < work:
                feature = feature_lane % S
                row = feature_lane // S
                base = Int64(row) * T * S + feature
                carry_max = Float32(float("-inf"))
                carry_sum = Float32(0.0)

                # Each warp scans a 32-timestep tile.  Its lane-31 prefix is
                # broadcast as the carry for the following tile, so arbitrary
                # sequence lengths require no CTA-wide synchronization.
                for tile_start in cutlass.range(0, T, 32, unroll=1):
                    pos = tile_start + lane
                    value_max = Float32(float("-inf"))
                    value_sum = Float32(0.0)
                    if pos < T:
                        value_max = Float32(mS[base + Int64(pos) * S])
                        value_sum = Float32(1.0)

                    for offset in (1, 2, 4, 8, 16):
                        # For a full-width ``shfl.up``, the packed segment-mask
                        # and clamp operand is zero (not the idx/down default).
                        lhs_max = cute.arch.shuffle_sync_up(value_max, offset, mask_and_clamp=0)
                        lhs_sum = cute.arch.shuffle_sync_up(value_sum, offset, mask_and_clamp=0)
                        if lane >= offset:
                            value_max, value_sum = _logaddexp_pair(
                                lhs_max,
                                lhs_sum,
                                value_max,
                                value_sum,
                                use_fast_ops=use_fast_ops,
                            )

                    value_max, value_sum = _logaddexp_pair(
                        carry_max,
                        carry_sum,
                        value_max,
                        value_sum,
                        use_fast_ops=use_fast_ops,
                    )
                    if pos < T:
                        mZ[base + Int64(pos) * S] = cute.math.log(value_sum, fastmath=use_fast_ops) + value_max

                    carry_max = cute.arch.shuffle_sync(value_max, 31)
                    carry_sum = cute.arch.shuffle_sync(value_sum, 31)

    s_fake = cute.runtime.make_fake_tensor(input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
    z_fake = cute.runtime.make_fake_tensor(Float32, (cute.sym_int(),), stride=(1,), assumed_align=16)
    return cute.compile(
        LogCumSumExpForward(),
        s_fake,
        z_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def logcumsumexp_fwd_cute(s: torch.Tensor, *, use_fast_ops: bool = False) -> torch.Tensor:
    shape = s.shape
    T, S = shape[-2:]
    N = s.numel() // (T * S)
    z = torch.empty_like(s, dtype=torch.float32)
    compiled = _compile_logcumsumexp(_torch_to_cute_dtype(s.dtype), use_fast_ops=use_fast_ops)
    compiled(s.view(-1), z.view(-1), N, T, S)
    return z


__all__ = ["logcumsumexp_fwd_cute"]
