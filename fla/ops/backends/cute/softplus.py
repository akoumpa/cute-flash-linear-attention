# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


@cute.jit
def softplus_cute(x: Float32) -> Float32:
    """Evaluate the natural-base softplus used by CuTe gate kernels."""
    result = x
    if x <= Float32(20.0):
        exponent = cute.math.exp2(x * Float32(_LOG2E), fastmath=True)
        result = cute.math.log2(exponent + Float32(1.0), fastmath=True) * Float32(_LN2)
    return result


@cute.jit
def softplus2_cute(x: Float32) -> Float32:
    """Evaluate the base-2 softplus used by CuTe kernels."""
    result = x
    if x <= Float32(15.0):
        exponent = cute.math.exp2(x, fastmath=True)
        result = cute.math.log2(exponent + Float32(1.0), fastmath=True)
    return result


__all__ = ["softplus2_cute", "softplus_cute"]
