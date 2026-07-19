# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import os

import cutlass
import cutlass.cute as cute
from cutlass import Float32

_USE_FAST_OPS = os.environ.get("FLA_USE_FAST_OPS", "0") == "1"


@cute.jit
def exp(x: Float32, *, use_fast_ops: cutlass.Constexpr = _USE_FAST_OPS) -> Float32:
    """Evaluate the natural exponential in FP32."""
    return cute.math.exp(x, fastmath=use_fast_ops)


@cute.jit
def exp2(x: Float32, *, use_fast_ops: cutlass.Constexpr = _USE_FAST_OPS) -> Float32:
    """Evaluate the base-2 exponential in FP32."""
    return cute.math.exp2(x, fastmath=use_fast_ops)


@cute.jit
def log(x: Float32, *, use_fast_ops: cutlass.Constexpr = _USE_FAST_OPS) -> Float32:
    """Evaluate the natural logarithm in FP32."""
    return cute.math.log(x, fastmath=use_fast_ops)


@cute.jit
def log2(x: Float32, *, use_fast_ops: cutlass.Constexpr = _USE_FAST_OPS) -> Float32:
    """Evaluate the base-2 logarithm in FP32."""
    return cute.math.log2(x, fastmath=use_fast_ops)


@cute.jit
def tanh(x: Float32, *, use_fast_ops: cutlass.Constexpr = _USE_FAST_OPS) -> Float32:
    """Evaluate the hyperbolic tangent in FP32."""
    return cute.math.tanh(x, fastmath=use_fast_ops)


__all__ = ["exp", "exp2", "log", "log2", "tanh"]
