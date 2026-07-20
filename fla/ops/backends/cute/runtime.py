# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import importlib.util
import os
from functools import cache

from fla.utils import IS_NVIDIA_BLACKWELL, IS_NVIDIA_HOPPER, has_usable_nvcc


@cache
def _host_supports_cute_dsl() -> bool:
    if not (IS_NVIDIA_HOPPER or IS_NVIDIA_BLACKWELL):
        return False
    if not has_usable_nvcc():
        return False
    try:
        return importlib.util.find_spec("cutlass.cute") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def is_cute_dsl_available() -> bool:
    """Return whether CuTe DSL is enabled and can be compiled on this host."""
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1" or os.environ.get("FLA_CUTE_DSL", "1") == "0":
        return False
    return _host_supports_cute_dsl()


__all__ = ["is_cute_dsl_available"]
