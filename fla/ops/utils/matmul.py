# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# code adapted from
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html


import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, input_guard

_CUTE_MATMUL_MIN_N = 2
_CUTE_MATMUL_MAX_N = 4
_CUTE_MATMUL_MIN_K = 16
_CUTE_MATMUL_MAX_K = 256
_CUTE_MATMUL_MIN_OUTPUTS = 128
_CUTE_MATMUL_MAX_OUTPUTS = 8192


def _can_use_cute_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    activation: str,
    *,
    x: torch.Tensor | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> bool:
    if torch.compiler.is_compiling() or not a.is_cuda or not b.is_cuda:
        return False
    if a.device != b.device or a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if not a.is_contiguous() or not b.is_contiguous() or activation not in ("", "leaky_relu", "relu"):
        return False
    if a.dim() not in (2, 3) or b.dim() != 2 or a.shape[-1] != b.shape[0]:
        return False
    B = a.shape[0] if a.dim() == 3 else 1
    M, K, N = a.shape[-2], a.shape[-1], b.shape[1]
    output_count = B * M * N
    if not (
        _CUTE_MATMUL_MIN_N <= N <= _CUTE_MATMUL_MAX_N
        and _CUTE_MATMUL_MIN_K <= K <= _CUTE_MATMUL_MAX_K
        and _CUTE_MATMUL_MIN_OUTPUTS <= output_count <= _CUTE_MATMUL_MAX_OUTPUTS
    ):
        return False
    if alpha is not None and not isinstance(alpha, int | float):
        return False
    if beta is not None and not isinstance(beta, int | float):
        return False
    if x is not None:
        if not x.is_cuda or x.device != a.device or x.dtype != a.dtype or not x.is_contiguous() or x.dim() not in (1, 2):
            return False
        if x.dim() == 1 and x.shape[0] != N:
            return False
        if x.dim() == 2 and x.shape != (M, N):
            return False
    from fla.ops.backends.cute.runtime import is_cute_dsl_available

    return is_cute_dsl_available()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BM`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.heuristics(
    {
        "HAS_ALPHA": lambda args: args["alpha"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BK": 64, "BN": 256, "G": 4}, num_stages=3, num_warps=8),
        triton.Config({"BM": 64, "BK": 32, "BN": 256, "G": 4}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BK": 32, "BN": 128, "G": 4}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BK": 32, "BN": 64, "G": 4}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BK": 32, "BN": 128, "G": 4}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BK": 32, "BN": 32, "G": 4}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BK": 32, "BN": 32, "G": 4}, num_stages=5, num_warps=2),
        triton.Config({"BM": 32, "BK": 32, "BN": 64, "G": 4}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        # triton.Config({'BM': 128, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8),
        # triton.Config({'BM': 256, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=3, num_warps=8),
        # triton.Config({'BM': 256, 'BK': 128, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 64, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 128, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 64, 'BK': 64, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 128, 'BK': 64, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4)
    ],
    key=["M", "N", "K"],
    **autotune_cache_kwargs,
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    input,
    alpha,
    beta,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `s_am` is how much to increase `a`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,  # a: batch, M, K
    stride_bk,
    stride_bn,  # b: K, N
    stride_cb,
    stride_cm,
    stride_cn,  # c: batch, M, N
    # Meta-parameters
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BETA: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    X_DIM: tl.constexpr = 1,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    i_b, i_m, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NM, NN = tl.num_programs(1), tl.num_programs(2)
    i_m, i_n = tl.swizzle2d(i_m, i_n, NM, NN, G)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details
    a_batch_ptr = a + i_b * stride_ab
    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    p_a = a_batch_ptr + (o_am[:, None] * stride_am + o_k[None, :] * stride_ak)
    p_b = b + (o_k[:, None] * stride_bk + o_bn[None, :] * stride_bn)

    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        # We accumulate along the K dimension.
        b_acc = tl.dot(b_a, b_b, acc=b_acc, allow_tf32=ALLOW_TF32)
        # Advance the ptrs to the next K block.
        p_a += BK * stride_ak
        p_b += BK * stride_bk

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cm[:, None] < M) & (o_cn[None, :] < N)

    b_c = b_acc
    # You can fuse arbitrary activation functions here
    # while the b_acc is still in FP32!
    if ACTIVATION == "leaky_relu":
        b_c = leaky_relu(b_c)
    elif ACTIVATION == "relu":
        b_c = relu(b_c)
    elif ACTIVATION == "sigmoid":
        b_c = sigmoid(b_c)
    elif ACTIVATION == "tanh":
        b_c = tanh(b_c)

    if HAS_ALPHA:
        b_c *= tl.load(alpha)

    if HAS_INPUT:
        p_i = input + (stride_cm * o_cm[:, None] if X_DIM == 2 else 0) + stride_cn * o_cn[None, :]
        mask_p = (o_cn[None, :] < N) if X_DIM == 1 else mask
        b_i = tl.load(p_i, mask=mask_p, other=0.0).to(tl.float32)
        if HAS_BETA:
            b_i *= tl.load(beta)
        b_c += b_i

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_batch_ptr = c + i_b * stride_cb
    p_c = c_batch_ptr + stride_cm * o_cm[:, None] + stride_cn * o_cn[None, :]
    tl.store(p_c, b_c.to(c.dtype.element_ty), mask=mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def sigmoid(x):
    # σ(x) = 1 / (1 + exp(-x))
    return 1.0 / (1.0 + exp(-x))


@triton.jit
def tanh(x):
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # 2 * sigmoid(2x) - 1
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


@triton.jit
def relu(x):
    # ReLU(x) = max(0, x)
    return tl.maximum(x, 0.0)


@input_guard
def matmul(a, b, activation=""):
    assert a.dim() in [2, 3], "a must be 2D or 3D"
    assert b.dim() == 2, "b must be 2D"
    assert a.shape[-1] == b.shape[0], f"Incompatible dimensions: A {a.shape}, B {b.shape}"

    if _can_use_cute_matmul(a, b, activation):
        from fla.ops.backends.cute.matmul import narrow_matmul_cute

        return narrow_matmul_cute(a, b, activation=activation)

    if a.dim() == 2:
        a_dim = 2
        a = a.unsqueeze(0).contiguous()  # (1, M, K)
    else:
        a_dim = 3
    allow_tf32 = False if a.dtype == torch.float32 else True

    B, M, K = a.shape[0], a.shape[1], a.shape[2]
    K_b, N = b.shape
    assert K_b == K, f"Incompatible K dimension: A {K} vs B {K_b}"
    c = a.new_empty(B, M, N)

    def grid(meta):
        return (B, triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))

    matmul_kernel[grid](
        a,
        b,
        c,
        None,
        None,
        None,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),  # stride_ab, stride_am, stride_ak
        b.stride(0),
        b.stride(1),  # stride_bk, stride_bn (b.dim() == 2)
        c.stride(0),
        c.stride(1),
        c.stride(2),  # stride_cb, stride_cm, stride_cn
        ACTIVATION=activation,
        ALLOW_TF32=allow_tf32,
        HAS_INPUT=False,
    )
    return c.squeeze(0) if a_dim == 2 else c


@input_guard
def addmm(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float | None = None,
    beta: float | None = None,
) -> torch.Tensor:
    assert a.dim() in [2, 3], "a must be 2D or 3D"
    assert b.dim() == 2, "b must be 2D"
    assert a.shape[-1] == b.shape[0], f"Incompatible dimensions: A {a.shape}, B {b.shape}"

    if _can_use_cute_matmul(a, b, "", x=x, alpha=alpha, beta=beta):
        from fla.ops.backends.cute.matmul import narrow_matmul_cute

        return narrow_matmul_cute(a, b, x=x, alpha=alpha, beta=beta)

    if a.dim() == 2:
        a_dim = 2
        a = a.unsqueeze(0).contiguous()  # (1, M, K)
    else:
        a_dim = 3
    allow_tf32 = False if a.dtype == torch.float32 else True

    B, M, K = a.shape[0], a.shape[1], a.shape[2]
    K_b, N = b.shape
    assert K_b == K, f"Incompatible K dimension: A {K} vs B {K_b}"
    c = a.new_empty(B, M, N)

    def grid(meta):
        return (B, triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))

    matmul_kernel[grid](
        a,
        b,
        c,
        x,
        alpha,
        beta,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),  # stride_ab, stride_am, stride_ak
        b.stride(0),
        b.stride(1),  # stride_bk, stride_bn (b.dim() == 2)
        c.stride(0),
        c.stride(1),
        c.stride(2),  # stride_cb, stride_cm, stride_cn
        ACTIVATION=None,
        ALLOW_TF32=allow_tf32,
        HAS_INPUT=True,
        X_DIM=x.dim(),
    )
    return c.squeeze(0) if a_dim == 2 else c
