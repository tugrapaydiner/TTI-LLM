# Kernels module
from .api import (
    gemv_int8,
    gemm_int8,
    gemv_int4,
    softmax_kernel,
    layer_norm_kernel,
    gelu_kernel,
    get_kernel,
    KERNELS,
)
from .numba_kernels import (
    gemv_int8_numba,
    gemv_int4_numba,
    get_backend_info,
    HAS_NUMBA,
)

# Try to import C++ kernels if built
try:
    from .cpp_kernels import gemv_int8_avx2
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    gemv_int8_avx2 = None

__all__ = [
    # Reference kernels
    "gemv_int8",
    "gemm_int8",
    "gemv_int4",
    "softmax_kernel",
    "layer_norm_kernel",
    "gelu_kernel",
    "get_kernel",
    "KERNELS",
    # Numba kernels
    "gemv_int8_numba",
    "gemv_int4_numba",
    "get_backend_info",
    "HAS_NUMBA",
    # C++ kernels
    "gemv_int8_avx2",
    "HAS_CPP",
]

