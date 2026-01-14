# Kernel API - optimized compute kernels with NumPy fallbacks
import numpy as np
from typing import Tuple


# Int8 Kernels
def gemv_int8(Wq: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:  # x @ dequant(Wq)
    W = Wq.astype(np.float32) * scales[:, np.newaxis]
    return x @ W


def gemm_int8(Wq: np.ndarray, scales: np.ndarray, X: np.ndarray) -> np.ndarray:  # alias for gemv_int8
    return gemv_int8(Wq, scales, X)


# Int4 Kernels
def gemv_int4(Wq_packed: np.ndarray, scales: np.ndarray, orig_shape: Tuple[int, int], x: np.ndarray, group_size: int = 64) -> np.ndarray:
    rows, cols = orig_shape
    low_nibble = (Wq_packed & 0x0F).astype(np.int8) - 8   # unpack [-8, 7]
    high_nibble = ((Wq_packed >> 4) & 0x0F).astype(np.int8) - 8
    cols_packed = Wq_packed.shape[1]
    W_int4 = np.zeros((rows, cols_packed * 2), dtype=np.int8)
    W_int4[:, 0::2], W_int4[:, 1::2] = low_nibble, high_nibble
    W_int4 = W_int4[:, :cols]
    
    W_flat = W_int4.flatten().astype(np.float32)
    n_elements = len(W_flat)
    pad_size = (group_size - (n_elements % group_size)) % group_size
    if pad_size > 0: W_flat = np.concatenate([W_flat, np.zeros(pad_size, dtype=np.float32)])
    
    n_groups = len(W_flat) // group_size
    W_dequant = (W_flat.reshape(n_groups, group_size) * scales[:, np.newaxis]).flatten()[:n_elements].reshape(rows, cols)
    return x @ W_dequant


# Utility Kernels
def softmax_kernel(x: np.ndarray, axis: int = -1) -> np.ndarray:  # numerically stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm_kernel(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * ((x - mean) / np.sqrt(var + eps)) + beta


def gelu_kernel(x: np.ndarray) -> np.ndarray:  # GELU activation (tanh approx)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# Kernel Registry
KERNELS = {"gemv_int8": gemv_int8, "gemm_int8": gemm_int8, "gemv_int4": gemv_int4, 
           "softmax": softmax_kernel, "layer_norm": layer_norm_kernel, "gelu": gelu_kernel}


def get_kernel(name: str):
    if name not in KERNELS: raise ValueError(f"Unknown kernel: {name}. Available: {list(KERNELS.keys())}")
    return KERNELS[name]

