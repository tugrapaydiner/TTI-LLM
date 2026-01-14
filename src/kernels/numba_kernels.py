# Optimized kernels using Numba JIT, falls back to NumPy
import numpy as np
from typing import Tuple

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range


# Int8 Kernels (Numba optimized)
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _gemv_int8_numba(Wq: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    T, in_features = x.shape
    _, out_features = Wq.shape
    y = np.zeros((T, out_features), dtype=np.float32)
    for t in range(T):
        for j in prange(out_features):
            acc = np.float32(0.0)
            for i in range(in_features):
                acc += x[t, i] * np.float32(Wq[i, j]) * scales[i]  # fused dequant + MAC
            y[t, j] = acc
    return y


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _gemv_int8_numba_tiled(Wq: np.ndarray, scales: np.ndarray, x: np.ndarray, tile_size: int = 32) -> np.ndarray:
    T, in_features = x.shape
    _, out_features = Wq.shape
    y = np.zeros((T, out_features), dtype=np.float32)
    for t in range(T):
        for j_start in prange(0, out_features, tile_size):
            j_end = min(j_start + tile_size, out_features)
            for i_start in range(0, in_features, tile_size):
                i_end = min(i_start + tile_size, in_features)
                for j in range(j_start, j_end):
                    acc = np.float32(0.0)
                    for i in range(i_start, i_end):
                        acc += x[t, i] * np.float32(Wq[i, j]) * scales[i]
                    y[t, j] += acc
    return y


def gemv_int8_numba(Wq: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:  # int8 GEMV with Numba
    if not HAS_NUMBA:
        W = Wq.astype(np.float32) * scales[:, np.newaxis]
        return x @ W
    Wq = np.ascontiguousarray(Wq)
    scales = np.ascontiguousarray(scales.astype(np.float32))
    x = np.ascontiguousarray(x.astype(np.float32))
    return _gemv_int8_numba(Wq, scales, x)


# Int4 Kernels (Numba optimized)
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _unpack_int4_numba(Wq_packed: np.ndarray, rows: int, cols: int) -> np.ndarray:  # uint8 -> int8
    cols_packed = Wq_packed.shape[1]
    W_int8 = np.zeros((rows, cols_packed * 2), dtype=np.int8)
    for i in prange(rows):
        for j in range(cols_packed):
            W_int8[i, j * 2] = np.int8((Wq_packed[i, j] & 0x0F)) - 8
            W_int8[i, j * 2 + 1] = np.int8((Wq_packed[i, j] >> 4) & 0x0F) - 8
    return W_int8[:, :cols]


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _gemv_int4_numba(W_int8: np.ndarray, scales: np.ndarray, x: np.ndarray, group_size: int) -> np.ndarray:
    T, in_features = x.shape
    _, out_features = W_int8.shape
    y = np.zeros((T, out_features), dtype=np.float32)
    for t in range(T):
        for j in prange(out_features):
            acc = np.float32(0.0)
            for i in range(in_features):
                flat_idx = i * out_features + j
                group_idx = flat_idx // group_size
                if group_idx < len(scales):
                    acc += x[t, i] * np.float32(W_int8[i, j]) * scales[group_idx]
            y[t, j] = acc
    return y


def gemv_int4_numba(Wq_packed: np.ndarray, scales: np.ndarray, orig_shape: Tuple[int, int], x: np.ndarray, group_size: int = 64) -> np.ndarray:
    rows, cols = orig_shape
    if not HAS_NUMBA:
        from kernels.api import gemv_int4
        return gemv_int4(Wq_packed, scales, orig_shape, x, group_size)
    Wq_packed = np.ascontiguousarray(Wq_packed)
    scales = np.ascontiguousarray(scales.astype(np.float32))
    x = np.ascontiguousarray(x.astype(np.float32))
    W_int8 = _unpack_int4_numba(Wq_packed, rows, cols)
    return _gemv_int4_numba(W_int8, scales, x, group_size)


def get_backend_info():
    return {"numba_available": HAS_NUMBA, "numba_version": None if not HAS_NUMBA else __import__("numba").__version__, "backend": "numba" if HAS_NUMBA else "numpy"}

