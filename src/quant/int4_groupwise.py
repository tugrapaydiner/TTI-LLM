# Int4 groupwise quantization - reduces memory by ~8x, group_size=64, range [-8,7]
import numpy as np
from typing import Tuple

GROUP_SIZE = 64


def quantize_int4_groupwise(W: np.ndarray, group_size: int = GROUP_SIZE) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    orig_shape = W.shape
    rows, cols = W.shape
    W_flat = W.flatten()
    n_elements = len(W_flat)
    
    pad_size = (group_size - (n_elements % group_size)) % group_size  # pad to group_size
    if pad_size > 0:
        W_flat = np.concatenate([W_flat, np.zeros(pad_size, dtype=W.dtype)])
    
    n_groups = len(W_flat) // group_size
    W_grouped = W_flat.reshape(n_groups, group_size)
    abs_max = np.maximum(np.abs(W_grouped).max(axis=1, keepdims=True), 1e-8)
    scales = (abs_max / 7.0).flatten()  # scale per group
    
    W_int4 = np.round(W_grouped / abs_max * 7.0).clip(-8, 7).astype(np.int8)
    W_int4_flat = W_int4.flatten()[:n_elements] if pad_size > 0 else W_int4.flatten()
    W_int4_2d = W_int4_flat.reshape(rows, cols)
    
    W_shifted = (W_int4_2d + 8).astype(np.uint8)  # shift to [0,15] for packing
    if cols % 2 != 0:  # pad to even columns
        W_shifted = np.concatenate([W_shifted, np.zeros((rows, 1), dtype=np.uint8)], axis=1)
    
    Wq_packed = (W_shifted[:, 0::2] & 0x0F) | ((W_shifted[:, 1::2] & 0x0F) << 4)  # pack 2 int4 per byte
    return Wq_packed, scales, orig_shape


def dequantize_int4_groupwise(Wq_packed: np.ndarray, scales: np.ndarray, orig_shape: Tuple[int, int], group_size: int = GROUP_SIZE) -> np.ndarray:
    rows, cols = orig_shape
    low_nibble = (Wq_packed & 0x0F).astype(np.int8) - 8   # unpack low [-8,7]
    high_nibble = ((Wq_packed >> 4) & 0x0F).astype(np.int8) - 8  # unpack high
    
    cols_packed = Wq_packed.shape[1]
    W_int4 = np.zeros((rows, cols_packed * 2), dtype=np.int8)
    W_int4[:, 0::2], W_int4[:, 1::2] = low_nibble, high_nibble
    W_int4 = W_int4[:, :cols]
    
    W_flat = W_int4.flatten().astype(np.float32)
    n_elements = len(W_flat)
    pad_size = (group_size - (n_elements % group_size)) % group_size
    if pad_size > 0:
        W_flat = np.concatenate([W_flat, np.zeros(pad_size, dtype=np.float32)])
    
    n_groups = len(W_flat) // group_size
    W_dequant = (W_flat.reshape(n_groups, group_size) * scales[:, np.newaxis]).flatten()[:n_elements]
    return W_dequant.reshape(rows, cols)


def matmul_int4(x: np.ndarray, Wq_packed: np.ndarray, scales: np.ndarray, orig_shape: Tuple[int, int], group_size: int = GROUP_SIZE) -> np.ndarray:
    return x @ dequantize_int4_groupwise(Wq_packed, scales, orig_shape, group_size)


def compute_int4_error(W: np.ndarray, group_size: int = GROUP_SIZE) -> dict:
    Wq_packed, scales, orig_shape = quantize_int4_groupwise(W, group_size)
    W_reconstructed = dequantize_int4_groupwise(Wq_packed, scales, orig_shape, group_size)
    error = W - W_reconstructed
    return {
        "max_abs_error": np.abs(error).max(),
        "mean_abs_error": np.abs(error).mean(),
        "rmse": np.sqrt((error ** 2).mean()),
        "relative_error": np.abs(error).mean() / np.abs(W).mean(),
        "compression_ratio": W.nbytes / (Wq_packed.nbytes + scales.nbytes),
        "original_bytes": W.nbytes,
        "quantized_bytes": Wq_packed.nbytes + scales.nbytes,
    }

