# Int8 per-row quantization - reduces memory by ~4x
import numpy as np
from typing import Tuple


def quantize_per_row(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # W -> (Wq, scales)
    abs_max = np.maximum(np.abs(W).max(axis=1, keepdims=True), 1e-8)  # per-row max
    scales = abs_max.flatten() / 127.0
    Wq = np.round(W / abs_max * 127.0).astype(np.int8)
    return Wq, scales


def dequantize_per_row(Wq: np.ndarray, scales: np.ndarray) -> np.ndarray:  # Wq -> W
    return Wq.astype(np.float32) * scales[:, np.newaxis]


def matmul_int8(x: np.ndarray, Wq: np.ndarray, scales: np.ndarray) -> np.ndarray:  # x @ dequant(Wq)
    return x @ dequantize_per_row(Wq, scales)


class QuantizedLinear:
    def __init__(self, Wq: np.ndarray, scales: np.ndarray, bias: np.ndarray = None):
        self.Wq, self.scales, self.bias = Wq, scales, bias
    
    @classmethod
    def from_float(cls, W: np.ndarray, bias: np.ndarray = None) -> "QuantizedLinear":
        Wq, scales = quantize_per_row(W)
        return cls(Wq, scales, bias)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = matmul_int8(x, self.Wq, self.scales)
        return out + self.bias if self.bias is not None else out
    
    @property
    def memory_bytes(self) -> int:
        return self.Wq.nbytes + self.scales.nbytes + (self.bias.nbytes if self.bias else 0)


def compute_quantization_error(W: np.ndarray) -> dict:
    Wq, scales = quantize_per_row(W)
    W_reconstructed = dequantize_per_row(Wq, scales)
    error = W - W_reconstructed
    return {
        "max_abs_error": np.abs(error).max(),
        "mean_abs_error": np.abs(error).mean(),
        "rmse": np.sqrt((error ** 2).mean()),
        "relative_error": np.abs(error).mean() / np.abs(W).mean(),
        "compression_ratio": W.nbytes / (Wq.nbytes + scales.nbytes),
    }

