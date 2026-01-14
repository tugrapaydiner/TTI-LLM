# Minimal tensor operations using NumPy
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:  # numerically stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:  # normalize, scale, shift
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return w * ((x - mean) / np.sqrt(var + eps)) + b


def gelu(x: np.ndarray) -> np.ndarray:  # GELU activation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # matrix multiplication
    return a @ b

