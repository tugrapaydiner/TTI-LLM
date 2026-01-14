# MLP (Feed-Forward) with GELU, supports int8/int4 quantization
import numpy as np
import sys
sys.path.insert(0, str(__file__).replace("model/mlp.py", ""))
from utils.tensor import gelu, matmul


class MLP:
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = None              # [d_model, d_ff]
        self.b1 = None              # [d_ff]
        self.W2 = None              # [d_ff, d_model]
        self.b2 = None              # [d_model]
        self.use_quant = False      # int8 flag
        self.W1_q = None
        self.W1_scales = None
        self.W2_q = None
        self.W2_scales = None
        self.use_int4 = False       # int4 flag
        self.W1_int4 = None
        self.W1_int4_scales = None
        self.W1_int4_shape = None
        self.W2_int4 = None
        self.W2_int4_scales = None
        self.W2_int4_shape = None
    
    def load_weights(self, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
    
    def _matmul_up(self, x: np.ndarray) -> np.ndarray:  # up-projection
        if self.use_int4:
            from kernels import gemv_int4
            return gemv_int4(self.W1_int4, self.W1_int4_scales, self.W1_int4_shape, x) + self.b1
        elif self.use_quant:
            from kernels import gemv_int8
            return gemv_int8(self.W1_q, self.W1_scales, x) + self.b1
        return matmul(x, self.W1) + self.b1
    
    def _matmul_down(self, x: np.ndarray) -> np.ndarray:  # down-projection
        if self.use_int4:
            from kernels import gemv_int4
            return gemv_int4(self.W2_int4, self.W2_int4_scales, self.W2_int4_shape, x) + self.b2
        elif self.use_quant:
            from kernels import gemv_int8
            return gemv_int8(self.W2_q, self.W2_scales, x) + self.b2
        return matmul(x, self.W2) + self.b2
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self._matmul_up(x))  # up-project + GELU [T, d_ff]
        return self._matmul_down(h)   # down-project [T, d_model]


