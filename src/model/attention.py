# Multi-head self-attention with KV-cache and int8 quantization support
import numpy as np
import sys
sys.path.insert(0, str(__file__).replace("model/attention.py", ""))
from utils.tensor import softmax, matmul
from typing import Optional, Tuple


class Attention:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wqkv = None      # [d_model, 3*d_model]
        self.bqkv = None      # [3*d_model]
        self.Wo = None        # [d_model, d_model]
        self.bo = None        # [d_model]
        self.use_quant = False  # int8 quantization flag
        self.Wqkv_q = None
        self.Wqkv_scales = None
        self.Wo_q = None
        self.Wo_scales = None
    
    def load_weights(self, Wqkv: np.ndarray, bqkv: np.ndarray, Wo: np.ndarray, bo: np.ndarray):
        self.Wqkv = Wqkv
        self.bqkv = bqkv
        self.Wo = Wo
        self.bo = bo
    
    def _matmul_qkv(self, x: np.ndarray) -> np.ndarray:  # QKV projection
        if self.use_quant:
            from kernels import gemv_int8
            return gemv_int8(self.Wqkv_q, self.Wqkv_scales, x) + self.bqkv
        return matmul(x, self.Wqkv) + self.bqkv
    
    def _matmul_out(self, x: np.ndarray) -> np.ndarray:  # output projection
        if self.use_quant:
            from kernels import gemv_int8
            return gemv_int8(self.Wo_q, self.Wo_scales, x) + self.bo
        return matmul(x, self.Wo) + self.bo
    
    def __call__(self, x: np.ndarray, kv_cache: Optional[object] = None, layer_idx: int = 0) -> np.ndarray:
        T, d_model = x.shape
        qkv = self._matmul_qkv(x)                         # [T, 3*d_model]
        q, k, v = np.split(qkv, 3, axis=-1)               # split into q, k, v
        
        q = q.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)  # [n_heads, T, d_head]
        k = k.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        
        if kv_cache is not None:                          # use cache for incremental generation
            k, v = kv_cache.update(layer_idx, k, v)
            T_kv = k.shape[1]
        else:
            T_kv = T
        
        scale = 1.0 / np.sqrt(self.d_head)
        scores = matmul(q, k.transpose(0, 2, 1)) * scale  # attention scores [n_heads, T, T_kv]
        
        if kv_cache is None:                              # causal mask for full sequence
            mask = np.triu(np.ones((T, T_kv)), k=1) * -1e9
            scores = scores + mask
        
        attn = softmax(scores, axis=-1)                   # softmax over keys
        out = matmul(attn, v)                             # weighted sum [n_heads, T, d_head]
        out = out.transpose(1, 0, 2).reshape(T, d_model)  # reshape to [T, d_model]
        return self._matmul_out(out)                      # output projection


