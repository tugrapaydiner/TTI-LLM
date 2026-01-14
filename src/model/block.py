# Transformer block: Pre-norm with LN -> Attention/MLP -> residual
import numpy as np
import sys
from typing import Optional
sys.path.insert(0, str(__file__).replace("model/block.py", ""))
from utils.tensor import layer_norm
from model.attention import Attention
from model.mlp import MLP


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)
        self.ln1_w = None  # [d_model]
        self.ln1_b = None
        self.ln2_w = None  # [d_model]
        self.ln2_b = None
    
    def load_weights(self, ln1_w, ln1_b, ln2_w, ln2_b, attn_Wqkv, attn_bqkv, attn_Wo, attn_bo, mlp_W1, mlp_b1, mlp_W2, mlp_b2):
        self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b = ln1_w, ln1_b, ln2_w, ln2_b
        self.attn.load_weights(attn_Wqkv, attn_bqkv, attn_Wo, attn_bo)
        self.mlp.load_weights(mlp_W1, mlp_b1, mlp_W2, mlp_b2)
    
    def __call__(self, x: np.ndarray, kv_cache: Optional[object] = None, layer_idx: int = 0) -> np.ndarray:
        h = layer_norm(x, self.ln1_w, self.ln1_b, self.eps)
        x = x + self.attn(h, kv_cache=kv_cache, layer_idx=layer_idx)  # attention + residual
        h = layer_norm(x, self.ln2_w, self.ln2_b, self.eps)
        return x + self.mlp(h)  # MLP + residual


