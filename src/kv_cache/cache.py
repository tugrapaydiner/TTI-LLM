# KV-cache for efficient autoregressive generation
import numpy as np
from typing import Tuple, Optional


class KVCache:
    def __init__(self, n_layers: int, n_heads: int, max_seq_len: int, d_head: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_head = d_head
        self.k_cache = np.zeros((n_layers, n_heads, max_seq_len, d_head), dtype=np.float32)  # [layers, heads, T, d]
        self.v_cache = np.zeros((n_layers, n_heads, max_seq_len, d_head), dtype=np.float32)
        self.cur_pos = 0  # current position pointer
    
    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T_new = k.shape[1]
        start_pos, end_pos = self.cur_pos, self.cur_pos + T_new
        self.k_cache[layer_idx, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx, :, start_pos:end_pos, :] = v
        return self.k_cache[layer_idx, :, :end_pos, :], self.v_cache[layer_idx, :, :end_pos, :]
    
    def advance(self, n_tokens: int = 1): self.cur_pos += n_tokens
    
    def reset(self):
        self.k_cache.fill(0)
        self.v_cache.fill(0)
        self.cur_pos = 0
    
    def get_seq_len(self) -> int: return self.cur_pos
    
    def rollback(self, n_tokens: int):  # for speculative decoding rejection
        self.cur_pos -= min(n_tokens, self.cur_pos)


def create_cache(n_layers: int = 12, n_heads: int = 12, max_seq_len: int = 1024, d_head: int = 64) -> KVCache:
    return KVCache(n_layers, n_heads, max_seq_len, d_head)

