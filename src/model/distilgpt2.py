# DistilGPT-2 - faster draft model for speculative decoding (6 layers vs 12)
import numpy as np
import sys
from typing import Optional
sys.path.insert(0, str(__file__).replace("model/distilgpt2.py", ""))
from utils.tensor import layer_norm, matmul
from model.block import TransformerBlock


class DistilGPT2:
    VOCAB_SIZE = 50257
    D_MODEL = 768
    N_HEADS = 12
    D_FF = 3072
    N_LAYERS = 6  # half of GPT-2
    MAX_SEQ_LEN = 1024
    EPS = 1e-5
    D_HEAD = D_MODEL // N_HEADS
    
    def __init__(self):
        self.token_embed = None   # [vocab_size, d_model]
        self.pos_embed = None     # [max_seq_len, d_model]
        self.blocks = [TransformerBlock(self.D_MODEL, self.N_HEADS, self.D_FF, self.EPS) for _ in range(self.N_LAYERS)]
        self.ln_f_w = None
        self.ln_f_b = None
        self.tie_weights = True
    
    def load_embeddings(self, token_embed: np.ndarray, pos_embed: np.ndarray):
        self.token_embed, self.pos_embed = token_embed, pos_embed
    
    def load_final_ln(self, ln_f_w: np.ndarray, ln_f_b: np.ndarray):
        self.ln_f_w, self.ln_f_b = ln_f_w, ln_f_b
    
    def embed(self, token_ids: np.ndarray) -> np.ndarray:  # [T] -> [T, d_model]
        return self.token_embed[token_ids] + self.pos_embed[:len(token_ids)]
    
    def embed_one(self, token_id: int, pos: int) -> np.ndarray:  # single token at pos
        return self.token_embed[token_id:token_id+1] + self.pos_embed[pos:pos+1]
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:  # full forward pass
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = layer_norm(x, self.ln_f_w, self.ln_f_b, self.EPS)
        return matmul(x, self.token_embed.T)
    
    def prefill(self, token_ids: np.ndarray, kv_cache) -> np.ndarray:  # fill cache, return last logits
        x = self.embed(token_ids)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=layer_idx)
        kv_cache.advance(len(token_ids))
        x = layer_norm(x, self.ln_f_w, self.ln_f_b, self.EPS)
        return matmul(x[-1:], self.token_embed.T)[0]
    
    def decode_one(self, token_id: int, kv_cache) -> np.ndarray:  # decode single token
        pos = kv_cache.get_seq_len()
        x = self.embed_one(token_id, pos)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=layer_idx)
        kv_cache.advance(1)
        x = layer_norm(x, self.ln_f_w, self.ln_f_b, self.EPS)
        return matmul(x, self.token_embed.T)[0]
    
    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)

