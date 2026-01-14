# Quantize GPT-2 model weights to int8 or int4
import numpy as np
import sys
sys.path.insert(0, str(__file__).replace("quant/quantize_model.py", ""))

from quant.int8 import quantize_per_row, dequantize_per_row
from quant.int4_groupwise import quantize_int4_groupwise


def quantize_attention_weights(attn):  # int8 for Wqkv and Wo
    if attn.Wqkv is not None and attn.Wqkv.dtype != np.int8:
        attn.Wqkv_q, attn.Wqkv_scales = quantize_per_row(attn.Wqkv)
        attn.Wqkv = None
        attn.use_quant = True
    if attn.Wo is not None and attn.Wo.dtype != np.int8:
        attn.Wo_q, attn.Wo_scales = quantize_per_row(attn.Wo)
        attn.Wo = None


def quantize_mlp_weights(mlp):  # int8 for W1 and W2
    if mlp.W1 is not None and mlp.W1.dtype != np.int8:
        mlp.W1_q, mlp.W1_scales = quantize_per_row(mlp.W1)
        mlp.W1 = None
        mlp.use_quant = True
    if mlp.W2 is not None and mlp.W2.dtype != np.int8:
        mlp.W2_q, mlp.W2_scales = quantize_per_row(mlp.W2)
        mlp.W2 = None


def quantize_mlp_weights_int4(mlp):  # int4 for W1 and W2
    if mlp.W1 is not None:
        mlp.W1_int4, mlp.W1_int4_scales, mlp.W1_int4_shape = quantize_int4_groupwise(mlp.W1)
        mlp.W1 = None
        mlp.use_int4 = True
    if mlp.W2 is not None:
        mlp.W2_int4, mlp.W2_int4_scales, mlp.W2_int4_shape = quantize_int4_groupwise(mlp.W2)
        mlp.W2 = None


def quantize_block_weights(block):
    quantize_attention_weights(block.attn)
    quantize_mlp_weights(block.mlp)


def quantize_block_weights_int4(block, mlp_only=True):  # int4 MLP, int8 attention
    if mlp_only:
        quantize_attention_weights(block.attn)
        quantize_mlp_weights_int4(block.mlp)
    else:
        quantize_mlp_weights_int4(block.mlp)


def quantize_model(model, quantize_lm_head=False):  # int8 all blocks
    print("Quantizing model weights to int8...")
    for i, block in enumerate(model.blocks):
        quantize_block_weights(block)
    if quantize_lm_head and model.token_embed is not None:
        model.token_embed_q, model.token_embed_scales = quantize_per_row(model.token_embed)
        model.use_quant_lm_head = True
    print(f"  Quantized {len(model.blocks)} blocks")
    print("  Done!")


def quantize_model_int4(model, mlp_only=True):  # int4 MLP, int8 attention
    print("Quantizing model weights (MLP: int4, Attention: int8)...")
    for i, block in enumerate(model.blocks):
        quantize_block_weights_int4(block, mlp_only=mlp_only)
    print(f"  Quantized {len(model.blocks)} blocks")
    print("  Done!")


def get_model_memory_bytes(model):  # estimate total memory
    total = 0
    if model.token_embed is not None: total += model.token_embed.nbytes
    if model.pos_embed is not None: total += model.pos_embed.nbytes
    if model.ln_f_w is not None: total += model.ln_f_w.nbytes
    if model.ln_f_b is not None: total += model.ln_f_b.nbytes
    
    for block in model.blocks:
        if block.ln1_w is not None: total += block.ln1_w.nbytes
        if block.ln1_b is not None: total += block.ln1_b.nbytes
        if block.ln2_w is not None: total += block.ln2_w.nbytes
        if block.ln2_b is not None: total += block.ln2_b.nbytes
        
        attn = block.attn
        if hasattr(attn, 'use_quant') and attn.use_quant:
            total += attn.Wqkv_q.nbytes + attn.Wqkv_scales.nbytes + attn.Wo_q.nbytes + attn.Wo_scales.nbytes
        else:
            if attn.Wqkv is not None: total += attn.Wqkv.nbytes
            if attn.Wo is not None: total += attn.Wo.nbytes
        if attn.bqkv is not None: total += attn.bqkv.nbytes
        if attn.bo is not None: total += attn.bo.nbytes
        
        mlp = block.mlp
        if hasattr(mlp, 'use_int4') and mlp.use_int4:
            total += mlp.W1_int4.nbytes + mlp.W1_int4_scales.nbytes + mlp.W2_int4.nbytes + mlp.W2_int4_scales.nbytes
        elif hasattr(mlp, 'use_quant') and mlp.use_quant:
            total += mlp.W1_q.nbytes + mlp.W1_scales.nbytes + mlp.W2_q.nbytes + mlp.W2_scales.nbytes
        else:
            if mlp.W1 is not None: total += mlp.W1.nbytes
            if mlp.W2 is not None: total += mlp.W2.nbytes
        if mlp.b1 is not None: total += mlp.b1.nbytes
        if mlp.b2 is not None: total += mlp.b2.nbytes
    return total


