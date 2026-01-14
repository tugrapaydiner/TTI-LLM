# HuggingFace weight loader - extracts GPT-2 weights as NumPy arrays
import numpy as np
from typing import Dict, Any


def load_gpt2_weights(model_id: str = "gpt2") -> Dict[str, np.ndarray]:  # model -> weight dict
    try:
        from transformers import GPT2LMHeadModel
        import torch
    except ImportError:
        raise ImportError("Install transformers and torch: pip install transformers torch")
    print(f"Loading model: {model_id}")
    model = GPT2LMHeadModel.from_pretrained(model_id)
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
        print(f"  Loaded: {name} {weights[name].shape}")
    return weights


def get_model_config(model_id: str = "gpt2") -> Dict[str, Any]:  # get model hyperparameters
    try:
        from transformers import GPT2Config
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")
    config = GPT2Config.from_pretrained(model_id)
    return {
        "vocab_size": config.vocab_size, "d_model": config.n_embd, "n_heads": config.n_head,
        "d_ff": config.n_inner or 4 * config.n_embd, "n_layers": config.n_layer,
        "max_seq_len": config.n_positions, "eps": config.layer_norm_epsilon,
    }


def map_weights_to_model(weights: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, Any]:
    mapped = {"token_embed": None, "pos_embed": None, "ln_f_w": None, "ln_f_b": None, "blocks": []}
    for name, w in weights.items():
        if "wte.weight" in name:
            mapped["token_embed"] = w
            print(f"Token embedding: {w.shape}")
        elif "wpe.weight" in name:
            mapped["pos_embed"] = w
            print(f"Position embedding: {w.shape}")
        elif "ln_f.weight" in name:
            mapped["ln_f_w"] = w
            print(f"Final LN weight: {w.shape}")
        elif "ln_f.bias" in name:
            mapped["ln_f_b"] = w
            print(f"Final LN bias: {w.shape}")
    for i in range(config["n_layers"]):
        mapped["blocks"].append(extract_layer_weights(weights, i, config))
    return mapped


def extract_layer_weights(weights: Dict[str, np.ndarray], layer_idx: int, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    prefix = f"transformer.h.{layer_idx}."
    layer = {"ln1_w": None, "ln1_b": None, "ln2_w": None, "ln2_b": None,
             "attn_Wqkv": None, "attn_bqkv": None, "attn_Wo": None, "attn_bo": None,
             "mlp_W1": None, "mlp_b1": None, "mlp_W2": None, "mlp_b2": None}
    for name, w in weights.items():
        if prefix not in name: continue
        if "ln_1.weight" in name: layer["ln1_w"] = w
        elif "ln_1.bias" in name: layer["ln1_b"] = w
        elif "ln_2.weight" in name: layer["ln2_w"] = w
        elif "ln_2.bias" in name: layer["ln2_b"] = w
        elif "attn.c_attn.weight" in name: layer["attn_Wqkv"] = w
        elif "attn.c_attn.bias" in name: layer["attn_bqkv"] = w
        elif "attn.c_proj.weight" in name: layer["attn_Wo"] = w
        elif "attn.c_proj.bias" in name: layer["attn_bo"] = w
        elif "mlp.c_fc.weight" in name: layer["mlp_W1"] = w
        elif "mlp.c_fc.bias" in name: layer["mlp_b1"] = w
        elif "mlp.c_proj.weight" in name: layer["mlp_W2"] = w
        elif "mlp.c_proj.bias" in name: layer["mlp_b2"] = w
    return layer


def load_tokenizer(model_id: str = "gpt2"):
    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")
    return GPT2Tokenizer.from_pretrained(model_id)


if __name__ == "__main__":
    config = get_model_config()
    print("\nGPT-2 Small config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

