# Text generation driver - supports fp32, int8, int4 modes
import sys
import time
import numpy as np
from typing import Optional

sys.path.insert(0, str(__file__).replace("run_generate.py", ""))

from utils.tokenizer import Tokenizer
from utils.hf_loader import load_gpt2_weights, get_model_config, map_weights_to_model
from model.gpt2 import GPT2
from kv_cache import create_cache
from sampling import greedy, top_k


def load_model(mode: str = "fp32") -> GPT2:  # load GPT-2 with optional quantization
    print(f"Loading GPT-2 model (mode: {mode})...")
    weights = load_gpt2_weights("gpt2")
    config = get_model_config("gpt2")
    mapped = map_weights_to_model(weights, config)
    
    model = GPT2()
    model.load_embeddings(mapped["token_embed"], mapped["pos_embed"])
    model.load_final_ln(mapped["ln_f_w"], mapped["ln_f_b"])
    
    for i, bw in enumerate(mapped["blocks"]):
        model.blocks[i].load_weights(
            ln1_w=bw["ln1_w"], ln1_b=bw["ln1_b"], ln2_w=bw["ln2_w"], ln2_b=bw["ln2_b"],
            attn_Wqkv=bw["attn_Wqkv"], attn_bqkv=bw["attn_bqkv"], attn_Wo=bw["attn_Wo"], attn_bo=bw["attn_bo"],
            mlp_W1=bw["mlp_W1"], mlp_b1=bw["mlp_b1"], mlp_W2=bw["mlp_W2"], mlp_b2=bw["mlp_b2"],
        )
    
    if mode == "int8":
        from quant import quantize_model
        quantize_model(model)
    elif mode == "int4":
        from quant import quantize_model_int4
        quantize_model_int4(model)
    
    print("Model loaded!")
    return model


def generate(prompt: str, max_new_tokens: int = 50, model: Optional[GPT2] = None, tokenizer: Optional[Tokenizer] = None,
             mode: str = "fp32", sampling: str = "greedy", top_k_k: int = 40, temperature: float = 1.0,
             use_cache: bool = True, verbose: bool = True) -> str:
    if model is None: model = load_model(mode)
    if tokenizer is None: tokenizer = Tokenizer()
    
    prompt_tokens = tokenizer.encode(prompt)
    if verbose:
        print(f"\nPrompt: {prompt!r}\nTokens: {len(prompt_tokens)}\nMode: {mode}\nGenerating {max_new_tokens} tokens...\n")
    
    start_time = time.perf_counter()
    
    if use_cache:  # KV-cache for efficient generation
        cache = create_cache(n_layers=model.N_LAYERS, n_heads=model.N_HEADS, max_seq_len=model.MAX_SEQ_LEN, d_head=model.D_HEAD)
        logits = model.prefill(prompt_tokens, cache)
        next_token = greedy(logits) if sampling == "greedy" else top_k(logits, k=top_k_k, temp=temperature)
        generated_tokens = [next_token]
        if verbose: print(tokenizer.decode_single(next_token), end="", flush=True)
        
        for _ in range(max_new_tokens - 1):
            logits = model.decode_one(next_token, cache)
            next_token = greedy(logits) if sampling == "greedy" else top_k(logits, k=top_k_k, temp=temperature)
            generated_tokens.append(next_token)
            if verbose: print(tokenizer.decode_single(next_token), end="", flush=True)
            if next_token == tokenizer.eos_token_id: break
        
        all_tokens = np.concatenate([prompt_tokens, generated_tokens])
    else:  # non-cached generation
        token_ids = prompt_tokens.copy()
        for _ in range(max_new_tokens):
            logits = model(token_ids)
            next_logits = logits[-1]
            next_token = greedy(next_logits) if sampling == "greedy" else top_k(next_logits, k=top_k_k, temp=temperature)
            token_ids = np.append(token_ids, next_token)
            if verbose: print(tokenizer.decode_single(next_token), end="", flush=True)
            if next_token == tokenizer.eos_token_id: break
        all_tokens = token_ids
    
    elapsed = time.perf_counter() - start_time
    n_generated = len(all_tokens) - len(prompt_tokens)
    if verbose: print(f"\n\n[Generated {n_generated} tokens in {elapsed:.2f}s = {n_generated/elapsed:.1f} tok/s]")
    return tokenizer.decode(all_tokens)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate text with TTI-LLM")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument("--mode", type=str, default="fp32", choices=["fp32", "int8", "int4"], help="Quantization mode")
    parser.add_argument("--sampling", type=str, default="greedy", choices=["greedy", "top_k"], help="Sampling strategy")
    parser.add_argument("--top_k", type=int, default=40, help="k for top-k sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--no_cache", action="store_true", help="Disable KV-cache")
    args = parser.parse_args()
    
    model = load_model(args.mode)
    tokenizer = Tokenizer()
    output = generate(prompt=args.prompt, max_new_tokens=args.max_new_tokens, model=model, tokenizer=tokenizer,
                      mode=args.mode, sampling=args.sampling, top_k_k=args.top_k, temperature=args.temperature, use_cache=not args.no_cache)
    print("=" * 50)
    print("Full output:")
    print(output)


if __name__ == "__main__":
    main()


