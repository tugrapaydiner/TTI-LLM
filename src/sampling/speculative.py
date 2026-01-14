# Speculative decoding - draft proposes m tokens, main verifies, accept until mismatch
import numpy as np
from typing import Tuple, List, Optional


def speculative_decode(draft_model, main_model, draft_cache, main_cache, prompt_tokens: np.ndarray,
                       max_tokens: int = 100, num_speculative: int = 4, temperature: float = 1.0) -> Tuple[np.ndarray, dict]:
    generated = list(prompt_tokens)
    stats = {"total_tokens": 0, "accepted_tokens": 0, "rejected_tokens": 0, "main_forward_calls": 0, "draft_forward_calls": 0}
    
    _ = draft_model.prefill(prompt_tokens, draft_cache)  # prefill both models
    _ = main_model.prefill(prompt_tokens, main_cache)
    tokens_generated = 0
    
    while tokens_generated < max_tokens:
        draft_tokens, draft_logits_list = [], []
        for _ in range(num_speculative):  # draft proposes m tokens
            if len(generated) == 0: break
            last_token = generated[-1] if len(draft_tokens) == 0 else draft_tokens[-1]
            logits = draft_model.decode_one(last_token, draft_cache)
            stats["draft_forward_calls"] += 1
            draft_tokens.append(int(np.argmax(logits)))
            draft_logits_list.append(logits)
        
        if len(draft_tokens) == 0: break
        
        accepted_count = 0
        for i, draft_token in enumerate(draft_tokens):  # main verifies
            prev_token = generated[-1] if i == 0 else draft_tokens[i - 1]
            main_logits = main_model.decode_one(prev_token, main_cache)
            stats["main_forward_calls"] += 1
            main_top = int(np.argmax(main_logits))
            
            if draft_token == main_top:  # accept
                accepted_count += 1
                generated.append(draft_token)
                tokens_generated += 1
                stats["accepted_tokens"] += 1
            else:  # reject and sample from main
                sampled = main_top if temperature == 0 else int(np.random.choice(len(main_logits), p=softmax_with_temp(main_logits, temperature)))
                generated.append(sampled)
                tokens_generated += 1
                stats["rejected_tokens"] += 1
                break
        
        if accepted_count < len(draft_tokens):  # rollback draft cache
            draft_cache.rollback(len(draft_tokens) - accepted_count)
            if accepted_count < len(draft_tokens):
                _ = draft_model.decode_one(generated[-1], draft_cache)
                stats["draft_forward_calls"] += 1
        
        stats["total_tokens"] = tokens_generated
        if generated[-1] == 50256: break  # EOS
    
    return np.array(generated), stats


def softmax_with_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def speculative_decode_simple(draft_model, main_model, draft_cache, main_cache, prompt_tokens: np.ndarray,
                               max_tokens: int = 100, num_speculative: int = 4) -> Tuple[np.ndarray, dict]:
    generated = list(prompt_tokens)
    stats = {"total_tokens": 0, "accepted_tokens": 0, "rejected_tokens": 0, "acceptance_rate": 0.0}
    
    _ = draft_model.prefill(prompt_tokens, draft_cache)
    _ = main_model.prefill(prompt_tokens, main_cache)
    tokens_generated = 0
    
    while tokens_generated < max_tokens:
        draft_tokens, current_token = [], generated[-1]
        for _ in range(num_speculative):  # draft proposes
            logits = draft_model.decode_one(current_token, draft_cache)
            next_token = int(np.argmax(logits))
            draft_tokens.append(next_token)
            current_token = next_token
        
        accepted = 0
        for i, draft_tok in enumerate(draft_tokens):  # main verifies
            prev_tok = generated[-1] if i == 0 else draft_tokens[i-1]
            main_logits = main_model.decode_one(prev_tok, main_cache)
            main_top = int(np.argmax(main_logits))
            
            if draft_tok == main_top:  # accept
                generated.append(draft_tok)
                accepted += 1
                tokens_generated += 1
            else:  # reject
                generated.append(main_top)
                tokens_generated += 1
                rollback = len(draft_tokens) - i - 1
                if rollback > 0: draft_cache.rollback(rollback)
                if main_top != draft_tok: _ = draft_model.decode_one(main_top, draft_cache)
                break
        else:  # all accepted
            main_logits = main_model.decode_one(draft_tokens[-1], main_cache)
            generated.append(int(np.argmax(main_logits)))
            tokens_generated += 1
        
        stats["accepted_tokens"] += accepted
        stats["rejected_tokens"] += (min(len(draft_tokens), tokens_generated) - accepted)
        stats["total_tokens"] = tokens_generated
        if generated[-1] == 50256: break
    
    total = stats["accepted_tokens"] + stats["rejected_tokens"]
    stats["acceptance_rate"] = stats["accepted_tokens"] / max(total, 1)
    return np.array(generated), stats

