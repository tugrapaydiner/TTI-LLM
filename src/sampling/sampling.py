# Sampling strategies: greedy, top-k, top-p
import numpy as np
import sys
sys.path.insert(0, str(__file__).replace("sampling/sampling.py", ""))
from utils.tensor import softmax


def greedy(logits: np.ndarray) -> int:  # return highest probability token
    return int(np.argmax(logits))


def top_k(logits: np.ndarray, k: int = 40, temp: float = 1.0) -> int:  # sample from top-k tokens
    logits = logits / temp
    top_k_indices = np.argpartition(logits, -k)[-k:]
    probs = softmax(logits[top_k_indices])
    return int(top_k_indices[np.random.choice(len(top_k_indices), p=probs)])


def top_p(logits: np.ndarray, p: float = 0.9, temp: float = 1.0) -> int:  # nucleus sampling
    logits = logits / temp
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cutoff_idx = np.searchsorted(np.cumsum(sorted_probs), p) + 1
    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_probs = sorted_probs[:cutoff_idx] / sorted_probs[:cutoff_idx].sum()
    return int(top_p_indices[np.random.choice(len(top_p_indices), p=top_p_probs)])

