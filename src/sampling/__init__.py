# Sampling module
from .sampling import greedy, top_k, top_p
from .speculative import speculative_decode, speculative_decode_simple

__all__ = ["greedy", "top_k", "top_p", "speculative_decode", "speculative_decode_simple"]

