# GPT-2 tokenizer wrapper using HuggingFace
from typing import List, Union
import numpy as np


class Tokenizer:
    def __init__(self, model_id: str = "gpt2"):
        try:
            from transformers import GPT2Tokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.eos_token_id = self._tokenizer.eos_token_id
        self.bos_token_id = self._tokenizer.bos_token_id
        self.pad_token_id = self._tokenizer.pad_token_id
        self.vocab_size = self._tokenizer.vocab_size
    
    def encode(self, text: str) -> np.ndarray:  # text -> token IDs
        return np.array(self._tokenizer.encode(text), dtype=np.int64)
    
    def decode(self, token_ids: Union[List[int], np.ndarray]) -> str:  # token IDs -> text
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids)
    
    def decode_single(self, token_id: int) -> str:  # single token -> text
        return self._tokenizer.decode([token_id])


def get_tokenizer(model_id: str = "gpt2") -> Tokenizer:
    return Tokenizer(model_id)

