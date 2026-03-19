import os
from functools import lru_cache
from pathlib import Path

from sentencepiece import SentencePieceProcessor

from word_ferry.path import get_data_dir


class Tokenizer:
    """词表处理器"""
    tokenizer: SentencePieceProcessor

    bos_token_id: int = 1
    eos_token_id: int = 2

    def __init__(self, vocab_file: Path = get_data_dir() / "vocab/word_ferry.model"):
        if os.path.exists(vocab_file):
            self.tokenizer = SentencePieceProcessor()
            self.tokenizer.Load(str(vocab_file))
        else:
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        """编码文本为token ids"""
        return self.tokenizer.Encode(text, add_bos=add_special_tokens, add_eos=add_special_tokens)

    def decode(self, token_ids: list[int]) -> str:
        """解码token ids为文本"""
        return self.tokenizer.Decode(token_ids)

    @property
    @lru_cache(maxsize=1)
    def vocab_size(self):
        return self.tokenizer.GetPieceSize()
