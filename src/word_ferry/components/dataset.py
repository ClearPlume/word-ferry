import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, Generator
from torch.utils.data import Dataset, Subset, random_split

from word_ferry.components.config import Config
from word_ferry.components.tokenizer import Tokenizer
from word_ferry.core.constants import VAL_RATIO, TEST_RATIO, RANDOM_SEED


@dataclass
class TransSample:
    """
    单个原始翻译样本
    
    Attributes:
        input: How are you?
        target: <zh>你怎么样呀？
    """

    input: str
    target: str


@dataclass
class TokenizedTransSample:
    """
    单个翻译样本，已tokenized
    
    Attributes:
        input: 编码后的输入tokens
        target: 编码后的输出tokens
    """
    input: Tensor
    target: Tensor


@dataclass
class BatchedTransSample:
    """
    分组后，批次内数据padding之后的结构
    
    Attributes:
        src: [batch, src_max_len]
        src_attention_mask: [batch, src_max_len]
        tgt_in: [batch, tgt_max_len]
        tgt_out: [batch, tgt_max_len]
        tgt_attention_mask: [batch, tgt_max_len]
    """

    src: Tensor
    src_attention_mask: Tensor
    tgt_in: Tensor
    tgt_out: Tensor
    tgt_attention_mask: Tensor


class WordFerryDataset(Dataset):
    """翻译数据集"""

    sample_file: Path
    tokenizer: Tokenizer
    config: Config
    offsets: list[int]

    def __init__(self, sample_file: Path, tokenizer: Tokenizer, config: Config):
        self.sample_file = sample_file
        self.tokenizer = tokenizer
        self.config = config
        self._prepare_offset()

    def _prepare_offset(self):
        cache_file = self.sample_file.with_suffix(".offset.pkl")

        if cache_file.exists() and cache_file.stat().st_mtime > self.sample_file.stat().st_mtime:
            with open(cache_file, "rb") as cache:
                self.offsets = pickle.load(cache)
            return

        print("cache not exists or failure, rebuilding.")
        self.offsets = []

        with open(self.sample_file, "rb") as _file:
            while True:
                pos = _file.tell()
                if not _file.readline():
                    break
                _file.readline()
                _file.readline()
                self.offsets.append(pos)

        cache_file.write_bytes(pickle.dumps(self.offsets))

    def __len__(self):
        return len(self.offsets)

    # 样本文件结构为：source\n<zh>目标\n\n源\n<en>target
    def __getitem__(self, idx) -> TokenizedTransSample:
        with open(self.sample_file, "rb") as _file:
            _file.seek(self.offsets[idx])
            source = _file.readline().decode("utf-8").strip()
            target = _file.readline().decode("utf-8").strip()

        # [seq_len]
        seq_encoded = self.tokenizer.encode(source, False)

        # [min(max_len, seq_len)]
        if len(seq_encoded) > self.config.max_len:
            seq_encoded = seq_encoded[:self.config.max_len]

        # [tgt_len]
        tgt_encoded = self.tokenizer.encode(target, True)

        # [min(max_len, tgt_len)]
        if len(tgt_encoded) > self.config.max_len:
            tgt_encoded = tgt_encoded[:self.config.max_len]
            tgt_encoded[-1] = self.tokenizer.eos_token_id

        return TokenizedTransSample(torch.tensor(seq_encoded), torch.tensor(tgt_encoded))

    def split(self) -> list[Subset["WordFerryDataset"]]:
        total = len(self.offsets)
        val_size = int(total * VAL_RATIO)
        test_size = int(total * TEST_RATIO)
        train_size = total - val_size - test_size

        return random_split(self, [train_size, val_size, test_size], Generator().manual_seed(RANDOM_SEED))
