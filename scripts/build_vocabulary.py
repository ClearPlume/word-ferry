import math
import random
from typing import Callable

import numpy as np
import sentencepiece as spm

from scripts.d_model_calculator import d_model_calculator
from word_ferry.core.constants import PAD_TOKEN_ID, PAD_TOKEN, VOCAB_SIZE, ZH_TOKEN, EN_TOKEN, FR_TOKEN, \
    TOKEN_PER_PARAM, N_ENCODER_LAYERS, N_DECODER_LAYERS
from word_ferry.path import get_data_dir

data_dir = get_data_dir()
(data_dir / "vocab").mkdir(parents=True, exist_ok=True)

# 样本文件路径
samples_dir = data_dir / "samples"

if not samples_dir.exists():
    raise FileNotFoundError("Samples directory does not exist")


def train():
    sample_file = f"{samples_dir}/samples.txt"

    spm.SentencePieceTrainer.Train(
        input=sample_file,
        model_prefix=f"{data_dir}/vocab/word_ferry",
        model_type="unigram",
        vocab_size=VOCAB_SIZE,
        max_sentencepiece_length=256,
        max_sentence_length=8000,
        pad_id=PAD_TOKEN_ID,
        pad_piece=PAD_TOKEN,
        user_defined_symbols=[ZH_TOKEN, EN_TOKEN, FR_TOKEN],
        shuffle_input_sentence=True,
        character_coverage=1,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        split_digits=True,
        add_dummy_prefix=False,
    )


def verify(
        full_sample: list[str],
        tokenizer: spm.SentencePieceProcessor,
        vocab_size: int,
        extractors: list[tuple[str, Callable[[list[str]], list[str]]]] | None = None,
):
    """
    Tokenizer分析，评估分词器编码效率

    :param full_sample: 全部样本数据
    :param tokenizer: Tokenizer
    :param vocab_size: 词表大小
    :param extractors: 多组(名称, 数据筛选器)对，筛选器从完整数据中按条件过滤子集，再由sample_size控制采样量
    """
    if extractors == []:
        raise ValueError("extractors不能为空列表，传入None以使用默认配置")

    if extractors is None:
        extractors = [("默认数据集", lambda data: data)]

    sample_size = 10000

    # 多 extractor 时汇总词表使用率
    all_used_token_ids = set()

    for name, extractor in extractors:
        filtered_sample = extractor(full_sample)

        actual_size = min(len(filtered_sample), sample_size)
        test_cases = random.sample(filtered_sample, k=actual_size)

        # 评估指标
        char_lengths = []
        token_lengths = []
        used_token_ids = set()

        for test_case in test_cases:
            encoded = tokenizer.Encode(test_case)
            token_lengths.append(len(encoded))
            char_lengths.append(len(test_case))
            used_token_ids.update(encoded)

        all_used_token_ids.update(used_token_ids)

        # 统计分析
        vocab_usage = len(used_token_ids) / vocab_size * 100
        zip_ratio = sum(char_lengths) / sum(token_lengths)

        # 子报告
        print()
        print(f"{'=' * 60}")
        print(f"  [{name}] 词表评估报告")
        print(f"{'=' * 60}")
        print(f"  词表大小: {vocab_size}")
        print(f"  测试样本数: {len(test_cases)}")
        print(f"-" * 60)
        print(f"  压缩率 (chars/token): {zip_ratio:.2f}")
        print(f"    参考: 中文 ~1.5-2.0 | 拉丁语系 ~3.0-5.0")
        print(f"  本组词表使用率: {vocab_usage:.2f}% ({len(used_token_ids)}/{vocab_size})")
        if len(extractors) == 1:
            print(f"    参考: 小于 50% 可能冗余 | 超出 95% 可能不够")
        print(f"{'=' * 60}")

    # 汇总报告（多 extractor 时）
    if len(extractors) > 1:
        overall_usage = len(all_used_token_ids) / vocab_size * 100
        print()
        print(f"{'=' * 60}")
        print(f"  汇总")
        print(f"{'=' * 60}")
        print(f"  词表总使用率: {overall_usage:.2f}% ({len(all_used_token_ids)}/{vocab_size})")
        print(f"    参考: 小于 50% 可能冗余 | 超出 95% 可能不够")
        print(f"{'=' * 60}")


def extract_target_with_locale(locale: str, data: list[str]) -> list[str]:
    targets: list[str] = list(map(lambda line: line.split("\n")[1], data))
    return list(filter(lambda line: line.startswith(locale), targets))


def main():
    # train()
    full_sample = (samples_dir / "samples.txt").read_text(encoding="utf-8").split("\n\n")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f"{data_dir}/vocab/word_ferry.model")

    verify(
        full_sample,
        tokenizer,
        VOCAB_SIZE,
        [
            ("en", lambda data: extract_target_with_locale("<en>", data)),
            ("zh", lambda data: extract_target_with_locale("<zh>", data)),
            ("fr", lambda data: extract_target_with_locale("<fr>", data)),
        ]
    )

    sample_size = min(10000, len(full_sample))
    overall_cases = random.sample(full_sample, k=min(len(full_sample), sample_size))
    lengths = [len(tokenizer.Encode(s)) for s in overall_cases]

    # 百分位分布
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    values = np.percentile(lengths, percentiles)

    print("\n长度分布:")
    for p, v in zip(percentiles, values):
        print(f"  P{p:>5}: {v:.0f}")

    # 从 P95/P99 推导候选 max_len（向上取最近的 2 的幂）
    candidates = set()
    for p in [95, 99]:
        v = np.percentile(lengths, p)
        power = 2 ** math.ceil(math.log2(v))
        candidates.add(power)
        candidates.add(power // 2)  # 再紧凑一档也列出来

    candidates = sorted(candidates)
    print("\n候选 max_len 截断分析:")
    for c in candidates:
        truncated = (np.array(lengths) > c).mean()
        print(f"  max_len={c:>5} → 截断 {truncated:.2%} 样本")

    print(f"{'=' * 60}")

    test_chars = [chr(c) for c in range(0x4E00, 0x9FFF + 1)]  # CJK统一汉字基本区，20992字，远超日常需要
    unk_chars = [c for c in test_chars if tokenizer.PieceToId(c) == tokenizer.unk_id()]
    print(f"UNK汉字: {len(unk_chars)}/{len(test_chars)}")
    if unk_chars:
        print(f"示例: {''.join(random.sample(unk_chars, 200))}")

    print(f"{'=' * 60}")

    total_token = len(full_sample) * (sum(lengths) // len(lengths))
    print(f"total_token: {total_token}")
    d_model = d_model_calculator(
        tokenizer.vocab_size(),
        total_token,
        TOKEN_PER_PARAM,
        N_ENCODER_LAYERS,
        N_DECODER_LAYERS,
    )
    print(f"推荐维度：{d_model}")


if __name__ == "__main__":
    main()
