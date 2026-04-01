import torch
from torch import Tensor

from word_ferry.components.config import Config
from word_ferry.components.model import Model
from word_ferry.components.tokenizer import Tokenizer
from word_ferry.path import get_data_dir, get_models_dir


def run_test(model: Model, src: Tensor, attn_mask: Tensor, lang: Tensor) -> Tensor:
    device = model.config.device

    with torch.no_grad():
        generated = model.generate(src.to(device), attn_mask.to(device), lang.to(device))

    return generated


def interactive_test(model: Model, remove_lang_token=True):
    """交互测试模式：支持单样本实时预测"""
    tokenizer = model.tokenizer
    start_index = 1 if remove_lang_token else 0

    print("🎯 交互测试模式")
    print("=" * 60)
    print("输入以开始测试 (输入 'quit' 退出):")

    while True:
        print()
        data = input("输入> ").strip()

        if data.lower() in ["quit", "exit", "q"]:
            print("退出交互测试")
            break

        if not data:
            continue

        lang = input("目标语言(zh/en/fr)> ").strip()

        if lang not in ["zh", "en", "fr"]:
            print(f"available lang is: zh/en/fr, got <{lang}>")
            continue

        input_ids = torch.tensor(tokenizer.encode(data, False)[:model.config.max_len]).unsqueeze(0)
        attn_mask = torch.ones(input_ids.shape[1]).unsqueeze(0)
        lang_tokens = torch.tensor(tokenizer.encode(f"<{lang}>", False)).unsqueeze(0)

        # generated: [batch_size, seq_len]，每行是 [lang, t1, t2, ..., EOS, PAD, PAD, ...]
        generated = run_test(model, input_ids, attn_mask, lang_tokens)

        result = []
        for seq in generated:
            token_ids = seq.tolist()
            # 去掉语言标签，截断到 EOS
            if tokenizer.eos_token_id in token_ids:
                token_ids = token_ids[start_index:token_ids.index(tokenizer.eos_token_id)]
            else:
                token_ids = token_ids[start_index:]
            result.append(tokenizer.decode(token_ids))

        print(result)


def main(name: str):
    config = Config.default(
        learning_rate=3e-4,
        initial_dropout=0.2,
        dropout_factor=0.1,
        max_dropout=0.4,
        dropout_cooldown=3,
        batch_size=40,
        max_len=256,
    )
    tokenizer = Tokenizer()

    model = Model(tokenizer, config)
    model.load(name, True, "word_ferry")
    model.eval()

    print("=" * 60)
    print("? 加载模型")
    print(f"    Checkpoint: {name}")
    print(f"    模型架构: {config.arch_str}")
    print(f"    训练参数: {config.train_str}")
    print(f"    参数规模: {model.param_num}")
    print("=" * 60)
    interactive_test(model, False)


if __name__ == '__main__':
    data_dir = get_data_dir()
    model_dir = get_models_dir()
    main("2026-03-25")
