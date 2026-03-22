import torch

from word_ferry.components.config import Config
from word_ferry.components.logger import setup_logger
from word_ferry.components.model import Model
from word_ferry.components.tokenizer import Tokenizer
from word_ferry.core.constants import EN_TOKEN
from word_ferry.path import get_data_dir, get_models_dir


def interactive_test(name: str):
    """交互测试模式：支持单样本实时预测"""
    config = Config.default(
        learning_rate=3e-4,
        initial_dropout=0.2,
        dropout_factor=0.1,
        max_dropout=0.4,
        dropout_window=3,
        dropout_cooldown=3,
        batch_size=40,
        max_len=256,
    )
    device = config.device
    tokenizer = Tokenizer()
    logger = setup_logger(name, "train")

    model = Model(tokenizer, config)
    model.load(name)
    model.eval()

    logger.info("=" * 60)
    logger.info("🎯 交互测试模式")
    logger.info(f"    Checkpoint: {name}")
    logger.info(f"    模型架构: {config.arch_str}")
    logger.info(f"    训练参数: {config.train_str}")
    logger.info(f"    参数规模: {model.param_num}")
    logger.info("=" * 60)
    logger.info("输入以开始测试 (输入 'quit' 退出):")

    while True:
        data = input("\n输入> ").strip()

        if data.lower() in ["quit", "exit", "q"]:
            print("退出交互测试")
            break

        if not data:
            continue

        # Tokenize
        input_ids = torch.tensor(tokenizer.encode(data, False)[:config.max_len]).unsqueeze(0)
        attention_mask = torch.ones(input_ids.shape[1]).unsqueeze(0)
        lang_tokens = torch.tensor(tokenizer.encode(EN_TOKEN, False)).unsqueeze(0)

        # 推理
        with torch.no_grad():
            generated = model.generate(input_ids.to(device), attention_mask.to(device), lang_tokens.to(device))

        result = []
        # generated: [batch_size, seq_len]，每行是 [lang, t1, t2, ..., EOS, PAD, PAD, ...]
        for seq in generated:
            token_ids = seq.tolist()
            # 去掉语言标签，截断到 EOS
            if tokenizer.eos_token_id in token_ids:
                token_ids = token_ids[1:token_ids.index(tokenizer.eos_token_id)]
            else:
                token_ids = token_ids[1:]
            result.append(tokenizer.decode(token_ids))

        # 输出结果
        print(result)


def main():
    interactive_test("2026-03-20")


if __name__ == '__main__':
    data_dir = get_data_dir()
    model_dir = get_models_dir()
    main()
