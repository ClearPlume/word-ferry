import torch
from torch.nn import functional as torch_f

from word_ferry.components.dataset import TokenizedTransSample, BatchedTransSample
from word_ferry.core.constants import PAD_TOKEN_ID


def collate_fn(batch: list[TokenizedTransSample]) -> BatchedTransSample:
    """
    对批次内的token序列进行padding对齐，并生成padding_mask

    :param batch: 一个批次中，元素为单个token的列表
    :return 经过了padding，设置了mask，重组为张量的训练语料
    """

    # 分离源序列与目标序列
    src_list = [batch.input for batch in batch]
    tgt_list = [batch.target for batch in batch]

    # Padding源序列
    src_max_len = max(len(src) for src in src_list)
    padded_src = []
    src_attention_mask = []

    for src in src_list:
        padding_len = src_max_len - len(src)
        # 文本右填充
        padded = torch_f.pad(src, (0, padding_len), value=PAD_TOKEN_ID)
        # mask = 原始长度的1 + 填充长度的0
        mask = torch.cat([torch.ones(len(src)), torch.zeros(padding_len)])

        padded_src.append(padded)
        src_attention_mask.append(mask)

    # Padding目标序列，需分为输入和输出两部分
    tgt_max_len = max(len(tgt) for tgt in tgt_list) - 1  # 去除BOS或者EOS之后的最大长度
    padded_tgt_in = []
    padded_tgt_out = []
    tgt_attention_mask = []

    for tgt in tgt_list:
        # 去掉最后一个token作为解码器输入 [BOS] + tokens[:-1]
        tgt_in = tgt[:-1]
        # 去掉第一个token作为目标输出 tokens + [EOS]
        tgt_out = tgt[1:]

        # Padding
        in_padding_len = tgt_max_len - len(tgt_in)
        out_padding_len = tgt_max_len - len(tgt_out)

        # 文本右填充
        padded_in = torch_f.pad(tgt_in, (0, in_padding_len), value=PAD_TOKEN_ID)
        padded_out = torch_f.pad(tgt_out, (0, out_padding_len), value=PAD_TOKEN_ID)

        # mask = 原始长度的1 + 填充长度的0
        mask = torch.cat([torch.ones(len(tgt_in)), torch.zeros(out_padding_len)])

        padded_tgt_in.append(padded_in)
        padded_tgt_out.append(padded_out)
        tgt_attention_mask.append(mask)

    return BatchedTransSample(
        torch.stack(padded_src),
        torch.stack(src_attention_mask),
        torch.stack(padded_tgt_in),
        torch.stack(padded_tgt_out),
        torch.stack(tgt_attention_mask),
    )
