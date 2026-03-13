import math


def d_model_calculator(
        vocab_size: int,
        total_token: int,
        token_per_param: float,
        n_encoder_layers: int,
        n_decoder_layers: int = 0,
        sharing_embedding: bool = False,
        ffn_ratio: int = 4,
        separate_lm_head: bool = False,
) -> int:
    """
    依据层数反推对应维度
    
    :param vocab_size: 词表大小
    :param total_token: 总token数(样本数 * 样本平均token长度)
    :param token_per_param: token参数比(5/10/20/50/100)，每参数从多少token处学习；样本信息密度极高时，可能出现一个参数只需要不到1个token即可充分学习的情况——也就是说，一个token即可让多个参数学习
    :param n_encoder_layers: 基于任务复杂度推断层数
    :param n_decoder_layers: 基于任务复杂度推断层数
    :param sharing_embedding: 编码和解码是否共享嵌入空间
    :param ffn_ratio: FFN内部维度之于d_model的倍数
    :param separate_lm_head: 输出投影层是否独立

    :return: 对齐到64倍数的维度
    """
    encoder_layer_coef = 4 + 2 * ffn_ratio
    decoder_layer_coef = 8 + 2 * ffn_ratio

    # 二次方程系数
    if n_decoder_layers > 0:
        # encoder-decoder架构
        a = encoder_layer_coef * n_encoder_layers + decoder_layer_coef * n_decoder_layers
        # 无论是否编解码是否共享权重，lm_head都可独立，需要单独计算
        b = vocab_size * ((1 if sharing_embedding else 2) + (1 if separate_lm_head else 0))
    else:
        # encoder-only架构
        if separate_lm_head:
            print("encoder-only架构下不存在 lm_head 结构，参数 `separate_lm_head=True` 无作用")

        a = encoder_layer_coef * n_encoder_layers
        b = vocab_size

    c = -(total_token / token_per_param)

    # 求根公式
    discriminant = b ** 2 - 4 * a * c
    d_model_raw = (-b + math.sqrt(discriminant)) / (2 * a)

    # 对齐到64倍
    if d_model_raw < 32:
        raise ValueError("任务可能不适合transformer，考虑更简单的架构")
    elif d_model_raw < 64:
        print("警告：接近transformer下限，建议检查配置")
        d_model = 64
    else:
        d_model = math.ceil(d_model_raw / 64) * 64

    return d_model
