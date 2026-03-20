from dataclasses import dataclass
from enum import Enum

import torch

from word_ferry.core.constants import D_MODEL, N_HEAD, N_ENCODER_LAYERS, FFN_RATIO, N_DECODER_LAYERS


class CorpusType(Enum):
    # 1500W
    EN_ZH = "en-zh"
    # 2500W
    EN_FR = "en-fr"
    # 1500W
    FR_ZH = "fr-zh"


class ResumeStrategy(Enum):
    """
    训练恢复策略枚举
      
    定义从检查点恢复训练时需要加载哪些组件的策略。
    用于控制模型权重、优化器状态、学习率调度器和Dropout调度器等组件的恢复行为。
    
    Attributes:
        ALL_COMPONENTS: 恢复所有组件
        EXCLUDE_OPTIMIZATION: 排除优化相关（优化器+LR调度器）
        EXCLUDE_REGULARIZATION: 排除正则化相关（Dropout调度器）
        MODEL_WEIGHTS_ONLY: 仅模型权重
    """

    ALL_COMPONENTS = 0
    EXCLUDE_OPTIMIZATION = 1
    EXCLUDE_REGULARIZATION = 2
    MODEL_WEIGHTS_ONLY = 3


@dataclass
class Config:
    """模型配置信息"""

    d_model: int
    n_head: int
    n_encoder_layers: int
    ffn_ratio: int
    n_decoder_layers: int
    learning_rate: float
    dropout: float
    max_dropout: float
    batch_size: int
    max_len: int

    min_improvement: float = 1e-6
    max_grad_norm: float = 1.0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def arch_str(self):
        layers_parts = []
        if self.n_encoder_layers > 0:
            layers_parts.append(f"enc={self.n_encoder_layers}")
        if self.n_decoder_layers > 0:
            layers_parts.append(f"dec={self.n_decoder_layers}")

        return f"dim={self.d_model} | head={self.n_head} | {' | '.join(layers_parts)} | ffn={self.ffn_ratio}"

    @property
    def train_str(self):
        return (
            f"lr={self.learning_rate:.2e} | dropout={self.dropout}→{self.max_dropout} | "
            f"batch={self.batch_size} | max_len={self.max_len} | grad_norm={self.max_grad_norm}"
        )

    @staticmethod
    def default(learning_rate: float, dropout: float, max_dropout: float, batch_size: int, max_len: int):
        return Config(
            d_model=D_MODEL,
            n_head=N_HEAD,
            n_encoder_layers=N_ENCODER_LAYERS,
            ffn_ratio=FFN_RATIO,
            n_decoder_layers=N_DECODER_LAYERS,
            learning_rate=learning_rate,
            dropout=dropout,
            max_dropout=max_dropout,
            batch_size=batch_size,
            max_len=max_len,
        )
