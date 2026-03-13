from dataclasses import dataclass
from enum import Enum

import torch


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

    min_delta: float = 1e-6
    max_len: int = 2048
    max_grad_norm: float = 1.0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self,
            d_model: int,
            n_head: int,
            n_encoder_layers: int,
            ffn_ratio: int,
            n_decoder_layers: int,
            learning_rate: float,
            dropout: float,
            max_dropout: float,
            batch_size: int,
    ):
        self.d_model = d_model
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.ffn_ratio = ffn_ratio
        self.n_decoder_layers = n_decoder_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_dropout = max_dropout
        self.batch_size = batch_size

    def __str__(self):
        return (
            f"{self.d_model}d×{self.n_head}h×{self.n_encoder_layers}L | "
            f"lr={self.learning_rate} | dropout={self.dropout}→{self.max_dropout} | "
            f"batch={self.batch_size}"
        )

    @staticmethod
    def default(learning_rate=8e-5, dropout=0.2, max_dropout=0.4, batch_size=32):
        return Config(
            d_model=128,
            n_head=2,
            n_encoder_layers=4,
            ffn_ratio=4,
            n_decoder_layers=4,
            learning_rate=learning_rate,
            dropout=dropout,
            max_dropout=max_dropout,
            batch_size=batch_size
        )
