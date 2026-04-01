from typing import Callable

import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Linear, Dropout, LayerNorm

from word_ferry.components.infer.cached_multihead_attention import CachedMultiheadAttention


class CachedDecoderLayer(nn.Module):
    """支持KV Cached的CachedDecoderLayer"""

    self_attn: CachedMultiheadAttention
    multihead_attn: CachedMultiheadAttention

    linear1: Linear
    dropout: Dropout
    linear2: Linear

    norm1: LayerNorm
    norm2: LayerNorm
    norm3: LayerNorm
    dropout1: Dropout
    dropout2: Dropout
    dropout3: Dropout

    activation: Callable[[Tensor, bool], Tensor]

    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.self_attn = CachedMultiheadAttention(d_model, n_head, dropout)
        self.multihead_attn = CachedMultiheadAttention(d_model, n_head, dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = F.relu

    def forward(
            self,
            decoder_in: Tensor,
            decoder_in_causal_mask: Tensor,
            decoder_in_valid_mask: Tensor,
            memory: Tensor,
            memory_valid_mask: Tensor,
    ) -> Tensor:
        x = decoder_in

        x = x + self._self_attn_block(self.norm1(x), decoder_in_causal_mask, decoder_in_valid_mask)
        x = x + self._cross_attn_block(self.norm2(x), memory, memory_valid_mask)
        x = x + self._feed_forward_block(self.norm3(x))

        return x

    def _self_attn_block(self, x: Tensor, decoder_in_causal_mask: Tensor, decoder_in_valid_mask: Tensor) -> Tensor:
        x = self.self_attn(x, x, decoder_in_causal_mask, decoder_in_valid_mask)[0]
        return self.dropout1(x)

    def _cross_attn_block(self, x: Tensor, memory: Tensor, memory_valid_mask: Tensor) -> Tensor:
        x = self.multihead_attn(x, memory, None, memory_valid_mask)[0]
        return self.dropout2(x)

    def _feed_forward_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
