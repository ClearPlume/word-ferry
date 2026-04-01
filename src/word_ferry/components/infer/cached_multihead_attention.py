from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.modules.linear import Linear


class CachedMultiheadAttention(nn.Module):
    """支持KV Cached的CachedMultiheadAttention"""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model

        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_model // n_head

        self.in_proj_weight = Parameter(torch.empty((3 * d_model, d_model)))
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)

        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)

        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, query: Tensor, kv: Tensor, query_causal_mask: Optional[Tensor], kv_valid_mask: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        d_head = self.d_model // self.n_head
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = kv.shape

        attn_mask = kv_valid_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = torch.where(attn_mask, torch.finfo(query.dtype).min, 0.0)

        if query is kv:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            w_q, w_kv = self.in_proj_weight.split([self.d_model, self.d_model * 2])
            b_q, b_kv = self.in_proj_bias.split([self.d_model, self.d_model * 2])
            q = F.linear(query, w_q, b_q)
            k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)

        if query_causal_mask is not None:
            causal = query_causal_mask.unsqueeze(0).unsqueeze(0)
            causal = torch.where(causal, torch.finfo(query.dtype).min, 0.0)
            attn_mask = attn_mask + causal

        q = q.view(batch_size, q_len, self.n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.n_head, d_head).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, None
