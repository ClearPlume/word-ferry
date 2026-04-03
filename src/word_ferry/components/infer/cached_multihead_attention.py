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

    def forward(
            self,
            is_self_attention: bool,
            query: Tensor,
            kv: Tensor,
            query_causal_mask: Optional[Tensor],
            kv_valid_mask: Tensor,
            cache: tuple[Optional[Tensor], Optional[Tensor]],
    ) -> tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]]]:
        d_head = self.d_model // self.n_head
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = kv.shape

        attn_mask = kv_valid_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = torch.where(attn_mask, torch.finfo(query.dtype).min, 0.0)

        w_q, w_kv = self.in_proj_weight.split([self.d_model, self.d_model * 2])
        b_q, b_kv = self.in_proj_bias.split([self.d_model, self.d_model * 2])
        q = F.linear(query, w_q, b_q)

        if query_causal_mask is not None:
            causal = query_causal_mask.unsqueeze(0).unsqueeze(0)
            causal = torch.where(causal, torch.finfo(query.dtype).min, 0.0)
            attn_mask = attn_mask + causal

        q = q.view(batch_size, q_len, self.n_head, d_head).transpose(1, 2)

        cache_k, cache_v = cache
        if is_self_attention:
            k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
            k = k.view(batch_size, kv_len, self.n_head, d_head).transpose(1, 2)
            v = v.view(batch_size, kv_len, self.n_head, d_head).transpose(1, 2)
            if cache_k is not None and cache_v is not None:
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)

            cache = (k, v)
        else:
            if cache_k is not None and cache_v is not None:
                k = cache_k
                v = cache_v
            else:
                k, v = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
                k = k.view(batch_size, kv_len, self.n_head, d_head).transpose(1, 2)
                v = v.view(batch_size, kv_len, self.n_head, d_head).transpose(1, 2)
                cache = (k, v)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, cache
