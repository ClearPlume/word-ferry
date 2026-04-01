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
            query: Tensor,
            kv: Tensor,
            query_causal_mask: Tensor | None = None,
            kv_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if query is kv:
            query = kv = value = query.transpose(1, 0)
        else:
            query, kv = (x.transpose(1, 0) for x in (query, kv))
            value = kv

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            kv,
            value,
            self.d_model,
            self.n_head,
            self.in_proj_weight,
            self.in_proj_bias,
            None,
            None,
            False,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=kv_valid_mask,
            need_weights=False,
            attn_mask=query_causal_mask,
            average_attn_weights=True,
            is_causal=query_causal_mask is not None,
        )

        return attn_output.transpose(1, 0), attn_output_weights
