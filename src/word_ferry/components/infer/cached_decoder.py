import copy
from typing import Optional

from torch import Tensor
from torch.nn import ModuleList, Module

from word_ferry.components.infer.cached_decoder_layer import CachedDecoderLayer


class CachedDecoder(Module):
    """支持KV Cached的Decoder"""

    layers: ModuleList

    def __init__(self, decoder_layer: CachedDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(
            self,
            decoder_in: Tensor,
            decoder_in_causal_mask: Tensor,
            decoder_in_valid_mask: Tensor,
            memory: Tensor,
            memory_valid_mask: Tensor,
            caches: list[Optional[tuple[Tensor, Tensor, Tensor, Tensor]]],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor, Tensor, Tensor]]]:
        output = decoder_in

        for idx, layer in enumerate(self.layers):
            output, layer_cache = layer(
                output,
                decoder_in_causal_mask,
                decoder_in_valid_mask,
                memory,
                memory_valid_mask,
                caches[idx],
            )
            caches[idx] = layer_cache

        caches: list[tuple[Tensor, Tensor, Tensor, Tensor]]
        return output, caches
