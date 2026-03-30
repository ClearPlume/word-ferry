import copy

from torch import Tensor
from torch.nn import TransformerDecoderLayer, ModuleList, Module


class CachedDecoder(Module):
    """支持KV Cached的Decoder"""

    layers: ModuleList

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(
            self,
            decoder_in_embedded: Tensor,
            memory: Tensor,
            decoder_in_causal_mask: Tensor,
            decoder_in_valid_mask: Tensor,
            memory_valid_mask: Tensor,
    ) -> Tensor:
        output = decoder_in_embedded

        for layer in self.layers:
            output = layer(
                output,
                memory,
                decoder_in_causal_mask,
                tgt_key_padding_mask=decoder_in_valid_mask,
                memory_key_padding_mask=memory_valid_mask,
            )

        return output
