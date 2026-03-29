import copy

from torch import Tensor
from torch.nn import TransformerDecoderLayer, ModuleList, Module


class CachedDecoder(Module):
    """支持KV Cached的Decoder"""

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_key_padding_mask: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=True,
            )

        return output
