from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Embedding, TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, Linear
from torch.nn.functional import softmax

from word_ferry.components.cached_decoder import CachedDecoder
from word_ferry.components.config import Config
from word_ferry.components.tokenizer import Tokenizer
from word_ferry.core.constants import PAD_TOKEN_ID
from word_ferry.path import get_models_dir


class Model(Module):
    tokenizer: Tokenizer
    config: Config
    embedding: Embedding
    pos_encoding: Embedding
    encoder: TransformerEncoder
    decoder: CachedDecoder

    def __init__(self, tokenizer: Tokenizer, config: Config):
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config

        # 词嵌入，将词表中的每个token都映射到d_model维度的向量空间，初始值为随机浮点值
        # [[e0_0, e0_1, e0_2, ..., e0_{d_model-1}]  <- tokens[v0]
        #  [e1_0, e1_1, e1_2, ..., e1_{d_model-1}]  <- tokens[v1]
        #  [e2_0, e2_1, e2_2, ..., e2_{d_model-1}]  <- tokens[v2]
        #                     ...
        #  [eN_0, eN_1, eN_2, ..., eN_{d_model-1}]] <- tokens[v_{vocab_size-1}]
        # [vocab_size, d_model]
        self.embedding = Embedding(tokenizer.vocab_size, config.d_model, PAD_TOKEN_ID)
        # 位置编码
        # [max_len, d_model], batch维广播自动补齐
        self.pos_encoding = Embedding(config.max_len, config.d_model)

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_model * config.ffn_ratio,
            dropout=config.initial_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, config.n_encoder_layers, enable_nested_tensor=False)

        # Transformer解码器
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_model * config.ffn_ratio,
            dropout=config.initial_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = CachedDecoder(decoder_layer, config.n_decoder_layers)

        # 输出投影层
        self.output_projection = Linear(config.d_model, tokenizer.vocab_size)

        # 词嵌入和输出投影共享权重
        self.output_projection.weight = self.embedding.weight

        self.to(config.device)

    def forward(
            self,
            src: Tensor,
            src_attention_mask: Tensor,
            tgt: Optional[Tensor] = None,
            tgt_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        前向传播
        
        :param src: 源序列 [batch_size, src_len]
        :param src_attention_mask: 源序列attention_mask
        :param tgt: 目标序列 [batch_size, tgt_len]
        :param tgt_attention_mask: 目标序列attention_mask
        :return: 词汇表上的原始分数 [batch_size, tgt_len, vocab_size]
        """
        memory, src_kpm = self.encode(src, src_attention_mask)

        if tgt is None:
            return memory

        return self.decode(tgt, memory, src_kpm, tgt_attention_mask)

    def encode(self, src: Tensor, src_attention_mask: Tensor):
        src_attention_mask = src_attention_mask == 0

        # 编码
        src_len = src.shape[1]
        src_embedded = self.embedding(src)
        src_pos = torch.arange(src_len, device=src.device)
        src_pe = self.pos_encoding(src_pos)
        src_embedded = src_embedded + src_pe

        memory = self.encoder(src_embedded, src_key_padding_mask=src_attention_mask)
        return memory, src_attention_mask

    def decode(
            self,
            tgt: Tensor,
            memory: Tensor,
            src_attention_mask: Tensor,
            tgt_attention_mask: Tensor = None,
            last_only: bool = False,
    ) -> Tensor:
        if tgt_attention_mask is None:
            tgt_attention_mask = torch.ones(tgt.shape, device=tgt.device)

        tgt_attention_mask = tgt_attention_mask == 0

        tgt_len = tgt.shape[1]
        tgt_embedded = self.embedding(tgt)
        tgt_pos = torch.arange(tgt_len, device=tgt.device)
        tgt_pe = self.pos_encoding(tgt_pos)
        tgt_embedded = tgt_embedded + tgt_pe

        # 目标因果mask
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()

        decoder_output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask,
            tgt_key_padding_mask=tgt_attention_mask,
            memory_key_padding_mask=src_attention_mask,
        )

        if last_only:
            decoder_output = decoder_output[:, -1:, :]

        # 输出投影
        return self.output_projection(decoder_output)

    @torch.no_grad()
    def generate(
            self,
            src: Tensor,
            src_mask: Tensor,
            lang_tokens: Tensor,
            do_sample: bool = False,
            temperature: float = 1.0,
    ) -> Tensor:
        """一次完整推理"""
        device = src.device
        batch_size = src.size(0)

        # encoder 只跑一次
        memory, src_key_padding_mask = self.encode(src, src_mask)

        # 初始化解码序列 [zh/en/fr]
        generated = torch.full(
            (batch_size, self.config.max_len + 1),  # +1 给 lang token
            PAD_TOKEN_ID, dtype=torch.long, device=device,
        )
        generated[:, 0] = lang_tokens
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        length = 1  # 已经有 lang token
        for t in range(1, self.config.max_len + 1):
            logits = self.decode(generated[:, :t], memory, src_key_padding_mask, last_only=True, )
            next_logits = logits.squeeze(1)  # [b, 48000]

            if do_sample:
                next_tokens = torch.multinomial(softmax(next_logits / temperature), 1, ).squeeze(1)
            else:
                next_tokens = next_logits.argmax(dim=-1)

            next_tokens[finished] = PAD_TOKEN_ID
            generated[:, t] = next_tokens
            finished |= (next_tokens == self.tokenizer.eos_token_id)

            length = t + 1
            if finished.all():
                break

        return generated[:, :length]

    @property
    def param_num(self) -> str:
        return f"{sum(p.numel() for p in self.parameters()) / 1e6:.1f}M"

    def load(self, checkpoint_name: str, weight_only: bool = False, model_name: str = None):
        if weight_only:
            checkpoint_dir = get_models_dir() / f"{model_name}.pt"
        else:
            checkpoint_dir = get_models_dir() / f"checkpoint/{checkpoint_name}/checkpoint_{checkpoint_name}_best.pt"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        checkpoint = torch.load(checkpoint_dir)
        if weight_only:
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint["model_state"])
