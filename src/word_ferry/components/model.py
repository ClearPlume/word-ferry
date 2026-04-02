import torch
from torch import Tensor
from torch.nn import Module, Embedding, TransformerEncoder, TransformerEncoderLayer, Linear
from torch.nn.functional import softmax

from word_ferry.components.config import Config
from word_ferry.components.infer.cached_decoder import CachedDecoder
from word_ferry.components.infer.cached_decoder_layer import CachedDecoderLayer
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
        decoder_layer = CachedDecoderLayer(config.d_model, config.n_head, config.d_model * config.ffn_ratio, config.initial_dropout)
        self.decoder = CachedDecoder(decoder_layer, config.n_decoder_layers)

        # 输出投影层
        self.output_projection = Linear(config.d_model, tokenizer.vocab_size)

        # 词嵌入和输出投影共享权重
        self.output_projection.weight = self.embedding.weight

        self.to(config.device)

    def forward(
            self,
            encoder_in: Tensor,
            encoder_in_valid_masks: Tensor,
            decoder_in: Tensor,
            decoder_in_valid_mask: Tensor,
    ) -> Tensor:
        """
        前向传播
        
        :param encoder_in: 源序列 [batch_size, src_len]
        :param encoder_in_valid_masks: 源序列attention_mask
        :param decoder_in: 目标序列 [batch_size, tgt_len]
        :param decoder_in_valid_mask: 目标序列attention_mask
        :return: 词汇表上的原始分数 [batch_size, tgt_len, vocab_size]
        """
        memory, memory_valid_mask = self.encode(encoder_in, encoder_in_valid_masks)

        if decoder_in is None:
            return memory

        return self.decode(decoder_in, decoder_in_valid_mask, memory, memory_valid_mask)

    def encode(self, src: Tensor, src_valid_mask: Tensor):
        memory_valid_mask = src_valid_mask == 0

        # 编码
        src_len = src.shape[1]
        src_embedded = self.embedding(src)
        src_pos = torch.arange(src_len, device=src.device)
        src_pe = self.pos_encoding(src_pos)
        src_embedded = src_embedded + src_pe

        memory = self.encoder(src_embedded, src_key_padding_mask=memory_valid_mask)
        return memory, memory_valid_mask

    def decode(
            self,
            decoder_in: Tensor,
            decoder_in_valid_mask: Tensor,
            memory: Tensor,
            memory_valid_mask: Tensor,
            last_only: bool = False,
    ) -> Tensor:
        decoder_in_len = decoder_in.shape[1]
        decoder_in_embedded = self.embedding(decoder_in)
        decoder_in_pos = torch.arange(decoder_in_len, device=decoder_in.device)
        decoder_in_pe = self.pos_encoding(decoder_in_pos)
        decoder_in_embedded = decoder_in_embedded + decoder_in_pe

        decoder_in_valid_mask = decoder_in_valid_mask == 0

        # 目标因果mask
        decoder_in_causal_mask = torch.triu(torch.ones(decoder_in_len, decoder_in_len, device=decoder_in.device), diagonal=1).bool()

        decoder_output = self.decoder(
            decoder_in_embedded,
            decoder_in_causal_mask,
            decoder_in_valid_mask,
            memory,
            memory_valid_mask,
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
            temperature: float = 0.3,
    ) -> Tensor:
        """一次完整推理"""
        device = src.device
        batch_size = src.size(0)

        # encoder 只跑一次
        memory, memory_valid_mask = self.encode(src, src_mask)

        # 初始化解码序列 [zh/en/fr]
        generated = lang_tokens
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.config.max_len):
            g_mask = torch.ones(generated.shape, device=generated.device)
            # [batch, seq, vocab]
            logits = self.decode(generated, g_mask, memory, memory_valid_mask, True)

            if do_sample:
                next_tokens = torch.multinomial(softmax(logits[:, -1, :] / temperature, 1), 1)
            else:
                next_tokens = logits.argmax(2)

            next_tokens[finished] = PAD_TOKEN_ID
            generated = torch.cat([generated, next_tokens], 1)
            finished |= (next_tokens.squeeze(0) == self.tokenizer.eos_token_id)

            if finished.all():
                break

        return generated

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
