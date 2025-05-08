import torch
import torch.nn as nn

from block.decoder import Decoder, DecoderBlock
from block.encoder import Encoder, EncoderBlock
from block.feed_forward import FeedForwardBlock
from block.input_embedding import InputEmbedding
from block.multihead_attention import MultiHeadAttention
from block.positional_encoding import SinusoidalPositionEncoding
from block.projection import ProjectionLayer


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        max_seq_len: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_heads: int,
        num_layer: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = src_vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layer
        self.src_input_embedding = InputEmbedding(d_model, src_vocab_size)
        self.tgt_input_embedding = InputEmbedding(d_model, tgt_vocab_size)
        self.position_encoding = SinusoidalPositionEncoding(d_model, max_seq_len, dropout)
        self.encoder = Encoder(
            d_model,
            nn.ModuleList(
                [
                    EncoderBlock(
                        d_model,
                        MultiHeadAttention(d_model, num_heads, dropout),
                        FeedForwardBlock(d_model, d_ff, dropout),
                        dropout,
                    )
                    for _ in range(num_layer)
                ]
            ),
        )
        self.decoder = Decoder(
            d_model,
            nn.ModuleList(
                [
                    DecoderBlock(
                        d_model,
                        MultiHeadAttention(d_model, num_heads, dropout),
                        MultiHeadAttention(d_model, num_heads, dropout),
                        FeedForwardBlock(d_model, d_ff, dropout),
                        dropout,
                    )
                    for _ in range(num_layer)
                ]
            ),
        )
        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src = self.src_input_embedding(src)
        src = self.position_encoding(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_input_embedding(tgt)
        tgt = self.position_encoding(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt

    def projection(self, x):
        return self.projection_layer(x)
