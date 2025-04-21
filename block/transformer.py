import torch
import torch.nn as nn

from block.decoder import Decoder, DecoderBlock
from block.encoder import Encoder, EncoderBlock
from block.feed_forward import FeedForwardBlock
from block.input_embedding import InputEmbedding
from block.multihead_attention import MultiHeadAttention
from block.positional_encoding import PositionEncoding
from block.projection import ProjectionLayer


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        max_seq_len: int,
        vocab_size: int,
        num_head: int,
        num_layer: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_head = num_head
        self.num_layers = num_layer
        self.input_embedding = InputEmbedding(d_model, vocab_size)
        self.position_encoding = PositionEncoding(d_model, max_seq_len)
        self.encoder = Encoder(
            nn.ModuleList(
                [
                    EncoderBlock(
                        MultiHeadAttention(d_model, num_head, dropout),
                        FeedForwardBlock(d_model, d_ff),
                        dropout,
                    )
                    for _ in range(num_layer)
                ]
            )
        )
        self.decoder = Decoder(
            nn.ModuleList(
                [
                    DecoderBlock(
                        MultiHeadAttention(d_model, num_head, dropout),
                        MultiHeadAttention(d_model, num_head, dropout),
                        FeedForwardBlock(d_model, d_ff),
                        dropout,
                    )
                    for _ in range(num_layer)
                ]
            )
        )
        self.projection = ProjectionLayer(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, src, src_mask, tgt, tgt_mask):
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        src = self.decoder(src, tgt, src_mask, tgt_mask)
        src = self.projection(src)
        return src
