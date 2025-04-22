import torch
import torch.nn as nn
from block.feed_forward import FeedForwardBlock
from block.layer_norm import LayerNormalization
from block.multihead_attention import MultiHeadAttention
from block.residual import ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_att_block: MultiHeadAttention,
        cross_att_block: MultiHeadAttention,
        feed_forwad: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_att_block = self_att_block
        self.cross_att_block = cross_att_block
        self.ff = feed_forwad
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_att_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_att_block(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connections[2](x, self.ff)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
