import torch
import torch.nn as nn
from block.feed_forward import FeedForwardBlock
from block.layer_norm import LayerNormalization
from block.multihead_attention import MultiHeadAttention
from block.residual import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_att_block: MultiHeadAttention,
        feed_forwad: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_att_block = self_att_block
        self.ff = feed_forwad
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_att_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.features = features
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
