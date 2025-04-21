import torch
import torch.nn as nn
from feed_forward import FeedForwardBlock
from layer_norm import LayerNormalization
from multihead_attention import MultiHeadAttention
from residual import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(
        self,
        mha_block: MultiHeadAttention,
        feed_forwad: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.mha_block = mha_block
        self.ff = feed_forwad
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.mha_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
