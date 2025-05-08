import torch
import torch.nn as nn
from block.flexscale_attention import DownScaleSeqAttention
from block.feed_forward import FeedForwardBlock
from block.latent_compressor import LatentCompressor
from block.layer_norm import LayerNormalization
from block.multihead_attention import MultiHeadAttention
from block.positional_encoding import PositionEmbedding
from block.residual import ResidualConnection, ResidualConnectionDifferentScale


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
        # print(f"Sublayer forward time: {time.time() - start:.6f}s")
        return x


class LatentEncoderBlock(nn.Module):
    def __init__(
        self,
        latent_tokens: int,
        d_model: int,
        d_ff: int,
        num_head: int,
        dropout: float,
    ):
        super().__init__()
        self.compressor = LatentCompressor(
            latent_tokens=latent_tokens, d_model=d_model, num_head=num_head
        )
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connection_ff = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        x = self.compressor(x)  # 👈 no residual here, x now becomes latent
        x = self.residual_connection_ff(x, self.ff)
        return x

class FlexScaleEncoderBlock(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        in_seq: int,
        out_seq: int,
        d_ff: int,
        num_head: int,
        dropout: float,
    ):
        super().__init__()
        self.d_attention = DownScaleSeqAttention(
            dmodel_in=in_feature,
            dmodel_out=out_feature,
            in_seq=in_seq,
            out_seq=out_seq,
            num_head=num_head,
        )

        self.ff = FeedForwardBlock(out_feature, d_ff, dropout)

        self.residual_1 = ResidualConnectionDifferentScale(
            in_feature, out_feature, in_seq, out_seq, dropout
        )

        self.residual_2 = ResidualConnection(out_feature, dropout)

        # self.output_pos_emb = PositionEmbedding(out_feature, out_seq, dropout)

        self.out_seq = out_seq
        self.out_feature = out_feature
        self.in_seq = in_seq
        self.in_feature = in_feature

    def forward(self, x, mask=None):
        x = self.residual_1(x, self.d_attention)
        # x = self.d_attention(x)
        x = self.residual_2(x, self.ff)
        # x = self.output_pos_emb(x)
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
