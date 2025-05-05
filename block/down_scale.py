import torch
import torch.nn as nn
from block.feed_forward import FeedForwardBlock
from block.layer_norm import LayerNormalization
from block.multihead_attention import MultiHeadAttention
from block.residual import ResidualConnection
import math

class DownScaleSeqAttention(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, in_seq: int, out_seq: int, dropout: float):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(in_channel, out_channel)
        self.w_k = nn.Linear(in_channel, out_channel)
        self.w_v = nn.Linear(in_channel, out_channel)

        self.downscale_proj = nn.Linear(in_seq, out_seq)

    def foward(self, x):
        assert x.shape[1] == self.in_seq
        assert x.shape[2] == self.in_channel

        query = self.w_q(x)  # batch_size, in_seq, out_channel
        key = self.w_k(x)
        value = self.w_v(x)

        attention_score = (query @ key.transpose(-1, -2)) / math.sqrt(
            self.channel
        )  # batch_size, in_seq, in_seq
        attention_score = attention_score.softmax(dim=-1)
        down_scale = self.downscale_proj(attention_score)  # batch_size, in_seq, out_seq
        output = torch.bmm(
            down_scale.transpose(1, 2), value
        )  # batch_size, out_seq, channel
        return output

