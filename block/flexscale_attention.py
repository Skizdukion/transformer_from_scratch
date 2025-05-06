import torch
import torch.nn as nn
from block.feed_forward import FeedForwardBlock
from block.layer_norm import LayerNormalization
from block.multihead_attention import MultiHeadAttention
from block.residual import ResidualConnection
import math


class FlexScaleSeqAttention(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        in_seq: int,
        out_seq: int,
        num_head: int,
    ):
        super().__init__()

        assert (
            out_channel % num_head == 0
        ), "out_channel must be divisible by num_heads"

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.num_heads = num_head
        self.head_dim = out_channel // num_head
        
        self.w_q = nn.Linear(in_channel, out_channel)
        self.w_k = nn.Linear(in_channel, out_channel)
        self.w_v = nn.Linear(in_channel, out_channel)

        self.flexscale_proj = nn.Linear(in_seq, out_seq)

    def forward(self, x):
        batch_size = x.size(0)
        assert x.shape[1] == self.in_seq
        assert x.shape[2] == self.in_channel

        # Project inputs
        q = self.w_q(x)  # [B, in_seq, out_channel]
        k = self.w_k(x)
        v = self.w_v(x)

        # Split into heads: [B, in_seq, num_heads, head_dim] -> [B, num_heads, in_seq, head_dim]
        q = q.view(batch_size, self.in_seq, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, self.in_seq, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, self.in_seq, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Attention: [B, num_heads, in_seq, head_dim] x [B, num_heads, head_dim, in_seq] = [B, num_heads, in_seq, in_seq]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)

        # Downscale across the sequence dimension
        # [B, num_heads, in_seq, out_seq]
        flexscale = self.flexscale_proj(attn)  # projects in_seq → out_seq per head

        # Apply attention: [B, num_heads, out_seq, in_seq] x [B, num_heads, in_seq, head_dim]
        out = torch.matmul(
            flexscale.transpose(-2, -1), v
        )  # [B, num_heads, out_seq, head_dim]

        # Merge heads: [B, out_seq, out_channel]
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.out_seq, self.out_channel)
        )
        return out
