import torch
import math
import torch.nn as nn
import time

from block.multihead_attention import MultiHeadAttention


class LatentCompressor(nn.Module):
    def __init__(self, latent_tokens, d_model, num_head):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, latent_tokens, d_model))
        self.attn = MultiHeadAttention(d_model, num_head, 0.1)

    def forward(self, x):
        # x: (B, N, D)
        B = x.size(0)
        latent = self.latent.expand(B, -1, -1)  # (B, M, D)
        out, _ = self.attn(latent, x, x)  # Q = latent, K/V = x
        return out  # (B, M, D)
