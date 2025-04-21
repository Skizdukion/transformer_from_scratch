import torch
import math
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # mutiplied
        self.bias = nn.Parameter(torch.ones(1))  # additive

    def forward(self, x: torch.Tensor):
        mean = x.mean(
            dim=-1, keepdim=True
        )  # d_model dims ? new dims: (batch_size, seq_len)
        std = x.mean(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
