import torch
import torch.nn as nn
import math


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        return torch.log_softmax(x, dim=-1)
