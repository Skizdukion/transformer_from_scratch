import torch
import math
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        # self.d_model = d_model
        # self.d_ff = d_ff
        self.ln_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        x = self.ln_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.ln_2(x)