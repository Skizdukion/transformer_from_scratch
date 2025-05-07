import torch
import torch.nn as nn
from block.layer_norm import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        normed = self.norm(x)
        value = sublayer(normed) 
        # print(f"Sublayer forward time: {time.time() - start:.6f}s")
        return x + self.dropout(value)


class ResidualConnectionDifferentScale(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_feature: int,
        in_seq: int,
        out_seq: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.out_seq = out_seq
        self.out_feature = out_feature
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(in_features)
        self.feature_proj = nn.Linear(in_features, out_feature)

        self.seq_proj = nn.Linear(in_seq, out_seq)

    def forward(self, x, sublayer):
        """
        x: [batch, in_seq, in_features]
        sublayer(norm(x)): [batch, out_seq, out_feature]
        """
        normed = self.norm(x)
        value = sublayer(normed)  # Expected shape: [batch, out_seq, out_feature]

        # Step 1: transpose to project across sequence length
        x_proj = self.feature_proj(x)  # [batch, in_seq, out_feature]
        x_proj = x_proj.transpose(1, 2)  # [batch, out_feature, in_seq]
        x_proj = self.seq_proj(x_proj)  # [batch, out_feature, out_seq]
        x_proj = x_proj.transpose(1, 2)  # [batch, out_seq, out_feature]

        assert x_proj.shape == value.shape
        return x_proj + self.dropout(value)
