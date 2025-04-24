import torch
import math
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to input tensor.

    This module adds fixed sinusoidal positional information to the input embeddings.
    Useful for transformer models including Vision Transformers (ViT) to provide the model
    with information about the position of tokens/patches in a sequence.

    Args:
        d_model (int): The dimension of the embedding.
        seq_len (int): The maximum length of the sequence (number of patches or tokens).
        dropout (float): Dropout rate applied after adding positional encoding.

    Attributes:
        pe (Tensor): Positional encoding matrix of shape (1, seq_len, d_model), non-trainable.

    Input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

    Output:
        Tensor: Positionally encoded tensor of the same shape as input.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create matrix dims: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # DIMS: (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(100000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.shape[1]
        assert seq_len <= self.seq_len, "Sequence len is not compatible"
        x = x + (self.pe[:, :seq_len, :]).requires_grad_(
            False
        )  # Broadcasting rule: 1 -> batch_size
        return self.dropout(x)


class PositionEmbedding(nn.Module):
    """
    Learned positional embedding module for Vision Transformer (ViT) or similar architectures.

    This module adds learnable positional encodings to the input embeddings, allowing the model
    to incorporate information about the order of input tokens or image patches.

    Args:
        d_model (int): The dimensionality of each input embedding vector (e.g., patch embedding size).
        seq_len (int): The maximum sequence length (e.g., number of patches or tokens).
        dropout (float): Dropout probability applied after adding positional embeddings.

    Attributes:
        embedding (nn.Embedding): Learnable embedding layer for positions.
        position_ids (Tensor): Fixed buffer of position indices with shape (1, seq_len).
        dropout (nn.Dropout): Dropout layer applied after position addition.

    Input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

    Output:
        Tensor: Output tensor of shape (batch_size, seq_len, d_model) with positional embeddings added.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "position_ids", torch.arange(self.seq_len).unsqueeze(0), persistent=False
        )

    def forward(self, x):
        seq_len = x.shape[1]
        assert seq_len <= self.seq_len, "Sequence len is not compatible"
        x = x + self.pos_emb(self.position_ids[:, :seq_len])
        return self.dropout(x)
