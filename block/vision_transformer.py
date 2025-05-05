import torch
import torch.nn as nn

from block.encoder import Encoder, EncoderBlock
from block.feed_forward import FeedForwardBlock
from block.input_embedding import InputEmbedding
from block.multihead_attention import MultiHeadAttention
from block.patch_embedding import PatchEmbedding
from block.positional_encoding import PositionEmbedding, SinusoidalPositionEncoding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_encoder: Encoder,
        # text_encoder: Encoder,
        image_size: int,
        patch_size: int,
        # txt_vocab_size: int,
        # txt_seq_len: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        # self.text_encoder = text_encoder

        self.image_emb = PatchEmbedding(
            self.image_encoder.features, image_size, patch_size
        )

        self.image_pos_emb = PositionEmbedding(
            self.image_encoder.features, self.image_emb.num_patch + 1, dropout
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.image_encoder.features))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # self.txt_emb = InputEmbedding(self.text_encoder.features, txt_vocab_size)
        # self.txt_pos_enc = SinusoidalPositionEncoding(
        #     self.text_encoder.features, txt_seq_len, dropout
        # )

        self.classifier = nn.Linear(self.image_encoder.features, num_classes)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # (batch_len, )
    def encode_image(self, x, mask):
        x = self.image_emb(x)
        x = x.flatten(2).transpose(1, 2)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.image_pos_emb(x)
        x = self.image_encoder(x, mask)
        return x

    def forward(self, x, mask=None):
        encoded = self.encode_image(x, mask)  # (batch, seq_len, d_model)
        cls_token = encoded[:, 0]  # (batch, d_model)
        logits = self.classifier(cls_token)  # (batch, num_classes)
        return logits

    # def encode_text(self, x, mask):
    #     x = self.txt_emb(x)
    #     x = self.txt_pos_enc(x)
    #     x = self.text_encoder(x, mask)
    #     return x
