import torch
import torch.nn as nn

from block.encoder import Encoder, EncoderBlock
from block.feed_forward import FeedForwardBlock
from block.input_embedding import InputEmbedding
from block.multihead_attention import MultiHeadAttention
from block.patch_embedding import LocalEmbedding, PatchEmbedding
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

        self.num_params = 0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            if p.requires_grad:
                self.num_params += p.numel()

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


class CustomClassifyVisionTransformer(nn.Module):
    def __init__(
        self,
        image_encoder: Encoder,
        num_classes: int,
    ):
        super().__init__()

        self.image_encoder = image_encoder

        self.local_emb = LocalEmbedding(image_encoder.layers[0].in_feature)

        self.classifier = nn.Linear(self.image_encoder.features, num_classes)
        
        self.pos_emb = PositionEmbedding(image_encoder.layers[0].in_feature, image_encoder.layers[0].in_seq, 0.1)

        self.num_params = 0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            if p.requires_grad:
                self.num_params += p.numel()

    def forward(self, x):
        x = self.local_emb(x)
        x = x.view(x.size(0), x.size(1), -1)  # flatten width and height
        x = x.transpose(1, 2)  # pixel become sequence
        x = self.pos_emb(x)
        encoded = self.image_encoder(x, None)  # (batch, 1, features)
        logits = self.classifier(encoded).squeeze(1)  # (batch, num_classes)
        return logits
