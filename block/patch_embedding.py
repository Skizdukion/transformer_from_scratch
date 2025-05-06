import torch
import math
import torch.nn as nn


class LocalEmbedding(nn.Module):
    """
    Patch Embedding Layer for Vision Transformers.

    This layer splits the input image into non-overlapping patches and projects them into a higher dimensional embedding space using a convolution.

    Args:
        d_model (int): Dimension of the embedding (output channels of the Conv2d layer).
        image_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).

    Input:
        x (Tensor): Input tensor of shape (batch_size, 3, image_size, image_size)

    Output:
        Tensor: Output tensor of shape (batch_size, d_model, num_patches_h, num_patches_w)
            where num_patches_h = num_patches_w = image_size // patch_size
    """

    def __init__(self, channel_out: int):
        super().__init__()
        self.d_model = channel_out
        self.local_emb = nn.Conv2d(
            in_channels=3,
            out_channels=channel_out,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        assert x.size(1) == 3
        return self.local_emb(x)


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer for Vision Transformers.

    This layer splits the input image into non-overlapping patches and projects them into a higher dimensional embedding space using a convolution.

    Args:
        d_model (int): Dimension of the embedding (output channels of the Conv2d layer).
        image_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).

    Input:
        x (Tensor): Input tensor of shape (batch_size, 3, image_size, image_size)

    Output:
        Tensor: Output tensor of shape (batch_size, d_model, num_patches_h, num_patches_w)
            where num_patches_h = num_patches_w = image_size // patch_size
    """

    def __init__(self, d_model: int, image_size: int, patch_size: int):
        super().__init__()
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size % patch_size == 0, "Patch size is not divisible by image size"
        self.num_patch = (image_size // patch_size) ** 2  # height and width dimension
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        assert x.size(1) == 3
        assert x.size(2) == self.image_size, "Image height dims is not correct"
        assert x.size(3) == self.image_size, "Image witdh dims is not correct"
        return self.patch_embedding(x)
