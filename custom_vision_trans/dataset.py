from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from custom_vision_trans.config import get_config


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, dataset_split, resize=(32, 32), normalize=True):
        self.dataset = dataset_split
        self.resize = resize
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load image using PIL
        image = self.dataset[idx]["img"]

        # Resize the image manually
        image = image.resize(self.resize)

        # Convert the image to numpy array (HWC)
        image = np.array(image)

        # Normalize the image (if specified)
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
            std = np.array([0.229, 0.224, 0.225])  # ImageNet std
            image = (
                image / 255.0 - mean
            ) / std  # Normalize to [0, 1] and apply mean/std

        # Convert to PyTorch tensor (CHW format)
        image = (
            torch.tensor(image).permute(2, 0, 1).float()
        )  # HWC -> CHW and convert to float

        # Get the label
        label = self.dataset[idx]["label"]

        return image, label


def get_ds():
    dataset = load_dataset("cifar10")
    config = get_config()
    train_dataset = CustomCIFAR10Dataset(dataset["train"])
    val_dataset = CustomCIFAR10Dataset(dataset["test"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, val_loader, 10
