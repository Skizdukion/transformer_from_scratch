from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from custom_vision_trans.config import get_config


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, dataset_split, resize=(32, 32), normalize=True, augment=False):
        self.dataset = dataset_split
        self.resize = resize
        self.normalize = normalize
        self.augment = augment

        # Define transforms
        transform_list = []
        if augment:
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(resize, padding=4),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                ]
            )
        transform_list.append(transforms.Resize(resize))
        transform_list.append(transforms.ToTensor())  # Converts to [0,1] and CHW
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["img"]
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_ds():
    dataset = load_dataset("cifar10")
    config = get_config()
    train_dataset = CustomCIFAR10Dataset(dataset["train"], augment=True)
    val_dataset = CustomCIFAR10Dataset(dataset["test"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, val_loader, 10
