from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 60,
        "lr": 10**-4,
        "image_size": 32,
        "patch_size": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_head": 8,
        "num_layer": 6,
        "dropout": 0.1,
        "datasource": "cifar10",
        "model_folder": "vision_trans",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "runs/tmodel",
    }
