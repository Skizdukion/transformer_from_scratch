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


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
