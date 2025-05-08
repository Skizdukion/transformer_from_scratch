import os
from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 60,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 256,
        "d_ff": 2048,
        "num_heads": 8,
        "num_layer": 6,
        "dropout": 0.1,
        "datasource": "IWSLT/mt_eng_vietnamese",
        "data_name": "default",
        "data_revision": "refs/convert/parquet",
        "lang_src": "vi",
        "lang_tgt": "en",
        "model_folder": "language_translation_trans",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer/{0}.json",
        "experiment_name": "runs/tmodel",
    }
