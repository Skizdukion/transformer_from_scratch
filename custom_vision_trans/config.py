from pathlib import Path


def get_config():
    flexscale_layer = get_flexscale_layer()
    normal_layer = get_normal_layer()
    return {
        "batch_size": 512,
        "num_epochs": 60,
        "lr": 10**-4,
        "flexscale_layer": flexscale_layer,
        "normal_layer": normal_layer,
        "datasource": "cifar10",
        "model_folder": "custom_vision_trans",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "custom_vision_trans/tmodel",
    }


def get_flexscale_layer():
    return [
        {
            "in_feature": 3,
            "out_feature": 128,
            "in_seq": 1024,
            "out_seq": 128,
            "d_ff": 1024,
            "num_head": 8,
            "dropout": 0.1,
        },
    ]


def get_normal_layer():
    return {
        "seq_len": 128,
        "d_model": 128,
        "d_ff": 1024,
        "num_layer": 5,
        "num_head": 8,
        "dropout": 0.1,
    }


# def get_layer_config():
#     return [
#         {
#             "in_feature": 32,
#             "out_feature": 64,
#             "in_seq": 1024,
#             "hidden_seq": 256,
#             "out_seq": 512,
#             "d_ff": 1024,
#             "num_head": 8,
#             "dropout": 0.1,
#         },
#         {
#             "in_feature": 64,
#             "out_feature": 96,
#             "in_seq": 512,
#             "hidden_seq": 256,
#             "out_seq": 256,
#             "d_ff": 1024,
#             "num_head": 8,
#             "dropout": 0.1,
#         },
#         {
#             "in_feature": 96,
#             "out_feature": 128,
#             "in_seq": 256,
#             "hidden_seq": 256,
#             "out_seq": 128,
#             "d_ff": 1024,
#             "num_head": 8,
#             "dropout": 0.1,
#         },
#         {
#             "in_feature": 128,
#             "out_feature": 128,
#             "in_seq": 128,
#             "hidden_seq": 256,
#             "out_seq": 64,
#             "d_ff": 1024,
#             "num_head": 8,
#             "dropout": 0.1,
#         },
#         {
#             "in_feature": 128,
#             "out_feature": 192,
#             "in_seq": 64,
#             "hidden_seq": 256,
#             "out_seq": 32,
#             "d_ff": 1024,
#             "num_head": 8,
#             "dropout": 0.1,
#         },
#         {
#             "in_feature": 192,
#             "out_feature": 192,
#             "in_seq": 32,
#             "hidden_seq": 256,
#             "out_seq": 1,
#             "d_ff": 1024,
#             "num_head": 8,
#             "dropout": 0.1,
#         },
#     ]
