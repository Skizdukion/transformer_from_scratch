from pathlib import Path
import torch
from tqdm import tqdm
from block.encoder import EncoderBlock, FlexScaleEncoderBlock, Encoder
from block.feed_forward import FeedForwardBlock
from block.multihead_attention import MultiHeadAttention
from block.vision_transformer import CustomClassifyVisionTransformer
from utils.weight_retrieve import get_weights_file_path, latest_weights_file_path
from custom_vision_trans.config import get_config
from custom_vision_trans.dataset import get_ds
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import warnings


def run_validation(model, validation_loader, device, print_msg=print):
    """
    Validates the Vision Transformer model on the validation set.

    Args:
        model (nn.Module): The trained Vision Transformer model.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on.
        loss_fn (nn.Module): Loss function (e.g., CrossEntropyLoss).
        print_msg (Callable): Function to print output (defaults to print).

    Returns:
        float: average loss over the validation set.
        float: accuracy on the validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # shape: (batch_size, seq_len, d_model)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print_msg(f"Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
    return avg_loss, accuracy


def train_model():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    (train_data_loader, val_data_loader, num_classes) = get_ds()

    flexscale_blocks = []

    for layer_config in config["flexscale_layer"]:
        block = FlexScaleEncoderBlock(
            layer_config["in_feature"],
            layer_config["out_feature"],
            layer_config["in_seq"],
            layer_config["hidden_seq"],
            layer_config["out_seq"],
            layer_config["d_ff"],
            layer_config["num_head"],
            layer_config["dropout"],
        )
        flexscale_blocks.append(block)

    flex_scale_encoder = Encoder(
        flexscale_blocks[len(flexscale_blocks) - 1].out_feature,
        nn.ModuleList(flexscale_blocks),
    )

    d_model = config["normal_layer"]["d_model"]
    num_head = config["normal_layer"]["num_head"]
    dropout = config["normal_layer"]["dropout"]
    d_ff = config["normal_layer"]["d_ff"]
    num_layer = config["normal_layer"]["num_layer"]

    encoder = Encoder(
        d_model,
        nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    MultiHeadAttention(d_model, num_head, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(num_layer)
            ]
        ),
    )

    model = CustomClassifyVisionTransformer(flex_scale_encoder, encoder, num_classes)

    model.to(device)

    print(f"Model total params {model.num_params}")

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_data_loader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            mask = None

            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(
            model,
            val_data_loader,
            device,
            lambda msg: batch_iterator.write(msg),
        )

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    train_model()
