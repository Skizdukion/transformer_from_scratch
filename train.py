from pathlib import Path
import torch
from tqdm import tqdm
from block.transformer import Transformer
from config import get_config, get_weights_file_path
from dataset import get_ds
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import warnings


def train_model():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    (
        train_data_loader,
        test_data_loader,
        val_data_loader,
        tokenizer_src,
        tokenizer_tgt,
    ) = get_ds()

    model = Transformer(
        config["d_model"],
        config["d_ff"],
        config["seq_len"],
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["num_head"],
        config["num_layer"],
        config["dropout"],
    )

    model.to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config.get("preload") is not None:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )

            projection_output = model.projection_layer(decoder_output)

            label = batch["label"].to(device)
            loss = loss_fn(
                projection_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )
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
