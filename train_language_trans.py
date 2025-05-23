from pathlib import Path
import torch
from tqdm import tqdm
from block.transformer import Transformer
from language_translation.config import get_config
from language_translation.dataset import causal_mask, get_ds
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import warnings

from utils.weight_retrieve import get_weights_file_path, latest_weights_file_path


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Decoder dims: 1 (batch_size), cur_seq_len -> outdim batch_size, cur_seq_len, d_model
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Take only the last sequence to predict the next token, exactly like in training
        prob = model.projection_layer(out[:, -1])

        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    num_examples=2,
):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"]
            target_text = batch["tgt_text"]

            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg("-" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICT: {model_out_text}")

            if count == num_examples:
                break

    # if writer:


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

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_data_loader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # batch, seq_len, d_model

            projection_output = model.projection_layer(
                decoder_output
            )  # batch, seq_len, vocab_size

            label = batch["label"].to(device)  # batch, seq_len

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
            if global_step % 100 == 0:
                run_validation(
                    model,
                    val_data_loader,
                    tokenizer_src,
                    tokenizer_tgt,
                    config["seq_len"],
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
