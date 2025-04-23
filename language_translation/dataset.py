from pathlib import Path
import torch
import torch.nn as nn

from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader

from language_translation.config import get_config


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2

        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # Ignore padding token in encoder
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            # Ignore padding token and ignore after token
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def pruning_ds(ds_dict, pre_tokenizer, seq_len, max_samples=10000):
    pruned = {}

    for split, ds in ds_dict.items():
        # Step 1: Filter by tokenized length across all languages
        filtered = ds.filter(
            lambda example: all(
                len(pre_tokenizer.pre_tokenize_str(text)) < (seq_len - 2)
                for text in example["translation"].values()
            )
        )

        # Step 2: Truncate to max_samples if needed
        if len(filtered) > max_samples:
            filtered = filtered.select(range(max_samples))

        pruned[split] = filtered

    return DatasetDict(pruned)


def get_all_sentences(ds, lang):
    for item in ds["train"]:
        sentence = item["translation"][lang]
        yield sentence

    for item in ds["validation"]:
        sentence = item["translation"][lang]
        yield sentence

    for item in ds["test"]:
        sentence = item["translation"][lang]
        yield sentence


def get_or_build_tokenizer(ds, lang):
    config = get_config()

    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=5
        )
        tokenizer.train_from_iterator(
            get_all_sentences(
                ds,
                lang,
            ),
            trainer=trainer,
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


config = {}


def get_ds():
    config = get_config()

    ds_raw = load_dataset(
        config["datasource"], config["data_name"], revision=config["data_revision"]
    )

    ds_pruned = pruning_ds(ds_raw, Whitespace(), config["seq_len"])

    src_lan = config["lang_src"]
    tgt_lan = config["lang_tgt"]

    tokenizer_src = get_or_build_tokenizer(ds_pruned, src_lan)
    tokenizer_tgt = get_or_build_tokenizer(ds_pruned, tgt_lan)

    train_ds = BilingualDataset(
        ds_pruned["train"],
        tokenizer_src,
        tokenizer_tgt,
        src_lan,
        tgt_lan,
        config["seq_len"],
    )

    test_ds = BilingualDataset(
        ds_pruned["test"],
        tokenizer_src,
        tokenizer_tgt,
        src_lan,
        tgt_lan,
        config["seq_len"],
    )

    val_ds = BilingualDataset(
        ds_pruned["validation"],
        tokenizer_src,
        tokenizer_tgt,
        src_lan,
        tgt_lan,
        config["seq_len"],
    )

    # max_len_src, max_len_tgt = get_max_len(
    #     ds_raw, src_lan, tgt_lan, tokenizer_src, tokenizer_tgt
    # )

    # print(f"Max length of source sentence: {max_len_src}")
    # print(f"Max length of target sentence: {max_len_tgt}")

    train_data_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    test_data_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return (
        train_data_loader,
        test_data_loader,
        val_data_loader,
        tokenizer_src,
        tokenizer_tgt,
    )
