from huggingface_hub import hf_hub_download
from datasets import load_dataset

# Download files
train_path = hf_hub_download(
    repo_id="IWSLT/mt_eng_vietnamese",
    repo_type="dataset",
    revision="refs/convert/parquet",
    filename="iwslt2015-en-vi/train/0000.parquet"
)
val_path = hf_hub_download(
    repo_id="IWSLT/mt_eng_vietnamese",
    repo_type="dataset",
    revision="refs/convert/parquet",
    filename="iwslt2015-en-vi/validation/0000.parquet"
)
test_path = hf_hub_download(
    repo_id="IWSLT/mt_eng_vietnamese",
    repo_type="dataset",
    revision="refs/convert/parquet",
    filename="iwslt2015-en-vi/test/0000.parquet"
)

# Load using explicit splits
ds = load_dataset(
    "parquet",
    data_files={
        "train": train_path,
        "validation": val_path,
        "test": test_path
    }
)
