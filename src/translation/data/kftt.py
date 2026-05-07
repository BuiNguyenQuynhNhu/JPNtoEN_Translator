from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MAX_LEN = config['data']['MAX_LEN']
DATASET_NAME = config['data']['DATASET']
TOKENIZER_NAME = config['model']['TOKENIZER']

dataset = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def preprocess(example):
    model_inputs = tokenizer(
        example["ja"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=example["en"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names 
)

print(f"Max length used: {MAX_LEN}")


tokenized_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=16,
    shuffle=True
)

batch = next(iter(train_loader))

print(batch["input_ids"].shape)