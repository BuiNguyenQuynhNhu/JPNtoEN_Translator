import torch
import yaml
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from preprocessing.tokenizer import TranslationTokenizer
from data_loading.loader import TranslationDatasetLoader

config = yaml.safe_load(open("configs/baseline.yaml", "r"))
tokenizer = TranslationTokenizer(
    model_name=config["model"]["model_name"],
    src_lang=config["model"]["src_lang"],
    tgt_lang=config["model"]["tgt_lang"],
    max_length=config["model"]["max_length"]
)
loader = TranslationDatasetLoader(config["data"], tokenizer)
dataloaders = loader.get_dataloaders()
batch = next(iter(dataloaders["train"]))

model_config = AutoConfig.from_pretrained(config["model"]["model_name"])
model = AutoModelForSeq2SeqLM.from_config(model_config)

out = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
    output_hidden_states=True,
    return_dict=True
)
print("Loss:", out.loss)
