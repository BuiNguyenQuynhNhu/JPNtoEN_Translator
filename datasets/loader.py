"""
datasets/loader.py

This module provides a unified interface for loading machine translation datasets
(KFTT, JESC, ASPEC, OpenSubtitles). It handles:
- Downloading or streaming via HuggingFace datasets
- Creating sliding context windows for document-level translation (prepared for future)
- Tokenizing the text using the custom TranslationTokenizer

How to test:
    import yaml
    from preprocessing.tokenizer import TranslationTokenizer
    from datasets.loader import TranslationDatasetLoader
    
    config = yaml.safe_load(open("configs/baseline.yaml"))
    tokenizer = TranslationTokenizer(config["model"]["model_name"], config["model"]["src_lang"], config["model"]["tgt_lang"])
    loader = TranslationDatasetLoader(config["data"], tokenizer)
    dataloaders = loader.get_dataloaders()
    for batch in dataloaders["train"]:
        print(batch["input_ids"].shape) # Expected: [B, MAX_LENGTH]
        print(batch["labels"].shape)    # Expected: [B, MAX_LENGTH]
        break
"""

import torch
from typing import Optional, Dict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from preprocessing.tokenizer import TranslationTokenizer

class TranslationDatasetLoader:
    def __init__(self, config: dict, tokenizer: TranslationTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name = config.get("dataset_name", "kftt")
        self.dataset_path = config.get("dataset_path", "may-ohta/kftt")
        self.document_level = config.get("document_level", False)
        self.context_size = config.get("context_window_size", 1)
        
    def load_hf_dataset(self) -> Dict[str, Dataset]:
        """
        Loads the dataset from HuggingFace.
        """
        raw_dataset = load_dataset(self.dataset_path)
        
        splits = ["train", "validation", "test"]
        dataset_dict = {}
        
        # Rename 'dev' to 'validation' if necessary
        if "dev" in raw_dataset:
            raw_dataset["validation"] = raw_dataset["dev"]
            
        for split in splits:
            if split in raw_dataset:
                ds = raw_dataset[split]
                subset_key = f"{split if split != 'validation' else 'val'}_subset_size"
                subset_size = self.config.get(subset_key)
                if subset_size is not None and subset_size < len(ds):
                    ds = ds.select(range(subset_size))
                dataset_dict[split] = ds
                
        return dataset_dict
        
    def preprocess_function(self, examples):
        """
        Tokenizes the examples.
        Handles formatting differently depending on if dataset has 'translation' key
        or direct 'ja'/'en' columns.
        """
        inputs = []
        targets = []
        
        # Extract texts
        if "translation" in examples:
            for ex in examples["translation"]:
                inputs.append(ex["ja"])
                targets.append(ex["en"])
        else:
            # Fallback for simpler datasets
            inputs = examples.get("ja", examples.get("ja_JP", []))
            targets = examples.get("en", examples.get("en_US", []))
            
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for ja_text, en_text in zip(inputs, targets):
            src_enc = self.tokenizer.tokenize_source(ja_text)
            tgt_enc = self.tokenizer.tokenize_target(en_text)
            
            model_inputs["input_ids"].append(src_enc["input_ids"].tolist())
            model_inputs["attention_mask"].append(src_enc["attention_mask"].tolist())
            model_inputs["labels"].append(tgt_enc["labels"].tolist())
            
        return model_inputs

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Returns PyTorch DataLoaders for train, validation, and test splits.
        """
        dataset_dict = self.load_hf_dataset()
        
        dataloaders = {}
        # We need the training config to get batch size
        batch_size = 8 # Default, should ideally be passed from training config but simplified here
        
        for split, ds in dataset_dict.items():
            tokenized_ds = ds.map(
                self.preprocess_function,
                batched=True,
                remove_columns=ds.column_names,
                desc=f"Running tokenizer on {split} dataset"
            )
            
            tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            
            shuffle = True if split == "train" else False
            dataloaders[split] = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=shuffle)
            
        return dataloaders

# Future improvements:
# - Implement stateful dataset parsing to retain document boundaries.
# - Yield previous context sentences when document_level is enabled.
