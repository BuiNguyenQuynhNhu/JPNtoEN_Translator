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

import os
import torch
from typing import Optional, Dict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from preprocessing.tokenizer import TranslationTokenizer
from models.graph.builder import GraphBuilder
from models.graph.batching import batch_graphs

class TranslationDatasetLoader:
    def __init__(self, config: dict, tokenizer: TranslationTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name = config.get("dataset_name", "kftt")
        self.dataset_path = config.get("dataset_path", "may-ohta/kftt")
        self.document_level = config.get("document_level", False)
        self.context_size = config.get("context_window_size", 1)
        self.graph_builder = GraphBuilder()
        
    def load_dataset_splits(self) -> Dict[str, Dataset]:
        """
        Loads the dataset either from local directory or HuggingFace.
        """
        if os.path.isdir(self.dataset_path):
            raw_dataset = self._load_local_dataset()
        else:
            raw_dataset = load_dataset(self.dataset_path)
            # Rename 'dev' to 'validation' if necessary
            if "dev" in raw_dataset:
                raw_dataset["validation"] = raw_dataset.pop("dev")
                
        dataset_dict = {}
        splits = ["train", "validation", "test"]
        for split in splits:
            if split in raw_dataset:
                ds = raw_dataset[split]
                subset_key = f"{split if split != 'validation' else 'val'}_subset_size"
                subset_size = self.config.get(subset_key)
                if subset_size is not None and subset_size < len(ds):
                    ds = ds.select(range(subset_size))
                dataset_dict[split] = ds
                
        return dataset_dict

    def _load_local_dataset(self) -> Dict[str, Dataset]:
        """
        Parses local parallel corpus files (.en and .ja) into a dictionary of Datasets.
        """
        raw_dataset = {}
        splits_map = {
            "train": ["train"],
            "validation": ["dev", "tune"],
            "test": ["test"]
        }
        
        for split, suffixes in splits_map.items():
            for suffix in suffixes:
                en_path = None
                ja_path = None
                for f in os.listdir(self.dataset_path):
                    if suffix in f and f.endswith(".en"):
                        en_path = os.path.join(self.dataset_path, f)
                    elif suffix in f and f.endswith(".ja"):
                        ja_path = os.path.join(self.dataset_path, f)
                        
                if en_path and ja_path:
                    with open(en_path, "r", encoding="utf-8") as fe, open(ja_path, "r", encoding="utf-8") as fj:
                        en_lines = [l.strip() for l in fe]
                        ja_lines = [l.strip() for l in fj]
                    
                    if len(en_lines) == len(ja_lines):
                        raw_dataset[split] = Dataset.from_dict({"en": en_lines, "ja": ja_lines})
                        break
                        
        return raw_dataset
        
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
            
        model_inputs = {
            "input_ids": [], "attention_mask": [], "labels": [],
            "graph": [], "offset_mapping": []
        }
        
        for ja_text, en_text in zip(inputs, targets):
            src_enc = self.tokenizer.tokenize_source(ja_text)
            tgt_enc = self.tokenizer.tokenize_target(en_text)
            
            model_inputs["input_ids"].append(src_enc["input_ids"].tolist())
            model_inputs["attention_mask"].append(src_enc["attention_mask"].tolist())
            model_inputs["labels"].append(tgt_enc["labels"].tolist())
            model_inputs["offset_mapping"].append(src_enc["offset_mapping"])
            
            graph = self.graph_builder.build_graph(ja_text)
            model_inputs["graph"].append(graph)
            
        return model_inputs

    def collate_fn(self, features):
        """
        Custom collate function to handle standard token tensors and variable-sized graph structures.
        """
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        
        graphs = [f["graph"] for f in features]
        offset_mappings = [f["offset_mapping"] for f in features]
        
        batched_graphs = batch_graphs(graphs, offset_mappings)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph": batched_graphs
        }

    def get_dataloaders(self, batch_size: int = 8) -> Dict[str, DataLoader]:
        """
        Returns PyTorch DataLoaders for train, validation, and test splits.
        """
        import os
        from datasets import DatasetDict, load_from_disk
        
        tokenized_path = self.config.get("tokenized_path", "data/tokenized_dataset")
        
        if os.path.exists(tokenized_path):
            print(f"Loading tokenized dataset from {tokenized_path}...")
            tokenized_dataset_dict = load_from_disk(tokenized_path)
        else:
            print("Tokenized dataset not found. Generating...")
            dataset_dict = self.load_dataset_splits()
            
            tokenized_dataset_dict = DatasetDict()
            for split, ds in dataset_dict.items():
                tokenized_ds = ds.map(
                    self.preprocess_function,
                    batched=True,
                    remove_columns=ds.column_names,
                    desc=f"Running tokenizer on {split} dataset"
                )
                
                tokenized_ds.set_format(type="python", columns=["input_ids", "attention_mask", "labels", "graph", "offset_mapping"])
                tokenized_dataset_dict[split] = tokenized_ds
                
            print(f"Saving tokenized dataset to {tokenized_path}...")
            tokenized_dataset_dict.save_to_disk(tokenized_path)
            
        dataloaders = {}
        
        for split, ds in tokenized_dataset_dict.items():
            shuffle = True if split == "train" else False
            dataloaders[split] = DataLoader(
                ds, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                collate_fn=self.collate_fn
            )
            
        return dataloaders

# Future improvements:
# - Implement stateful dataset parsing to retain document boundaries.
# - Yield previous context sentences when document_level is enabled.
