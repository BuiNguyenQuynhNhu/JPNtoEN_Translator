"""
preprocessing/tokenizer.py

This module provides a tokenizer wrapper for NLLB/mBART models.
It handles tokenization of Japanese and English texts, while preserving
token offsets and sentence indices, which are crucial for future
graph construction stages.

How to test:
    from preprocessing.tokenizer import TranslationTokenizer
    tok = TranslationTokenizer("facebook/nllb-200-distilled-600M", "jpn_Jpan", "eng_Latn")
    src = tok.tokenize_source("こんにちは世界")
    tgt = tok.tokenize_target("Hello World")
    print(src["input_ids"].shape) # Expected: [MAX_LENGTH]
    print(tgt["labels"].shape)    # Expected: [MAX_LENGTH]
"""

import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer

class TranslationTokenizer:
    def __init__(self, model_name: str, src_lang: str, tgt_lang: str, max_length: int = 128):
        """
        Initializes the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def tokenize_source(self, text: str) -> Dict[str, Any]:
        """
        Tokenizes Japanese source text, returning offset mappings.
        
        Expected Tensor Shapes:
        - input_ids: [MAX_LENGTH]
        - attention_mask: [MAX_LENGTH]
        """
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "offset_mapping": enc["offset_mapping"].squeeze(0).tolist() # [MAX_LENGTH, 2]
        }
        
    def tokenize_target(self, text: str) -> Dict[str, Any]:
        """
        Tokenizes English target text.
        
        Expected Tensor Shapes:
        - labels: [MAX_LENGTH]
        """
        enc = self.tokenizer(
            text_target=text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Replace pad_token_id with -100 for CrossEntropyLoss calculation
        labels = enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "labels": labels,
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes token ids back to string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Filter out -100 if present
        valid_ids = [t for t in token_ids if t != -100]
        return self.tokenizer.decode(valid_ids, skip_special_tokens=skip_special_tokens)

# Future improvements:
# - Add support for external morphological analyzers (MeCab/Sudachi) for Japanese.
# - Better handling of subword to word alignment logic.
