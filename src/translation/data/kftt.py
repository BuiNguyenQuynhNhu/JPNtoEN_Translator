import os
from typing import Optional
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

class KFTTDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 sp_model_path: Optional[str] = None,
                 train_sp_model: bool = True,
                 model_prefix: str = "kftt_spm",
                 vocab_size: int = 16000,
                 character_coverage: float = 0.9995,
                 model_type: str = "unigram",
                 max_length: int = 128,
                 add_bos_eos: bool = True,
                 ignore_index: int = -100,
                 tokenizer: Optional[object] = None):
        assert split in ("train", "dev", "test", "tune"), "split must be train/dev/test/tune"
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.add_bos_eos = add_bos_eos
        self.ignore_index = ignore_index

        # prefer tokenized .tok files if present, else fall back to orig files
        tok_ja = os.path.join(data_dir, f"tok/kyoto-{split}.ja")
        tok_en = os.path.join(data_dir, f"tok/kyoto-{split}.en")
        orig_ja = os.path.join(data_dir, f"orig/kyoto-{split}.ja")
        orig_en = os.path.join(data_dir, f"orig/kyoto-{split}.en")

        if os.path.exists(tok_ja) and os.path.exists(tok_en):
            ja_path, en_path = tok_ja, tok_en
        elif os.path.exists(orig_ja) and os.path.exists(orig_en):
            ja_path, en_path = orig_ja, orig_en
        else:
            raise FileNotFoundError(f"Missing files for split {split}: checked {tok_ja},{tok_en},{orig_ja},{orig_en}")

        # read lines
        with open(ja_path, encoding="utf-8") as f:
            self.ja_lines = [l.rstrip("\n") for l in f]
        with open(en_path, encoding="utf-8") as f:
            self.en_lines = [l.rstrip("\n") for l in f]

        assert len(self.ja_lines) == len(self.en_lines), "Source and target line counts differ"

        # tokenizer handling: user-provided or SentencePiece auto
        if tokenizer is not None:
            self.tokenizer_obj = tokenizer
            self._is_spm = isinstance(tokenizer, spm.SentencePieceProcessor)
            self._is_hf = hasattr(tokenizer, "__call__") and not self._is_spm
        else:
            # determine model path
            self.model_file = sp_model_path if sp_model_path is not None else os.path.join(data_dir, f"{model_prefix}.model")
            if not os.path.exists(self.model_file):
                if not train_sp_model:
                    raise FileNotFoundError(f"SentencePiece model not found at {self.model_file} and train_sp_model=False")
                # need train files: prefer tok train if exists else orig train
                train_tok_ja = os.path.join(data_dir, "tok/kyoto-train.ja")
                train_tok_en = os.path.join(data_dir, "tok/kyoto-train.en")
                train_orig_ja = os.path.join(data_dir, "orig/kyoto-train.ja")
                train_orig_en = os.path.join(data_dir, "orig/kyoto-train.en")

                if os.path.exists(train_tok_ja) and os.path.exists(train_tok_en):
                    tja, ten = train_tok_ja, train_tok_en
                elif os.path.exists(train_orig_ja) and os.path.exists(train_orig_en):
                    tja, ten = train_orig_ja, train_orig_en
                else:
                    raise FileNotFoundError("Cannot train SentencePiece: missing train split files (tok or orig)")

                combined_txt = os.path.join(data_dir, f"{model_prefix}.train_combined.txt")
                with open(combined_txt, "w", encoding="utf-8") as outf:
                    with open(tja, encoding="utf-8") as fja:
                        for l in fja:
                            outf.write(l)
                    with open(ten, encoding="utf-8") as fen:
                        for l in fen:
                            outf.write(l)
                # train with stable special ids
                spm.SentencePieceTrainer.Train(
                    input=combined_txt,
                    model_prefix=os.path.join(data_dir, model_prefix),
                    vocab_size=vocab_size,
                    character_coverage=character_coverage,
                    model_type=model_type,
                    pad_id=0, unk_id=1, bos_id=2, eos_id=3
                )
                try:
                    os.remove(combined_txt)
                except:
                    pass

            # load SPM
            self.tokenizer_obj = spm.SentencePieceProcessor()
            self.tokenizer_obj.load(self.model_file)
            self._is_spm = True
            self._is_hf = False

        # set special ids
        if self._is_spm:
            self.pad_id = 0
            self.unk_id = self.tokenizer_obj.unk_id()
            self.bos_id = 2
            self.eos_id = 3
        elif self._is_hf:
            self.pad_id = getattr(self.tokenizer_obj, "pad_token_id", None)
            self.unk_id = getattr(self.tokenizer_obj, "unk_token_id", None)
            self.bos_id = getattr(self.tokenizer_obj, "bos_token_id", None)
            self.eos_id = getattr(self.tokenizer_obj, "eos_token_id", None)
            if self.pad_id is None:
                self.pad_id = 0
        else:
            self.pad_id = 0
            self.unk_id = 1
            self.bos_id = None
            self.eos_id = None

    def __len__(self):
        return len(self.ja_lines)

    def _encode_spm(self, text: str):
        ids = self.tokenizer_obj.encode(text, out_type=int)
        if self.add_bos_eos and (self.bos_id is not None) and (self.eos_id is not None):
            ids = [self.bos_id] + ids + [self.eos_id]
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        return ids

    def _encode_hf(self, text: str):
        enc = self.tokenizer_obj(text,
                                 truncation=True,
                                 padding=False,
                                 max_length=self.max_length,
                                 return_tensors=None)
        return enc["input_ids"]

    def _pad_ids(self, ids: list):
        if len(ids) >= self.max_length:
            return ids[:self.max_length]
        return ids + [self.pad_id] * (self.max_length - len(ids))

    def __getitem__(self, idx):
        ja_text = self.ja_lines[idx].strip()
        en_text = self.en_lines[idx].strip()

        if self._is_spm:
            src_ids = self._encode_spm(ja_text)
            tgt_ids = self._encode_spm(en_text)
        elif self._is_hf:
            src_ids = self._encode_hf(ja_text)
            tgt_ids = self._encode_hf(en_text)
        else:
            raise RuntimeError("Unsupported tokenizer type")

        src_ids = self._pad_ids(src_ids)
        tgt_ids = self._pad_ids(tgt_ids)

        src_mask = [0 if x == self.pad_id else 1 for x in src_ids]
        tgt_mask = [0 if x == self.pad_id else 1 for x in tgt_ids]

        labels = [(x if x != self.pad_id else self.ignore_index) for x in tgt_ids]

        return {
            "input_ids": torch.LongTensor(src_ids),
            "attention_mask": torch.LongTensor(src_mask),
            "labels": torch.LongTensor(labels),
            "labels_mask": torch.LongTensor(tgt_mask),
            "raw_ja": ja_text,
            "raw_en": en_text
        }
