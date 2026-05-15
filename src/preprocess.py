import os
import yaml
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json

def load_config():
    with open("configs/translation.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    
    print("Loading spaCy and Tokenizer...")
    nlp = spacy.load("ja_ginza")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["MODEL_NAME"])
    tokenizer.src_lang = model_cfg["SRC_LANG"]
    tokenizer.tgt_lang = model_cfg["TGT_LANG"]
    
    rel2id = {"<pad>": 0, "ROOT": 1}
    
    print(f"Loading Dataset: {data_cfg['DATASET_NAME']}")
    dataset = load_dataset(data_cfg["DATASET_NAME"])
    
    def process_split(split_name, max_samples=None):
        ds = dataset[split_name]
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
            
        processed_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "edge_index_head": [],
            "edge_index_dep": [],
            "edge_type": []
        }
        
        for ex in tqdm(ds, desc=f"Processing {split_name}"):
            ja_text = ex['translation']['ja']
            en_text = ex['translation']['en']
            
            # Tokenize
            enc = tokenizer(ja_text, max_length=model_cfg["MAX_LENGTH"], truncation=True, padding="max_length")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(en_text, max_length=model_cfg["MAX_LENGTH"], truncation=True, padding="max_length")
                
            input_ids = enc["input_ids"]
            
            # Graph parsing
            doc = nlp(ja_text)
            encoded_offsets = tokenizer(
                ja_text, return_offsets_mapping=True, add_special_tokens=True,
                max_length=model_cfg["MAX_LENGTH"], truncation=True
            )
            offsets = encoded_offsets["offset_mapping"]
            special_token_mask = [id in tokenizer.all_special_ids for id in input_ids]
            
            word_to_subwords = {token.i: [] for token in doc}
            subword_idx = 0
            num_subwords = len(offsets)
            
            for token in doc:
                token_start = token.idx
                token_end = token.idx + len(token.text)
                
                while subword_idx < num_subwords:
                    start_char, end_char = offsets[subword_idx]
                    
                    if special_token_mask[subword_idx] or start_char == end_char:
                        subword_idx += 1
                        continue
                        
                    if start_char >= token_end:
                        break 
                        
                    if end_char <= token_start:
                        subword_idx += 1 
                        continue
                        
                    word_to_subwords[token.i].append(subword_idx)
                    subword_idx += 1
                    
            head_indices = []
            dep_indices = []
            edge_types = []
            
            for token in doc:
                if token.dep_ not in rel2id:
                    rel2id[token.dep_] = len(rel2id)
                rel_id = rel2id[token.dep_]
                
                head_subwords = word_to_subwords.get(token.head.i, [])
                dep_subwords = word_to_subwords.get(token.i, [])
                
                for head_idx in head_subwords:
                    for dep_idx in dep_subwords:
                        if head_idx < model_cfg["MAX_LENGTH"] and dep_idx < model_cfg["MAX_LENGTH"]:
                            head_indices.append(head_idx)
                            dep_indices.append(dep_idx)
                            edge_types.append(rel_id)
                            
            processed_data["input_ids"].append(input_ids)
            processed_data["attention_mask"].append(enc["attention_mask"])
            processed_data["labels"].append(labels["input_ids"])
            processed_data["edge_index_head"].append(head_indices)
            processed_data["edge_index_dep"].append(dep_indices)
            processed_data["edge_type"].append(edge_types)
            
        from datasets import Dataset
        return Dataset.from_dict(processed_data)
        
    train_ds = process_split('train', data_cfg.get("SUBSET_SIZE"))
    val_ds = process_split('test', data_cfg.get("VAL_SUBSET_SIZE"))
    
    save_dir = "data/processed_dataset"
    os.makedirs(save_dir, exist_ok=True)
    train_ds.save_to_disk(f"{save_dir}/train")
    val_ds.save_to_disk(f"{save_dir}/test")
    
    # Save relation mapping
    with open(f"{save_dir}/rel2id.json", "w") as f:
        json.dump(rel2id, f)
        
    print(f"Preprocessing complete. Saved to {save_dir}")
    print(f"Total relation types discovered: {len(rel2id)}")

if __name__ == "__main__":
    main()
