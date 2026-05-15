import yaml
import torch
import os
import json
import spacy
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from src.models.decoder import GraphAugmentedNLLB

class OnlineSparseGraphBuilder:
    def __init__(self, tokenizer, rel2id_path):
        self.nlp = spacy.load("ja_ginza")
        self.tokenizer = tokenizer
        
        if os.path.exists(rel2id_path):
            with open(rel2id_path, "r") as f:
                self.rel2id = json.load(f)
        else:
            self.rel2id = {"<pad>": 0, "ROOT": 1}
            print(f"Warning: {rel2id_path} not found. Relation mapping might be misaligned.")
            
    def build_sparse_graph(self, text, max_length):
        doc = self.nlp(text)
        
        encoded = self.tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            add_special_tokens=True, max_length=max_length, truncation=True
        )
        input_ids = encoded["input_ids"][0]
        offsets = encoded["offset_mapping"][0]
        special_token_mask = [id in self.tokenizer.all_special_ids for id in input_ids.tolist()]
        
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
                if start_char >= token_end: break 
                if end_char <= token_start:
                    subword_idx += 1 
                    continue
                word_to_subwords[token.i].append(subword_idx)
                subword_idx += 1
                
        head_indices = []
        dep_indices = []
        edge_types = []
        
        for token in doc:
            # At inference time, unknown relations are treated as PAD/0 to avoid crashing
            rel_id = self.rel2id.get(token.dep_, 0)
            
            head_subwords = word_to_subwords.get(token.head.i, [])
            dep_subwords = word_to_subwords.get(token.i, [])
            
            for head_idx in head_subwords:
                for dep_idx in dep_subwords:
                    if head_idx < max_length and dep_idx < max_length:
                        head_indices.append(head_idx)
                        dep_indices.append(dep_idx)
                        edge_types.append(rel_id)
                        
        if len(edge_types) == 0:
            # Fallback to prevent empty tensor issues
            head_indices = [0]
            dep_indices = [0]
            edge_types = [0]
            
        return (
            torch.tensor([head_indices], dtype=torch.long),
            torch.tensor([dep_indices], dtype=torch.long),
            torch.tensor([edge_types], dtype=torch.long)
        )

class TranslationPipeline:
    def __init__(self, config_path="configs/translation.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        self.model_cfg = cfg["model"]
        self.device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["MODEL_NAME"])
        self.tokenizer.src_lang = self.model_cfg["SRC_LANG"]
        self.tokenizer.tgt_lang = self.model_cfg["TGT_LANG"]
        
        rel2id_path = "data/processed_dataset/rel2id.json"
        self.graph_builder = OnlineSparseGraphBuilder(self.tokenizer, rel2id_path)
        
        print("Loading Base NLLB Model...")
        base_model = M2M100ForConditionalGeneration.from_pretrained(self.model_cfg["MODEL_NAME"])
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=True, 
            r=self.model_cfg["LORA_R"], 
            lora_alpha=self.model_cfg["LORA_ALPHA"], 
            lora_dropout=self.model_cfg["LORA_DROPOUT"],
            target_modules=["q_proj", "v_proj"]
        )
        peft_model = get_peft_model(base_model, peft_config)
        
        class ModelConfig:
            NUM_RELATIONS = len(self.graph_builder.rel2id)
            MEMORY_WINDOW_SIZE = self.model_cfg["MEMORY_WINDOW_SIZE"]
            
        self.model = GraphAugmentedNLLB(peft_model, ModelConfig()).to(self.device)
        
        checkpoint_dir = cfg["infer"]["checkpoints"]
        if checkpoint_dir and os.path.exists(f"{checkpoint_dir}/pytorch_model.bin"):
            try:
                state_dict = torch.load(f"{checkpoint_dir}/pytorch_model.bin", map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded custom weights from {checkpoint_dir}")
            except Exception as e:
                print(f"Warning: Could not load weights from {checkpoint_dir}. Error: {e}")
        else:
            print("No fine-tuned checkpoint found, using base architecture.")
                
        self.model.eval()
        
    def translate_subtitles(self, texts):
        translations = []
        
        # Reset streaming memory buffer (deque handles bounding automatically, but we clear it for a new scene)
        self.model.memory_adapter.memory_k.clear()
        self.model.memory_adapter.memory_v.clear()
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.model_cfg["MAX_LENGTH"], truncation=True, padding="max_length").to(self.device)
            
            # Online Sparse Graph Generation
            edge_head, edge_dep, edge_type = self.graph_builder.build_sparse_graph(text, self.model_cfg["MAX_LENGTH"])
            
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    edge_index_head=edge_head.to(self.device),
                    edge_index_dep=edge_dep.to(self.device),
                    edge_type=edge_type.to(self.device),
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.model_cfg["TGT_LANG"]],
                    max_length=self.model_cfg["MAX_LENGTH"]
                )
                
            trans = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translations.append(trans)
            
        return translations

if __name__ == "__main__":
    pipeline = TranslationPipeline()
    sample_texts = [
        "これはテストです。",
        "次の文です。",
        "グラフとメモリを使用しています。"
    ]
    
    print("Testing translation pipeline...")
    results = pipeline.translate_subtitles(sample_texts)
    for src, tgt in zip(sample_texts, results):
        print(f"SRC: {src}")
        print(f"TGT: {tgt}\n")
