import yaml
import torch
import json
from datasets import load_from_disk
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType
from src.models.decoder import GraphAugmentedNLLB
from dataclasses import dataclass

def load_config():
    with open("configs/translation.yaml", "r") as f:
        return yaml.safe_load(f)

@dataclass
class SparseGraphDataCollator:
    tokenizer: AutoTokenizer
    max_length: int
    
    def __call__(self, features):
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "labels": [f["labels"] for f in features]
        }
        
        batch = self.tokenizer.pad(batch, return_tensors="pt", padding="max_length", max_length=self.max_length)
        
        # Sparse graphs have variable number of edges per sentence. We need to pad them to the max edges in the batch.
        max_edges = max(len(f["edge_type"]) for f in features)
        if max_edges == 0:
            max_edges = 1 # Prevent empty tensor errors
            
        bsz = len(features)
        
        edge_index_head = torch.full((bsz, max_edges), 0, dtype=torch.long)
        edge_index_dep = torch.full((bsz, max_edges), 0, dtype=torch.long)
        edge_type = torch.full((bsz, max_edges), 0, dtype=torch.long) # 0 is <pad>
        
        for i, f in enumerate(features):
            n_edges = len(f["edge_type"])
            if n_edges > 0:
                edge_index_head[i, :n_edges] = torch.tensor(f["edge_index_head"], dtype=torch.long)
                edge_index_dep[i, :n_edges] = torch.tensor(f["edge_index_dep"], dtype=torch.long)
                edge_type[i, :n_edges] = torch.tensor(f["edge_type"], dtype=torch.long)
                
        batch["edge_index_head"] = edge_index_head
        batch["edge_index_dep"] = edge_index_dep
        batch["edge_type"] = edge_type
        
        return batch

def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["MODEL_NAME"])
    tokenizer.src_lang = model_cfg["SRC_LANG"]
    tokenizer.tgt_lang = model_cfg["TGT_LANG"]
    
    save_dir = "data/processed_dataset"
    print(f"Loading Preprocessed Offline Dataset from {save_dir}...")
    tokenized_train = load_from_disk(f"{save_dir}/train")
    tokenized_val = load_from_disk(f"{save_dir}/test")
    
    with open(f"{save_dir}/rel2id.json", "r") as f:
        rel2id = json.load(f)
    num_relations = len(rel2id)
    print(f"Loaded {num_relations} relation types.")
    
    print("Loading Base NLLB Model...")
    base_model = M2M100ForConditionalGeneration.from_pretrained(model_cfg["MODEL_NAME"])
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=model_cfg["LORA_R"], 
        lora_alpha=model_cfg["LORA_ALPHA"], 
        lora_dropout=model_cfg["LORA_DROPOUT"],
        target_modules=["q_proj", "v_proj"]
    )
    peft_model = get_peft_model(base_model, peft_config)
    
    class ModelConfig:
        NUM_RELATIONS = num_relations
        MEMORY_WINDOW_SIZE = model_cfg["MEMORY_WINDOW_SIZE"]
        
    model = GraphAugmentedNLLB(peft_model, ModelConfig()).to(device)
    print("Architecture Initialized.")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["infer"]["output_dir"],
        evaluation_strategy="epoch",
        learning_rate=float(model_cfg["LEARNING_RATE"]),
        per_device_train_batch_size=model_cfg["BATCH_SIZE"],
        per_device_eval_batch_size=model_cfg["BATCH_SIZE"],
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=model_cfg["EPOCHS"],
        fp16=True, 
        predict_with_generate=True,
        gradient_accumulation_steps=model_cfg["GRADIENT_ACCUMULATION_STEPS"],
        remove_unused_columns=False,
        report_to="none",
        label_smoothing_factor=0.1
    )
    
    data_collator = SparseGraphDataCollator(tokenizer, max_length=model_cfg["MAX_LENGTH"])
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting Optimized Memory-Safe Training...")
    trainer.train()
    
    torch.save(model.state_dict(), f"{cfg['infer']['output_dir']}/pytorch_model.bin")
    tokenizer.save_pretrained(cfg["infer"]["output_dir"])
    print(f"Model saved to {cfg['infer']['output_dir']}")

if __name__ == "__main__":
    main()
