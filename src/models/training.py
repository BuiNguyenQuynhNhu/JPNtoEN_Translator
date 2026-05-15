import yaml
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from src.models.encoder import GraphBuilder
from src.models.decoder import GraphAugmentedNLLB

def load_config():
    with open("configs/translation.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["MODEL_NAME"])
    tokenizer.src_lang = model_cfg["SRC_LANG"]
    tokenizer.tgt_lang = model_cfg["TGT_LANG"]
    
    graph_builder = GraphBuilder(tokenizer)
    
    print(f"Loading Dataset: {data_cfg['DATASET_NAME']}")
    dataset = load_dataset(data_cfg["DATASET_NAME"])
    
    if data_cfg.get("SUBSET_SIZE"):
        train_data = dataset['train'].select(range(data_cfg["SUBSET_SIZE"]))
        val_data = dataset['test'].select(range(data_cfg.get("VAL_SUBSET_SIZE", 100)))
    else:
        train_data = dataset['train']
        val_data = dataset['test']
        
    def preprocess_function(examples):
        inputs = [ex['ja'] for ex in examples['translation']]
        targets = [ex['en'] for ex in examples['translation']]
        
        model_inputs = tokenizer(inputs, max_length=model_cfg["MAX_LENGTH"], truncation=True, padding="max_length")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=model_cfg["MAX_LENGTH"], truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        
        batch_graphs = []
        for text in inputs:
            bias_matrix = graph_builder.build_dense_graph(text, model_cfg["MAX_LENGTH"])
            batch_graphs.append(bias_matrix.tolist())
            
        model_inputs["edge_bias_matrix"] = batch_graphs
        return model_inputs

    print("Tokenizing and building graphs...")
    tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=['translation'])
    tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=['translation'])
    
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
        NUM_RELATIONS = model_cfg["NUM_RELATIONS"]
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
        report_to = 'None',
        label_smoothing_factor=0.1
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting Training...")
    trainer.train()
    
    # Save the custom model state dict
    torch.save(model.state_dict(), f"{cfg['infer']['output_dir']}/pytorch_model.bin")
    tokenizer.save_pretrained(cfg["infer"]["output_dir"])
    print(f"Model saved to {cfg['infer']['output_dir']}")

if __name__ == "__main__":
    main()
