import argparse
import yaml
import torch
from preprocessing.tokenizer import TranslationTokenizer
from data_loading.loader import TranslationDatasetLoader
from models.full_model.baseline import BaselineTranslator
from training.trainer import BaselineTrainer
import os

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Translation Baseline Training")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--test_dataloader", action="store_true", help="Only test the dataloader")
    parser.add_argument("--test_model", action="store_true", help="Only test the model forward pass")
    parser.add_argument("--eval_bleu", action="store_true", help="Run BLEU and chrF evaluation on validation set")
    parser.add_argument("--eval_comet", action="store_true", help="Run COMET evaluation on validation set (SLOW)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = args.device
    
    # Force accelerate to use fp16 if specified in config, regardless of launch command
    if config.get("training", {}).get("mixed_precision", True):
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    
    print("Initializing Tokenizer...")
    tokenizer = TranslationTokenizer(
        model_name=config["model"]["model_name"],
        src_lang=config["model"]["src_lang"],
        tgt_lang=config["model"]["tgt_lang"],
        max_length=config["model"]["max_length"]
    )
    
    print("Initializing Dataset Loader...")
    dataset_loader = TranslationDatasetLoader(config["data"], tokenizer)
    batch_size = config.get("training", {}).get("batch_size", 8)
    dataloaders = dataset_loader.get_dataloaders(batch_size=batch_size)
    
    if args.test_dataloader:
        for batch in dataloaders["train"]:
            print(f"Batch input_ids shape: {batch['input_ids'].shape}")
            print(f"Batch labels shape: {batch['labels'].shape}")
            break
        print("Dataloader test passed!")
        return

    print("Initializing Baseline Model...")
    model = BaselineTranslator(model_name=config["model"]["model_name"])
    
    if args.test_model:
        model = model.to(device)
        for batch in dataloaders["train"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Send graph tensors to device
            graph = batch["graph"]
            for k, v in graph.items():
                graph[k] = v.to(device)
            
            outputs = model(input_ids, attention_mask, labels, graph=graph)
            print(f"Forward pass successful. Loss: {outputs.loss.item()}")
            print(f"Encoder hidden states shape: {outputs.encoder_last_hidden_state.shape}")
            if hasattr(outputs, 'node_features'):
                print(f"Node features shape: {outputs.node_features.shape}")
            break
        print("Model test passed!")
        return
        
    print("Initializing Trainer...")
    trainer = BaselineTrainer(
        model=model,
        tokenizer=tokenizer.tokenizer, # Pass the HF tokenizer object
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("validation", []),
        config=config["training"],
        eval_bleu=args.eval_bleu,
        eval_comet=args.eval_comet
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
