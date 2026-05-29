"""
main.py

Entry point for the Graph-Augmented Translation Project.
Currently runs the baseline training pipeline.
"""

import argparse
import yaml
import torch
from preprocessing.tokenizer import TranslationTokenizer
from datasets.loader import TranslationDatasetLoader
from models.full_model.baseline import BaselineTranslator
from training.trainer import BaselineTrainer

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Translation Baseline Training")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--test_dataloader", action="store_true", help="Only test the dataloader")
    parser.add_argument("--test_model", action="store_true", help="Only test the model forward pass")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Initializing Tokenizer...")
    tokenizer = TranslationTokenizer(
        model_name=config["model"]["model_name"],
        src_lang=config["model"]["src_lang"],
        tgt_lang=config["model"]["tgt_lang"],
        max_length=config["model"]["max_length"]
    )
    
    print("Initializing Dataset Loader...")
    dataset_loader = TranslationDatasetLoader(config["data"], tokenizer)
    dataloaders = dataset_loader.get_dataloaders()
    
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
            
            outputs = model(input_ids, attention_mask, labels)
            print(f"Forward pass successful. Loss: {outputs.loss.item()}")
            print(f"Encoder hidden states shape: {outputs.encoder_last_hidden_state.shape}")
            break
        print("Model test passed!")
        return
        
    print("Initializing Trainer...")
    trainer = BaselineTrainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("validation", []),
        config=config["training"],
        device=device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
