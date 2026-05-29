"""
training/trainer.py

This module provides the training loop for the baseline translation model.
It supports:
- Mixed precision training (AMP)
- Gradient clipping
- Checkpointing
- Simple validation loss/logging
- Colab/Kaggle friendly execution
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sacrebleu

class BaselineTrainer:
    def __init__(self, model: nn.Module, tokenizer, train_loader, val_loader, config: dict, device: str = "cuda", eval_bleu: bool = False):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.eval_bleu = eval_bleu
        
        self.best_val_metric = float('inf') # if loss, lower is better. if bleu, higher is better.
        if self.eval_bleu:
            self.best_val_metric = 0.0
        
        self.epochs = config.get("epochs", 3)
        self.lr = config.get("learning_rate", 2e-4)
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        self.mixed_precision = config.get("mixed_precision", True)
        self.output_dir = config.get("output_dir", "./checkpoints/baseline")
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler(enabled=self.mixed_precision)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self):
        """
        Executes the training loop.
        """
        print(f"Starting training on {self.device} for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                graph = batch["graph"]
                for k, v in graph.items():
                    graph[k] = v.to(self.device)
                
                with autocast(enabled=self.mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        graph=graph
                    )
                    loss = outputs.loss / self.grad_accum_steps
                    
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                total_loss += loss.item() * self.grad_accum_steps
                progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})
                
            avg_train_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}")
            
            # Validation
            val_loss, val_bleu = self.evaluate()
            
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            if val_bleu is not None:
                print(f"Epoch {epoch+1} Validation BLEU: {val_bleu:.2f}")
                
            # Checkpoint logic
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}.pt")
            self.save_checkpoint(checkpoint_path)
            
            # Save best checkpoint
            is_best = False
            if self.eval_bleu and val_bleu is not None:
                if val_bleu > self.best_val_metric:
                    self.best_val_metric = val_bleu
                    is_best = True
            else:
                if val_loss < self.best_val_metric:
                    self.best_val_metric = val_loss
                    is_best = True
                    
            if is_best:
                best_path = os.path.join(self.output_dir, f"checkpoint-best.pt")
                self.save_checkpoint(best_path)
                print(f"Saved new best checkpoint to {best_path} (Metric: {self.best_val_metric:.4f})")
            
    def evaluate(self):
        """
        Evaluates the model on the validation set.
        Returns the average loss, and optionally the BLEU score.
        """
        self.model.eval()
        total_loss = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                graph = batch.get("graph", None)
                if graph is not None:
                    for k, v in graph.items():
                        graph[k] = v.to(self.device)
                
                # 1. Compute Loss
                with autocast(enabled=self.mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        graph=graph
                    )
                    total_loss += outputs.loss.item()
                    
                # 2. Compute BLEU (if enabled)
                if self.eval_bleu:
                    generated_tokens = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        graph=graph,
                        max_length=self.config.get("max_length", 128)
                    )
                    
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    
                    # Replace -100 in labels with pad token id for decoding
                    labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
                    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    all_preds.extend(decoded_preds)
                    all_labels.extend(decoded_labels)
                    
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        
        bleu_score = None
        if self.eval_bleu and len(all_preds) > 0:
            # sacrebleu requires list of list for references
            bleu = sacrebleu.corpus_bleu(all_preds, [all_labels])
            bleu_score = bleu.score
            
        return avg_loss, bleu_score
        
    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

# Future improvements:
# - Add BLEU, COMET, chrF calculation during evaluation
# - Implement early stopping mechanism
