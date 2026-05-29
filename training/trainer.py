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
from tqdm import tqdm
import sacrebleu
from accelerate import Accelerator
import sacrebleu

class BaselineTrainer:
    def __init__(self, model: nn.Module, tokenizer, train_loader, val_loader, config: dict, eval_bleu: bool = False, eval_comet: bool = False):
        self.tokenizer = tokenizer
        self.config = config
        self.eval_bleu = eval_bleu
        self.eval_comet = eval_comet
        
        self.best_val_metric = float('inf') # if loss, lower is better. if bleu, higher is better.
        if self.eval_bleu:
            self.best_val_metric = 0.0
        
        self.epochs = config.get("epochs", 3)
        self.lr = config.get("learning_rate", 2e-4)
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        self.mixed_precision = config.get("mixed_precision", True)
        self.output_dir = config.get("output_dir", "./checkpoints/baseline")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.grad_accum_steps,
            mixed_precision="fp16" if self.mixed_precision else "no"
        )
        
        self.optimizer = AdamW(model.parameters(), lr=self.lr)
        
        # Accelerate takes over device placement and DDP wrapping
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            model, self.optimizer, train_loader, val_loader
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self):
        """
        Executes the training loop using Accelerate.
        """
        self.accelerator.print(f"Starting training on {self.accelerator.device} for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Only show progress bar on main process
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", disable=not self.accelerator.is_local_main_process)
            
            for step, batch in enumerate(progress_bar):
                
                # Accelerate automatically places standard tensors on device,
                # but we must manually move our custom graph dictionary.
                graph = batch.get("graph", None)
                if graph is not None:
                    for k, v in graph.items():
                        if isinstance(v, torch.Tensor):
                            graph[k] = v.to(self.accelerator.device)
                
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        graph=graph
                    )
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            # Gather average loss across all processes
            avg_train_loss = total_loss / len(self.train_loader)
            self.accelerator.print(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}")
            
            # Checkpoint logic (only main process should save)
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:
                # Validation
                val_loss, val_bleu, val_chrf, val_comet = self.evaluate()
                
                print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
                if val_bleu is not None:
                    print(f"Epoch {epoch+1} Validation BLEU: {val_bleu:.2f} | chrF: {val_chrf:.2f}")
                if val_comet is not None:
                    print(f"Epoch {epoch+1} Validation COMET: {val_comet:.4f}")
                    
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
        Runs only on the main process to simplify metric aggregation during autoregressive decoding.
        """
        # Unwrap model for native Generation functionality if needed
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()
        total_loss = 0
        
        all_preds = []
        all_labels = []
        all_srcs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                graph = batch.get("graph", None)
                if graph is not None:
                    for k, v in graph.items():
                        if isinstance(v, torch.Tensor):
                            graph[k] = v.to(self.accelerator.device)
                
                # 1. Compute Loss
                outputs = unwrapped_model(
                    input_ids=batch["input_ids"].to(self.accelerator.device),
                    attention_mask=batch["attention_mask"].to(self.accelerator.device),
                    labels=batch["labels"].to(self.accelerator.device),
                    graph=graph
                )
                total_loss += outputs.loss.item()
                    
                # 2. Compute BLEU (if enabled)
                if self.eval_bleu:
                    generated_tokens = unwrapped_model.generate(
                        input_ids=batch["input_ids"].to(self.accelerator.device),
                        attention_mask=batch["attention_mask"].to(self.accelerator.device),
                        graph=graph,
                        max_length=self.config.get("max_length", 128)
                    )
                    
                    # Gather across processes to compute global BLEU
                    generated_tokens = self.accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id)
                    labels = self.accelerator.pad_across_processes(batch["labels"].to(self.accelerator.device), dim=1, pad_index=-100)
                    
                    generated_tokens, labels = self.accelerator.gather_for_metrics((generated_tokens, labels))
                    
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
                    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_srcs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    
                    all_preds.extend(decoded_preds)
                    all_labels.extend(decoded_labels)
                    all_srcs.extend(decoded_srcs)
                    
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        
        bleu_score = None
        chrf_score = None
        comet_score_val = None
        
        if self.eval_bleu and len(all_preds) > 0:
            bleu = sacrebleu.corpus_bleu(all_preds, [all_labels])
            bleu_score = bleu.score
            
            chrf = sacrebleu.corpus_chrf(all_preds, [all_labels])
            chrf_score = chrf.score
            
        if self.eval_comet and len(all_preds) > 0:
            try:
                from comet import download_model, load_from_checkpoint
                comet_path = download_model("Unbabel/wmt22-comet-da")
                comet_model = load_from_checkpoint(comet_path)
                data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(all_srcs, all_preds, all_labels)]
                c_score = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
                comet_score_val = c_score.system_score
            except Exception as e:
                print(f"COMET evaluation failed: {e}")
            
        return avg_loss, bleu_score, chrf_score, comet_score_val
        
    def save_checkpoint(self, path: str):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save({
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

# Future improvements:
# - Add BLEU, COMET, chrF calculation during evaluation
# - Implement early stopping mechanism
