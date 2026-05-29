"""
scripts/compare_models.py

Compares the raw NLLB baseline model with the Graph-Augmented NLLB model
using BLEU, chrF, and COMET metrics on a subset of the test dataset.
"""

import argparse
import yaml
import torch
from tqdm import tqdm
import sacrebleu
from tabulate import tabulate
from transformers import AutoModelForSeq2SeqLM
from preprocessing.tokenizer import TranslationTokenizer
from data_loading.loader import TranslationDatasetLoader
from models.full_model.baseline import BaselineTranslator

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_evaluation(model, dataloader, tokenizer, device, mixed_precision=True, use_graph=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_srcs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {'Graph Model' if use_graph else 'Base Model'}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            graph = None
            if use_graph and "graph" in batch:
                graph = batch["graph"]
                for k, v in graph.items():
                    if isinstance(v, torch.Tensor):
                        graph[k] = v.to(device)
            
            kwargs = {}
            if use_graph:
                kwargs["graph"] = graph
                
            if mixed_precision:
                with torch.autocast(device_type=device if device != "cpu" else "cpu"):
                    generated_tokens = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=128,
                        **kwargs
                    )
            else:
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    **kwargs
                )
                
            # Decode
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # The original japanese strings
            decoded_srcs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
            all_srcs.extend(decoded_srcs)
            
    return all_preds, all_labels, all_srcs

def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline/checkpoint-best.pt")
    parser.add_argument("--num_examples", type=int, default=50, help="Number of examples to test on")
    parser.add_argument("--skip_comet", action="store_true", help="Skip heavy COMET evaluation")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Temporarily override test subset size for fast comparison
    config["data"]["test_subset_size"] = args.num_examples
    
    print("Loading tokenizer and dataloader...")
    tokenizer = TranslationTokenizer(
        model_name=config["model"]["model_name"],
        src_lang=config["model"]["src_lang"],
        tgt_lang=config["model"]["tgt_lang"],
        max_length=config["model"]["max_length"]
    )
    dataset_loader = TranslationDatasetLoader(config["data"], tokenizer)
    test_loader = dataset_loader.get_dataloaders().get("test")
    if test_loader is None:
        print("No test dataset found!")
        return

    # 1. Base NLLB Model
    print("Initializing Raw NLLB Model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["model_name"]).to(device)
    base_preds, base_labels, base_srcs = run_evaluation(
        base_model, test_loader, tokenizer.tokenizer, device, 
        mixed_precision=config["training"].get("mixed_precision", True), 
        use_graph=False
    )
    del base_model # free VRAM
    torch.cuda.empty_cache()

    # 2. Graph NLLB Model
    print("Initializing Graph-Augmented NLLB Model...")
    graph_model = BaselineTranslator(model_name=config["model"]["model_name"])
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        graph_model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded graph model checkpoint.")
    except Exception as e:
        print(f"Warning: Failed to load checkpoint ({e}). Comparing against untrained graph weights.")
        
    graph_model = graph_model.to(device)
    graph_preds, _, _ = run_evaluation(
        graph_model, test_loader, tokenizer.tokenizer, device, 
        mixed_precision=config["training"].get("mixed_precision", True), 
        use_graph=True
    )
    del graph_model
    torch.cuda.empty_cache()

    # 3. Compute Metrics
    print("Computing sacrebleu metrics...")
    base_bleu = sacrebleu.corpus_bleu(base_preds, [base_labels]).score
    base_chrf = sacrebleu.corpus_chrf(base_preds, [base_labels]).score
    
    graph_bleu = sacrebleu.corpus_bleu(graph_preds, [base_labels]).score
    graph_chrf = sacrebleu.corpus_chrf(graph_preds, [base_labels]).score
    
    base_comet = "Skipped"
    graph_comet = "Skipped"
    
    if not args.skip_comet:
        print("Loading COMET model (Unbabel/wmt22-comet-da)...")
        from comet import download_model, load_from_checkpoint
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        
        base_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(base_srcs, base_preds, base_labels)]
        graph_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(base_srcs, graph_preds, base_labels)]
        
        print("Scoring Base Model with COMET...")
        base_comet_score = comet_model.predict(base_data, batch_size=8, gpus=1 if device=="cuda" else 0)
        base_comet = f"{base_comet_score.system_score:.4f}"
        
        print("Scoring Graph Model with COMET...")
        graph_comet_score = comet_model.predict(graph_data, batch_size=8, gpus=1 if device=="cuda" else 0)
        graph_comet = f"{graph_comet_score.system_score:.4f}"

    # 4. Print Comparison Table
    print("\n" + "="*50)
    print("METRICS COMPARISON")
    print("="*50)
    metrics_table = [
        ["Model", "SacreBLEU", "chrF", "COMET"],
        ["Base NLLB-200", f"{base_bleu:.2f}", f"{base_chrf:.2f}", base_comet],
        ["Graph-Augmented NLLB", f"{graph_bleu:.2f}", f"{graph_chrf:.2f}", graph_comet]
    ]
    print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
    
    print("\n" + "="*50)
    print("SAMPLE TRANSLATIONS")
    print("="*50)
    
    samples_to_show = min(5, len(base_srcs))
    for i in range(samples_to_show):
        print(f"Example {i+1}:")
        print(f"Source (JA): {base_srcs[i]}")
        print(f"Reference:   {base_labels[i]}")
        print(f"Base NLLB:   {base_preds[i]}")
        print(f"Graph-NLLB:  {graph_preds[i]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
