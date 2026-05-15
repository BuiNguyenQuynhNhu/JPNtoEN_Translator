import yaml
import evaluate
from datasets import load_dataset
from src.models.inference import TranslationPipeline
import time

def load_config():
    with open("configs/translation.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    
    pipeline = TranslationPipeline()
    
    print("Loading evaluation metrics...")
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    try:
        comet = evaluate.load("comet")
    except Exception as e:
        print(f"Warning: Failed to load comet metric. Ensure 'unbabel-comet' is installed. Error: {e}")
        comet = None
    
    print(f"Loading Dataset: {data_cfg['DATASET_NAME']}")
    dataset = load_dataset(data_cfg["DATASET_NAME"])
    
    # Evaluate on the validation/test subset
    val_size = data_cfg.get("VAL_SUBSET_SIZE", 100)
    val_data = dataset['test'].select(range(val_size))
    
    sources = [ex['ja'] for ex in val_data['translation']]
    references = [[ex['en']] for ex in val_data['translation']]
    
    print(f"Starting evaluation on {len(sources)} sentences...")
    start_time = time.time()
    
    predictions = pipeline.translate_subtitles(sources)
    
    end_time = time.time()
    
    print("Computing metrics...")
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    chrf_results = chrf.compute(predictions=predictions, references=references)
    
    comet_score = "N/A"
    if comet is not None:
        # Comet expects a flat list of references
        flat_references = [ref[0] for ref in references]
        comet_results = comet.compute(predictions=predictions, references=flat_references, sources=sources)
        comet_score = f"{comet_results['mean_score']:.4f}"
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"SacreBLEU Score: {bleu_results['score']:.2f}")
    print(f"chrF Score:      {chrf_results['score']:.2f}")
    print(f"COMET Score:     {comet_score}")
    print(f"Time Taken:      {end_time - start_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    main()
