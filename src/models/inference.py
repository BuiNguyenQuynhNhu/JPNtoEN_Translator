import yaml
import torch
import os
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from src.models.encoder import GraphBuilder
from src.models.decoder import GraphAugmentedNLLB

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
        
        self.graph_builder = GraphBuilder(self.tokenizer)
        
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
            NUM_RELATIONS = self.model_cfg["NUM_RELATIONS"]
            MEMORY_WINDOW_SIZE = self.model_cfg["MEMORY_WINDOW_SIZE"]
            
        self.model = GraphAugmentedNLLB(peft_model, ModelConfig()).to(self.device)
        
        checkpoint_dir = cfg["infer"]["checkpoints"]
        if checkpoint_dir and os.path.exists(f"{checkpoint_dir}/pytorch_model.bin"):
            try:
                state_dict = torch.load(f"{checkpoint_dir}/pytorch_model.bin", map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded custom weights from {checkpoint_dir}")
            except Exception as e:
                print(f"Warning: Could not load weights from {checkpoint_dir}. Using initial architecture. Error: {e}")
        else:
            print("No fine-tuned checkpoint found, using base architecture with untrained adapters.")
                
        self.model.eval()
        
    def translate_subtitles(self, texts):
        translations = []
        
        # Reset streaming memory
        self.model.memory_adapter.memory_k = []
        self.model.memory_adapter.memory_v = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.model_cfg["MAX_LENGTH"], truncation=True, padding="max_length").to(self.device)
            edge_bias_matrix = self.graph_builder.build_dense_graph(text, self.model_cfg["MAX_LENGTH"]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    edge_bias_matrix=edge_bias_matrix,
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
