# Graph-Augmented Japanese-to-English Translation Pipeline

This repository contains a full, end-to-end research prototype for translating Japanese text into English using **NLLB-200** augmented with a **Semantic Discourse Graph**.

By extracting Event and Entity nodes from the source text and feeding them into a custom Graph Transformer, we construct contextual graph memory that is injected directly into the NLLB autoregressive decoder via Cross-Attention!

## Features
- **Semantic Graph Construction:** Uses `spaCy` to extract Entities, Events, and Dependencies.
- **Sparse-to-Dense Routing:** Highly optimized native PyTorch tensor operations (no heavy PyTorch Geometric dependency).
- **Custom Graph Transformer:** Computes relational message passing over the semantic graph.
- **Non-Destructive Decoder Adapter:** Wraps the HuggingFace `lm_head` to inject graph memory during both training and autoregressive `.generate()` decoding without rewriting the core `AutoModelForSeq2SeqLM`.
- **BLEU Evaluation Loop:** Fully integrated `sacrebleu` metrics and automated best-checkpoint saving.

---

## 🚀 How to Run on Google Colab

To easily train and evaluate this model on a Google Colab GPU, create a new notebook, select a **T4 GPU** or better, and copy-paste the following command into a single cell:

```bash
# 1. Clone the repository
!git clone https://github.com/BuiNguyenQuynhNhu/JPNtoEN_Translator.git
%cd JPNtoEN_Translator

# 2. Install required dependencies
!pip install -r requirements.txt
!python -m spacy download ja_core_news_sm

# 3. (Optional) Run a quick local test to verify the forward pass and graph processing
!python main.py --config configs/baseline.yaml --test_model

# 4. Start the full training loop with SacreBLEU evaluation!
!python main.py --config configs/baseline.yaml --eval_bleu
```

### Configuration
You can adjust hyperparameters (batch size, learning rate, max length, subset sizes for quick testing) inside `configs/baseline.yaml`.

---

### File Structure
- `configs/`: YAML configuration files.
- `data_loading/`: Unified dataloader that handles HF datasets, local parallel files, tokenization, and `spaCy` graph construction.
- `models/graph/`: Contains the `GraphBuilder`, `batch_graphs`, `NodeFeatureExtractor`, `GraphTransformer`, and `GraphMemoryAdapter`.
- `models/full_model/baseline.py`: The core wrapper that stitches NLLB-200 and the Graph modules together.
- `training/trainer.py`: The main loop handling mixed-precision, gradient clipping, evaluation, and checkpointing.
