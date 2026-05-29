# SOTA Graph-Augmented Subtitle Translation 🚀

A production-grade, memory-efficient Japanese $\rightarrow$ English subtitle translation system. This repository combines the **NLLB-200** backbone with a **Graph Bias Adapter** (Syntax-Aware) and an **ALiBi-enhanced KV Memory Adapter** (Long-context Streaming) to deliver state-of-the-art subtitle translation.

## 🌟 Key Features

*   **Offline Graph Preprocessing**: Achieves 0% NLP parsing overhead during training by precomputing Japanese dependency structures via `spaCy` and saving them as sparse indices.
*   **Sparse In-Place Scatter Attention**: Computes graph bias directly inside the $QK^T$ matrix, reducing Graph Memory complexity from $O(S^2)$ to $O(E)$ (Out-Of-Memory free).
*   **Bounded ALiBi Memory**: Employs `collections.deque` and strict `.detach()` rules for infinite-context streaming translation without RAM leakage or positional drift.
*   **End-to-End SRT Retiming**: Smart subtitle adjustment using a Characters-Per-Second (CPS) algorithm to extend short Japanese timestamps for English viewers.

## 🛠️ Installation

Make sure your environment is activated and install the necessary dependencies:

```bash
pip install torch transformers datasets evaluate peft sacrebleu
pip install spacy
pip install ja_ginza  # spaCy Japanese Model
pip install unbabel-comet # For COMET SOTA evaluation metric
```

## 🚀 Running the Pipeline

Follow these steps to train, evaluate, and use the translation system.

### 1. Data Preprocessing (Offline Graph Building)
Before training, you must process the raw dataset to extract dependency parsing information. This script will tokenize the text, run `spaCy`, and save the sparse graphs to `data/processed_dataset/`.

```bash
python -m src.preprocess
```

### 2. Training
Run the memory-optimized training pipeline. It utilizes HuggingFace `Seq2SeqTrainer` and LoRA. The model dynamically loads the precomputed sparse graphs from Step 1.

```bash
python -m src.models.training
```
*(Note: Check `configs/translation.yaml` to adjust hyperparameters like `BATCH_SIZE`, `EPOCHS`, and `LORA_R`).*

### 3. Evaluation
Evaluate your trained custom checkpoint on the validation set. It computes standard metrics: `SacreBLEU`, `chrF`, and `COMET`.

```bash
python -m src.models.evaluate
```

### 4. End-to-End SRT Translation (Inference)
Translate a raw Japanese `.srt` subtitle file directly into English. It automatically batches the translation and handles **Smart Retiming** (default 20 Characters Per Second) so the subtitles fit the English reading speed seamlessly.

```bash
python -m src.utils.srt_utils --input video_jp.srt --output video_en.srt --cps 20 --batch 8
```
