# Project: Manga translator

## Roadmap

### Part 1: Machine Translation Module

[x] **Preprocessing**
    [x] Implement `KFTTDataset` and `DataLoader`
    [ ] Integrate **OpenSubtitles** or **JParaCrawl** for conversational variety.
    [] Preprocessing dataset
    [] Train subword tokenizer for JA and EN
[] **Model development **
[] **Training & Optimization **
[] **Inference**


### Part 2: Subtitle Processing & Context

[ ] **File I/O Handling:**
[ ] **Context-Aware Translation:**
[ ] **Post-Processing:**

## Project architecture

```text
sub-translator/
├── data/
│   ├── raw/                # KFTT, OpenSubtitles, and sample .srt files
│   ├── processed/          # Tokenized data & cleaned pairs
│   └── vocab/              # BPE/SentencePiece models
├── src/
│   ├── translation/        # Core NMT Module
│   │   ├── data/           # kftt_loader.py, sub_loader.py
│   │   ├── models/         # transformer.py, modules.py
│   │   └── training/       # trainer.py, scheduler.py
│   ├── sub_processing/     # Subtitle specific logic
│   │   ├── parser.py       # Reads .srt/.ass
│   │   ├── formatter.py    # Writes translated .srt/.ass
│   │   └── context.py      # Logic for multi-sentence context
│   └── utils/              # Metrics (BLEU) & general helpers
├── configs/                # Hyperparameters (config.yaml)
├── checkpoints/            # Saved model weights (.pth)
├── main.py                 # Pipeline: Subtitle File -> Translation -> Output File
└── requirements.txt
