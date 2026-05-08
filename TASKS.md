# Project: Manga translator

## Roadmap

### Part 1: Japanese to English translate

- [] **Preprocessing**
    - [x] Implement `KFTTDataset` and `DataLoader`
    - [] Train subword tokenizer for JA and EN
- [] **Model development **
- [] **Training & Optimization **
- [] **Inference**


### Part 2: OCR & Text Extractrion

- []**Preprocessing**
- []**Text Detection**
- []**Text Recognition**
- []**Post processing**

## Project architecture

```text
manga-translate/
├── data/
│    
├── src/
│   ├── ocr/                # Module for detection & recognition
│   │   ├── detector.py
│   │   └── recognizer.py
│   ├── translation/        # Module for NMT
│   │   ├── data/
│   │   └── ...
│   └── utils/              # Image processing & logging helpers
├── checkpoints/            # Model weights for OCR & NMT
├── configs/                # Hyperparameter
├── main.py                 # End-to-end Pipeline (Image -> English Text)
└── requirements.txt
