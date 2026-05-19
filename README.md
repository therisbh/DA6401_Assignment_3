# DA6401 Assignment 3 

**Name:** Rishabh Gupta  
**Roll Number:** DA25M024 

Implementation of the Transformer architecture from the paper **"Attention Is All You Need"** using PyTorch for Neural Machine Translation on the Multi30k dataset.

---

## 🔗 Links

| Resource | URL |
|---|---|
| **W&B project link** | https://wandb.ai/da25m024-iitm/da6401-a3/ |
| **W&B Report link** | https://api.wandb.ai/links/da25m024-iitm/h7n9m7wg |
| **GitHub Repo link** | https://github.com/therisbh/DA6401_Assignment_3 |

---

## Project Overview

This project implements a complete Transformer-based sequence-to-sequence model for German-to-English translation using:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Sinusoidal Positional Encoding
- Encoder-Decoder Transformer Architecture
- Label Smoothing
- Noam Learning Rate Scheduler
- Greedy Decoding
- BLEU Score Evaluation

The implementation was built entirely using basic PyTorch modules without using `torch.nn.MultiheadAttention`.

---

## Dataset

Dataset used:
- **Multi30k Dataset**
- Source Language: German
- Target Language: English

Dataset link:
https://huggingface.co/datasets/bentrevett/multi30k

---

## Project Structure

```bash
.
├── dataset.py          # Dataset loading, tokenization, vocab building
├── lr_scheduler.py     # Noam learning rate scheduler
├── model.py            # Transformer architecture implementation
├── train.py            # Training pipeline and evaluation
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Features Implemented

### Transformer Components
- Scaled Dot-Product Attention
- Multi-Head Attention
- Padding Mask
- Causal Mask
- Positional Encoding
- Feed Forward Network
- Residual Connections
- Layer Normalization

### Training Features
- Label Smoothing (\(\epsilon = 0.1\))
- Noam Scheduler
- Gradient Clipping
- Greedy Decoding
- BLEU Score Evaluation
- Weights & Biases Logging

---

## Model Configuration

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Encoder Layers | 3 |
| Decoder Layers | 3 |
| Attention Heads | 8 |
| Feed Forward Dimension | 512 |
| Dropout | 0.1 |
| Batch Size | 128 |
| Warmup Steps | 4000 |
| Label Smoothing | 0.1 |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install spaCy models:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

Run training using:

```bash
python train.py
```

The training pipeline:
- Loads the Multi30k dataset
- Builds vocabularies
- Trains the Transformer model
- Saves checkpoints
- Evaluates BLEU score
- Logs metrics to Weights & Biases

---

## Inference

Example usage:

```python
from model import Transformer

model = Transformer(checkpoint_path="best_checkpoint.pt")

sentence = "ein hund spielt im park"
translation = model.infer(sentence)

print(translation)
```

