# DA6401 Assignment 3 - Neural Machine Translation with Transformer

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

```bash
python train.py
```

## Model Architecture

- d_model = 256
- N = 3 encoder/decoder layers
- num_heads = 8
- d_ff = 512
- dropout = 0.1

## Inference

The Transformer.infer() method accepts a German sentence and returns an English translation.

```python
from model import Transformer
model = Transformer()
model.eval()
english = model.infer("Ein Hund spielt im Garten.")
print(english)
```

## Notes on model weights

In Transformer.__init__, set gdrive_id to your Google Drive file ID and checkpoint_path to the output filename. The weights are downloaded automatically using gdown.

Example:
```python
model = Transformer(
    checkpoint_path="best_checkpoint.pt",
    gdrive_id="YOUR_DRIVE_FILE_ID_HERE"
)
```
