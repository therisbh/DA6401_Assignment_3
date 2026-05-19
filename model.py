import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    tgt_len = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
    mask = pad_mask | causal_mask
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        x, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(x)
        if return_attn:
            return output, attn_w
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        checkpoint_path: str = "best_checkpoint.pt",
        pad_idx: int = 1,
        gdrive_id: str = "1eWKrZAalQmfn3ib2q78JxGEOhwe2LGWV",
    ) -> None:
        super().__init__()

        import spacy
        import pickle
        from datasets import load_dataset

        self.pad_idx = pad_idx
        self.d_model = d_model

        import subprocess
        import sys
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
            self._spacy_de = spacy.load("de_core_news_sm")

        try:
            self._spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self._spacy_en = spacy.load("en_core_web_sm")

        self.src_vocab = None
        self.tgt_vocab = None

        if checkpoint_path is not None:
            ckpt_path = checkpoint_path
            if gdrive_id is not None and not os.path.exists(ckpt_path):
                import gdown
                gdown.download(id=gdrive_id, output=ckpt_path, quiet=False)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "src_vocab" in ckpt and "tgt_vocab" in ckpt:
                    self.src_vocab = ckpt["src_vocab"]
                    self.tgt_vocab = ckpt["tgt_vocab"]

        if self.src_vocab is None or self.tgt_vocab is None:
            vocab_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab")
            src_vocab_path = os.path.join(vocab_dir, "src_vocab.pkl")
            tgt_vocab_path = os.path.join(vocab_dir, "tgt_vocab.pkl")
            if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
                with open(src_vocab_path, "rb") as f:
                    self.src_vocab = pickle.load(f)
                with open(tgt_vocab_path, "rb") as f:
                    self.tgt_vocab = pickle.load(f)
            else:
                from dataset import Multi30kDataset
                ds = Multi30kDataset(split="train")
                ds.build_vocab()
                self.src_vocab = ds.src_vocab
                self.tgt_vocab = ds.tgt_vocab


        if src_vocab_size is None:
            src_vocab_size = len(self.src_vocab)
        if tgt_vocab_size is None:
            tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_pe = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if checkpoint_path is not None:
            ckpt_path = checkpoint_path
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.load_state_dict(ckpt["model_state_dict"])

    def _tokenize_de(self, sentence: str):
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(sentence)]

    def _tokenize_en(self, sentence: str):
        return [tok.text.lower() for tok in self._spacy_en.tokenizer(sentence)]

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.src_pe(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_pe(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        self.eval()
        device = next(self.parameters()).device

        sos_idx = self.tgt_vocab["<sos>"]
        eos_idx = self.tgt_vocab["<eos>"]
        unk_idx = self.src_vocab.get("<unk>", 0)

        tokens = self._tokenize_de(src_sentence)
        src_indices = [self.src_vocab.get("<sos>", unk_idx)] + \
                      [self.src_vocab.get(t, unk_idx) for t in tokens] + \
                      [self.src_vocab.get("<eos>", unk_idx)]

        src = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys, self.pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == eos_idx:
                    break

        tgt_itos = {v: k for k, v in self.tgt_vocab.items()}
        tokens_out = []
        for idx in ys[0, 1:].tolist():
            if idx == eos_idx:
                break
            word = tgt_itos.get(idx, "<unk>")
            tokens_out.append(word)

        import re
        sentence = " ".join(tokens_out)
        sentence = re.sub(r" ([.,!?;:'\"])", r"\1", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()

        return sentence
