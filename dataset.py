import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
import pickle
import os


class Multi30kDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_data = None
        self.tgt_data = None

    def tokenize_de(self, sentence):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(sentence)]

    def tokenize_en(self, sentence):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(sentence)]

    def build_vocab(self):
        special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        src_counter = {}
        tgt_counter = {}

        train_data = load_dataset("bentrevett/multi30k", split="train")
        for item in train_data:
            for tok in self.tokenize_de(item["de"]):
                src_counter[tok] = src_counter.get(tok, 0) + 1
            for tok in self.tokenize_en(item["en"]):
                tgt_counter[tok] = tgt_counter.get(tok, 0) + 1

        src_vocab = {tok: i for i, tok in enumerate(special_tokens)}
        for word in sorted(src_counter.keys()):
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)

        tgt_vocab = {tok: i for i, tok in enumerate(special_tokens)}
        for word in sorted(tgt_counter.keys()):
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def process_data(self):
        unk_src = self.src_vocab.get("<unk>", 0)
        unk_tgt = self.tgt_vocab.get("<unk>", 0)
        sos_src = self.src_vocab.get("<sos>", 2)
        eos_src = self.src_vocab.get("<eos>", 3)
        sos_tgt = self.tgt_vocab.get("<sos>", 2)
        eos_tgt = self.tgt_vocab.get("<eos>", 3)

        self.src_data = []
        self.tgt_data = []
        for item in self.dataset:
            src_tokens = [sos_src] + [self.src_vocab.get(t, unk_src) for t in self.tokenize_de(item["de"])] + [eos_src]
            tgt_tokens = [sos_tgt] + [self.tgt_vocab.get(t, unk_tgt) for t in self.tokenize_en(item["en"])] + [eos_tgt]
            self.src_data.append(src_tokens)
            self.tgt_data.append(tgt_tokens)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), \
               torch.tensor(self.tgt_data[idx], dtype=torch.long)


def collate_fn(batch, pad_idx=1):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
