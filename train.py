import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth_dist[:, self.pad_idx] = 0.0
            non_pad_mask = (target != self.pad_idx)
            smooth_dist[~non_pad_mask] = 0.0
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad_count = non_pad_mask.sum().float()
        return loss.sum() / non_pad_count.clamp(min=1)


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    pad_idx = model.pad_idx

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for src, tgt in tqdm(data_iter, desc=f"Epoch {epoch_num} {'train' if is_train else 'val'}"):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            batch_size, tgt_len, vocab_size = logits.size()
            logits_flat = logits.contiguous().view(-1, vocab_size)
            tgt_flat = tgt_output.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad = (tgt_flat != pad_idx).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long).to(device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, model.pad_idx)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    import sacrebleu

    model.eval()
    pad_idx = model.pad_idx

    if isinstance(tgt_vocab, dict):
        itos = {v: k for k, v in tgt_vocab.items()}
        sos_idx = tgt_vocab.get("<sos>", 2)
        eos_idx = tgt_vocab.get("<eos>", 3)
    else:
        itos = {}
        sos_idx = 2
        eos_idx = 3

    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, pad_idx)
                out = greedy_decode(model, src_i, src_mask, max_len, sos_idx, eos_idx, device)

                pred_tokens = []
                for idx in out[0, 1:].tolist():
                    if idx == eos_idx:
                        break
                    pred_tokens.append(itos.get(idx, "<unk>"))

                ref_tokens = []
                for idx in tgt[i, 1:].tolist():
                    if idx == eos_idx:
                        break
                    if idx == pad_idx:
                        break
                    ref_tokens.append(itos.get(idx, "<unk>"))

                predictions.append(" ".join(pred_tokens))
                references.append(" ".join(ref_tokens))

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "src_vocab": model.src_vocab,
        "tgt_vocab": model.tgt_vocab,
        "model_config": {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": len(model.encoder.layers),
            "num_heads": model.encoder.layers[0].self_attn.num_heads,
            "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
            "dropout": 0.1,
        }
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def run_training_experiment() -> None:
    config = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 30,
        "warmup_steps": 4000,
        "label_smoothing": 0.1,
    }

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_idx = 1

    vocab_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab")
    src_vocab_path = os.path.join(vocab_dir, "src_vocab.pkl")
    tgt_vocab_path = os.path.join(vocab_dir, "tgt_vocab.pkl")

    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        with open(src_vocab_path, "rb") as f:
            src_vocab = pickle.load(f)
        with open(tgt_vocab_path, "rb") as f:
            tgt_vocab = pickle.load(f)
        train_ds = Multi30kDataset(split="train")
        train_ds.src_vocab = src_vocab
        train_ds.tgt_vocab = tgt_vocab
        train_ds.process_data()
        val_ds = Multi30kDataset(split="validation")
        val_ds.src_vocab = src_vocab
        val_ds.tgt_vocab = tgt_vocab
        val_ds.process_data()
        test_ds = Multi30kDataset(split="test")
        test_ds.src_vocab = src_vocab
        test_ds.tgt_vocab = tgt_vocab
        test_ds.process_data()
    else:
        train_ds = Multi30kDataset(split="train")
        train_ds.build_vocab()
        src_vocab = train_ds.src_vocab
        tgt_vocab = train_ds.tgt_vocab
        os.makedirs(vocab_dir, exist_ok=True)
        with open(src_vocab_path, "wb") as f:
            pickle.dump(src_vocab, f)
        with open(tgt_vocab_path, "wb") as f:
            pickle.dump(tgt_vocab, f)
        train_ds.process_data()
        val_ds = Multi30kDataset(split="validation")
        val_ds.src_vocab = src_vocab
        val_ds.tgt_vocab = tgt_vocab
        val_ds.process_data()
        test_ds = Multi30kDataset(split="test")
        test_ds.src_vocab = src_vocab
        test_ds.tgt_vocab = tgt_vocab
        test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx, cfg.label_smoothing)

    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pt")

    save_checkpoint(model, optimizer, scheduler, cfg.num_epochs - 1, "last_checkpoint.pt")

    best_epoch = load_checkpoint("best_checkpoint.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
    wandb.log({"test_bleu": bleu})
    print(f"Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
