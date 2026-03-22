import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_utils import VerboseLogger
from metrics_utils import directory_size_and_count, length_stats, write_json


PAD = "<pad>"
UNK = "<unk>"


@dataclass
class Example:
    input_ids: List[int]
    target_ids: List[int]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(base.in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scaling
        return base_out + lora_out


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, rank: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.out = LoRALinear(nn.Linear(hidden, vocab_size), rank=rank)
        for p in self.emb.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, target_len: int) -> torch.Tensor:
        emb = self.emb(input_ids)
        pooled = emb.mean(dim=1)
        hidden = torch.tanh(self.proj(pooled))
        expanded = hidden.unsqueeze(1).expand(-1, target_len, -1)
        return self.out(expanded)


class TinyTeacher(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor, target_len: int) -> torch.Tensor:
        emb = self.emb(input_ids)
        pooled = emb.mean(dim=1)
        hidden = torch.relu(self.proj(pooled))
        expanded = hidden.unsqueeze(1).expand(-1, target_len, -1)
        return self.out(expanded)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_vocab(rows: List[Dict[str, str]]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for r in rows:
        for key in ("input", "target"):
            for tok in r[key].lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def encode_rows(rows: List[Dict[str, str]], vocab: Dict[str, int]) -> List[Example]:
    out = []
    for r in rows:
        inp = [vocab.get(tok, vocab[UNK]) for tok in r["input"].lower().split()]
        tgt = [vocab.get(tok, vocab[UNK]) for tok in r["target"].lower().split()]
        out.append(Example(input_ids=inp, target_ids=tgt))
    return out


def collate(batch: List[Example], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_in = max(len(x.input_ids) for x in batch)
    max_tgt = max(len(x.target_ids) for x in batch)
    inp = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, ex in enumerate(batch):
        inp[i, : len(ex.input_ids)] = torch.tensor(ex.input_ids)
        tgt[i, : len(ex.target_ids)] = torch.tensor(ex.target_ids)
    return inp, tgt


def make_batches(data: List[Example], batch_size: int) -> List[List[Example]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/tiny_train.jsonl")
    p.add_argument("--eval", default="data/tiny_eval.jsonl")
    p.add_argument("--out-dir", default="outputs/checkpoint")
    p.add_argument("--metrics-path", default="outputs/metrics/train_metrics.json")
    p.add_argument("--runner-label", default="macos-15")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logger = VerboseLogger(runner_label=args.runner_label)
    logger.env_report()
    logger.memory_snapshot("process_start")
    set_seed(args.seed)

    t_data0 = time.perf_counter()
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval)
    vocab = build_vocab(train_rows + eval_rows)
    train_data = encode_rows(train_rows, vocab)
    data_load_time = time.perf_counter() - t_data0
    logger.log("data_load", seconds=round(data_load_time, 4), train_rows=len(train_rows), eval_rows=len(eval_rows), vocab_size=len(vocab))

    input_lens = [len(ex.input_ids) for ex in train_data]
    target_lens = [len(ex.target_ids) for ex in train_data]
    in_stats = length_stats(input_lens)
    tgt_stats = length_stats(target_lens)
    logger.log("sequence_stats", input=in_stats, target=tgt_stats)

    teacher = TinyTeacher(vocab_size=len(vocab), hidden=args.hidden)
    student = TinyStudent(vocab_size=len(vocab), hidden=args.hidden, rank=args.rank)
    teacher.eval()
    logger.memory_snapshot("after_model_load")
    logger.memory_snapshot("after_lora_wrapping")

    trainable = [n for n, p_ in student.named_parameters() if p_.requires_grad]
    logger.log("trainable_parameters", names=trainable, count=sum(p_.numel() for p_ in student.parameters() if p_.requires_grad))

    optim = torch.optim.AdamW((p_ for p_ in student.parameters() if p_.requires_grad), lr=1e-2)
    batches = make_batches(train_data, args.batch_size)

    per_device_batch_size = args.batch_size
    effective_samples = args.batch_size * args.grad_accum
    logger.log(
        "training_shape",
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        effective_samples_per_optimizer_step=effective_samples,
    )

    step_records = []
    preprocess_s = 0.0
    first_fb_logged = False
    samples_seen = 0
    token_seen = 0
    optimizer_step = 0

    for epoch in range(args.epochs):
        logger.log("epoch_start", epoch=epoch)
        running = 0.0
        optim.zero_grad()
        for bi, batch in enumerate(batches):
            t_pre = time.perf_counter()
            inp, tgt = collate(batch, vocab[PAD])
            preprocess_s += time.perf_counter() - t_pre

            t_step = time.perf_counter()
            with torch.no_grad():
                t_logits = teacher(inp, tgt.shape[1])
            s_logits = student(inp, tgt.shape[1])

            ce_loss = F.cross_entropy(
                s_logits.reshape(-1, s_logits.shape[-1]),
                tgt.reshape(-1),
                ignore_index=vocab[PAD],
            )
            kd_loss = F.mse_loss(s_logits, t_logits)
            loss = 0.7 * ce_loss + 0.3 * kd_loss
            (loss / args.grad_accum).backward()

            if not first_fb_logged:
                logger.memory_snapshot("after_first_forward_backward")
                first_fb_logged = True

            samples_seen += inp.shape[0]
            token_seen += int((tgt != vocab[PAD]).sum().item())
            running += loss.item()

            if (bi + 1) % args.grad_accum == 0 or (bi + 1) == len(batches):
                optim.step()
                optim.zero_grad()
                optimizer_step += 1
                dt = time.perf_counter() - t_step
                rec = {
                    "optimizer_step": optimizer_step,
                    "step_duration_s": round(dt, 4),
                    "elapsed_s": round(logger.elapsed(), 4),
                    "samples_seen": samples_seen,
                    "tokens_seen": token_seen,
                    "tokens_per_optimizer_step": int(token_seen / optimizer_step),
                    "loss": round(running / max(1, args.grad_accum), 6),
                }
                step_records.append(rec)
                logger.log("optimizer_step", **rec)
                logger.memory_snapshot("optimizer_step", extra={"optimizer_step": optimizer_step})
                running = 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.memory_snapshot("before_save")
    save_start = time.perf_counter()
    save_start_iso = logger.now_iso()
    torch.save({"lora_a": student.out.lora_a.detach(), "lora_b": student.out.lora_b.detach(), "vocab": vocab}, out_dir / "adapter.pt")
    save_end = time.perf_counter()
    ckpt_stats = directory_size_and_count(str(out_dir))
    logger.log(
        "checkpoint_saved",
        save_start_utc=save_start_iso,
        save_end_utc=logger.now_iso(),
        save_duration_s=round(save_end - save_start, 4),
        checkpoint_bytes=ckpt_stats["bytes"],
        checkpoint_file_count=ckpt_stats["file_count"],
    )
    logger.memory_snapshot("after_save")

    eval_start = time.perf_counter()
    logger.memory_snapshot("before_eval")
    # Lightweight eval signal to include in train metrics.
    with torch.no_grad():
        ex = train_data[0]
        inp = torch.tensor([ex.input_ids], dtype=torch.long)
        logits = student(inp, len(ex.target_ids))
        _ = torch.argmax(logits, dim=-1)
    eval_duration = time.perf_counter() - eval_start
    logger.memory_snapshot("after_eval")

    train_time = logger.elapsed()
    save_time = save_end - save_start
    preprocess_ratio = preprocess_s / max(train_time, 1e-9)
    save_ratio = save_time / max(train_time, 1e-9)
    eval_ratio = eval_duration / max(train_time, 1e-9)

    bottleneck = "wall_clock"
    if save_ratio > 0.25:
        bottleneck = "checkpoint_io"
    elif preprocess_ratio > 0.25:
        bottleneck = "preprocessing_overhead"
    elif eval_ratio > 0.25:
        bottleneck = "evaluation_overhead"

    logger.log(
        "bottleneck_summary",
        first_likely_limit=bottleneck,
        total_train_s=round(train_time, 4),
        preprocess_s=round(preprocess_s, 4),
        eval_probe_s=round(eval_duration, 4),
        save_s=round(save_time, 4),
    )

    metrics = {
        "env": logger.env_report(),
        "data": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "vocab_size": len(vocab),
            "input_length_stats": in_stats,
            "target_length_stats": tgt_stats,
            "data_load_s": round(data_load_time, 6),
            "preprocess_s": round(preprocess_s, 6),
        },
        "training": {
            "epochs": args.epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_samples_per_optimizer_step": effective_samples,
            "optimizer_steps": optimizer_step,
            "step_records": step_records,
            "tokens_total": token_seen,
            "samples_total": samples_seen,
        },
        "checkpoint": {
            "path": str(out_dir),
            "save_duration_s": round(save_time, 6),
            "directory_bytes": ckpt_stats["bytes"],
            "file_count": ckpt_stats["file_count"],
        },
        "eval_probe": {"duration_s": round(eval_duration, 6)},
        "summary": {
            "first_likely_limit": bottleneck,
            "total_train_s": round(train_time, 6),
            "runner_label": args.runner_label,
        },
    }
    write_json(args.metrics_path, metrics)
    logger.log("metrics_written", path=args.metrics_path)


if __name__ == "__main__":
    main()
