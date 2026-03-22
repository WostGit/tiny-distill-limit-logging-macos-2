import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_utils import collect_env_info, configure_logging, log_env_info, memory_snapshot, utc_now_iso, write_json
from metrics_utils import compute_stats, summarize_bottlenecks


@dataclass
class Example:
    prompt: str
    target: str


class CharTokenizer:
    def __init__(self, texts: List[str]):
        vocab = sorted(set("".join(texts)))
        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.itos = [self.pad, self.bos, self.eos] + vocab
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi[self.bos]] + [self.stoi.get(ch, self.stoi[self.pad]) for ch in text[: max_len - 2]] + [self.stoi[self.eos]]
        if len(ids) < max_len:
            ids += [self.stoi[self.pad]] * (max_len - len(ids))
        return ids


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        out, _ = self.rnn(x)
        return self.proj(out)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(base.in_features, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, base.out_features))

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = x @ self.lora_a @ self.lora_b
        return base_out + self.scale * lora_out


def load_jsonl(path: str) -> List[Example]:
    records: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(Example(prompt=row["prompt"], target=row["target"]))
    return records


def build_batches(
    records: List[Example],
    tokenizer: CharTokenizer,
    max_prompt_len: int,
    max_target_len: int,
    batch_size: int,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict]:
    t0 = time.perf_counter()
    encoded = []
    input_lengths, target_lengths = [], []

    for rec in records:
        prompt_ids = tokenizer.encode(rec.prompt, max_prompt_len)
        target_ids = tokenizer.encode(rec.target, max_target_len)
        input_lengths.append(sum(1 for x in prompt_ids if x != tokenizer.stoi[tokenizer.pad]))
        target_lengths.append(sum(1 for x in target_ids if x != tokenizer.stoi[tokenizer.pad]))
        encoded.append((prompt_ids, target_ids))

    tokenization_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    batches = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        prompt_batch = torch.tensor([x[0] for x in chunk], dtype=torch.long)
        target_batch = torch.tensor([x[1] for x in chunk], dtype=torch.long)
        batches.append({"input_ids": prompt_batch, "labels": target_batch})
    data_load_sec = time.perf_counter() - t1

    seq_stats = {
        "input": compute_stats(input_lengths),
        "target": compute_stats(target_lengths),
    }
    return batches, {
        "tokenization_sec": round(tokenization_sec, 6),
        "data_load_sec": round(data_load_sec, 6),
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "sequence_stats": seq_stats,
    }


def checkpoint_size_info(path: str) -> Tuple[int, int]:
    total = 0
    files = 0
    for root, _, names in os.walk(path):
        for name in names:
            files += 1
            total += os.path.getsize(os.path.join(root, name))
    return total, files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="data/tiny_train.jsonl")
    p.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--metrics-out", default="outputs/metrics/train_metrics.json")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    run_start = time.perf_counter()
    torch.manual_seed(args.seed)

    env_info = collect_env_info()
    log_env_info(logger, env_info)

    metrics = {
        "run_started_utc": utc_now_iso(),
        "memory_snapshots": [],
        "step_logs": [],
        "step_durations_sec": [],
        "env": env_info,
    }
    metrics["memory_snapshots"].append(memory_snapshot(logger, "process_start", run_start))

    train_records = load_jsonl(args.train_data)
    eval_records = load_jsonl(args.eval_data)
    tokenizer = CharTokenizer([x.prompt + x.target for x in train_records + eval_records])

    train_batches, data_metrics = build_batches(train_records, tokenizer, 96, 64, args.batch_size)
    metrics["data_pipeline"] = {
        "tokenization_sec": data_metrics["tokenization_sec"],
        "data_load_sec": data_metrics["data_load_sec"],
    }
    metrics["sequence_stats"] = data_metrics["sequence_stats"]

    logger.info(
        "SEQUENCE STATS | input min/mean/max/p95=%.1f/%.1f/%.1f/%.1f | target min/mean/max/p95=%.1f/%.1f/%.1f/%.1f",
        metrics["sequence_stats"]["input"]["min"],
        metrics["sequence_stats"]["input"]["mean"],
        metrics["sequence_stats"]["input"]["max"],
        metrics["sequence_stats"]["input"]["p95"],
        metrics["sequence_stats"]["target"]["min"],
        metrics["sequence_stats"]["target"]["mean"],
        metrics["sequence_stats"]["target"]["max"],
        metrics["sequence_stats"]["target"]["p95"],
    )

    teacher = TinyLM(tokenizer.vocab_size, emb_dim=32, hidden_dim=64)
    student = TinyLM(tokenizer.vocab_size, emb_dim=24, hidden_dim=32)
    metrics["memory_snapshots"].append(memory_snapshot(logger, "after_model_load", run_start))

    student.proj = LoRALinear(student.proj, rank=4, alpha=8.0)
    for n, p in student.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    metrics["memory_snapshots"].append(memory_snapshot(logger, "after_lora_wrapping", run_start))

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    effective_batch = args.batch_size * args.grad_accum
    metrics["training_shape"] = {
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_samples_per_optimizer_step": effective_batch,
    }
    logger.info(
        "TRAINING SHAPE | per_device_batch_size=%d | gradient_accumulation_steps=%d | effective_samples_per_optimizer_step=%d",
        args.batch_size,
        args.grad_accum,
        effective_batch,
    )

    seen_samples = 0
    first_backward_done = False

    teacher.eval()
    student.train()
    for epoch in range(args.epochs):
        logger.info("EPOCH START | epoch=%d/%d", epoch + 1, args.epochs)
        optimizer.zero_grad()
        accum_count = 0
        opt_step_idx = 0
        for batch in train_batches:
            step_start = time.perf_counter()
            with torch.no_grad():
                teacher_logits = teacher(batch["input_ids"])

            student_logits = student(batch["input_ids"])

            min_t = min(student_logits.size(1), batch["labels"].size(1))
            student_slice = student_logits[:, :min_t, :]
            target_slice = batch["labels"][:, :min_t]
            teacher_slice = teacher_logits[:, :min_t, :]

            ce_loss = F.cross_entropy(student_slice.reshape(-1, tokenizer.vocab_size), target_slice.reshape(-1))
            kl_loss = F.kl_div(
                F.log_softmax(student_slice, dim=-1),
                F.softmax(teacher_slice, dim=-1),
                reduction="batchmean",
            )
            loss = 0.5 * ce_loss + 0.5 * kl_loss
            (loss / args.grad_accum).backward()

            if not first_backward_done:
                metrics["memory_snapshots"].append(memory_snapshot(logger, "after_first_forward_backward", run_start))
                first_backward_done = True

            accum_count += 1
            seen_samples += batch["input_ids"].shape[0]

            if accum_count == args.grad_accum or batch is train_batches[-1]:
                optimizer.step()
                optimizer.zero_grad()
                opt_step_idx += 1

                step_duration = time.perf_counter() - step_start
                step_log = {
                    "optimizer_step": opt_step_idx,
                    "elapsed_sec": round(time.perf_counter() - run_start, 4),
                    "step_duration_sec": round(step_duration, 4),
                    "cumulative_samples": seen_samples,
                    "tokens_this_step": int(batch["input_ids"].numel() + batch["labels"].numel()),
                    "loss": round(float(loss.item()), 6),
                }
                metrics["step_logs"].append(step_log)
                metrics["step_durations_sec"].append(step_duration)
                logger.info(
                    "OPT STEP | idx=%d | elapsed=%.4fs | step_duration=%.4fs | cumulative_samples=%d | tokens_this_step=%d | loss=%.6f",
                    step_log["optimizer_step"],
                    step_log["elapsed_sec"],
                    step_log["step_duration_sec"],
                    step_log["cumulative_samples"],
                    step_log["tokens_this_step"],
                    step_log["loss"],
                )
                metrics["memory_snapshots"].append(memory_snapshot(logger, f"optimizer_step_{opt_step_idx}", run_start))
                accum_count = 0

    ckpt_dir = Path(args.out_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics["memory_snapshots"].append(memory_snapshot(logger, "before_save", run_start))

    save_start = time.perf_counter()
    save_start_utc = utc_now_iso()
    lora_state = {k: v.detach().cpu() for k, v in student.state_dict().items() if "lora_" in k}
    torch.save(lora_state, ckpt_dir / "adapter.pt")
    save_end_utc = utc_now_iso()
    save_duration = time.perf_counter() - save_start

    ckpt_bytes, ckpt_files = checkpoint_size_info(str(ckpt_dir))
    metrics["checkpoint"] = {
        "save_start_utc": save_start_utc,
        "save_end_utc": save_end_utc,
        "save_duration_sec": round(save_duration, 6),
        "path": str(ckpt_dir),
        "bytes": ckpt_bytes,
        "file_count": ckpt_files,
    }
    logger.info(
        "CHECKPOINT | start=%s | end=%s | duration=%.4fs | bytes=%d | file_count=%d",
        save_start_utc,
        save_end_utc,
        save_duration,
        ckpt_bytes,
        ckpt_files,
    )
    metrics["memory_snapshots"].append(memory_snapshot(logger, "after_save", run_start))

    metrics["run_finished_utc"] = utc_now_iso()
    metrics["total_runtime_sec"] = round(time.perf_counter() - run_start, 6)
    write_json(args.metrics_out, metrics)

    eval_metrics_path = Path(args.out_dir) / "metrics" / "eval_metrics.json"
    if eval_metrics_path.exists():
        with open(eval_metrics_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)
        summary = summarize_bottlenecks(metrics, eval_metrics)
        logger.info("FINAL BOTTLENECK SUMMARY START")
        for line in summary:
            logger.info("BOTTLENECK | %s", line)
        logger.info("FINAL BOTTLENECK SUMMARY END")


if __name__ == "__main__":
    main()
