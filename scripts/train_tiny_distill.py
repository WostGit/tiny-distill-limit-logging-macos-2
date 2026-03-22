from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_utils import StepTimer, environment_snapshot, memory_snapshot, utc_now_iso
from metrics_utils import dir_stats, write_json

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_vocab(samples: List[Dict[str, str]]) -> Dict[str, int]:
    tokens = {PAD, BOS, EOS, UNK}
    for row in samples:
        tokens.update(row["input"].lower().split())
        tokens.update(row["target"].lower().split())
    idx = {tok: i for i, tok in enumerate(sorted(tokens))}
    return idx


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab[UNK]) for tok in text.lower().split()]


def pct(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = int((len(sorted_vals) - 1) * q)
    return float(sorted_vals[k])


@dataclass
class EncodedSample:
    src: List[int]
    tgt: List[int]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 2, alpha: float = 4.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + self.scaling * delta

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_a": self.lora_a.detach().cpu(), "lora_b": self.lora_b.detach().cpu()}

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.lora_a.data.copy_(state["lora_a"])
        self.lora_b.data.copy_(state["lora_b"])


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.gru(h)
        h = torch.relu(self.proj(h))
        return self.head(h)


def apply_lora(model: TinyLM, rank: int = 2) -> TinyLM:
    for p in model.parameters():
        p.requires_grad = False
    model.proj = LoRALinear(model.proj, rank=rank)
    model.head = LoRALinear(model.head, rank=rank)
    return model


def collate_batch(batch: List[EncodedSample], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x.src) + len(x.tgt) + 2 for x in batch)
    xs, ys = [], []
    for s in batch:
        seq = [s.src[0]] + s.src[1:] + s.tgt + [pad_id]
        tgt = seq[1:] + [pad_id]
        seq = seq[:max_len]
        tgt = tgt[:max_len]
        seq += [pad_id] * (max_len - len(seq))
        tgt += [pad_id] * (max_len - len(tgt))
        xs.append(seq)
        ys.append(tgt)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def run_eval(student: TinyLM, dataset: List[EncodedSample], pad_id: int) -> Dict[str, float]:
    student.eval()
    total_loss = 0.0
    total_tokens = 0
    gen_start = time.perf_counter()
    with torch.no_grad():
        for s in dataset:
            x, y = collate_batch([s], pad_id)
            logits = student(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id)
            total_loss += loss.item()
            total_tokens += int((y != pad_id).sum().item())
    gen_duration = time.perf_counter() - gen_start
    return {
        "eval_loss": total_loss / max(1, len(dataset)),
        "eval_tokens": total_tokens,
        "generation_duration_s": gen_duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/tiny_train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/tiny_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--runner-label", default="macos-15")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()

    run_start = time.perf_counter()
    set_seed(args.seed)
    memory_points = [memory_snapshot("process_start")]
    env = environment_snapshot(args.runner_label)

    data_load_start = time.perf_counter()
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval)
    data_load_s = time.perf_counter() - data_load_start
    print(f"[data] data_load_time_s={data_load_s:.4f} train_rows={len(train_rows)} eval_rows={len(eval_rows)}", flush=True)

    preprocess_start = time.perf_counter()
    vocab = build_vocab(train_rows + eval_rows)
    bos_id, eos_id, pad_id = vocab[BOS], vocab[EOS], vocab[PAD]

    def encode_rows(rows: List[Dict[str, str]]) -> List[EncodedSample]:
        out: List[EncodedSample] = []
        for row in rows:
            src = [bos_id] + encode_text(row["input"], vocab) + [eos_id]
            tgt = encode_text(row["target"], vocab) + [eos_id]
            out.append(EncodedSample(src=src, tgt=tgt))
        return out

    train_ds = encode_rows(train_rows)
    eval_ds = encode_rows(eval_rows)
    preprocess_s = time.perf_counter() - preprocess_start
    print(f"[data] preprocess_time_s={preprocess_s:.4f} vocab_size={len(vocab)}", flush=True)

    input_lens = [len(x.src) for x in train_ds]
    target_lens = [len(x.tgt) for x in train_ds]
    seq_stats = {
        "input_min": min(input_lens),
        "input_mean": mean(input_lens),
        "input_max": max(input_lens),
        "input_p95": pct(input_lens, 0.95),
        "target_min": min(target_lens),
        "target_mean": mean(target_lens),
        "target_max": max(target_lens),
        "target_p95": pct(target_lens, 0.95),
    }
    print(f"[seq] {json.dumps(seq_stats, sort_keys=True)}", flush=True)

    teacher = TinyLM(vocab_size=len(vocab), hidden_size=64)
    student = TinyLM(vocab_size=len(vocab), hidden_size=32)
    memory_points.append(memory_snapshot("after_model_load"))

    student = apply_lora(student, rank=2)
    memory_points.append(memory_snapshot("after_lora_wrapping"))

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-3)
    step_timer = StepTimer(run_start=run_start)
    teacher.eval()
    student.train()

    training_shape = {
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_samples_per_optimizer_step": args.batch_size * args.grad_accum,
    }
    print(f"[shape] {json.dumps(training_shape, sort_keys=True)}", flush=True)

    optimizer_steps = 0
    cumulative_samples = 0
    tokens_per_step: List[int] = []
    first_fb_logged = False
    loss_history: List[float] = []

    for i in range(0, len(train_ds), args.batch_size):
        micro_batch = train_ds[i : i + args.batch_size]
        x, y = collate_batch(micro_batch, pad_id)
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = student(x)
        ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), y.view(-1), ignore_index=pad_id)
        distill = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        )
        loss = 0.5 * ce + 0.5 * distill
        loss.backward()
        loss_history.append(float(loss.item()))

        if not first_fb_logged:
            memory_points.append(memory_snapshot("after_first_forward_backward"))
            first_fb_logged = True

        step_boundary = ((i // args.batch_size + 1) % args.grad_accum == 0) or (i + args.batch_size >= len(train_ds))
        if step_boundary:
            step_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_duration = time.perf_counter() - step_start
            optimizer_steps += 1
            cumulative_samples += len(micro_batch) * args.grad_accum
            tokens_this_step = int((y != pad_id).sum().item())
            tokens_per_step.append(tokens_this_step)
            step_timer.log_optimizer_step(optimizer_steps, step_duration, cumulative_samples, tokens_this_step)
            memory_points.append(memory_snapshot(f"optimizer_step_{optimizer_steps}"))

    ckpt_dir = args.output_dir / "checkpoints" / "tiny_lora_adapter"
    adapter_path = ckpt_dir / "adapter.pt"
    vocab_path = ckpt_dir / "vocab.json"
    ckpt_start_iso = utc_now_iso()
    ckpt_start = time.perf_counter()
    memory_points.append(memory_snapshot("before_save"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proj": student.proj.lora_state_dict(),
            "head": student.head.lora_state_dict(),
            "config": {"rank": 2, "alpha": 4.0},
        },
        adapter_path,
    )
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ckpt_end = time.perf_counter()
    ckpt_end_iso = utc_now_iso()
    memory_points.append(memory_snapshot("after_save"))

    reload_start = time.perf_counter()
    reloaded = TinyLM(vocab_size=len(vocab), hidden_size=32)
    reloaded = apply_lora(reloaded, rank=2)
    state = torch.load(adapter_path, map_location="cpu")
    reloaded.proj.load_lora_state_dict(state["proj"])
    reloaded.head.load_lora_state_dict(state["head"])
    adapter_reload_s = time.perf_counter() - reload_start

    memory_points.append(memory_snapshot("before_eval"))
    eval_start = time.perf_counter()
    eval_stats = run_eval(reloaded, eval_ds, pad_id)
    eval_duration_s = time.perf_counter() - eval_start
    memory_points.append(memory_snapshot("after_eval"))

    checkpoint_stats = dir_stats(ckpt_dir)
    checkpoint_info = {
        "save_start_utc": ckpt_start_iso,
        "save_end_utc": ckpt_end_iso,
        "save_duration_s": ckpt_end - ckpt_start,
        **checkpoint_stats,
    }
    print(f"[checkpoint] {json.dumps(checkpoint_info, sort_keys=True)}", flush=True)

    timing_breakdown = {
        "data_load_s": data_load_s,
        "preprocess_s": preprocess_s,
        "checkpoint_save_s": ckpt_end - ckpt_start,
        "adapter_reload_s": adapter_reload_s,
        "eval_duration_s": eval_duration_s,
        "generation_duration_s": eval_stats["generation_duration_s"],
        "total_run_s": time.perf_counter() - run_start,
    }

    dominant = max(timing_breakdown.items(), key=lambda kv: kv[1])
    summary = {
        "first_observed_bottleneck": dominant[0],
        "value_s": dominant[1],
        "notes": "For this tiny run, largest measured duration is used as a simple bottleneck proxy.",
    }
    print(f"[summary] {json.dumps(summary, sort_keys=True)}", flush=True)

    metrics = {
        "environment": env,
        "training_shape": training_shape,
        "sequence_stats": seq_stats,
        "tokens_per_optimizer_step": tokens_per_step,
        "optimizer_steps": optimizer_steps,
        "loss_history": loss_history,
        "timing": timing_breakdown,
        "checkpoint": checkpoint_info,
        "eval": eval_stats,
        "memory_snapshots": memory_points,
        "summary": summary,
    }

    write_json(args.output_dir / "metrics" / "train_metrics.json", metrics)
    print("[done] wrote outputs/metrics/train_metrics.json", flush=True)


if __name__ == "__main__":
    main()
