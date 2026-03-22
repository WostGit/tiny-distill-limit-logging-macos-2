import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from logging_utils import VerboseLogger
from metrics_utils import length_stats, write_json

PAD = "<pad>"
UNK = "<unk>"


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + ((x @ self.lora_a @ self.lora_b) * self.scaling)


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 32, rank: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.out = LoRALinear(nn.Linear(hidden, vocab_size), rank=rank)

    def forward(self, input_ids: torch.Tensor, target_len: int) -> torch.Tensor:
        hidden = torch.tanh(self.proj(self.emb(input_ids).mean(dim=1)))
        expanded = hidden.unsqueeze(1).expand(-1, target_len, -1)
        return self.out(expanded)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="data/tiny_eval.jsonl")
    p.add_argument("--adapter", default="outputs/checkpoint/adapter.pt")
    p.add_argument("--metrics-path", default="outputs/metrics/eval_metrics.json")
    p.add_argument("--runner-label", default="macos-15")
    args = p.parse_args()

    logger = VerboseLogger(runner_label=args.runner_label)
    logger.env_report()
    logger.memory_snapshot("process_start")

    rows = load_jsonl(args.eval)
    t_reload = time.perf_counter()
    blob = torch.load(args.adapter, map_location="cpu")
    vocab = blob["vocab"]
    model = TinyStudent(vocab_size=len(vocab))
    model.out.lora_a.data.copy_(blob["lora_a"])
    model.out.lora_b.data.copy_(blob["lora_b"])
    model.eval()
    reload_s = time.perf_counter() - t_reload
    logger.log("adapter_reloaded", duration_s=round(reload_s, 6), adapter=args.adapter)
    logger.memory_snapshot("after_model_load")

    eval_start = time.perf_counter()
    generation_total = 0.0
    correct = 0
    total = 0
    input_lens = []
    target_lens = []

    for idx, row in enumerate(rows):
        in_tokens = [vocab.get(tok, vocab[UNK]) for tok in row["input"].lower().split()]
        tgt_tokens = [vocab.get(tok, vocab[UNK]) for tok in row["target"].lower().split()]
        input_lens.append(len(in_tokens))
        target_lens.append(len(tgt_tokens))

        t_gen = time.perf_counter()
        with torch.no_grad():
            logits = model(torch.tensor([in_tokens], dtype=torch.long), len(tgt_tokens))
            pred = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        generation_total += time.perf_counter() - t_gen

        for p_tok, y_tok in zip(pred, tgt_tokens):
            correct += int(p_tok == y_tok)
            total += 1
        logger.log(
            "eval_example",
            index=idx,
            input_tokens=len(in_tokens),
            target_tokens=len(tgt_tokens),
            generation_s=round(generation_total, 6),
            cumulative_accuracy=round(correct / max(total, 1), 6),
        )

    eval_s = time.perf_counter() - eval_start
    acc = correct / max(total, 1)
    logger.log(
        "eval_summary",
        eval_duration_s=round(eval_s, 6),
        generation_duration_s=round(generation_total, 6),
        token_accuracy=round(acc, 6),
        input_length_stats=length_stats(input_lens),
        target_length_stats=length_stats(target_lens),
    )
    logger.memory_snapshot("after_eval")

    metrics = {
        "adapter_reload_s": round(reload_s, 6),
        "eval_duration_s": round(eval_s, 6),
        "generation_duration_s": round(generation_total, 6),
        "token_accuracy": round(acc, 6),
        "input_length_stats": length_stats(input_lens),
        "target_length_stats": length_stats(target_lens),
        "num_examples": len(rows),
        "runner_label": args.runner_label,
    }
    write_json(args.metrics_path, metrics)
    logger.log("metrics_written", path=args.metrics_path)


if __name__ == "__main__":
    main()
