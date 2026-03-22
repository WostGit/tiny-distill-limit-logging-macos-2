from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_utils import environment_snapshot, memory_snapshot
from metrics_utils import write_json

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 2, alpha: float = 4.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * ((x @ self.lora_a.t()) @ self.lora_b.t())

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


def apply_lora(model: TinyLM) -> TinyLM:
    model.proj = LoRALinear(model.proj)
    model.head = LoRALinear(model.head)
    return model


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab[UNK]) for tok in text.lower().split()]


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collate_seq(src: List[int], tgt: List[int], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq = src + tgt + [pad_id]
    y = seq[1:] + [pad_id]
    return torch.tensor([seq], dtype=torch.long), torch.tensor([y], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=Path("data/tiny_eval.jsonl"))
    parser.add_argument("--adapter", type=Path, default=Path("outputs/checkpoints/tiny_lora_adapter/adapter.pt"))
    parser.add_argument("--vocab", type=Path, default=Path("outputs/checkpoints/tiny_lora_adapter/vocab.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/eval_metrics.json"))
    parser.add_argument("--runner-label", default="macos-15")
    args = parser.parse_args()

    start = time.perf_counter()
    env = environment_snapshot(args.runner_label)
    memory_points = [memory_snapshot("eval_process_start")]

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    pad_id, bos_id, eos_id = vocab[PAD], vocab[BOS], vocab[EOS]

    reload_start = time.perf_counter()
    model = apply_lora(TinyLM(vocab_size=len(vocab), hidden_size=32))
    state = torch.load(args.adapter, map_location="cpu")
    model.proj.load_lora_state_dict(state["proj"])
    model.head.load_lora_state_dict(state["head"])
    adapter_reload_s = time.perf_counter() - reload_start
    print(f"[eval] adapter_reload_time_s={adapter_reload_s:.6f}", flush=True)
    memory_points.append(memory_snapshot("eval_after_adapter_reload"))

    rows = load_jsonl(args.eval)
    encoded = [([bos_id] + encode_text(r["input"], vocab) + [eos_id], encode_text(r["target"], vocab) + [eos_id]) for r in rows]
    input_lens = [len(s[0]) for s in encoded]
    target_lens = [len(s[1]) for s in encoded]
    print(
        "[eval-seq] input(min/mean/max)=({}/{:.2f}/{}) target(min/mean/max)=({}/{:.2f}/{})".format(
            min(input_lens), mean(input_lens), max(input_lens), min(target_lens), mean(target_lens), max(target_lens)
        ),
        flush=True,
    )

    eval_start = time.perf_counter()
    gen_start = time.perf_counter()
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for src, tgt in encoded:
            x, y = collate_seq(src, tgt, pad_id)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id)
            total_loss += float(loss.item())
            total_tokens += int((y != pad_id).sum().item())
    generation_duration_s = time.perf_counter() - gen_start
    eval_duration_s = time.perf_counter() - eval_start
    memory_points.append(memory_snapshot("eval_after_eval"))

    metrics = {
        "environment": env,
        "adapter_reload_s": adapter_reload_s,
        "eval_duration_s": eval_duration_s,
        "generation_duration_s": generation_duration_s,
        "eval_loss": total_loss / max(1, len(encoded)),
        "eval_tokens": total_tokens,
        "total_script_s": time.perf_counter() - start,
        "memory_snapshots": memory_points,
    }
    write_json(args.output, metrics)
    print(f"[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
