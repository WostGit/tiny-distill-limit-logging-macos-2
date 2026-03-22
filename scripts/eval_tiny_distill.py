import argparse
import json
import time
from typing import List

import torch
import torch.nn as nn

from logging_utils import collect_env_info, configure_logging, log_env_info, memory_snapshot, utc_now_iso, write_json


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
        return self.base(x) + self.scale * (x @ self.lora_a @ self.lora_b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="data/tiny_train.jsonl")
    p.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    p.add_argument("--adapter-path", default="outputs/checkpoint/adapter.pt")
    p.add_argument("--metrics-out", default="outputs/metrics/eval_metrics.json")
    return p.parse_args()


def read_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    run_start = time.perf_counter()

    env_info = collect_env_info()
    log_env_info(logger, env_info)

    metrics = {
        "run_started_utc": utc_now_iso(),
        "memory_snapshots": [],
        "env": env_info,
    }
    metrics["memory_snapshots"].append(memory_snapshot(logger, "before_eval", run_start))

    train_records = read_jsonl(args.train_data)
    eval_records = read_jsonl(args.eval_data)
    tokenizer = CharTokenizer([r["prompt"] + r["target"] for r in train_records + eval_records])

    model = TinyLM(tokenizer.vocab_size, emb_dim=24, hidden_dim=32)
    model.proj = LoRALinear(model.proj, rank=4, alpha=8.0)

    reload_start = time.perf_counter()
    adapter_state = torch.load(args.adapter_path, map_location="cpu")
    model.load_state_dict(adapter_state, strict=False)
    reload_sec = time.perf_counter() - reload_start
    metrics["adapter_reload_sec"] = round(reload_sec, 6)
    logger.info("EVAL | adapter_reload_sec=%.6f", reload_sec)

    model.eval()
    eval_start = time.perf_counter()

    correct = 0
    generation_total_sec = 0.0
    with torch.no_grad():
        for rec in eval_records:
            prompt = torch.tensor([tokenizer.encode(rec["prompt"], 96)], dtype=torch.long)
            gen_start = time.perf_counter()
            logits = model(prompt)
            generation_total_sec += time.perf_counter() - gen_start

            pred_ids = logits[0, :32].argmax(dim=-1).tolist()
            pred_text = "".join(tokenizer.itos[i] for i in pred_ids if i > 2).strip()
            tgt = rec["target"].strip().lower()
            if tgt and tgt in pred_text.lower():
                correct += 1

    eval_duration = time.perf_counter() - eval_start
    accuracy = correct / max(1, len(eval_records))

    metrics["eval_duration_sec"] = round(eval_duration, 6)
    metrics["generation_duration_sec"] = round(generation_total_sec, 6)
    metrics["eval_accuracy_contains"] = round(accuracy, 6)
    metrics["num_eval_examples"] = len(eval_records)

    logger.info(
        "EVAL SUMMARY | eval_duration_sec=%.6f | generation_duration_sec=%.6f | accuracy_contains=%.4f | num_eval_examples=%d",
        eval_duration,
        generation_total_sec,
        accuracy,
        len(eval_records),
    )

    metrics["memory_snapshots"].append(memory_snapshot(logger, "after_eval", run_start))
    metrics["run_finished_utc"] = utc_now_iso()
    metrics["total_runtime_sec"] = round(time.perf_counter() - run_start, 6)
    write_json(args.metrics_out, metrics)


if __name__ == "__main__":
    main()
