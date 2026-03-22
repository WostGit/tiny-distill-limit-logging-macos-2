from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import env_snapshot, snapshots_to_dicts, summarize_lengths, write_json, RunLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tiny LoRA distilled adapter with verbose logs")
    p.add_argument("--eval-file", default="data/tiny_eval.jsonl")
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--adapter-dir", default="outputs/checkpoints/tiny_adapter")
    p.add_argument("--metrics-path", default="outputs/metrics/eval_metrics.json")
    p.add_argument("--runner-label", default="macos-15")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=96)
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def tokenize_rows(rows: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int):
    tokenized, lengths = [], []
    for row in rows:
        prompt = f"Instruction: {row['prompt']}\nResponse:"
        answer = f" {row['response']}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[:max_length]
        labels = ([-100] * min(len(prompt_ids), len(ids)) + answer_ids)[: len(ids)]
        tokenized.append({"input_ids": ids, "labels": labels})
        lengths.append(len(ids))
    return tokenized, lengths


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids, labels, mask = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        mask.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    logger = RunLogger("eval")

    logger.log("environment", **env_snapshot(args.runner_label))
    logger.memory("process_start")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prep_start = time.perf_counter()
    rows = load_jsonl(args.eval_file)
    tok, lens = tokenize_rows(rows, tokenizer, args.max_length)
    prep_s = time.perf_counter() - prep_start
    logger.log("data_pipeline", rows=len(rows), preprocess_s=f"{prep_s:.4f}", length_stats=summarize_lengths(lens))

    loader = DataLoader(tok, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    load_start = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    reload_s = time.perf_counter() - load_start
    logger.log("adapter_reload", adapter_reload_s=f"{reload_s:.4f}")
    logger.memory("after_adapter_reload")

    model.eval()
    eval_start = time.perf_counter()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            b0 = time.perf_counter()
            out = model(**batch)
            dur = time.perf_counter() - b0
            losses.append(float(out.loss.item()))
            logger.log("eval_batch", batch_index=i, batch_duration_s=f"{dur:.4f}", batch_tokens=int(batch['attention_mask'].sum().item()))
            logger.memory(f"eval_batch_{i}")
    eval_s = time.perf_counter() - eval_start

    gen_start = time.perf_counter()
    prompt = "Instruction: Answer briefly: What is 1+1?\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_s = time.perf_counter() - gen_start
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    logger.log("generation", generation_duration_s=f"{gen_s:.4f}", generated_preview=decoded[:80])
    logger.memory("after_eval")

    mean_loss = float(np.mean(losses)) if losses else 0.0
    metrics = {
        "adapter_reload_s": reload_s,
        "preprocess_s": prep_s,
        "eval_duration_s": eval_s,
        "generation_duration_s": gen_s,
        "mean_eval_loss": mean_loss,
        "sequence_length_stats": summarize_lengths(lens),
        "memory_snapshots": snapshots_to_dicts(logger.memory_snapshots),
    }
    write_json(Path(args.metrics_path), metrics)
    logger.log("final_bottleneck_summary", likely_eval_limit_seconds=f"{max(eval_s, gen_s):.4f}", mean_eval_loss=f"{mean_loss:.5f}")
    logger.log("metrics_written", path=args.metrics_path)


if __name__ == "__main__":
    main()
