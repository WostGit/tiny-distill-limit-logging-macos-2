from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import (
    dir_stats,
    env_snapshot,
    snapshots_to_dicts,
    summarize_lengths,
    write_json,
    RunLogger,
)
from metrics_utils import bottleneck_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny verbose LoRA distillation smoke test")
    p.add_argument("--train-file", default="data/tiny_train.jsonl")
    p.add_argument("--eval-file", default="data/tiny_eval.jsonl")
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--output-dir", default="outputs/checkpoints/tiny_adapter")
    p.add_argument("--metrics-path", default="outputs/metrics/train_metrics.json")
    p.add_argument("--runner-label", default="macos-15")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def tokenize_rows(rows: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int) -> List[Dict[str, List[int]]]:
    tokenized = []
    input_lengths, target_lengths = [], []
    for row in rows:
        prompt = f"Instruction: {row['prompt']}\nResponse:"
        answer = f" {row['response']}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:max_length]
        labels = [-100] * min(len(prompt_ids), len(input_ids)) + answer_ids
        labels = labels[: len(input_ids)]

        tokenized.append({"input_ids": input_ids, "labels": labels})
        input_lengths.append(len(input_ids))
        target_lengths.append(max(0, len(labels) - min(len(prompt_ids), len(input_ids))))

    return tokenized, input_lengths, target_lengths


def collate(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def eval_loss(model: torch.nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            out = model(**batch)
            losses.append(float(out.loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    args = parse_args()
    logger = RunLogger("train")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    stage_times: Dict[str, float] = {}
    t0 = time.perf_counter()
    env = env_snapshot(args.runner_label)
    logger.log("environment", **env)
    logger.memory("process_start")

    data_load_start = time.perf_counter()
    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    stage_times["data_load_s"] = time.perf_counter() - data_load_start
    logger.log("loaded_rows", train_rows=len(train_rows), eval_rows=len(eval_rows), data_load_s=f"{stage_times['data_load_s']:.4f}")

    model_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.model_name)
    student_base = AutoModelForCausalLM.from_pretrained(args.model_name)
    stage_times["model_load_s"] = time.perf_counter() - model_load_start
    logger.memory("after_model_load")

    lora_start = time.perf_counter()
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student_base, lora_cfg)
    stage_times["lora_wrap_s"] = time.perf_counter() - lora_start
    logger.memory("after_lora_wrapping")

    preprocess_start = time.perf_counter()
    train_tok, input_lens, target_lens = tokenize_rows(train_rows, tokenizer, args.max_length)
    eval_tok, _, _ = tokenize_rows(eval_rows, tokenizer, args.max_length)
    stage_times["preprocess_s"] = time.perf_counter() - preprocess_start

    input_stats = summarize_lengths(input_lens)
    target_stats = summarize_lengths(target_lens)
    logger.log("sequence_length_stats", input_stats=input_stats, target_stats=target_stats)

    train_loader = DataLoader(
        train_tok,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    eval_loader = DataLoader(
        eval_tok,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    step_idx = 0
    cumulative_samples = 0
    step_times = []
    tokens_per_step = []

    logger.log(
        "training_shape",
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        effective_samples_per_step=args.batch_size * args.grad_accum,
    )

    teacher.eval()
    train_start = time.perf_counter()
    first_fb_done = False
    for epoch in range(args.epochs):
        logger.log("epoch_start", epoch=epoch)
        accum = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            batch_tokens = int(batch["attention_mask"].sum().item())
            with torch.no_grad():
                teacher_out = teacher(**batch)
            student_out = student(**batch)

            student_log_probs = F.log_softmax(student_out.logits, dim=-1)
            teacher_probs = F.softmax(teacher_out.logits, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            loss = 0.5 * student_out.loss + 0.5 * kl_loss
            loss.backward()
            accum += 1
            cumulative_samples += int(batch["input_ids"].shape[0])

            if not first_fb_done:
                logger.memory("after_first_forward_backward")
                first_fb_done = True

            if accum == args.grad_accum or batch_idx == (len(train_loader) - 1):
                step_start = time.perf_counter()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_dur = time.perf_counter() - step_start
                step_idx += 1
                step_times.append(step_dur)
                tokens_per_step.append(batch_tokens)
                logger.memory(f"optimizer_step_{step_idx}")
                logger.log(
                    "optimizer_step",
                    optimizer_step=step_idx,
                    elapsed_s=f"{logger.elapsed_s():.3f}",
                    step_duration_s=f"{step_dur:.4f}",
                    cumulative_samples=cumulative_samples,
                    tokens_this_step=batch_tokens,
                    loss=f"{float(loss.item()):.5f}",
                )
                accum = 0

    stage_times["train_loop_s"] = time.perf_counter() - train_start

    logger.memory("before_save")
    save_start = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_end = time.perf_counter()
    stage_times["checkpoint_save_s"] = save_end - save_start
    ckpt_stats = dir_stats(out_dir)
    logger.log(
        "checkpoint_saved",
        save_start_s=f"{save_start - t0:.3f}",
        save_end_s=f"{save_end - t0:.3f}",
        checkpoint_save_s=f"{stage_times['checkpoint_save_s']:.4f}",
        **ckpt_stats,
    )
    logger.memory("after_save")

    logger.memory("before_eval")
    reload_start = time.perf_counter()
    reloaded_base = AutoModelForCausalLM.from_pretrained(args.model_name)
    reloaded_student = PeftModel.from_pretrained(reloaded_base, out_dir)
    stage_times["adapter_reload_s"] = time.perf_counter() - reload_start

    eval_start = time.perf_counter()
    tiny_eval_loss = eval_loss(reloaded_student, eval_loader)
    stage_times["eval_s"] = time.perf_counter() - eval_start
    logger.memory("after_eval")

    stage_times["total_s"] = time.perf_counter() - t0
    bottleneck = bottleneck_summary(step_times, stage_times)

    logger.log(
        "final_bottleneck_summary",
        dominant_stage=bottleneck["dominant_stage"],
        dominant_stage_seconds=f"{bottleneck['dominant_stage_seconds']:.4f}",
        avg_optimizer_step_seconds=f"{bottleneck['avg_optimizer_step_seconds']:.4f}",
        tiny_eval_loss=f"{tiny_eval_loss:.5f}",
    )

    metrics = {
        "env": env,
        "stage_times_s": stage_times,
        "sequence_length_stats": {
            "input": input_stats,
            "target": target_stats,
            "tokens_per_optimizer_step": tokens_per_step,
        },
        "training_shape": {
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_samples_per_step": args.batch_size * args.grad_accum,
            "optimizer_steps": step_idx,
            "cumulative_samples": cumulative_samples,
        },
        "checkpoint_stats": ckpt_stats,
        "eval": {
            "tiny_eval_loss": tiny_eval_loss,
            "adapter_reload_s": stage_times["adapter_reload_s"],
            "eval_duration_s": stage_times["eval_s"],
        },
        "memory_snapshots": snapshots_to_dicts(logger.memory_snapshots),
        "bottleneck": bottleneck,
        "event_count": len(logger.events),
    }

    write_json(Path(args.metrics_path), metrics)
    logger.log("metrics_written", path=args.metrics_path)


if __name__ == "__main__":
    main()
