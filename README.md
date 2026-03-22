# tiny-distill-limit-logging-macos-2

This repository is a **self-contained tiny distillation smoke test** designed for **GitHub Actions first** execution on **GitHub-hosted macOS runners**. It trains a very small LoRA-style student adapter on a tiny checked-in dataset, reloads the adapter, runs a tiny evaluation pass, and emits intentionally verbose timing/memory/shape logs plus compact JSON metrics artifacts so a reviewer can quickly diagnose the first scaling limit.

## Why this exists

- Focused limit-diagnosis repo, not a general benchmark suite.
- CPU-only by design to match baseline GitHub-hosted macOS behavior.
- Small deterministic setup so CI remains stable and fast to inspect.

## Repository layout

- `data/tiny_train.jsonl`, `data/tiny_eval.jsonl`: checked-in tiny datasets
- `scripts/train_tiny_distill.py`: tiny LoRA-style distillation smoke test with verbose instrumentation
- `scripts/eval_tiny_distill.py`: adapter reload + tiny eval instrumentation
- `scripts/logging_utils.py`: environment/memory/timestamp logging helpers
- `scripts/metrics_utils.py`: sequence stats + bottleneck summary helper
- `.github/workflows/smoke-test-macos.yml`: fixed-label GitHub Actions workflow (`macos-15`)
- `outputs/metrics/`: JSON metric artifacts location

## Quick start (local)

```bash
python -m pip install -r requirements.txt
python scripts/train_tiny_distill.py --train-data data/tiny_train.jsonl --eval-data data/tiny_eval.jsonl --metrics-out outputs/metrics/train_metrics.json
python scripts/eval_tiny_distill.py --train-data data/tiny_train.jsonl --eval-data data/tiny_eval.jsonl --adapter-path outputs/checkpoint/adapter.pt --metrics-out outputs/metrics/eval_metrics.json
```

## How to read the verbose limit logs

Look for these sections in the training/eval output:

1. **Per optimizer step timing**
   - elapsed since run start
   - step duration
   - cumulative samples
   - optimizer step index
2. **Memory snapshots** at process start, model load, LoRA wrap, first backward, each optimizer step, save boundaries, and eval boundaries.
3. **Sequence stats** (`min/mean/max/p95`) for input and target token lengths plus tokens-per-step.
4. **Training shape metadata** (batch size, grad accumulation, effective samples per optimizer step).
5. **Checkpoint IO instrumentation** (start/end timestamps, save duration, directory size, file count).
6. **Eval instrumentation** (adapter reload time, eval duration, generation duration).
7. **Data pipeline instrumentation** (tokenization/preprocessing and data load timings).
8. **Environment instrumentation** (Python, Torch, macOS, architecture, thread settings, selected runner label).

## Expected bottlenecks on macOS

On standard GitHub-hosted Apple Silicon macOS runners, likely first limits are:

- wall-clock step time from CPU-only execution,
- memory ceiling (peak RSS),
- token/context length growth,
- checkpoint IO overhead,
- eval overhead relative to tiny train loops,
- preprocessing overhead if data prep grows,
- thread configuration effects on variance.

This repo intentionally exposes each category in logs and JSON metrics.

## What this smoke test proves and does not prove

### Proves

- End-to-end CI feasibility of tiny LoRA-style distillation + adapter reload + eval on `macos-15`.
- Visibility into first-order bottlenecks using verbose logs and compact machine-readable metrics.

### Does not prove

- Competitive model quality or production readiness.
- Scaling behavior for large models/datasets, long contexts, distributed training, or GPU-backed workflows.
- Generalizable benchmark numbers beyond this tiny diagnostic setup.
