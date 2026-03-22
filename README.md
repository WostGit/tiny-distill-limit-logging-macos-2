# tiny-distill-limit-logging-macos-2

This repository is a **self-contained tiny distillation smoke test** designed to run on **GitHub Actions first**, specifically on **GitHub-hosted macOS runners** (`macos-15`, CPU only). It intentionally uses a tiny checked-in dataset, a tiny LoRA-based student setup, and very verbose instrumentation so a single CI run can show which limit appears first: wall-clock time, RAM pressure, token length effects, checkpoint I/O, evaluation overhead, preprocessing overhead, or thread behavior.

> This is a smoke test and limit-diagnosis repo, **not** a full distillation benchmark.

## What is included

- Tiny checked-in dataset (`data/tiny_train.jsonl`, `data/tiny_eval.jsonl`)
- Training script with LoRA-style distillation + adapter save + adapter reload + tiny post-train eval
- Standalone evaluation script with reload/eval/generation timing
- Verbose logging utilities and compact JSON metrics utilities
- GitHub Actions workflow for fixed macOS runner label (`macos-15`)
- Metrics artifact upload from `outputs/metrics`

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_tiny_distill.py --runner-label local-macos
python scripts/eval_tiny_distill.py --runner-label local-macos
```

Metrics are written to:

- `outputs/metrics/train_metrics.json`
- `outputs/metrics/eval_metrics.json`

## GitHub Actions behavior

The workflow in `.github/workflows/smoke-test-macos.yml`:

1. Runs on `macos-15`
2. Installs dependencies from `requirements.txt`
3. Runs training then evaluation
4. Uploads JSON metrics artifacts

No secrets are required for the default path.

## How to read the verbose limit logs

Look for these log sections:

1. **Per-optimizer-step timing**
   - elapsed run time
   - step duration
   - cumulative samples
   - optimizer step index
2. **Memory snapshots** at key milestones
   - process start
   - after model load
   - after LoRA wrapping
   - after first forward/backward
   - every optimizer step
   - before/after save
   - before/after eval
3. **Sequence stats**
   - input/target length min/mean/max/p95
   - tokens per optimizer step
4. **Checkpoint instrumentation**
   - save start/end timing
   - checkpoint size and file count
5. **Eval instrumentation**
   - adapter reload time
   - eval duration
   - generation duration
6. **Data pipeline instrumentation**
   - data load time
   - tokenization/preprocessing time
7. **Environment instrumentation**
   - Python/Torch/macOS versions
   - architecture/thread settings
   - selected runner label

## Expected bottlenecks on macOS

On standard GitHub-hosted Apple Silicon runners, likely early bottlenecks are:

- CPU wall-clock latency per optimization step
- Preprocessing/tokenization overhead relative to tiny batch sizes
- Adapter reload/checkpoint I/O overhead (small but visible at this scale)
- Evaluation and generation overhead dominating short train loops
- RAM pressure if model/token settings are increased aggressively

Because this setup is intentionally tiny, overhead ratios are often more informative than absolute times.

## What this smoke test proves and does not prove

### Proves

- End-to-end LoRA-style tiny distillation flow works in CI on `macos-15`
- Key timing/memory/checkpoint/eval/thread signals are captured verbosely
- JSON metrics are produced for machine-readable analysis

### Does not prove

- SOTA quality or scaling for large models
- Throughput on GPU or high-memory Linux machines
- Production cost/performance optimization

Use this repo as an early warning and diagnosis harness before larger experiments.
