# tiny-distill-limit-logging-macos-2

This repository is a **self-contained, GitHub Actions–first smoke test** for tiny LoRA-style distillation on **GitHub-hosted macOS runners**. It intentionally keeps model/data/training tiny and emits very verbose logs so a reviewer can quickly identify whether first-order limits are wall-clock time, RAM pressure, token length effects, checkpoint I/O, evaluation overhead, preprocessing overhead, or thread behavior.

## Why this repo exists
- Diagnose scaling limits of tiny CPU-only distillation runs on macOS Actions workers.
- Keep every moving piece in one place: data, scripts, workflow, dependencies, and metrics outputs.
- Prioritize readability and diagnosability over abstraction or benchmark-quality training.

## Self-contained layout
- `data/tiny_train.jsonl`, `data/tiny_eval.jsonl`: tiny checked-in datasets.
- `scripts/train_tiny_distill.py`: one-epoch tiny LoRA-style distillation smoke test with verbose instrumentation.
- `scripts/eval_tiny_distill.py`: standalone adapter reload + tiny evaluation pass.
- `scripts/logging_utils.py`: environment and memory snapshot logging helpers.
- `scripts/metrics_utils.py`: JSON metric writing and checkpoint directory stats.
- `.github/workflows/smoke-test-macos.yml`: fixed-label macOS workflow (`macos-15`) that runs train+eval and uploads JSON artifacts.
- `outputs/metrics/`: location for JSON metrics artifacts.

## Quick start (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_tiny_distill.py --runner-label macos-15
python scripts/eval_tiny_distill.py --runner-label macos-15
```

## GitHub Actions behavior
The workflow is designed for **standard GitHub-hosted macOS CPU runners** and uses a fixed runner label (`macos-15`). It installs `requirements.txt`, runs training and evaluation scripts, prints concise metrics in logs, and uploads JSON metrics artifacts from `outputs/metrics/*.json`.

## How to read the verbose limit logs
Look for these log families in Actions output:
1. **Environment snapshot** (`[env]`): Python/Torch/macOS/arch/thread settings and selected runner label.
2. **Memory snapshots** (`[mem]`): process start, model load, LoRA wrap, first fwd/bwd, each optimizer step, save boundaries, eval boundaries.
3. **Data pipeline timings** (`[data]`): load time and preprocessing/tokenization time.
4. **Sequence stats** (`[seq]`): input/target min/mean/max/p95 lengths.
5. **Training shape** (`[shape]`): batch size, grad accumulation, effective samples per optimizer step.
6. **Per-step timings** (`[step]`): elapsed time since start, optimizer step duration, cumulative samples, tokens this step.
7. **Checkpoint instrumentation** (`[checkpoint]`): save start/end times, save duration, checkpoint file count and size.
8. **Eval instrumentation** (`[eval]` plus train metrics fields): adapter reload duration, eval duration, generation-loop duration.
9. **Bottleneck summary** (`[summary]`): simple first-order bottleneck proxy from measured timing components.

## Expected bottlenecks on macOS
For tiny smoke runs on standard hosted Apple Silicon runners, likely first bottlenecks are:
- Wall-clock overhead from Python + framework startup and data preprocessing.
- Checkpoint save/reload costs relative to tiny training compute.
- Evaluation overhead dominating tiny train loops.
- Thread behavior mismatch (too many threads can hurt tiny workloads).
- RAM spikes if sequence length or batch size is increased beyond conservative defaults.

## What this smoke test proves and does not prove
### Proves
- The repository can run end-to-end in GitHub Actions on macOS (`macos-15`) with CPU only.
- Verbose instrumentation is sufficient to identify likely first bottleneck categories.
- JSON metrics artifacts are emitted for post-run comparison.

### Does not prove
- Real-world model quality or benchmark competitiveness.
- Scalability to large models/datasets/long-context training.
- Throughput parity with Linux GPU or larger-memory runners.

## Smoke test disclaimer
This is intentionally a **smoke test and limit-diagnosis repository**, not a full distillation benchmark suite.
