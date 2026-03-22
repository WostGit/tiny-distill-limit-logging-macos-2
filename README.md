# tiny-distill-limit-logging-macos-2

This repository is a **self-contained smoke test** for tiny LoRA-style distillation on **GitHub-hosted macOS runners**. It is intentionally built for **GitHub Actions first**: a tiny checked-in dataset, explicit training/eval scripts, very verbose step-level instrumentation, and compact JSON metrics artifacts that help diagnose which resource hits the first limit.

> This is a limit-diagnosis smoke test repo, **not** a full distillation benchmark.

## What is included

- `data/tiny_train.jsonl` and `data/tiny_eval.jsonl`: tiny checked-in dataset.
- `scripts/train_tiny_distill.py`: tiny LoRA-style distillation smoke training.
- `scripts/eval_tiny_distill.py`: adapter reload + tiny evaluation.
- `scripts/logging_utils.py`: verbose JSONL-style logging and memory snapshots.
- `scripts/metrics_utils.py`: compact metrics/stat helpers.
- `.github/workflows/smoke-test-macos.yml`: macOS-15 CPU-only smoke workflow.
- `outputs/metrics/`: JSON artifacts destination.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_tiny_distill.py --runner-label macos-15
python scripts/eval_tiny_distill.py --runner-label macos-15
```

Metrics are written to:
- `outputs/metrics/train_metrics.json`
- `outputs/metrics/eval_metrics.json`

## GitHub Actions smoke test

The workflow is fixed to `runs-on: macos-15`, installs dependencies from `requirements.txt`, runs train then eval, and uploads JSON metrics artifacts from `outputs/metrics/*.json`.

## How to read the verbose limit logs

Each log line is structured JSON for easy grep/filtering in CI:

1. **Per optimizer step timing**
   - `optimizer_step`, `step_duration_s`, `elapsed_s`, `samples_seen`, `tokens_seen`, `tokens_per_optimizer_step`.
2. **Memory snapshots**
   - Emitted at process start, model load, LoRA wrap, first forward/backward, every optimizer step, before/after save, before/after eval.
3. **Sequence length statistics**
   - Input/target token length min/mean/max/p95, plus tokens-per-step trends.
4. **Training shape metadata**
   - Per-device batch size, grad accumulation, effective samples per optimizer step.
5. **Checkpoint instrumentation**
   - Save start/end timestamps, save duration, checkpoint byte size, and file count.
6. **Eval instrumentation**
   - Adapter reload duration, eval duration, cumulative generation duration, per-example eval traces.
7. **Data pipeline instrumentation**
   - Data load time and preprocessing/tokenization time contribution.
8. **Environment instrumentation**
   - Python/torch versions, macOS/platform metadata, architecture, thread settings, runner label.

## Expected bottlenecks on macOS

On standard GitHub-hosted Apple Silicon macOS runners, likely first limits are:
- **Wall-clock overhead** from setup + Python interpreter + framework startup.
- **Checkpoint IO proportion** in very short jobs.
- **Preprocessing overhead** when tokenization dominates tiny train loops.
- **Thread behavior variance** when thread settings are not conservative.
- **RAM pressure** if model/token lengths are increased beyond smoke-test bounds.

This repo keeps defaults conservative to avoid OOM while still surfacing these ratios clearly.

## What this smoke test proves and does not prove

### Proves
- End-to-end train/save/reload/eval works in CPU-only macOS GitHub Actions.
- Verbose instrumentation can identify which limit appears first in a tiny run.
- JSON metrics are emitted for run-to-run comparison.

### Does not prove
- Real-world quality of large-scale distillation.
- Throughput ceilings for larger models, longer contexts, or mixed precision.
- Production-grade optimization strategy beyond first-limit diagnosis.
