# AGENTS.md

## Repository mission
This repository is a self-contained smoke-test harness for diagnosing tiny LoRA-style distillation limits on GitHub-hosted macOS runners.

## Guardrails for contributors/agents
- Preserve verbose, limit-focused instrumentation.
- Keep the repository self-contained.
- Prefer small, reviewable changes.
- Do not replace explicit logging with opaque abstractions.
- Do not remove useful timing or memory logs without a better replacement.
- Keep GitHub Actions compatibility as a top priority.
- Be conservative about RAM because standard GitHub-hosted Apple Silicon macOS runners are resource-constrained.
