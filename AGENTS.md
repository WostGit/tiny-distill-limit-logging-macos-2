# AGENTS.md

## Purpose
This repository is a tiny, self-contained smoke-test harness for diagnosing early scaling limits of LoRA-style distillation on GitHub-hosted macOS runners.

## Change policy
- Preserve verbose, limit-focused instrumentation across training, evaluation, and workflow logs.
- Keep the repo fully self-contained: no hidden local dependencies, no reliance on external monorepos, and no unpublished packages.
- Prefer small, reviewable pull requests that keep behavior explicit and easy to inspect in CI logs.
- Do not replace explicit timing/memory/checkpoint logging with opaque abstractions.
- Do not remove useful timing or memory instrumentation without adding a better replacement.
- Keep GitHub Actions compatibility as a top priority.
- Be conservative about RAM usage because standard GitHub-hosted Apple Silicon macOS runners are resource-constrained.

## Practical guardrails
- Keep the model and dataset tiny enough to complete quickly on CPU-only GitHub Actions.
- Favor deterministic behavior where practical (fixed seed, stable ordering, simple settings).
- Keep metrics artifacts compact JSON so reviewers can compare runs quickly.
