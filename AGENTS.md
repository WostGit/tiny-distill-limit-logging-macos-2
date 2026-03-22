# AGENTS.md

## Purpose
This repository is a tiny, self-contained, verbose smoke test for diagnosing distillation scaling limits on GitHub-hosted macOS runners.

## Guardrails for future changes
- Preserve verbose, limit-focused instrumentation.
- Keep the repository self-contained.
- Prefer small, reviewable changes.
- Do not replace explicit logging with opaque abstractions.
- Do not remove useful timing or memory logs without a better replacement.
- Keep GitHub Actions compatibility as a top priority.
- Be conservative about RAM because standard GitHub-hosted Apple Silicon macOS runners are resource-constrained.

## Practical expectations
- CPU-only behavior should remain easy to run in CI.
- JSON metrics artifacts should remain stable and machine-readable.
- Any simplification should keep diagnosability as the top objective.
