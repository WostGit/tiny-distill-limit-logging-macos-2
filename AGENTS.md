# Repository Agent Guidance

This repository is a **tiny distillation smoke test focused on verbose limit diagnosis on GitHub-hosted macOS runners**.

## Non-negotiable priorities
- Preserve verbose, limit-focused instrumentation.
- Keep the repository self-contained and easy to clone/run.
- Prefer small, reviewable changes with explicit behavior.
- Do not replace explicit logging with opaque abstractions.
- Do not remove useful timing or memory logs without a better replacement.
- Keep GitHub Actions compatibility as a top priority.
- Be conservative about RAM because GitHub-hosted Apple Silicon macOS runners are resource-constrained.

## Implementation posture
- Favor deterministic behavior and tiny defaults.
- Keep all core scripts under `scripts/` and checked-in tiny datasets under `data/`.
- Keep JSON metrics in `outputs/metrics/` for artifact upload and regression inspection.
