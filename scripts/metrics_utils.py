"""Metrics helpers for JSON artifacts."""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def safe_sum(values: Iterable[float]) -> float:
    return float(sum(values))


def bottleneck_summary(step_times_s: List[float], stage_times_s: Dict[str, float]) -> Dict[str, object]:
    ranked = sorted(stage_times_s.items(), key=lambda kv: kv[1], reverse=True)
    dominant_stage = ranked[0][0] if ranked else "unknown"
    dominant_seconds = ranked[0][1] if ranked else 0.0

    return {
        "dominant_stage": dominant_stage,
        "dominant_stage_seconds": dominant_seconds,
        "avg_optimizer_step_seconds": safe_mean(step_times_s),
        "total_optimizer_seconds": safe_sum(step_times_s),
        "stage_time_rank_desc": [{"stage": k, "seconds": v} for k, v in ranked],
    }
