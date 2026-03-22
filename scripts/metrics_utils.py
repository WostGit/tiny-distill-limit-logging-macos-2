import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(round((q / 100.0) * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def length_stats(values: Iterable[int]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "min": float(min(vals)),
        "mean": float(mean(vals)),
        "max": float(max(vals)),
        "p95": percentile(vals, 95),
    }


def directory_size_and_count(path: str) -> Dict[str, int]:
    total = 0
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            total += os.path.getsize(full)
            count += 1
    return {"bytes": total, "file_count": count}


def write_json(path: str, data: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
