"""Verbose logging helpers for tiny distillation smoke tests."""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import psutil
import torch


@dataclass
class MemorySnapshot:
    tag: str
    elapsed_s: float
    rss_mb: float
    vms_mb: float


class RunLogger:
    """Structured logger that prints human-readable lines and stores JSON-ready events."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.start_time = time.perf_counter()
        self.process = psutil.Process(os.getpid())
        self.events: List[Dict] = []
        self.memory_snapshots: List[MemorySnapshot] = []

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.start_time

    def log(self, message: str, **fields: object) -> None:
        elapsed = self.elapsed_s()
        prefix = f"[{self.run_name}] +{elapsed:8.3f}s"
        if fields:
            kv = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"{prefix} {message} | {kv}", flush=True)
        else:
            print(f"{prefix} {message}", flush=True)
        self.events.append({"type": "log", "elapsed_s": elapsed, "message": message, "fields": fields})

    def memory(self, tag: str) -> MemorySnapshot:
        mem = self.process.memory_info()
        snapshot = MemorySnapshot(
            tag=tag,
            elapsed_s=self.elapsed_s(),
            rss_mb=mem.rss / (1024**2),
            vms_mb=mem.vms / (1024**2),
        )
        self.memory_snapshots.append(snapshot)
        self.log(
            "memory_snapshot",
            tag=tag,
            rss_mb=f"{snapshot.rss_mb:.2f}",
            vms_mb=f"{snapshot.vms_mb:.2f}",
        )
        return snapshot


def env_snapshot(runner_label: str) -> Dict[str, object]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "threading_active_count": threading.active_count(),
        "github_runner_label": runner_label,
    }


def summarize_lengths(lengths: Iterable[int]) -> Dict[str, float]:
    arr = np.array(list(lengths), dtype=np.int32)
    if arr.size == 0:
        return {"min": 0, "mean": 0.0, "max": 0, "p95": 0.0}
    return {
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "p95": float(np.percentile(arr, 95)),
    }


def dir_stats(path: Path) -> Dict[str, int]:
    total_size = 0
    file_count = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            file_count += 1
            total_size += file_path.stat().st_size
    return {"checkpoint_file_count": file_count, "checkpoint_size_bytes": total_size}


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def snapshots_to_dicts(snapshots: Iterable[MemorySnapshot]) -> List[Dict[str, object]]:
    return [asdict(s) for s in snapshots]
