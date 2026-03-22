"""Verbose logging helpers for tiny macOS distillation smoke tests."""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import psutil
import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ru_maxrss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def memory_snapshot(tag: str) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    snapshot = {
        "tag": tag,
        "timestamp_utc": utc_now_iso(),
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "ru_maxrss_mb": _ru_maxrss_mb(),
        "threads": proc.num_threads(),
    }
    print(f"[mem] {json.dumps(snapshot, sort_keys=True)}", flush=True)
    return snapshot


def environment_snapshot(selected_runner_label: str) -> Dict[str, Any]:
    info = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "unset"),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "selected_runner_label": selected_runner_label,
        "pid": os.getpid(),
        "python_threads_active": threading.active_count(),
    }
    print(f"[env] {json.dumps(info, sort_keys=True)}", flush=True)
    return info


@dataclass
class StepTimer:
    run_start: float

    def log_optimizer_step(
        self,
        step_idx: int,
        step_duration_s: float,
        cumulative_samples: int,
        tokens_this_step: int,
    ) -> None:
        payload = {
            "event": "optimizer_step",
            "step_idx": step_idx,
            "elapsed_since_start_s": time.perf_counter() - self.run_start,
            "step_duration_s": step_duration_s,
            "cumulative_samples": cumulative_samples,
            "tokens_this_step": tokens_this_step,
        }
        print(f"[step] {json.dumps(payload, sort_keys=True)}", flush=True)
