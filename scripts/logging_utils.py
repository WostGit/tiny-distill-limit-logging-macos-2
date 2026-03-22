import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch


class VerboseLogger:
    def __init__(self, runner_label: str) -> None:
        self.t0 = time.perf_counter()
        self.runner_label = runner_label

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, event: str, **payload: Any) -> None:
        base = {
            "ts_utc": self.now_iso(),
            "elapsed_s": round(self.elapsed(), 4),
            "event": event,
        }
        base.update(payload)
        print(json.dumps(base, sort_keys=True), flush=True)

    def memory_snapshot(self, stage: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # On macOS, ru_maxrss is in bytes. On Linux it is KiB.
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_mib = rss_raw / (1024 * 1024)
        else:
            rss_mib = rss_raw / 1024
        payload = {
            "stage": stage,
            "rss_mib": round(rss_mib, 3),
        }
        if extra:
            payload.update(extra)
        self.log("memory_snapshot", **payload)
        return payload

    def env_report(self) -> Dict[str, Any]:
        report = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "macos_version": platform.mac_ver()[0] or "not-macos",
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "runner_label": self.runner_label,
        }
        self.log("environment", **report)
        return report


def timed_block(logger: VerboseLogger, event_start: str, event_end: str):
    class _TimerCtx:
        def __enter__(self_inner):
            self_inner.t = time.perf_counter()
            logger.log(event_start)
            return self_inner

        def __exit__(self_inner, exc_type, exc_val, exc_tb):
            dt = time.perf_counter() - self_inner.t
            logger.log(event_end, duration_s=round(dt, 4), failed=exc_type is not None)

    return _TimerCtx()
