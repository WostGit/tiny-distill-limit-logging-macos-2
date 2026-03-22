import json
import logging
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("tiny_distill")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_rss_mb() -> float:
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    # Fallback for environments without psutil (ru_maxrss is KiB on macOS/Linux).
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kib / 1024.0


def memory_snapshot(logger: logging.Logger, stage: str, run_start: float) -> Dict[str, Any]:
    snapshot = {
        "stage": stage,
        "elapsed_sec": round(time.perf_counter() - run_start, 4),
        "rss_mb": round(get_rss_mb(), 2),
        "timestamp_utc": utc_now_iso(),
    }
    logger.info(
        "MEMORY SNAPSHOT | stage=%s | elapsed_sec=%.4f | rss_mb=%.2f",
        stage,
        snapshot["elapsed_sec"],
        snapshot["rss_mb"],
    )
    return snapshot


def collect_env_info() -> Dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "macos_version": platform.platform(),
        "architecture": platform.machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkldnn_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "github_runner_label": os.environ.get("RUNNER_OS", "unknown") + "/" + os.environ.get("ImageOS", "unknown"),
    }


def log_env_info(logger: logging.Logger, env_info: Dict[str, Any]) -> None:
    logger.info("ENVIRONMENT SNAPSHOT START")
    for key, value in env_info.items():
        logger.info("ENV | %s=%s", key, value)
    logger.info("ENVIRONMENT SNAPSHOT END")


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
