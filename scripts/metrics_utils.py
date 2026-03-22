"""Metrics utilities for JSON outputs and checkpoint inspection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dir_stats(path: Path) -> Dict[str, Any]:
    total_size = 0
    file_count = 0
    if not path.exists():
        return {"exists": False, "file_count": 0, "size_bytes": 0}
    for p in path.rglob("*"):
        if p.is_file():
            file_count += 1
            total_size += p.stat().st_size
    return {"exists": True, "file_count": file_count, "size_bytes": total_size}
