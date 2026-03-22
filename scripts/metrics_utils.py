import math
from typing import Dict, Iterable, List


def compute_stats(values: Iterable[int]) -> Dict[str, float]:
    series: List[int] = list(values)
    if not series:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0}

    ordered = sorted(series)
    index_95 = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": float(ordered[0]),
        "mean": float(sum(ordered) / len(ordered)),
        "max": float(ordered[-1]),
        "p95": float(ordered[index_95]),
    }


def summarize_bottlenecks(train_metrics: Dict, eval_metrics: Dict) -> List[str]:
    summaries = []

    per_step_durations = train_metrics.get("step_durations_sec", [])
    avg_step = sum(per_step_durations) / len(per_step_durations) if per_step_durations else 0.0
    save_sec = train_metrics.get("checkpoint", {}).get("save_duration_sec", 0.0)
    preprocess_sec = train_metrics.get("data_pipeline", {}).get("tokenization_sec", 0.0)
    eval_sec = eval_metrics.get("eval_duration_sec", 0.0)

    if avg_step > 1.0:
        summaries.append(f"Wall-clock training step time is notable (avg {avg_step:.3f}s/step).")
    else:
        summaries.append(f"Wall-clock training steps are short (avg {avg_step:.3f}s/step).")

    peak_rss = max((m.get("rss_mb", 0.0) for m in train_metrics.get("memory_snapshots", [])), default=0.0)
    summaries.append(f"Peak observed RSS was {peak_rss:.2f} MB (RAM pressure proxy).")

    summaries.append(
        "Token length p95 (input/target): "
        f"{train_metrics.get('sequence_stats', {}).get('input', {}).get('p95', 0):.1f}/"
        f"{train_metrics.get('sequence_stats', {}).get('target', {}).get('p95', 0):.1f}."
    )

    summaries.append(
        f"Checkpoint IO time was {save_sec:.3f}s with "
        f"{train_metrics.get('checkpoint', {}).get('file_count', 0)} files."
    )
    summaries.append(f"Eval runtime was {eval_sec:.3f}s.")
    summaries.append(f"Preprocessing/tokenization runtime was {preprocess_sec:.3f}s.")

    threads = train_metrics.get("env", {}).get("torch_num_threads", "?")
    summaries.append(f"Torch thread setting was {threads}; inspect per-step variance for thread behavior limits.")

    return summaries
