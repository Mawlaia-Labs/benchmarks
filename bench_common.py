"""Shared utilities for all benchmark scripts."""
import json
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import httpx

MAWLAIA_BASE = "https://api.mawlaia.com/v1"
OPENAI_BASE = "https://api.openai.com/v1"
MISTRAL_BASE = "https://api.mistral.ai/v1"
LLAMA_CLOUD_BASE = "https://api.cloud.llamaindex.ai/api"


def env(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        raise RuntimeError(f"Missing env var: {key}")
    return val


def mawlaia_headers() -> dict:
    return {"Authorization": f"Bearer {env('MWL_API_KEY')}", "Content-Type": "application/json"}


def openai_headers() -> dict:
    return {"Authorization": f"Bearer {env('OPENAI_API_KEY')}", "Content-Type": "application/json"}


def mistral_headers() -> dict:
    return {"Authorization": f"Bearer {env('MISTRAL_API_KEY')}", "Content-Type": "application/json"}


def timed_post(client: httpx.Client, url: str, headers: dict, body: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.post(url, headers=headers, json=body, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), ms


def latency_stats(times_ms: list[float]) -> dict:
    if not times_ms:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    s = sorted(times_ms)
    n = len(s)
    return {
        "p50_ms": round(s[n // 2], 1),
        "p95_ms": round(s[int(n * 0.95)], 1),
        "mean_ms": round(statistics.mean(s), 1),
    }


def save_results(product: str, n_samples: int, providers: dict[str, Any], outfile: str | None = None) -> str:
    payload = {
        "product": product,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_samples": n_samples,
        "providers": providers,
    }
    if outfile is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        outfile = f"benchmarks/results/{product}_{date}.json"
    with open(outfile, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {outfile}")
    return outfile


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_span_metrics(
    ground_truth: list[dict],   # [{type, value}]
    predicted: list[dict],      # [{type, value}]
    entity_types: list[str],
) -> dict:
    """Token-level F1 per entity type, plus macro average."""
    results = {}
    for et in entity_types:
        gt_values = {d["value"].lower() for d in ground_truth if d["type"] == et}
        pred_values = {d["value"].lower() for d in predicted if d["type"] == et}
        tp = len(gt_values & pred_values)
        fp = len(pred_values - gt_values)
        fn = len(gt_values - pred_values)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        results[et] = {"precision": round(p, 4), "recall": round(r, 4), "f1": f1(p, r)}
    macro_f1 = round(statistics.mean(v["f1"] for v in results.values()), 4)
    results["MACRO"] = {"f1": macro_f1}
    return results
