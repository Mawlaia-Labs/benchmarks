"""
Guardrail Public Benchmark — Aegis AI Content Safety Dataset
Dataset: nvidia/Aegis-AI-Content-Safety-Dataset-1.0
Providers: Mawlaia Guardrail (regex), Mawlaia Guardrail (hybrid), Mistral Moderation

Label mapping:
  "Safe" → safe (0)
  "Needs Caution" → unsafe (1)  [borderline content still requires moderation action]
  anything else → unsafe (1)

Usage:
  python3 benchmarks/public/bench_guardrail_aegis.py [--providers mawlaia mawlaia_hybrid mistral_moderation] [--n-items 1000]

Required env vars:
  MWL_API_KEY, OPENAI_API_KEY (for mawlaia_hybrid), MISTRAL_API_KEY (for mistral_moderation)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


from bench_common import MAWLAIA_BASE, MISTRAL_BASE, env, mawlaia_headers, mistral_headers, latency_stats
from bench_stats import bootstrap_ci, bootstrap_ci_pairs, mcnemar_test

SEED = 42
RESULTS_DIR = _THIS_DIR / "results"
DATASET_NAME = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _majority_label(row: dict) -> str:
    labels = [row.get(f"labels_{i}", "") for i in range(5)]
    labels = [l for l in labels if l]
    if not labels:
        return "Safe"
    c = Counter(labels)
    return c.most_common(1)[0][0]


def _is_unsafe(label: str) -> bool:
    return label != "Safe"


def load_aegis(n_items: int = 1000) -> list[dict]:
    """Load Aegis test split, return list of {text, label, unsafe, source}."""
    print(f"Loading {DATASET_NAME} (test split)…")
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="test")
    items: list[dict] = []
    for row in ds:
        label = _majority_label(row)
        text = row.get("text", "").strip()
        if not text:
            continue
        items.append({
            "text": text,
            "label": label,
            "unsafe": _is_unsafe(label),
            "text_type": row.get("text_type", ""),
        })

    total = len(items)
    n_unsafe = sum(1 for x in items if x["unsafe"])
    print(f"  {total} items: {n_unsafe} unsafe / {total - n_unsafe} safe")

    if n_items < total:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(total, size=n_items, replace=False)
        items = [items[i] for i in sorted(idx)]
        print(f"  Sampled {len(items)} items")

    return items


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def run_mawlaia(corpus: list[dict], client: httpx.Client) -> dict:
    print(f"\n  Running Mawlaia Guardrail (regex) on {len(corpus)} items…")
    latencies, predictions, errors = [], [], 0
    for i, item in enumerate(corpus):
        if (i + 1) % 100 == 0:
            print(f"    [{i + 1}/{len(corpus)}] errors={errors}")
        try:
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/guardrail/check",
                headers=mawlaia_headers(),
                json={"text": item["text"], "detectors": ["harmful_content"]},
                timeout=15,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            body = r.json()
            flagged = not body.get("passed", True)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    WARN [{i}] {exc}")
            flagged = False
        predictions.append({"pred": flagged, "gt": item["unsafe"], "text_type": item["text_type"]})
    return {"predictions": predictions, "latency": latency_stats(latencies), "errors": errors}


def run_mawlaia_hybrid(corpus: list[dict], client: httpx.Client) -> dict:
    print(f"\n  Running Mawlaia Guardrail (hybrid regex+LLM) on {len(corpus)} items…")
    latencies, predictions, errors = [], [], 0
    for i, item in enumerate(corpus):
        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(corpus)}] errors={errors}")
        time.sleep(0.3)
        try:
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/guardrail/check",
                headers=mawlaia_headers(),
                json={"text": item["text"], "detectors": ["harmful_content", "llm_classifier"]},
                timeout=30,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            flagged = not r.json().get("passed", True)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    WARN [{i}] {exc}")
            flagged = False
        predictions.append({"pred": flagged, "gt": item["unsafe"], "text_type": item["text_type"]})
    return {"predictions": predictions, "latency": latency_stats(latencies), "errors": errors}


def run_mistral(corpus: list[dict], client: httpx.Client) -> dict:
    print(f"\n  Running Mistral Moderation (1 req/s) on {len(corpus)} items…")
    latencies, predictions, errors = [], [], 0
    for i, item in enumerate(corpus):
        if (i + 1) % 100 == 0:
            print(f"    [{i + 1}/{len(corpus)}] errors={errors}")
        time.sleep(1.1)
        try:
            t0 = time.perf_counter()
            r = client.post(
                f"{MISTRAL_BASE}/moderations",
                headers=mistral_headers(),
                json={"model": "mistral-moderation-latest", "input": item["text"]},
                timeout=30,
            )
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 429:
                time.sleep(2.0)
                raise RuntimeError("429 Too Many Requests")
            r.raise_for_status()
            latencies.append(ms)
            cats = r.json()["results"][0]["categories"]
            flagged = any(cats.values())
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    WARN [{i}] {exc}")
            flagged = False
        predictions.append({"pred": flagged, "gt": item["unsafe"], "text_type": item["text_type"]})
    return {"predictions": predictions, "latency": latency_stats(latencies), "errors": errors}


PROVIDERS = {
    "mawlaia":            (run_mawlaia,        ["MWL_API_KEY"]),
    "mawlaia_hybrid":     (run_mawlaia_hybrid, ["MWL_API_KEY", "OPENAI_API_KEY"]),
    "mistral_moderation": (run_mistral,        ["MISTRAL_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list[dict]) -> dict:
    pairs_correct = []
    tp = fp = tn = fn = 0
    for p in predictions:
        pred, gt = p["pred"], p["gt"]
        pairs_correct.append((bool(gt), bool(pred)))
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1

    acc = (tp + tn) / len(predictions) if predictions else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1v = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    def _ci(vals):
        return bootstrap_ci(vals, seed=SEED)

    return {
        "accuracy": _ci([1.0 if (p["pred"] == p["gt"]) else 0.0 for p in predictions]),
        "f1": {"value": round(f1v, 4), "ci95_low": None, "ci95_high": None},
        "fpr": {"value": round(fpr, 4), "ci95_low": None, "ci95_high": None},
        "fnr": {"value": round(fnr, 4), "ci95_low": None, "ci95_high": None},
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(all_results: dict, n_items: int):
    W = 90
    print(f"\n── Guardrail — Aegis AI Safety (N={n_items}) " + "─" * (W - 40))
    print(f"{'Provider':<22} {'Accuracy':>10} {'[95% CI]':>16} {'F1':>8} {'FPR':>8} {'FNR':>8}")
    print("-" * W)
    for name, data in all_results.items():
        m = data["metrics"]
        acc = m["accuracy"]
        f1v = m["f1"]["value"]
        fpr = m["fpr"]["value"]
        fnr = m["fnr"]["value"]
        ci = f"[{acc['ci95_low']:.3f},{acc['ci95_high']:.3f}]"
        print(f"{name:<22} {acc['value']:>10.3f} {ci:>16} {f1v:>8.3f} {fpr:>8.3f} {fnr:>8.3f}")
    print("-" * W)
    print(f"Dataset: {DATASET_NAME} | N={n_items} | Seed: {SEED}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    parser.add_argument("--n-items", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}")

    corpus = load_aegis(n_items=args.n_items)
    n_unsafe = sum(1 for x in corpus if x["unsafe"])
    print(f"\nGuardrail Aegis Benchmark — {len(corpus)} samples ({n_unsafe} unsafe / {len(corpus)-n_unsafe} safe)\n")

    all_results: dict[str, dict] = {}
    with httpx.Client() as client:
        for name in args.providers:
            fn, required = PROVIDERS[name]
            missing = [k for k in required if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing: {', '.join(missing)})")
                continue
            raw = fn(corpus, client)
            metrics = compute_metrics(raw["predictions"])
            acc = metrics["accuracy"]["value"]
            f1v = metrics["f1"]["value"]
            fnr = metrics["fnr"]["value"]
            fpr = metrics["fpr"]["value"]
            print(f"  {name} → acc={acc:.3f} F1={f1v:.3f} FPR={fpr:.3f} FNR={fnr:.3f} | "
                  f"p50={raw['latency'].get('p50_ms')}ms | errors={raw['errors']}")
            all_results[name] = {**raw, "metrics": metrics}
            del all_results[name]["predictions"]

    if not all_results:
        print("No providers ran.")
        return

    print_table(all_results, len(corpus))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"guardrail_aegis_{date}.json")

    payload = {
        "product": "guardrail",
        "dataset": DATASET_NAME,
        "n_samples": len(corpus),
        "seed": SEED,
        "providers": {n: {k: v for k, v in d.items() if k != "predictions"} for n, d in all_results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
