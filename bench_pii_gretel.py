"""
PII Detection Benchmark — Gretel Synthetic PII Finance Dataset
Dataset: gretelai/synthetic_pii_finance_multilingual
Providers: Mawlaia PII Vault (regex), Mawlaia PII Vault + LLM fallback

Entity types: email, phone, ssn, credit_card, ip_address
Metric: value-level F1 per entity type + macro F1

Usage:
  python3 benchmarks/public/bench_pii_gretel.py [--n-items 1000] [--providers mawlaia mawlaia_llm]

Required env vars:
  MWL_API_KEY, OPENAI_API_KEY (only for mawlaia_llm)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


from bench_common import MAWLAIA_BASE, OPENAI_BASE, env, mawlaia_headers, openai_headers, timed_post, latency_stats
from bench_stats import bootstrap_ci

SEED = 42
RESULTS_DIR = _THIS_DIR / "results"
DATASET_NAME = "gretelai/synthetic_pii_finance_multilingual"

# Gretel label → Mawlaia entity type
ENTITY_MAP = {
    "email":               "EMAIL",
    "phone_number":        "PHONE",
    "ssn":                 "SSN",
    "credit_card_number":  "CREDIT_CARD",
    "ipv4":                "IP_ADDRESS",
}
TARGET_TYPES = list(ENTITY_MAP.values())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gretel(n_items: int = 1000) -> list[dict]:
    """
    Load Gretel PII dataset, filter English items with ≥1 target entity type.
    Returns list of dicts: {"text": str, "entities": [{type, value}]}
    """
    print(f"Loading {DATASET_NAME} (streaming)…")
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    strata: dict[str, list[dict]] = {}
    total_rows = 0
    target_pool = n_items * 2   # collect 2× target before sampling
    max_rows = n_items * 30     # hard cap to avoid indefinite streaming

    for row in ds:
        total_rows += 1
        if total_rows > max_rows:
            break
        if row.get("language", "") != "English":
            continue

        text = row["generated_text"]
        spans_raw = row["pii_spans"]
        if isinstance(spans_raw, str):
            try:
                spans = json.loads(spans_raw)
            except (json.JSONDecodeError, ValueError):
                continue
        else:
            spans = spans_raw

        entities = []
        for span in spans:
            gretel_type = span.get("label", "")
            mwl_type = ENTITY_MAP.get(gretel_type)
            if mwl_type is None:
                continue
            value = text[span["start"]:span["end"]]
            if value.strip():
                entities.append({"type": mwl_type, "value": value})

        if not entities:
            continue

        types_present = frozenset(e["type"] for e in entities)
        key = ",".join(sorted(types_present))
        strata.setdefault(key, []).append({"text": text, "entities": entities})

        pool_size = sum(len(v) for v in strata.values())
        if pool_size >= target_pool:
            break

    total_available = sum(len(v) for v in strata.values())
    print(f"  Collected {total_available} English items with target PII.")
    print(f"  Strata: { {k: len(v) for k, v in sorted(strata.items(), key=lambda x: -len(x[1]))[:6]} }")

    # Proportional stratified sample
    rng = np.random.default_rng(SEED)
    sampled: list[dict] = []
    if total_available <= n_items:
        for v in strata.values():
            sampled.extend(v)
    else:
        for key, bucket in strata.items():
            quota = max(1, round(n_items * len(bucket) / total_available))
            idx = rng.choice(len(bucket), size=min(quota, len(bucket)), replace=False)
            sampled.extend(bucket[i] for i in idx)
        # trim or pad to exact n_items
        rng.shuffle(sampled)
        sampled = sampled[:n_items]

    print(f"  Sampled {len(sampled)} items (stratified by entity-type set).")
    return sampled


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _parse_mawlaia_entities(body: dict) -> list[dict]:
    """Extract [{type, value}] from Mawlaia tokenize response."""
    raw = body.get("entities", body.get("tokens", []))
    entities = []
    for tok in raw:
        if tok.get("entity_type") and tok.get("original"):
            entities.append({"type": tok["entity_type"], "value": tok["original"]})
    return entities


def run_mawlaia(items: list[dict], client: httpx.Client, use_llm: bool = False) -> dict:
    label = "mawlaia_llm" if use_llm else "mawlaia"
    print(f"\n  Running {label} on {len(items)} items…")
    latencies: list[float] = []
    per_item: list[dict] = []
    errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 200 == 0:
            print(f"    … {i + 1}/{len(items)}")
        if use_llm:
            time.sleep(0.5)
        try:
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/pii-vault/tokenize",
                headers=mawlaia_headers(),
                json={
                    "text": item["text"],
                    "entity_types": TARGET_TYPES,
                    "use_llm": use_llm,
                },
                timeout=30,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            pred = _parse_mawlaia_entities(r.json())
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    WARN [{i}] {exc}")
            pred = []

        per_item.append({"pred": pred, "gt": item["entities"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors}


def run_mawlaia_llm(items: list[dict], client: httpx.Client) -> dict:
    return run_mawlaia(items, client, use_llm=True)


# TYPE_ALIASES maps any normalised form OpenAI might return → canonical TARGET_TYPES value
_TYPE_ALIASES = {
    "email": "EMAIL", "email_address": "EMAIL",
    "phone": "PHONE", "phone_number": "PHONE", "telephone": "PHONE",
    "ssn": "SSN", "social_security_number": "SSN", "social_security": "SSN",
    "credit_card": "CREDIT_CARD", "credit_card_number": "CREDIT_CARD", "creditcard": "CREDIT_CARD",
    "ip_address": "IP_ADDRESS", "ip": "IP_ADDRESS", "ipv4": "IP_ADDRESS",
}


def run_openai_zeroshot(items: list[dict], client: httpx.Client) -> dict:
    """GPT-4o-mini zero-shot PII extraction on Gretel items."""
    print(f"\n  Running OpenAI zero-shot on {len(items)} items…")
    PROMPT_PREFIX = (
        "Extract all PII from the following text. "
        'Return JSON: {"entities": [{"type": "EMAIL|PHONE|SSN|CREDIT_CARD|IP_ADDRESS", "value": "..."}]}\n'
        "Only include: email addresses, phone numbers, SSNs, credit card numbers, IP addresses.\n"
        "Text: "
    )
    latencies, per_item, errors = [], [], 0

    for i, item in enumerate(items):
        if (i + 1) % 100 == 0:
            print(f"    … {i + 1}/{len(items)}")
        time.sleep(1.2)
        try:
            resp, ms = timed_post(
                client,
                f"{OPENAI_BASE}/chat/completions",
                openai_headers(),
                {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": PROMPT_PREFIX + item["text"]}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 512,
                },
            )
            latencies.append(ms)
            raw = json.loads(resp["choices"][0]["message"]["content"])
            entities_raw = raw if isinstance(raw, list) else next(
                (v for v in raw.values() if isinstance(v, list)), []
            )
            pred = []
            for e in entities_raw:
                if not isinstance(e, dict):
                    continue
                t = _TYPE_ALIASES.get(str(e.get("type", "")).lower().strip())
                v = str(e.get("value", "")).strip()
                if t and v:
                    pred.append({"type": t, "value": v})
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    WARN [{i}] {exc}")
            pred = []
        per_item.append({"pred": pred, "gt": item["entities"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors}


PROVIDERS = {
    "mawlaia":          (lambda items, client: run_mawlaia(items, client, use_llm=False), ["MWL_API_KEY"]),
    "mawlaia_llm":      (run_mawlaia_llm,      ["MWL_API_KEY", "OPENAI_API_KEY"]),
    "openai_zeroshot":  (run_openai_zeroshot,  ["OPENAI_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(per_item: list[dict]) -> dict:
    """Compute per-type F1 + macro F1 with bootstrap CI."""
    from collections import defaultdict
    import numpy as np

    type_labels: dict[str, list[float]] = defaultdict(list)

    for item in per_item:
        gt_by_type: dict[str, set] = defaultdict(set)
        pred_by_type: dict[str, set] = defaultdict(set)
        for e in item["gt"]:
            gt_by_type[e["type"]].add(e["value"].lower())
        for e in item["pred"]:
            pred_by_type[e["type"]].add(e["value"].lower())

        for t in TARGET_TYPES:
            gt_set = gt_by_type[t]
            pred_set = pred_by_type[t]
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) else (1.0 if not gt_set else 0.0)
            r = tp / (tp + fn) if (tp + fn) else (1.0 if not gt_set else 0.0)
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            if gt_set or pred_set:
                type_labels[t].append(f)

    per_type = {}
    macro_vals = []
    for t in TARGET_TYPES:
        if type_labels[t]:
            ci = bootstrap_ci(type_labels[t], seed=SEED)
            per_type[t] = {
                "f1": round(ci["value"], 4),
                "ci95_low": round(ci["ci95_low"], 4),
                "ci95_high": round(ci["ci95_high"], 4),
                "n_items": len(type_labels[t]),
            }
            macro_vals.append(ci["value"])
        else:
            per_type[t] = {"f1": None, "n_items": 0}

    macro_ci = bootstrap_ci(macro_vals, seed=SEED) if macro_vals else {}
    return {"per_type": per_type, "macro_f1": macro_ci}


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(all_results: dict, n_items: int):
    W = 110
    print("\n" + "─" * W)
    print(f"  PII Detection — Gretel Finance PII (N={n_items})")
    print("─" * W)
    header = f"{'Provider':<22} " + " ".join(f"{'  '+t:<26}" for t in TARGET_TYPES) + f"  {'Macro F1':<24}"
    sub = f"{'':22} " + " ".join(f"  {'F1   [95% CI]':<24}" for _ in TARGET_TYPES) + f"  {'F1   [95% CI]'}"
    print(header)
    print(sub)
    print("─" * W)
    for name, data in all_results.items():
        m = data["metrics"]
        row = f"{name:<22}"
        for t in TARGET_TYPES:
            pt = m["per_type"].get(t, {})
            f1v = pt.get("f1")
            lo = pt.get("ci95_low")
            hi = pt.get("ci95_high")
            cell = f"{f1v:.3f} [{lo:.3f},{hi:.3f}]" if f1v is not None else "N/A"
            row += f"  {cell:<24}"
        mf = m["macro_f1"]
        row += f"  {mf.get('value', 0):.3f} [{mf.get('ci95_low', 0):.3f},{mf.get('ci95_high', 0):.3f}]"
        print(row)
    print("─" * W)
    print(f"Dataset: {DATASET_NAME} | N={n_items} stratified | Seed: {SEED}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PII benchmark — Gretel Finance PII")
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    parser.add_argument("--n-items", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}")

    items = load_gretel(n_items=args.n_items)

    print(f"\nPII Gretel Benchmark — {len(items)} items\n")
    all_results: dict[str, dict] = {}

    with httpx.Client() as client:
        for name in args.providers:
            fn, required = PROVIDERS[name]
            missing = [k for k in required if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing: {', '.join(missing)})")
                continue
            raw = fn(items, client)
            metrics = compute_metrics(raw["per_item"])
            print(f"  {name} → Macro F1={metrics['macro_f1'].get('value', 0):.4f} "
                  f"[{metrics['macro_f1'].get('ci95_low', 0):.4f}, {metrics['macro_f1'].get('ci95_high', 0):.4f}]")
            all_results[name] = {**raw, "metrics": metrics}
            del all_results[name]["per_item"]

    if not all_results:
        print("No providers ran.")
        return

    print_table({n: d for n, d in all_results.items()}, len(items))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"pii_gretel_{date}.json")

    payload = {
        "product": "pii",
        "dataset": DATASET_NAME,
        "n_samples": len(items),
        "seed": SEED,
        "entity_types": TARGET_TYPES,
        "providers": {n: {k: v for k, v in d.items() if k != "per_item"} for n, d in all_results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
