"""
PII Detection Benchmark — AI4Privacy Public Dataset
====================================================
Evaluates Mawlaia and OpenAI (zero-shot GPT-4o-mini) on the
ai4privacy/pii-masking-400k dataset (train split, streaming).

Usage:
    python3 benchmarks/public/bench_pii_public.py [options]

Options:
    --providers  mawlaia openai_zeroshot   (default: both)
    --n-items    2000                       (default: 2000)
    --output     path/to/results.json      (auto-generated if omitted)

Required env vars:
    MWL_API_KEY       — Mawlaia API key
    OPENAI_API_KEY    — OpenAI API key (only if openai_zeroshot requested)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

# Allow importing bench_common from the parent benchmarks/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import (
    MAWLAIA_BASE,
    OPENAI_BASE,
    env,
    latency_stats,
    mawlaia_headers,
    openai_headers,
    timed_post,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_stats import (
    bootstrap_ci_pairs,
    macro_f1,
    mcnemar_test,
    span_f1,
    stratified_sample,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mawlaia entity types used in this benchmark
ENTITY_TYPES = ["email", "phone", "credit_card", "ssn"]

# AI4Privacy label → Mawlaia entity type
LABEL_MAP = {
    "EMAIL": "email",
    "TELEPHONENUM": "phone",
    "CREDITCARDNUMBER": "credit_card",
    "SOCIALNUM": "ssn",
}

# Mawlaia API entity_type values → our normalized labels
_API_TYPE_MAP = {
    "EMAIL": "email",
    "PHONE": "phone",
    "PHONE_NUMBER": "phone",
    "TELEPHONENUM": "phone",
    "CREDIT_CARD": "credit_card",
    "CREDITCARD": "credit_card",
    "CREDITCARDNUMBER": "credit_card",
    "SSN": "ssn",
    "US_SSN": "ssn",
    "SOCIALNUM": "ssn",
}

OPENAI_EXTRACT_PROMPT_PREFIX = (
    "Extract all PII from the following text. "
    'Return JSON: {"entities": [{"type": "email|phone|ssn|credit_card", "value": "..."}]}\n'
    "Only include: email addresses, phone numbers, SSNs, credit card numbers.\n"
    "Text: "
)

DATASET_NAME = "ai4privacy/pii-masking-400k"
SEED = 42
N_OPENAI_VARIANCE_SUBSET = 200
N_OPENAI_RUNS = 3


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_ai4privacy(n_collect: int = 8000) -> list[dict]:
    """
    Stream the AI4Privacy train split and collect up to n_collect items
    that contain at least one of the 4 mapped entity types.

    Returns a list of dicts:
        {"text": str, "gt_entities": [{"type": str, "value": str}]}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the `datasets` package: pip install datasets")

    print(f"Loading {DATASET_NAME} (streaming)…")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    collected = []
    for row in ds:
        # privacy_mask may be a JSON string or already a list depending on dataset version
        mask = row.get("privacy_mask", [])
        if isinstance(mask, str):
            try:
                mask = json.loads(mask)
            except Exception:
                mask = []

        # Keep only entities we care about
        gt_entities = []
        for ent in mask:
            label = ent.get("label", ent.get("type", ""))
            mwl_type = LABEL_MAP.get(label)
            if mwl_type:
                gt_entities.append({"type": mwl_type, "value": ent.get("value", "")})

        if not gt_entities:
            continue

        collected.append({
            "text": row["source_text"],
            "gt_entities": gt_entities,
        })

        if len(collected) >= n_collect:
            break

    print(f"  Collected {len(collected)} items with mapped PII entities.")
    return collected


def _stratum_key(item: dict) -> str:
    """Stratum key = sorted frozenset of entity types present in an item."""
    types = frozenset(e["type"] for e in item["gt_entities"])
    return ",".join(sorted(types))


# ---------------------------------------------------------------------------
# Provider runners
# ---------------------------------------------------------------------------

def run_mawlaia(items: list[dict], client: httpx.Client) -> dict:
    """Run Mawlaia PII detection on all items."""
    print(f"  Running Mawlaia PII on {len(items)} items…")
    latencies: list[float] = []
    per_item_results: list[dict] = []
    errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 200 == 0:
            print(f"    … {i + 1}/{len(items)}")
        try:
            body, ms = timed_post(
                client,
                f"{MAWLAIA_BASE}/pii-vault/tokenize",
                mawlaia_headers(),
                {"text": item["text"], "format_preserving": False},
            )
            latencies.append(ms)
            # API returns: {"entities": [{"entity_type": "EMAIL", "original": "..."}]}
            pred_entities = [
                {"type": _API_TYPE_MAP.get(e.get("entity_type", "").upper(), e.get("entity_type", "").lower()),
                 "value": e.get("original", "")}
                for e in body.get("entities", [])
                if _API_TYPE_MAP.get(e.get("entity_type", "").upper())
            ]
            per_item_results.append({
                "gt": item["gt_entities"],
                "pred": pred_entities,
            })
        except Exception as exc:
            errors += 1
            if errors <= 10:
                print(f"    WARN item {i}: {exc}")
            per_item_results.append({"gt": item["gt_entities"], "pred": []})

    return {
        "per_item": per_item_results,
        "latency": latency_stats(latencies),
        "errors": errors,
        "n_items": len(items),
    }


def run_openai_zeroshot(items: list[dict], client: httpx.Client) -> dict:
    """Run OpenAI zero-shot GPT-4o-mini PII extraction on items."""
    print(f"  Running OpenAI zero-shot on {len(items)} items…")
    latencies: list[float] = []
    per_item_results: list[dict] = []
    errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 50 == 0:
            print(f"    … {i + 1}/{len(items)}")
        try:
            body_req = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": OPENAI_EXTRACT_PROMPT_PREFIX + item["text"],
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            resp, ms = timed_post(
                client, f"{OPENAI_BASE}/chat/completions", openai_headers(), body_req
            )
            latencies.append(ms)
            raw = resp["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            # Handle both list and dict response shapes
            if isinstance(parsed, list):
                entities_raw = parsed
            elif isinstance(parsed, dict):
                entities_raw = next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
            else:
                entities_raw = []

            pred_entities = [
                {"type": e.get("type", "").lower(), "value": e.get("value", "")}
                for e in entities_raw
                if isinstance(e, dict)
            ]
            per_item_results.append({
                "gt": item["gt_entities"],
                "pred": pred_entities,
            })
        except Exception as exc:
            errors += 1
            if errors <= 10:
                print(f"    WARN item {i}: {exc}")
            per_item_results.append({"gt": item["gt_entities"], "pred": []})

    return {
        "per_item": per_item_results,
        "latency": latency_stats(latencies),
        "errors": errors,
        "n_items": len(items),
    }


def run_mawlaia_llm(items: list[dict], client: httpx.Client) -> dict:
    """Run Mawlaia PII detection with LLM fallback (use_llm=True)."""
    print(f"  Running Mawlaia PII + LLM fallback on {len(items)} items…")
    import time
    latencies: list[float] = []
    per_item_results: list[dict] = []
    errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 100 == 0:
            print(f"    … {i + 1}/{len(items)}")
        time.sleep(0.5)  # LLM fallback calls OpenAI internally
        try:
            body, ms = timed_post(
                client,
                f"{MAWLAIA_BASE}/pii-vault/tokenize",
                mawlaia_headers(),
                {"text": item["text"], "format_preserving": False, "use_llm": True},
            )
            latencies.append(ms)
            pred_entities = [
                {"type": _API_TYPE_MAP.get(e.get("entity_type", "").upper(), e.get("entity_type", "").lower()),
                 "value": e.get("original", "")}
                for e in body.get("entities", [])
                if _API_TYPE_MAP.get(e.get("entity_type", "").upper())
            ]
            per_item_results.append({"gt": item["gt_entities"], "pred": pred_entities})
        except Exception as exc:
            errors += 1
            if errors <= 10:
                print(f"    WARN item {i}: {exc}")
            per_item_results.append({"gt": item["gt_entities"], "pred": []})

    return {
        "per_item": per_item_results,
        "latency": latency_stats(latencies),
        "errors": errors,
        "n_items": len(items),
    }


PROVIDERS = {
    "mawlaia": (run_mawlaia, ["MWL_API_KEY"]),
    "mawlaia_llm": (run_mawlaia_llm, ["MWL_API_KEY", "OPENAI_API_KEY"]),
    "openai_zeroshot": (run_openai_zeroshot, ["OPENAI_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _item_fully_correct(result: dict) -> bool:
    """True if all ground-truth entities in the item were detected."""
    gt = result["gt"]
    pred = result["pred"]
    for etype in ENTITY_TYPES:
        sf = span_f1(gt, pred, etype)
        if sf["fn"] > 0:
            return False
    return True


def compute_provider_metrics(provider_result: dict) -> dict:
    """
    Compute per-type F1 with bootstrap CI and macro F1 with CI.

    Returns:
        {
            "per_type": {
                "email": {"f1": float, "ci95_low": float, "ci95_high": float,
                          "precision": float, "recall": float},
                ...
            },
            "macro_f1": {"value": float, "ci95_low": float, "ci95_high": float},
            "item_correct": [bool, ...],   # item-level correctness
        }
    """
    per_item = provider_result["per_item"]

    # Aggregate all entities across items for global span F1
    all_gt = [e for r in per_item for e in r["gt"]]
    all_pred = [e for r in per_item for e in r["pred"]]

    per_type: dict[str, dict] = {}
    for etype in ENTITY_TYPES:
        sf = span_f1(all_gt, all_pred, etype)
        per_type[etype] = sf

    # Bootstrap CI per type: resample items, compute span F1 on aggregated lists
    rng = np.random.default_rng(SEED)
    n = len(per_item)

    for etype in ENTITY_TYPES:
        boot_f1s = []
        for _ in range(1000):
            indices = rng.integers(0, n, size=n)
            boot_gt = [e for i in indices for e in per_item[i]["gt"]]
            boot_pred = [e for i in indices for e in per_item[i]["pred"]]
            sf = span_f1(boot_gt, boot_pred, etype)
            boot_f1s.append(sf["f1"])
        per_type[etype]["ci95_low"] = float(np.percentile(boot_f1s, 2.5))
        per_type[etype]["ci95_high"] = float(np.percentile(boot_f1s, 97.5))

    # Macro F1 with bootstrap CI
    mf1 = macro_f1(per_type)
    boot_macros = []
    rng2 = np.random.default_rng(SEED + 1)
    for _ in range(1000):
        indices = rng2.integers(0, n, size=n)
        boot_gt = [e for i in indices for e in per_item[i]["gt"]]
        boot_pred = [e for i in indices for e in per_item[i]["pred"]]
        boot_per_type = {et: span_f1(boot_gt, boot_pred, et) for et in ENTITY_TYPES}
        boot_macros.append(macro_f1(boot_per_type))

    macro_ci = {
        "value": mf1,
        "ci95_low": float(np.percentile(boot_macros, 2.5)),
        "ci95_high": float(np.percentile(boot_macros, 97.5)),
    }

    item_correct = [_item_fully_correct(r) for r in per_item]

    return {
        "per_type": per_type,
        "macro_f1": macro_ci,
        "item_correct": item_correct,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt_f1_ci(metrics: dict, etype: str) -> str:
    """Format 'F1=X.XXX [lo, hi]' for a single entity type."""
    pt = metrics["per_type"][etype]
    f1v = pt["f1"]
    lo = pt.get("ci95_low", float("nan"))
    hi = pt.get("ci95_high", float("nan"))
    return f"{f1v:.3f} [{lo:.3f},{hi:.3f}]"


def _fmt_macro(metrics: dict) -> str:
    mf = metrics["macro_f1"]
    return f"{mf['value']:.3f} [{mf['ci95_low']:.3f},{mf['ci95_high']:.3f}]"


def print_table(
    computed: dict[str, dict],
    mcnemar_result: dict | None,
    n_items: int,
) -> None:
    col_w = 22
    type_labels = {
        "email": "Email",
        "phone": "Phone",
        "credit_card": "Credit Card",
        "ssn": "SSN",
    }

    bar = "─" * 110
    print(f"\n{bar}")
    print(f"  PII Detection — AI4Privacy (N={n_items})")
    print(bar)

    header = f"{'Provider':<20}"
    for etype in ENTITY_TYPES:
        header += f"  {type_labels[etype]:<{col_w}}"
    header += f"  {'Macro F1':<{col_w}}"
    print(header)

    sub = f"{'':20}"
    for _ in ENTITY_TYPES:
        sub += f"  {'F1   [95% CI]':<{col_w}}"
    sub += f"  {'F1   [95% CI]':<{col_w}}"
    print(sub)
    print("─" * 110)

    for provider, metrics in computed.items():
        row = f"{provider:<20}"
        for etype in ENTITY_TYPES:
            cell = _fmt_f1_ci(metrics, etype)
            row += f"  {cell:<{col_w}}"
        row += f"  {_fmt_macro(metrics):<{col_w}}"
        print(row)

    print(bar)

    if mcnemar_result:
        sig = "significant" if mcnemar_result["significant"] else "not significant"
        print(
            f"\nMcNemar's test (Mawlaia vs OpenAI): "
            f"χ²={mcnemar_result['chi2']:.2f}, "
            f"p={mcnemar_result['p_value']:.3f} ({sig})"
        )

    print(
        f"Dataset: {DATASET_NAME} | Sample: N={n_items} stratified | Seed: {SEED}"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PII benchmark on AI4Privacy public dataset"
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=list(PROVIDERS.keys()),
        choices=list(PROVIDERS.keys()),
        help="Providers to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=2000,
        help="Number of items to sample (default: 2000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (auto-generated if omitted)",
    )
    args = parser.parse_args()

    n_items: int = args.n_items
    selected_providers: list[str] = args.providers

    # ── 1. Load dataset ───────────────────────────────────────────────────
    raw_items = _load_ai4privacy(n_collect=max(8000, n_items * 4))

    # ── 2. Stratified sample ──────────────────────────────────────────────
    items = stratified_sample(raw_items, _stratum_key, n_total=n_items, seed=SEED)
    print(f"Sampled {len(items)} items (stratified by entity-type set).")

    # Summarize stratum distribution
    strata_counts = Counter(_stratum_key(it) for it in items)
    print(f"  Strata: {dict(sorted(strata_counts.items(), key=lambda x: -x[1])[:6])}")

    # ── 3. Run providers ──────────────────────────────────────────────────
    raw_results: dict[str, dict] = {}

    with httpx.Client(timeout=60.0) as client:
        for name in selected_providers:
            fn, required_keys = PROVIDERS[name]
            missing = [k for k in required_keys if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing env vars: {', '.join(missing)})")
                continue

            if name == "openai_zeroshot":
                # Run 3 times on a 200-item subset for variance estimation
                variance_subset = stratified_sample(
                    items, _stratum_key, n_total=min(N_OPENAI_VARIANCE_SUBSET, len(items)), seed=SEED + 10
                )
                run_results = []
                for run_idx in range(N_OPENAI_RUNS):
                    print(f"  OpenAI variance run {run_idx + 1}/{N_OPENAI_RUNS}…")
                    r = fn(variance_subset, client)
                    run_results.append(r)

                # Use run 1 as canonical (averaged per-item results for the variance subset)
                # For the main evaluation, also run on all n_items
                print(f"  Running OpenAI on full {len(items)}-item set…")
                full_run = fn(items, client)
                full_run["variance_runs"] = run_results
                raw_results[name] = full_run
            else:
                raw_results[name] = fn(items, client)

    if not raw_results:
        print("No providers ran successfully. Check env vars and try again.")
        sys.exit(1)

    # ── 4. Compute metrics ────────────────────────────────────────────────
    print("\nComputing metrics with bootstrap CIs (1000 resamples)…")
    computed: dict[str, dict] = {}
    for name, result in raw_results.items():
        print(f"  Computing {name}…")
        computed[name] = compute_provider_metrics(result)
        mf = computed[name]["macro_f1"]
        print(
            f"    Macro F1={mf['value']:.4f} "
            f"[{mf['ci95_low']:.4f}, {mf['ci95_high']:.4f}]"
        )

    # ── 5. McNemar's test ─────────────────────────────────────────────────
    mcnemar_result = None
    if "mawlaia" in computed and "openai_zeroshot" in computed:
        # Align item_correct lists — both should cover all n_items
        mwl_correct = computed["mawlaia"]["item_correct"]
        oai_correct = computed["openai_zeroshot"]["item_correct"]
        min_len = min(len(mwl_correct), len(oai_correct))
        gt_labels = [True] * min_len  # ground truth is always "should detect all"
        mcnemar_result = mcnemar_test(
            gt_labels,
            mwl_correct[:min_len],
            oai_correct[:min_len],
        )
        print(
            f"\nMcNemar's test: χ²={mcnemar_result['chi2']}, "
            f"p={mcnemar_result['p_value']}, "
            f"significant={mcnemar_result['significant']}"
        )

    # ── 6. Print academic table ───────────────────────────────────────────
    print_table(computed, mcnemar_result, n_items=len(items))

    # ── 7. Save JSON results ──────────────────────────────────────────────
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    if args.output:
        outfile = args.output
    else:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        outfile = str(results_dir / f"pii_public_{date_str}.json")

    def _serializable(obj):
        """Make numpy types JSON-serializable."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serializable(v) for v in obj]
        return obj

    payload = {
        "product": "pii",
        "dataset": DATASET_NAME,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_samples": len(items),
        "seed": SEED,
        "entity_types": ENTITY_TYPES,
        "providers": {},
    }

    for name in computed:
        raw = raw_results[name]
        met = computed[name]
        payload["providers"][name] = _serializable({
            "n_items": raw["n_items"],
            "errors": raw["errors"],
            "latency": raw["latency"],
            "per_type_metrics": met["per_type"],
            "macro_f1": met["macro_f1"],
        })

    if mcnemar_result:
        payload["mcnemar_mawlaia_vs_openai"] = _serializable(mcnemar_result)

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Results saved → {outfile}")


if __name__ == "__main__":
    main()
