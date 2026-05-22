"""
Guardrail Public Benchmark — Content Safety Detection
Datasets: Do-Not-Answer + ToxiGen + PKU-SafeRLHF
Providers: Mawlaia Guardrail, OpenAI Moderation, Mistral Moderation

Usage:
  python3 benchmarks/public/bench_guardrail_public.py [--providers mawlaia openai_moderation] [--n-items 2939]

Required env vars (only for providers you want):
  MWL_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

# Path resolution: allow running from project root or from benchmarks/public/
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))          # benchmarks/public/ (for bench_stats)
   # benchmarks/ (for bench_common)

from bench_common import (
    MAWLAIA_BASE, OPENAI_BASE, MISTRAL_BASE,
    env, mawlaia_headers, openai_headers, mistral_headers,
    timed_post, latency_stats, save_results,
)
from bench_stats import bootstrap_ci_pairs, mcnemar_test, stratified_sample

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_do_not_answer(max_items: int = 939) -> list[dict]:
    """LibrAI/do-not-answer — 939 unsafe questions across 5 risk areas."""
    print("  Loading Do-Not-Answer dataset...")
    from datasets import load_dataset
    ds = load_dataset("LibrAI/do-not-answer", split="train")
    items = []
    for row in ds:
        text = str(row.get("question", "")).strip()
        if not text:
            continue
        items.append({
            "text": text,
            "label": True,  # ALL are unsafe
            "source": "do_not_answer",
            "category": str(row.get("risk_area", "unknown")),
        })
        if len(items) >= max_items:
            break
    print(f"    Loaded {len(items)} items from Do-Not-Answer")
    return items


def load_toxigen(max_unsafe: int = 1000, max_safe: int = 500) -> list[dict]:
    """skg/toxigen-data — annotated split, threshold 3.5/2.0."""
    print("  Loading ToxiGen dataset...")
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("skg/toxigen-data", name="annotated", split="train")
    rng = __import__("numpy").random.default_rng(SEED)

    unsafe_items: list[dict] = []
    safe_items: list[dict] = []

    for row in ds:
        text = str(row.get("text", "")).strip()
        score = float(row.get("toxicity_human", 0.0))
        group = str(row.get("target_group", "unknown"))

        if not text:
            continue

        if score >= 3.5:
            unsafe_items.append({
                "text": text,
                "label": True,
                "source": "toxigen",
                "category": group,
                "_score": score,
            })
        elif score <= 2.0:
            safe_items.append({
                "text": text,
                "label": False,
                "source": "toxigen",
                "category": group,
                "_score": score,
            })
        # Exclude ambiguous (2.0 < score < 3.5)

    # Stratified sample by target_group
    sampled_unsafe = stratified_sample(
        unsafe_items, key_fn=lambda x: x["category"], n_total=max_unsafe, seed=SEED
    )
    sampled_safe = stratified_sample(
        safe_items, key_fn=lambda x: x["category"], n_total=max_safe, seed=SEED
    )

    # Remove internal score field
    result = []
    for item in sampled_unsafe + sampled_safe:
        result.append({k: v for k, v in item.items() if k != "_score"})

    print(f"    Loaded {len(sampled_unsafe)} unsafe + {len(sampled_safe)} safe from ToxiGen")
    return result


def load_pku_saferlhf(max_safe: int = 500) -> list[dict]:
    """PKU-Alignment/PKU-SafeRLHF — safe prompts only (FPR measurement)."""
    print("  Loading PKU-SafeRLHF dataset...")
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    safe_items: list[dict] = []

    for row in ds:
        # Only rows where is_response_0_safe == "True" → safe prompt
        is_safe = row.get("is_response_0_safe", False)
        # Field may be bool or string
        if isinstance(is_safe, str):
            is_safe = is_safe.strip().lower() == "true"

        if not is_safe:
            continue

        text = str(row.get("prompt", "")).strip()
        if len(text) < 20:  # exclude very short prompts
            continue

        safe_items.append({
            "text": text,
            "label": False,  # ground truth = safe
            "source": "pku_saferlhf",
            "category": "safe_prompt",
        })

    # Random sample (no stratification needed — single category)
    rng = __import__("numpy").random.default_rng(SEED)
    if len(safe_items) > max_safe:
        indices = rng.choice(len(safe_items), size=max_safe, replace=False)
        safe_items = [safe_items[i] for i in indices]

    print(f"    Loaded {len(safe_items)} safe prompts from PKU-SafeRLHF")
    return safe_items


def build_corpus(n_items: int | None = None) -> list[dict]:
    """Combine all three datasets into a single corpus."""
    print("\nBuilding combined corpus...")

    dna = load_do_not_answer(max_items=939)
    tox = load_toxigen(max_unsafe=1000, max_safe=500)
    pku = load_pku_saferlhf(max_safe=500)

    corpus = dna + tox + pku

    # Shuffle deterministically
    import numpy as np
    rng = np.random.default_rng(SEED)
    rng.shuffle(corpus)

    if n_items is not None and n_items < len(corpus):
        corpus = corpus[:n_items]

    n_unsafe = sum(1 for x in corpus if x["label"])
    n_safe = sum(1 for x in corpus if not x["label"])
    print(f"  Total corpus: {len(corpus)} items ({n_unsafe} unsafe, {n_safe} safe)")
    return corpus


# ---------------------------------------------------------------------------
# Provider runners
# ---------------------------------------------------------------------------

def run_mawlaia(corpus: list[dict], client: httpx.Client) -> dict:
    """Mawlaia Guardrail API (regex only): passed=False → predicted unsafe."""
    print("\n  Running Mawlaia Guardrail (regex)...")
    latencies: list[float] = []
    pairs: list[tuple[bool, bool]] = []
    per_source: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    errors = 0

    for i, item in enumerate(corpus):
        if i % 100 == 0 and i > 0:
            print(f"    [{i}/{len(corpus)}] errors={errors}")
        try:
            body, ms = timed_post(
                client,
                f"{MAWLAIA_BASE}/guardrail/check",
                mawlaia_headers(),
                {
                    "text": item["text"],
                    "direction": "input",
                    "detectors": ["harmful_content"],
                },
            )
            latencies.append(ms)
            predicted_unsafe = not body.get("passed", True)
            gt = item["label"]
            pairs.append((gt, predicted_unsafe))
            per_source[item["source"]].append((gt, predicted_unsafe))
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{item['source']}] {exc}")

    return {
        "pairs": pairs,
        "per_source": dict(per_source),
        "latency": latency_stats(latencies),
        "errors": errors,
    }


def run_mawlaia_hybrid(corpus: list[dict], client: httpx.Client) -> dict:
    """Mawlaia Guardrail API (regex + LLM classifier): semantic harm detection."""
    print("\n  Running Mawlaia Guardrail (hybrid regex+LLM)...")
    latencies: list[float] = []
    pairs: list[tuple[bool, bool]] = []
    per_source: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    errors = 0

    for i, item in enumerate(corpus):
        if i % 50 == 0 and i > 0:
            print(f"    [{i}/{len(corpus)}] errors={errors}")
        time.sleep(0.3)  # avoid OpenAI rate limits on the LLM classifier
        try:
            body, ms = timed_post(
                client,
                f"{MAWLAIA_BASE}/guardrail/check",
                mawlaia_headers(),
                {
                    "text": item["text"],
                    "direction": "input",
                    "detectors": ["harmful_content", "llm_classifier"],
                },
            )
            latencies.append(ms)
            predicted_unsafe = not body.get("passed", True)
            gt = item["label"]
            pairs.append((gt, predicted_unsafe))
            per_source[item["source"]].append((gt, predicted_unsafe))
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{item['source']}] {exc}")
            time.sleep(2.0)

    return {
        "pairs": pairs,
        "per_source": dict(per_source),
        "latency": latency_stats(latencies),
        "errors": errors,
    }


def run_openai_moderation(corpus: list[dict], client: httpx.Client) -> dict:
    """OpenAI omni-moderation-latest: results[0].flagged → predicted unsafe."""
    print("\n  Running OpenAI Moderation...")
    latencies: list[float] = []
    pairs: list[tuple[bool, bool]] = []
    per_source: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    errors = 0

    for i, item in enumerate(corpus):
        if i % 100 == 0 and i > 0:
            print(f"    [{i}/{len(corpus)}] errors={errors}")
        try:
            body, ms = timed_post(
                client,
                f"{OPENAI_BASE}/moderations",
                openai_headers(),
                {"input": item["text"], "model": "omni-moderation-latest"},
            )
            latencies.append(ms)
            flagged = body["results"][0]["flagged"]
            gt = item["label"]
            pairs.append((gt, flagged))
            per_source[item["source"]].append((gt, flagged))
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{item['source']}] {exc}")

    return {
        "pairs": pairs,
        "per_source": dict(per_source),
        "latency": latency_stats(latencies),
        "errors": errors,
    }


def run_mistral_moderation(corpus: list[dict], client: httpx.Client) -> dict:
    """Mistral moderation-latest: any(categories.values()) → predicted unsafe. 1 req/s."""
    print("\n  Running Mistral Moderation (1 req/s — free tier)...")
    latencies: list[float] = []
    pairs: list[tuple[bool, bool]] = []
    per_source: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    errors = 0

    for i, item in enumerate(corpus):
        if i % 100 == 0 and i > 0:
            print(f"    [{i}/{len(corpus)}] errors={errors}")
        try:
            body, ms = timed_post(
                client,
                f"{MISTRAL_BASE}/moderations",
                mistral_headers(),
                {"model": "mistral-moderation-latest", "input": item["text"]},
            )
            latencies.append(ms)
            categories = body["results"][0].get("categories", {})
            flagged = any(categories.values())
            gt = item["label"]
            pairs.append((gt, flagged))
            per_source[item["source"]].append((gt, flagged))
            time.sleep(1.1)  # free tier: 1 req/s
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{item['source']}] {exc}")
            time.sleep(2.0)  # back off on error

    return {
        "pairs": pairs,
        "per_source": dict(per_source),
        "latency": latency_stats(latencies),
        "errors": errors,
    }


PROVIDERS = {
    "mawlaia": (run_mawlaia, ["MWL_API_KEY"]),
    "mawlaia_hybrid": (run_mawlaia_hybrid, ["MWL_API_KEY", "OPENAI_API_KEY"]),
    "openai_moderation": (run_openai_moderation, ["OPENAI_API_KEY"]),
    "mistral_moderation": (run_mistral_moderation, ["MISTRAL_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    """Compute accuracy, F1, FPR, FNR with 95% bootstrap CIs."""
    metrics = {}
    for metric in ("accuracy", "f1", "fpr", "fnr"):
        ci = bootstrap_ci_pairs(pairs, metric=metric, seed=SEED)
        metrics[metric] = ci
    return metrics


def _fmt_ci(ci: dict, decimals: int = 3) -> str:
    v = ci.get("value", float("nan"))
    lo = ci.get("ci95_low", float("nan"))
    hi = ci.get("ci95_high", float("nan"))
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(v)}  [{fmt.format(lo)}, {fmt.format(hi)}]"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(corpus: list[dict], all_results: dict[str, dict]):
    n = len(corpus)
    n_unsafe = sum(1 for x in corpus if x["label"])
    n_safe = sum(1 for x in corpus if not x["label"])

    # Source counts
    src_counts: dict[str, dict] = defaultdict(lambda: {"unsafe": 0, "safe": 0})
    for item in corpus:
        key = "unsafe" if item["label"] else "safe"
        src_counts[item["source"]][key] += 1

    dna_unsafe = src_counts["do_not_answer"]["unsafe"]
    tox_unsafe = src_counts["toxigen"]["unsafe"]
    tox_safe = src_counts["toxigen"]["safe"]
    pku_safe = src_counts["pku_saferlhf"]["safe"]

    print(f"\n── Guardrail — Public Datasets (N={n}) ──────────────────────────────────────")
    print(f"Source breakdown: Do-Not-Answer({dna_unsafe} unsafe) | "
          f"ToxiGen({tox_unsafe} unsafe, {tox_safe} safe) | "
          f"PKU-SafeRLHF({pku_safe} safe)")
    print()
    print(f"{'Provider':<22} {'Accuracy':^22} {'F1':^22} {'FPR':^22} {'FNR':^22}")
    print(f"{'':22} {'val  [95% CI]':^22} {'val  [95% CI]':^22} {'val  [95% CI]':^22} {'val  [95% CI]':^22}")
    print("─" * 94)

    metric_results: dict[str, dict] = {}
    for provider, data in all_results.items():
        pairs = data["pairs"]
        m = compute_metrics(pairs)
        metric_results[provider] = m
        acc_s = _fmt_ci(m["accuracy"])
        f1_s = _fmt_ci(m["f1"])
        fpr_s = _fmt_ci(m["fpr"])
        fnr_s = _fmt_ci(m["fnr"])
        print(f"{provider:<22} {acc_s:<22} {f1_s:<22} {fpr_s:<22} {fnr_s:<22}")

    # Per-source breakdown
    print("\nPer-source breakdown:")
    sources = ["do_not_answer", "toxigen", "pku_saferlhf"]
    for src in sources:
        print(f"\n  {src}:")
        print(f"    {'Provider':<22} {'Accuracy':>10} {'F1':>10} {'FPR':>10} {'FNR':>10} {'N':>6}")
        print(f"    {'─'*62}")
        for provider, data in all_results.items():
            src_pairs = data.get("per_source", {}).get(src, [])
            if not src_pairs:
                print(f"    {provider:<22} {'n/a':>10}")
                continue
            src_metrics = compute_metrics(src_pairs)
            acc = src_metrics["accuracy"]["value"]
            f1v = src_metrics["f1"]["value"]
            fpr = src_metrics["fpr"]["value"]
            fnr = src_metrics["fnr"]["value"]
            print(f"    {provider:<22} {acc:>10.3f} {f1v:>10.3f} {fpr:>10.3f} {fnr:>10.3f} {len(src_pairs):>6}")

    # McNemar's tests
    if "mawlaia" in all_results:
        mwl_pairs = all_results["mawlaia"]["pairs"]
        gt_all = [p[0] for p in mwl_pairs]
        pred_mwl = [p[1] for p in mwl_pairs]

        print("\nMcNemar's tests:")
        competitors = [p for p in all_results if p != "mawlaia"]
        for comp in competitors:
            # Align predictions on the same items — mawlaia pairs define the base
            comp_pairs = all_results[comp]["pairs"]
            n_aligned = min(len(mwl_pairs), len(comp_pairs))
            if n_aligned == 0:
                print(f"  Mawlaia vs {comp}: insufficient data")
                continue
            gt_aligned = [mwl_pairs[i][0] for i in range(n_aligned)]
            pred_a = [mwl_pairs[i][1] for i in range(n_aligned)]
            pred_b = [comp_pairs[i][1] for i in range(n_aligned)]
            mn = mcnemar_test(gt_aligned, pred_a, pred_b)
            label = comp.replace("_moderation", "").replace("_", " ").title()
            sig = " *" if mn["significant"] else ""
            print(f"  Mawlaia vs {label}: χ²={mn['chi2']:.2f}, p={mn['p_value']:.3f}{sig}")

    print(f"\nDataset: Do-Not-Answer + ToxiGen + PKU-SafeRLHF | Seed: {SEED}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _serialize_results(all_results: dict) -> dict:
    """Convert results to JSON-serializable form (drop raw pairs list)."""
    out = {}
    for provider, data in all_results.items():
        pairs = data["pairs"]
        m = compute_metrics(pairs)
        out[provider] = {
            "metrics": {
                metric: {
                    "value": round(m[metric]["value"], 4),
                    "ci95_low": round(m[metric]["ci95_low"], 4),
                    "ci95_high": round(m[metric]["ci95_high"], 4),
                }
                for metric in ("accuracy", "f1", "fpr", "fnr")
            },
            "per_source_n": {
                src: len(pairs_)
                for src, pairs_ in data.get("per_source", {}).items()
            },
            "latency": data["latency"],
            "errors": data["errors"],
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Guardrail public benchmark")
    parser.add_argument(
        "--providers", nargs="*", default=list(PROVIDERS),
        help=f"Providers to run: {list(PROVIDERS)}",
    )
    parser.add_argument(
        "--n-items", type=int, default=None,
        help="Cap total corpus size (default: all ~2939 items)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: benchmarks/public/results/guardrail_public_YYYYMMDD.json)",
    )
    args = parser.parse_args()

    # Validate providers
    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}. Valid: {list(PROVIDERS)}")

    # Build corpus
    corpus = build_corpus(n_items=args.n_items)

    print(f"\nGuardrail Public Benchmark — {len(corpus)} samples\n")

    all_results: dict[str, dict] = {}

    with httpx.Client() as client:
        for name in args.providers:
            fn, required_keys = PROVIDERS[name]
            missing = [k for k in required_keys if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing env vars: {', '.join(missing)})")
                continue
            result = fn(corpus, client)
            all_results[name] = result
            pairs = result["pairs"]
            m = compute_metrics(pairs)
            print(
                f"  {name} → acc={m['accuracy']['value']:.3f} "
                f"F1={m['f1']['value']:.3f} "
                f"FPR={m['fpr']['value']:.3f} "
                f"FNR={m['fnr']['value']:.3f} "
                f"| p50={result['latency']['p50_ms']}ms "
                f"| errors={result['errors']}"
            )

    if not all_results:
        print("No providers ran successfully. Exiting.")
        return

    print_table(corpus, all_results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"guardrail_public_{date}.json")

    payload = {
        "product": "guardrail_public",
        "datasets": ["LibrAI/do-not-answer", "skg/toxigen-data", "PKU-Alignment/PKU-SafeRLHF"],
        "seed": SEED,
        "n_samples": len(corpus),
        "providers": _serialize_results(all_results),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
