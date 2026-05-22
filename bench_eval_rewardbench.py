"""
LLM Judge Benchmark — RewardBench
Dataset: allenai/reward-bench (filtered split)
Providers: Mawlaia pairwise_judge, GPT-4o-mini direct judge

RewardBench format: (prompt, chosen, rejected) — judge must pick chosen over rejected.
Pairwise accuracy = P(judge prefers chosen).

Position randomized to test for position bias.

Usage:
  python3 benchmarks/public/bench_eval_rewardbench.py [--providers mawlaia_pairwise gpt4o_mini_judge] [--n-items 500]

Required env vars:
  MWL_API_KEY, OPENAI_API_KEY
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
import numpy as np

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


from bench_common import MAWLAIA_BASE, OPENAI_BASE, env, mawlaia_headers, openai_headers, latency_stats
from bench_stats import bootstrap_ci, mcnemar_test

SEED = 42
RESULTS_DIR = _THIS_DIR / "results"
DATASET_NAME = "allenai/reward-bench"

# Subset groups for reporting
SUBSET_GROUPS = {
    "chat":      ["alpacaeval-easy", "alpacaeval-hard", "alpacaeval-length", "mt-bench-easy", "mt-bench-med"],
    "chat_hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "safety":    ["refusals-dangerous", "refusals-offensive", "xstest-should-respond", "xstest-should-refuse",
                  "donotanswer"],
    "reasoning": ["hep-python", "hep-go", "hep-cpp", "hep-js", "hep-rust", "hep-java", "math-prm"],
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_rewardbench(n_items: int = 500) -> list[dict]:
    """Load RewardBench filtered split. Returns list of (prompt, chosen, rejected, subset)."""
    print(f"Loading {DATASET_NAME} (filtered split)…")
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="filtered")
    all_items = [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "subset": row["subset"],
        }
        for row in ds
    ]
    total = len(all_items)
    print(f"  {total} items across {len(set(x['subset'] for x in all_items))} subsets")

    if n_items < total:
        # Stratified sample by subset
        by_subset: dict[str, list] = defaultdict(list)
        for item in all_items:
            by_subset[item["subset"]].append(item)

        rng = np.random.default_rng(SEED)
        sampled: list[dict] = []
        per_subset = max(1, n_items // len(by_subset))
        for subset, items in by_subset.items():
            n = min(len(items), per_subset)
            idx = rng.choice(len(items), size=n, replace=False)
            sampled.extend(items[i] for i in idx)

        # Fill remaining slots from largest subsets
        remaining = n_items - len(sampled)
        if remaining > 0:
            all_remaining = [x for x in all_items if x not in sampled]
            rng.shuffle(all_remaining)
            sampled.extend(all_remaining[:remaining])

        sampled = sampled[:n_items]
        print(f"  Sampled {len(sampled)} items (stratified by subset)")
        return sampled

    return all_items


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

PAIRWISE_CRITERIA = "overall quality, helpfulness, accuracy, and clarity"


def run_mawlaia_pairwise(pairs: list[dict], client: httpx.Client) -> dict:
    """
    Use Mawlaia pairwise_judge scorer. Randomize A/B position.
    Ground truth: chosen is always the better response.
    """
    print(f"\n  Running Mawlaia Pairwise Judge ({len(pairs)} pairs)…")
    rng = np.random.default_rng(SEED + 1)
    latencies, results, errors = [], [], 0

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(pairs)}] errors={errors}")
        time.sleep(1.2)

        # Randomize position
        chosen_is_a = bool(rng.integers(2))
        resp_a = pair["chosen"] if chosen_is_a else pair["rejected"]
        resp_b = pair["rejected"] if chosen_is_a else pair["chosen"]

        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                r = client.post(
                    f"{MAWLAIA_BASE}/eval/score",
                    headers=mawlaia_headers(),
                    json={
                        "scorer": "pairwise_judge",
                        "cases": [
                            {"input": pair["prompt"], "output": resp_a},
                            {"input": pair["prompt"], "output": resp_b},
                        ],
                        "criteria": PAIRWISE_CRITERIA,
                    },
                    timeout=30,
                )
                ms = (time.perf_counter() - t0) * 1000
                r.raise_for_status()
                latencies.append(ms)
                body = r.json()
                # Parse winner from rationale or scores
                res_list = body.get("results", [])
                winner = "TIE"
                if res_list:
                    rationale = str(res_list[0].get("rationale", ""))
                    if "Winner: A" in rationale or "winner: a" in rationale.lower():
                        winner = "A"
                    elif "Winner: B" in rationale or "winner: b" in rationale.lower():
                        winner = "B"
                    else:
                        # Fall back to score comparison
                        scores = [r2.get("score", 0.5) for r2 in res_list[:2]]
                        if len(scores) == 2:
                            if scores[0] > scores[1] + 0.01:
                                winner = "A"
                            elif scores[1] > scores[0] + 0.01:
                                winner = "B"

                # Determine if prediction is correct
                if winner == "TIE":
                    correct = False  # ties count as wrong
                elif winner == "A":
                    correct = chosen_is_a
                else:
                    correct = not chosen_is_a

                results.append({"correct": correct, "subset": pair["subset"], "winner": winner})
                break
            except Exception as exc:
                if attempt == 2:
                    errors += 1
                    if errors <= 5:
                        print(f"    WARN [{i}] {exc}")
                    results.append({"correct": False, "subset": pair["subset"], "winner": "ERR"})
                else:
                    time.sleep(2 ** attempt * 2)

    return {"results": results, "latency": latency_stats(latencies), "errors": errors}


def run_gpt4o_mini(pairs: list[dict], client: httpx.Client) -> dict:
    """Direct A-vs-B GPT-4o-mini judge with randomized position."""
    print(f"\n  Running GPT-4o-mini Judge ({len(pairs)} pairs)…")
    rng = np.random.default_rng(SEED + 2)
    latencies, results, errors = [], [], 0

    JUDGE_PROMPT = (
        "You are an expert judge evaluating two AI assistant responses.\n"
        "Given the prompt and two responses (A and B), determine which is better "
        "in terms of {criteria}.\n\n"
        "Prompt: {prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        'Return JSON only: {{"winner": "A" or "B" or "TIE", "reason": "one sentence"}}'
    )

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(pairs)}] errors={errors}")
        time.sleep(1.2)

        chosen_is_a = bool(rng.integers(2))
        resp_a = pair["chosen"] if chosen_is_a else pair["rejected"]
        resp_b = pair["rejected"] if chosen_is_a else pair["chosen"]

        try:
            prompt_text = JUDGE_PROMPT.format(
                criteria=PAIRWISE_CRITERIA,
                prompt=pair["prompt"][:1000],
                response_a=resp_a[:1000],
                response_b=resp_b[:1000],
            )
            t0 = time.perf_counter()
            r = client.post(
                f"{OPENAI_BASE}/chat/completions",
                headers=openai_headers(),
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 128,
                },
                timeout=30,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            data = json.loads(r.json()["choices"][0]["message"]["content"])
            winner = str(data.get("winner", "TIE")).upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE"

            if winner == "TIE":
                correct = False
            elif winner == "A":
                correct = chosen_is_a
            else:
                correct = not chosen_is_a

            results.append({"correct": correct, "subset": pair["subset"], "winner": winner})
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{i}] {exc}")
            results.append({"correct": False, "subset": pair["subset"], "winner": "ERR"})

    return {"results": results, "latency": latency_stats(latencies), "errors": errors}


PROVIDERS = {
    "mawlaia_pairwise": (run_mawlaia_pairwise, ["MWL_API_KEY"]),
    "gpt4o_mini_judge": (run_gpt4o_mini,        ["OPENAI_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    overall_correct = [1.0 if r["correct"] else 0.0 for r in results]
    overall_ci = bootstrap_ci(overall_correct, seed=SEED)

    by_group = {}
    for group, subsets in SUBSET_GROUPS.items():
        group_results = [r for r in results if r["subset"] in subsets]
        if group_results:
            ci = bootstrap_ci([1.0 if r["correct"] else 0.0 for r in group_results], seed=SEED)
            by_group[group] = {"value": round(ci["value"], 4), "ci95_low": round(ci["ci95_low"], 4),
                               "ci95_high": round(ci["ci95_high"], 4), "n": len(group_results)}
        else:
            by_group[group] = None

    return {"pairwise_acc": overall_ci, "by_group": by_group}


# ---------------------------------------------------------------------------
# Table + main
# ---------------------------------------------------------------------------

def print_table(all_results: dict, n_items: int):
    print(f"\n── LLM Judge — RewardBench (N={n_items}) ───────────────────────────────────")
    print(f"{'Provider':<22} {'Overall Acc':>12} {'[95% CI]':>16} {'Chat':>8} {'Chat-H':>8} {'Safety':>8} {'Reason':>8}")
    print("-" * 86)
    for name, data in all_results.items():
        m = data["metrics"]
        acc = m["pairwise_acc"]
        bg = m["by_group"]
        chat = bg.get("chat", {}) or {}
        chath = bg.get("chat_hard", {}) or {}
        safety = bg.get("safety", {}) or {}
        reason = bg.get("reasoning", {}) or {}
        ci = f"[{acc['ci95_low']:.3f},{acc['ci95_high']:.3f}]"
        print(f"{name:<22} {acc['value']:>12.3f} {ci:>16} "
              f"{chat.get('value', 0):>8.3f} {chath.get('value', 0):>8.3f} "
              f"{safety.get('value', 0):>8.3f} {reason.get('value', 0):>8.3f}")
    print("-" * 86)
    print(f"Dataset: {DATASET_NAME} | N={n_items} stratified | Seed: {SEED}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}")

    pairs = load_rewardbench(n_items=args.n_items)
    print(f"\nEval RewardBench — {len(pairs)} pairs\n")

    all_results: dict[str, dict] = {}
    with httpx.Client() as client:
        for name in args.providers:
            fn, required = PROVIDERS[name]
            missing = [k for k in required if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing: {', '.join(missing)})")
                continue
            raw = fn(pairs, client)
            metrics = compute_metrics(raw["results"])
            acc = metrics["pairwise_acc"]["value"]
            print(f"  {name} → pairwise_acc={acc:.3f} | p50={raw['latency'].get('p50_ms')}ms | errors={raw['errors']}")
            all_results[name] = {**raw, "metrics": metrics}
            del all_results[name]["results"]

    if not all_results:
        print("No providers ran.")
        return

    print_table(all_results, len(pairs))

    if "mawlaia_pairwise" in all_results and "gpt4o_mini_judge" in all_results:
        print("\nMcNemar's tests:")
        # Would need original results lists, skip for now

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"eval_rewardbench_{date}.json")

    payload = {
        "product": "eval",
        "dataset": DATASET_NAME,
        "n_samples": len(pairs),
        "seed": SEED,
        "providers": {n: {k: v for k, v in d.items() if k != "results"} for n, d in all_results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
