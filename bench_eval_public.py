"""
LLM Judge Public Benchmark — MT-Bench Human Pairwise Judgments
Providers: Mawlaia Eval, GPT-4o-mini judge

Measures pairwise accuracy (P(model preference == human winner)) and
Pearson r between score differences and human preference direction.

Usage:
  python3 benchmarks/public/bench_eval_public.py [--providers mawlaia gpt4o_mini_judge] [--n-items 400]

Required env vars (only for providers you want):
  MWL_API_KEY, OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Path resolution: allow running from project root or from benchmarks/public/
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))          # benchmarks/public/ (for bench_stats)
   # benchmarks/ (for bench_common)

from bench_common import (
    MAWLAIA_BASE, OPENAI_BASE,
    env, mawlaia_headers, openai_headers,
    timed_post, latency_stats, save_results,
)
from bench_stats import bootstrap_ci, mcnemar_test, stratified_sample

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"

# Margin for declaring a winner vs tie (score difference threshold)
SCORE_MARGIN = 0.1

GPT4O_MINI_JUDGE_PROMPT = """\
Given the question and two responses A and B, which is better?
Question: {prompt}
Response A: {response_a}
Response B: {response_b}
Return JSON: {{"winner": "A" or "B" or "tie", "score_a": 1-5, "score_b": 1-5}}"""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _parse_conversation(raw) -> list[dict]:
    """Parse conversation field — may be a list already or a JSON string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def load_mt_bench(max_items: int = 400) -> list[dict]:
    """
    Load lmsys/mt_bench_human_judgments, filter to turn==1, sample up to max_items.

    Returns list of dicts with keys:
      question_id, model_a, model_b, winner, prompt, response_a, response_b
    """
    print("  Loading MT-Bench Human Judgments dataset...")
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")

    pairs: list[dict] = []
    skipped = 0

    for row in ds:
        # Filter to first turn only
        if int(row.get("turn", 0)) != 1:
            continue

        winner = str(row.get("winner", "")).strip()
        if winner not in ("model_a", "model_b", "tie"):
            skipped += 1
            continue

        conv_a = _parse_conversation(row.get("conversation_a", []))
        conv_b = _parse_conversation(row.get("conversation_b", []))

        # Need at least 2 turns (user + assistant) in each conversation
        if len(conv_a) < 2 or len(conv_b) < 2:
            skipped += 1
            continue

        # Extract user prompt (turn 0) and assistant responses (turn 1)
        user_turn_a = next(
            (t for t in conv_a if str(t.get("role", "")).lower() == "user"), None
        )
        asst_turn_a = next(
            (t for t in conv_a if str(t.get("role", "")).lower() == "assistant"), None
        )
        asst_turn_b = next(
            (t for t in conv_b if str(t.get("role", "")).lower() == "assistant"), None
        )

        if not user_turn_a or not asst_turn_a or not asst_turn_b:
            skipped += 1
            continue

        prompt = str(user_turn_a.get("content", "")).strip()
        response_a = str(asst_turn_a.get("content", "")).strip()
        response_b = str(asst_turn_b.get("content", "")).strip()

        if not prompt or not response_a or not response_b:
            skipped += 1
            continue

        pairs.append({
            "question_id": row.get("question_id"),
            "model_a": row.get("model_a", ""),
            "model_b": row.get("model_b", ""),
            "winner": winner,
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
        })

    print(f"    Raw turn-1 pairs: {len(pairs)} (skipped {skipped})")

    # Stratified sample by winner (balance tie/non-tie)
    sampled = stratified_sample(
        pairs,
        key_fn=lambda x: x["winner"],
        n_total=max_items,
        seed=SEED,
    )

    n_tie = sum(1 for x in sampled if x["winner"] == "tie")
    n_non_tie = len(sampled) - n_tie
    print(f"    Sampled {len(sampled)} pairs: {n_non_tie} non-tie, {n_tie} tie")
    return sampled


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _preference_from_scores(score_a: float, score_b: float, margin: float = SCORE_MARGIN) -> str:
    """Convert two scores to a preference: 'model_a', 'model_b', or 'tie'."""
    if score_a > score_b + margin:
        return "model_a"
    if score_b > score_a + margin:
        return "model_b"
    return "tie"


def _human_direction(winner: str) -> float | None:
    """Convert human winner to numeric direction for correlation: +1, -1, or None for tie."""
    if winner == "model_a":
        return 1.0
    if winner == "model_b":
        return -1.0
    return None


# ---------------------------------------------------------------------------
# Provider runners
# ---------------------------------------------------------------------------

def run_mawlaia(pairs: list[dict], client: httpx.Client) -> dict:
    """
    Mawlaia Eval API: POST /eval/score for each response separately.
    preference = model_a if score_a > score_b + 0.1, model_b if vice-versa, else tie.
    """
    print("\n  Running Mawlaia Eval Judge...")
    latencies: list[float] = []
    preferences: list[str] = []   # model's preference
    human_winners: list[str] = []  # ground-truth (all, including tie)
    score_diffs: list[float] = []   # score_a - score_b
    human_directions: list[float] = []  # +1 / -1 (non-tie only)
    errors = 0

    for i, pair in enumerate(pairs):
        if i % 50 == 0 and i > 0:
            print(f"    [{i}/{len(pairs)}] errors={errors}")
        try:
            time.sleep(1.0)  # avoid 429 rate limit on eval/score API
            criteria = "Rate response quality: accuracy, relevance, clarity, and completeness."
            payload = {
                "scorer": "llm_judge",
                "cases": [
                    {"input": pair["prompt"], "output": pair["response_a"], "expected": ""},
                    {"input": pair["prompt"], "output": pair["response_b"], "expected": ""},
                ],
                "criteria": criteria,
            }
            for _attempt in range(3):
                try:
                    resp, ms = timed_post(client, f"{MAWLAIA_BASE}/eval/score", mawlaia_headers(), payload)
                    break
                except Exception as retry_exc:
                    if "429" in str(retry_exc) and _attempt < 2:
                        time.sleep(2 ** _attempt * 2)
                        continue
                    raise

            latencies.append(ms)
            results = resp.get("results", [{}, {}])
            score_a = float(results[0].get("score", 0.5))
            score_b = float(results[1].get("score", 0.5))
            pref = _preference_from_scores(score_a, score_b)
            preferences.append(pref)
            human_winners.append(pair["winner"])
            score_diffs.append(score_a - score_b)
            hdir = _human_direction(pair["winner"])
            if hdir is not None:
                human_directions.append(hdir)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{pair['question_id']}] {exc}")
            # Record as tie to keep alignment
            preferences.append("tie")
            human_winners.append(pair["winner"])
            score_diffs.append(0.0)

    return {
        "preferences": preferences,
        "human_winners": human_winners,
        "score_diffs": score_diffs,
        "latency": latency_stats(latencies),
        "errors": errors,
    }


def run_gpt4o_mini_judge(pairs: list[dict], client: httpx.Client) -> dict:
    """
    GPT-4o-mini as judge: single comparative call per pair (A vs B), derive preference.
    """
    print("\n  Running GPT-4o-mini Judge...")
    latencies: list[float] = []
    preferences: list[str] = []
    human_winners: list[str] = []
    score_diffs: list[float] = []
    errors = 0

    for i, pair in enumerate(pairs):
        if i % 50 == 0 and i > 0:
            print(f"    [{i}/{len(pairs)}] errors={errors}")
        time.sleep(1.5)  # avoid 429 rate limit on OpenAI
        try:
            body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": GPT4O_MINI_JUDGE_PROMPT.format(
                            prompt=pair["prompt"][:1000],
                            response_a=pair["response_a"][:500],
                            response_b=pair["response_b"][:500],
                        ),
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 128,
            }
            resp, ms = timed_post(client, f"{OPENAI_BASE}/chat/completions", openai_headers(), body)
            raw = resp["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            winner_label = parsed.get("winner", "tie").strip().upper()
            score_a = float(parsed.get("score_a", 3.0))
            score_b = float(parsed.get("score_b", 3.0))
            if winner_label == "A":
                pref = "model_a"
            elif winner_label == "B":
                pref = "model_b"
            else:
                pref = "tie"
            latencies.append(ms)
            preferences.append(pref)
            human_winners.append(pair["winner"])
            score_diffs.append(score_a - score_b)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{pair['question_id']}] {exc}")
            preferences.append("tie")
            human_winners.append(pair["winner"])
            score_diffs.append(0.0)

    return {
        "preferences": preferences,
        "human_winners": human_winners,
        "score_diffs": score_diffs,
        "latency": latency_stats(latencies),
        "errors": errors,
    }


def run_mawlaia_pairwise(pairs: list[dict], client: httpx.Client) -> dict:
    """
    Mawlaia pairwise_judge scorer: single A-vs-B LLM call per pair.
    Fixes the systematic sign-flip that plagued absolute score differencing.
    """
    print("\n  Running Mawlaia Pairwise Judge...")
    latencies: list[float] = []
    preferences: list[str] = []
    human_winners: list[str] = []
    score_diffs: list[float] = []
    errors = 0

    for i, pair in enumerate(pairs):
        if i % 50 == 0 and i > 0:
            print(f"    [{i}/{len(pairs)}] errors={errors}")
        time.sleep(1.2)  # avoid rate limits
        try:
            payload = {
                "scorer": "pairwise_judge",
                "criteria": "Rate response quality: accuracy, relevance, clarity, and completeness.",
                "cases": [
                    {"input": pair["prompt"], "output": pair["response_a"], "expected": ""},
                    {"input": pair["prompt"], "output": pair["response_b"], "expected": ""},
                ],
            }
            for _attempt in range(3):
                try:
                    resp, ms = timed_post(client, f"{MAWLAIA_BASE}/eval/score", mawlaia_headers(), payload)
                    break
                except Exception as retry_exc:
                    if "429" in str(retry_exc) and _attempt < 2:
                        time.sleep(2 ** _attempt * 3)
                        continue
                    raise

            latencies.append(ms)
            results = resp.get("results", [{}, {}])
            score_a = float(results[0].get("score", 0.5))
            score_b = float(results[1].get("score", 0.5))
            # winner is in the rationale of results[0]
            rationale = results[0].get("rationale", "")
            if "Winner: A" in rationale:
                pref = "model_a"
            elif "Winner: B" in rationale:
                pref = "model_b"
            else:
                pref = _preference_from_scores(score_a, score_b)
            preferences.append(pref)
            human_winners.append(pair["winner"])
            score_diffs.append(score_a - score_b)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{pair['question_id']}] {exc}")
            preferences.append("tie")
            human_winners.append(pair["winner"])
            score_diffs.append(0.0)

    return {
        "preferences": preferences,
        "human_winners": human_winners,
        "score_diffs": score_diffs,
        "latency": latency_stats(latencies),
        "errors": errors,
    }


PROVIDERS = {
    "mawlaia": (run_mawlaia, ["MWL_API_KEY"]),
    "mawlaia_pairwise": (run_mawlaia_pairwise, ["MWL_API_KEY"]),
    "gpt4o_mini_judge": (run_gpt4o_mini_judge, ["OPENAI_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    xarr = np.array(xs, dtype=float)
    yarr = np.array(ys, dtype=float)
    mx, my = xarr.mean(), yarr.mean()
    num = float(((xarr - mx) * (yarr - my)).sum())
    denom = float(np.sqrt(((xarr - mx) ** 2).sum() * ((yarr - my) ** 2).sum()))
    return num / denom if denom > 0 else float("nan")


def compute_pairwise_metrics(data: dict) -> dict:
    """
    Compute pairwise accuracy on non-tie ground-truth pairs, plus Pearson r.

    Returns dict with:
      - pairwise_acc: {"value", "ci95_low", "ci95_high"}
      - pearson_r: {"value", "ci95_low", "ci95_high"}
      - n_non_tie: int
      - n_total: int
    """
    preferences = data["preferences"]
    human_winners = data["human_winners"]
    score_diffs = data["score_diffs"]

    # Pairwise accuracy: exclude ties in ground truth
    correct_flags: list[float] = []
    for pref, hw in zip(preferences, human_winners):
        if hw == "tie":
            continue  # exclude ground-truth ties
        correct_flags.append(1.0 if pref == hw else 0.0)

    acc_ci = bootstrap_ci(correct_flags, stat_fn=np.mean, seed=SEED) if correct_flags else {
        "value": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")
    }

    # Pearson r: score_diff vs human direction (+1/-1), exclude ties
    paired_diffs: list[float] = []
    paired_dirs: list[float] = []
    for diff, hw in zip(score_diffs, human_winners):
        hdir = _human_direction(hw)
        if hdir is None:
            continue
        paired_diffs.append(diff)
        paired_dirs.append(hdir)

    # Bootstrap Pearson r
    if len(paired_diffs) >= 2:
        point_r = _pearson_r(paired_diffs, paired_dirs)

        rng = np.random.default_rng(SEED)
        boot_rs = []
        arr_diff = np.array(paired_diffs)
        arr_dir = np.array(paired_dirs)
        n = len(paired_diffs)
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            r = _pearson_r(arr_diff[idx].tolist(), arr_dir[idx].tolist())
            if not math.isnan(r):
                boot_rs.append(r)
        pearson_ci = {
            "value": point_r,
            "ci95_low": float(np.percentile(boot_rs, 2.5)) if boot_rs else float("nan"),
            "ci95_high": float(np.percentile(boot_rs, 97.5)) if boot_rs else float("nan"),
        }
    else:
        pearson_ci = {
            "value": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")
        }

    return {
        "pairwise_acc": acc_ci,
        "pearson_r": pearson_ci,
        "n_non_tie": len(correct_flags),
        "n_total": len(preferences),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_ci(ci: dict, decimals: int = 3) -> tuple[str, str]:
    """Return (value_str, ci_str) formatted."""
    v = ci.get("value", float("nan"))
    lo = ci.get("ci95_low", float("nan"))
    hi = ci.get("ci95_high", float("nan"))
    fmt = f"{{:.{decimals}f}}"
    nan_s = "N/A"

    def fmtv(x: float) -> str:
        return fmt.format(x) if not math.isnan(x) else nan_s

    return fmtv(v), f"[{fmtv(lo)}, {fmtv(hi)}]"


def print_table(pairs: list[dict], all_results: dict[str, dict]):
    n_total = len(pairs)
    n_non_tie = sum(1 for p in pairs if p["winner"] != "tie")

    print(f"\n── LLM Judge — MT-Bench Human Judgments (N={n_non_tie} non-tie pairs) ─────────────────")
    print(
        f"{'Provider':<20} {'Pairwise Acc':>14} {'[95% CI]':>18}    "
        f"{'Pearson r':>10} {'[95% CI]':>18}"
    )
    print("─" * 86)

    metric_results: dict[str, dict] = {}
    for provider, data in all_results.items():
        m = compute_pairwise_metrics(data)
        metric_results[provider] = m
        acc_v, acc_ci = _fmt_ci(m["pairwise_acc"])
        r_v, r_ci = _fmt_ci(m["pearson_r"])
        print(f"{provider:<20} {acc_v:>14} {acc_ci:>18}    {r_v:>10} {r_ci:>18}")

    # McNemar's tests: Mawlaia variants vs GPT-4o-mini
    mawlaia_providers = [p for p in all_results if p.startswith("mawlaia")]
    if mawlaia_providers and "gpt4o_mini_judge" in all_results:
        best_mwl_key = "mawlaia_pairwise" if "mawlaia_pairwise" in all_results else "mawlaia"
        mwl = all_results[best_mwl_key]
        gpt = all_results["gpt4o_mini_judge"]
    if "mawlaia" in all_results and "gpt4o_mini_judge" in all_results:
        mwl = all_results["mawlaia"]
        gpt = all_results["gpt4o_mini_judge"]

        # Align: use only pairs where ground truth is NOT a tie (for accuracy comparison)
        n_aligned = min(len(mwl["preferences"]), len(gpt["preferences"]))
        gt_list = mwl["human_winners"][:n_aligned]
        pred_mwl = mwl["preferences"][:n_aligned]
        pred_gpt = gpt["preferences"][:n_aligned]

        # McNemar needs boolean correct/wrong vectors on non-tie subset
        gt_bool: list[bool] = []
        preds_mwl_bool: list[bool] = []
        preds_gpt_bool: list[bool] = []
        for hw, pm, pg in zip(gt_list, pred_mwl, pred_gpt):
            if hw == "tie":
                continue
            gt_bool.append(True)        # dummy — we pass correctness booleans
            preds_mwl_bool.append(pm == hw)
            preds_gpt_bool.append(pg == hw)

        if gt_bool:
            # McNemar's: compare two binary prediction vectors against "ground truth" of all-True
            mn = mcnemar_test(
                gt=[True] * len(gt_bool),
                pred_a=preds_mwl_bool,
                pred_b=preds_gpt_bool,
            )
            sig = " *" if mn["significant"] else ""
            print(
                f"\nMcNemar's test (Mawlaia vs GPT-4o-mini): "
                f"χ²={mn['chi2']:.2f}, p={mn['p_value']:.3f}{sig}"
            )
        else:
            print("\nMcNemar's test: insufficient non-tie pairs")

    print(
        f"Dataset: lmsys/mt_bench_human_judgments | "
        f"N={n_total} sampled (excl. ties for acc) | Seed: {SEED}"
    )


# ---------------------------------------------------------------------------
# Serialize results
# ---------------------------------------------------------------------------

def _serialize_results(all_results: dict) -> dict:
    out = {}
    for provider, data in all_results.items():
        m = compute_pairwise_metrics(data)
        out[provider] = {
            "metrics": {
                "pairwise_accuracy": {k: round(v, 4) if not math.isnan(v) else None
                                      for k, v in m["pairwise_acc"].items()},
                "pearson_r": {k: round(v, 4) if not math.isnan(v) else None
                              for k, v in m["pearson_r"].items()},
                "n_non_tie": m["n_non_tie"],
                "n_total": m["n_total"],
            },
            "latency": data["latency"],
            "errors": data["errors"],
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM judge public benchmark (MT-Bench)")
    parser.add_argument(
        "--providers", nargs="*", default=list(PROVIDERS),
        help=f"Providers to run: {list(PROVIDERS)}",
    )
    parser.add_argument(
        "--n-items", type=int, default=400,
        help="Max pairwise comparisons to sample (default: 400)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: benchmarks/public/results/eval_public_YYYYMMDD.json)",
    )
    args = parser.parse_args()

    # Validate providers
    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}. Valid: {list(PROVIDERS)}")

    # Load dataset
    pairs = load_mt_bench(max_items=args.n_items)

    print(f"\nEval Public Benchmark — {len(pairs)} pairwise comparisons\n")

    all_results: dict[str, dict] = {}

    with httpx.Client() as client:
        for name in args.providers:
            fn, required_keys = PROVIDERS[name]
            missing = [k for k in required_keys if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing env vars: {', '.join(missing)})")
                continue
            result = fn(pairs, client)
            all_results[name] = result
            m = compute_pairwise_metrics(result)
            print(
                f"  {name} → pairwise_acc={m['pairwise_acc']['value']:.3f} "
                f"pearson_r={m['pearson_r']['value']:.3f} "
                f"| p50={result['latency']['p50_ms']}ms "
                f"| errors={result['errors']}"
            )

    if not all_results:
        print("No providers ran successfully. Exiting.")
        return

    print_table(pairs, all_results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"eval_public_{date}.json")

    payload = {
        "product": "eval_public",
        "dataset": "lmsys/mt_bench_human_judgments",
        "seed": SEED,
        "n_samples": len(pairs),
        "n_non_tie": sum(1 for p in pairs if p["winner"] != "tie"),
        "providers": _serialize_results(all_results),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
