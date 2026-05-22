"""
Statistical utilities for academic-quality benchmark evaluation.

Provides bootstrap confidence intervals, McNemar's test, stratified sampling,
span-level F1, macro F1, and pretty-printing helpers.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import numpy as np
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    stat_fn: Callable = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for a scalar statistic.

    Returns {"value": float, "ci95_low": float, "ci95_high": float}
    """
    if not values:
        return {"value": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    point_estimate = float(stat_fn(arr))

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[i] = stat_fn(sample)

    lo = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return {"value": point_estimate, "ci95_low": lo, "ci95_high": hi}


# ---------------------------------------------------------------------------
# Bootstrap CI for paired classification metrics
# ---------------------------------------------------------------------------

def _compute_metric(pairs: list[tuple[bool, bool]], metric: str) -> float:
    """Compute a single metric from (gt, pred) boolean pairs."""
    tp = sum(1 for gt, pr in pairs if gt and pr)
    fp = sum(1 for gt, pr in pairs if not gt and pr)
    fn = sum(1 for gt, pr in pairs if gt and not pr)
    tn = sum(1 for gt, pr in pairs if not gt and not pr)
    n = len(pairs)

    if metric == "accuracy":
        return (tp + tn) / n if n else float("nan")
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else 0.0
    if metric == "recall":
        return tp / (tp + fn) if (tp + fn) else 0.0
    if metric == "f1":
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    if metric == "fpr":
        return fp / (fp + tn) if (fp + tn) else 0.0
    if metric == "fnr":
        return fn / (fn + tp) if (fn + tp) else 0.0
    raise ValueError(f"Unknown metric: {metric!r}")


def bootstrap_ci_pairs(
    pairs: list[tuple[bool, bool]],
    metric: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI for a classification metric computed from (gt, pred) pairs.

    metric: one of "accuracy", "f1", "fpr", "fnr", "precision", "recall"
    Returns {"value": float, "ci95_low": float, "ci95_high": float}
    """
    valid_metrics = {"accuracy", "f1", "fpr", "fnr", "precision", "recall"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}")
    if not pairs:
        return {"value": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    arr = np.array(pairs, dtype=object)
    rng = np.random.default_rng(seed)
    point_estimate = _compute_metric(pairs, metric)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        sample = [pairs[j] for j in indices]
        boot_stats[i] = _compute_metric(sample, metric)

    lo = float(np.percentile(boot_stats, 2.5))
    hi = float(np.percentile(boot_stats, 97.5))
    return {"value": point_estimate, "ci95_low": lo, "ci95_high": hi}


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    gt: list[bool],
    pred_a: list[bool],
    pred_b: list[bool],
) -> dict:
    """
    McNemar's test comparing two classifiers A and B against ground truth.

    b = A correct, B wrong
    c = A wrong, B correct
    Edwards continuity correction applied when b+c < 25.

    Returns {"b": int, "c": int, "chi2": float, "p_value": float, "significant": bool}
    """
    if not (len(gt) == len(pred_a) == len(pred_b)):
        raise ValueError("gt, pred_a, pred_b must have the same length")

    b = sum(1 for g, a, bv in zip(gt, pred_a, pred_b) if a == g and bv != g)
    c = sum(1 for g, a, bv in zip(gt, pred_a, pred_b) if a != g and bv == g)

    use_correction = (b + c) < 25
    if use_correction:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    else:
        chi2_stat = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0.0

    p_value = float(1 - chi2.cdf(chi2_stat, df=1)) if (b + c) > 0 else 1.0

    return {
        "b": b,
        "c": c,
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    items: list,
    key_fn: Callable,
    n_total: int,
    seed: int = 42,
) -> list:
    """
    Proportional stratified sampling.

    Preserves category distribution of key_fn(item) across items.
    If n_total >= len(items), returns a shuffled copy of all items.
    """
    rng = np.random.default_rng(seed)

    if n_total >= len(items):
        result = list(items)
        rng.shuffle(result)
        return result

    # Group by stratum key
    strata: dict[str, list] = {}
    for item in items:
        key = str(key_fn(item))
        strata.setdefault(key, []).append(item)

    # Proportional allocation
    total = len(items)
    sampled = []
    remainder: list[tuple[float, str]] = []
    allocated = 0

    for key, group in strata.items():
        exact = n_total * len(group) / total
        floor_n = int(exact)
        allocated += floor_n
        remainder.append((exact - floor_n, key))

    # Distribute remaining slots by largest fractional part
    leftover = n_total - allocated
    remainder.sort(key=lambda x: -x[0])
    extra_keys = {key for _, key in remainder[:leftover]}

    allocation: dict[str, int] = {}
    for frac, key in remainder:
        allocation[key] = int(n_total * len(strata[key]) / total)
    for key in extra_keys:
        allocation[key] = allocation.get(key, 0) + 1

    for key, group in strata.items():
        n = min(allocation.get(key, 0), len(group))
        indices = rng.choice(len(group), size=n, replace=False)
        sampled.extend(group[i] for i in indices)

    # Shuffle the final result
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Span-level F1
# ---------------------------------------------------------------------------

def span_f1(
    gt_entities: list[dict],
    pred_entities: list[dict],
    entity_type: str,
) -> dict:
    """
    Span F1 for a single entity type.

    Matching is by normalized value (lowercase + strip). Type must also match.
    Returns {"tp", "fp", "fn", "precision", "recall", "f1"}
    """
    def normalize(v: str) -> str:
        return v.lower().strip()

    gt_values = [normalize(e["value"]) for e in gt_entities if e.get("type") == entity_type]
    pred_values = [normalize(e["value"]) for e in pred_entities if e.get("type") == entity_type]

    gt_counter = Counter(gt_values)
    pred_counter = Counter(pred_values)

    tp = sum((gt_counter & pred_counter).values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gt_counter.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score, 4),
    }


# ---------------------------------------------------------------------------
# Macro F1
# ---------------------------------------------------------------------------

def macro_f1(per_type_f1: dict[str, dict]) -> float:
    """
    Macro-average F1 across entity types.

    Skips types that had zero ground-truth entities (indicated by tp=fn=0
    when precision is also 0, i.e. the type was never present in the gold set).
    Uses only types with at least one ground-truth entity (tp + fn > 0).
    """
    scores = []
    for etype, metrics in per_type_f1.items():
        tp = metrics.get("tp", 0)
        fn = metrics.get("fn", 0)
        if (tp + fn) > 0:
            scores.append(metrics["f1"])
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_result(name: str, metrics: dict, ci_key: str = "f1") -> str:
    """
    Pretty-print benchmark result.

    Example output:
        mawlaia: F1=0.9577 [0.9412, 0.9721] Prec=0.961 Rec=0.955
    """
    f1_val = metrics.get(ci_key, metrics.get("f1", float("nan")))
    lo = metrics.get("ci95_low", float("nan"))
    hi = metrics.get("ci95_high", float("nan"))
    prec = metrics.get("precision", float("nan"))
    rec = metrics.get("recall", float("nan"))

    def fmt(v: float, ndigits: int = 4) -> str:
        return f"{v:.{ndigits}f}" if not math.isnan(v) else "N/A"

    return (
        f"{name}: F1={fmt(f1_val)} [{fmt(lo)}, {fmt(hi)}]"
        f" Prec={fmt(prec, 3)} Rec={fmt(rec, 3)}"
    )
