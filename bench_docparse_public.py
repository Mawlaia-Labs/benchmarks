"""
Document Parsing Public Benchmark — CORD-v2 Receipt Dataset
Providers: Mawlaia DocParse, LlamaParse, OpenAI GPT-4o-mini vision

Architecture:
  - LlamaParse: image → structured JSON (end-to-end, direct image upload)
  - OpenAI vision: base64 image → GPT-4o-mini structured extraction
  - Mawlaia pipeline: base64 image → GPT-4o-mini OCR text → Mawlaia DocParse

Target field: total_price (grand total on receipt)

Usage:
  python3 benchmarks/public/bench_docparse_public.py [--n-items 100] [--providers llamaparse openai_vision mawlaia_pipeline]

Required env vars:
  LLAMA_PARSE_API_KEY, OPENAI_API_KEY, MWL_API_KEY
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


from bench_common import MAWLAIA_BASE, OPENAI_BASE, env, mawlaia_headers, openai_headers, latency_stats
from bench_stats import bootstrap_ci, stratified_sample

SEED = 42
RESULTS_DIR = _THIS_DIR / "results"
DATASET_NAME = "naver-clova-ix/cord-v2"

# Target field mapping: our name → CORD gt_parse path
TARGET_FIELDS = {
    "total_price": ["total", "total_price"],
    "subtotal_price": ["sub_total", "subtotal_price"],
}

# LlamaParse endpoints
LLAMA_PARSE_UPLOAD_URL = "https://api.cloud.llamaindex.ai/api/parsing/upload"
LLAMA_PARSE_JOB_URL = "https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cord(n_items: int = 100) -> list[dict]:
    """
    Load CORD-v2 test split. Returns list of dicts:
      {"image_bytes": bytes, "image_format": "png", "ground_truth": dict}
    where ground_truth contains total_price and subtotal_price (string, e.g. "60.000").
    """
    print(f"Loading {DATASET_NAME} test split...")
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="test")
    print(f"  {len(ds)} items in test split")

    items: list[dict] = []
    skipped = 0

    for row in ds:
        img = row["image"]
        gt_raw = row["ground_truth"]

        if isinstance(gt_raw, str):
            try:
                gt = json.loads(gt_raw)
            except (json.JSONDecodeError, ValueError):
                skipped += 1
                continue
        else:
            gt = gt_raw

        gt_parse = gt.get("gt_parse", {})
        if not gt_parse:
            skipped += 1
            continue

        # Extract target field values
        fields: dict[str, str | None] = {}
        for field_name, path in TARGET_FIELDS.items():
            obj = gt_parse
            for key in path:
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    obj = None
                    break
            fields[field_name] = str(obj).strip() if obj is not None else None

        # Skip items missing total_price (core field)
        if not fields.get("total_price"):
            skipped += 1
            continue

        # Convert PIL image to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        items.append({
            "image_bytes": image_bytes,
            "image_format": "png",
            "ground_truth": fields,
        })

        if len(items) >= max(n_items * 2, 200):
            break

    print(f"  Collected {len(items)} items (skipped {skipped})")

    if n_items < len(items):
        # Deterministic shuffle + truncate
        import numpy as np
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(items), size=n_items, replace=False)
        items = [items[i] for i in sorted(indices)]
        print(f"  Sampled {len(items)} items")

    return items


# ---------------------------------------------------------------------------
# Field accuracy helpers
# ---------------------------------------------------------------------------

def _normalize_amount(s: str | None) -> float | None:
    """Normalize a price string to float. Returns None if unparseable."""
    if not s:
        return None
    # Remove currency symbols, spaces; normalize separators
    s = re.sub(r"[^\d.,]", "", s.strip())
    if not s:
        return None
    # Handle formats: 1,234.56 or 1.234,56 or 1234
    if "," in s and "." in s:
        # e.g. 1,234.56 → English; 1.234,56 → European
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Could be thousands separator or decimal
        if len(s) - s.rfind(",") == 4:
            s = s.replace(",", "")  # thousands
        else:
            s = s.replace(",", ".")  # decimal
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _field_match(pred: str | None, gt: str | None, tolerance: float = 0.01) -> bool:
    """True if the predicted value matches ground truth (numeric-aware)."""
    if pred is None or gt is None:
        return False
    pred_n = _normalize_amount(pred)
    gt_n = _normalize_amount(gt)
    if pred_n is not None and gt_n is not None:
        return abs(pred_n - gt_n) <= tolerance
    # Fall back to string match
    return pred.strip().lower() == gt.strip().lower()


def compute_field_metrics(per_item: list[dict], field: str) -> dict:
    """
    Compute accuracy + bootstrap CI for a single field.
    per_item: list of {"pred": dict, "gt": dict}
    """
    correct = [
        1.0 if _field_match(item["pred"].get(field), item["gt"].get(field)) else 0.0
        for item in per_item
        if item["gt"].get(field) is not None
    ]
    if not correct:
        return {"accuracy": None, "ci95_low": None, "ci95_high": None, "n": 0}
    ci = bootstrap_ci(correct, seed=SEED)
    return {"accuracy": ci["value"], "ci95_low": ci["ci95_low"], "ci95_high": ci["ci95_high"],
            "n": len(correct)}


# ---------------------------------------------------------------------------
# Provider: OpenAI Vision (direct image → structured JSON)
# ---------------------------------------------------------------------------

OPENAI_VISION_PROMPT = (
    "Extract these fields from the receipt image. "
    "Return JSON only, no extra text.\n"
    'Format: {"total_price": "...", "subtotal_price": "..."}\n'
    "- total_price: the grand total / final amount to pay\n"
    "- subtotal_price: subtotal before tax\n"
    "Use the exact number as it appears on the receipt. null if not found."
)


def _image_to_base64(image_bytes: bytes, image_format: str = "png") -> str:
    return base64.b64encode(image_bytes).decode()


def run_openai_vision(items: list[dict], client: httpx.Client) -> dict:
    print(f"\n  Running OpenAI GPT-4o-mini vision on {len(items)} receipts...")
    latencies: list[float] = []
    per_item: list[dict] = []
    errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 20 == 0:
            print(f"    [{i + 1}/{len(items)}] errors={errors}")
        time.sleep(2.0)  # avoid 429 rate limit on OpenAI vision
        try:
            b64 = _image_to_base64(item["image_bytes"], item["image_format"])
            mime = f"image/{item['image_format']}"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OPENAI_VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        ],
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 256,
            }
            t0 = time.perf_counter()
            r = client.post(
                f"{OPENAI_BASE}/chat/completions",
                headers=openai_headers(),
                json=payload,
                timeout=60,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            raw = r.json()["choices"][0]["message"]["content"]
            pred = json.loads(raw)
            per_item.append({"pred": pred, "gt": item["ground_truth"]})
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{i}] {exc}")
            per_item.append({"pred": {}, "gt": item["ground_truth"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors}


# ---------------------------------------------------------------------------
# Provider: LlamaParse (direct image upload → structured extraction)
# ---------------------------------------------------------------------------

def run_llamaparse(items: list[dict], client: httpx.Client) -> dict:
    """
    Upload each receipt image to LlamaParse, get markdown output,
    then extract total_price and subtotal_price from the markdown using regex.
    """
    print(f"\n  Running LlamaParse on {len(items)} receipts...")
    api_key = env("LLAMA_PARSE_API_KEY")
    latencies: list[float] = []
    per_item: list[dict] = []
    errors = 0

    _llm_extract_prompt = (
        "From the following receipt text (may be Korean or English), extract:\n"
        '- total_price: the grand total / final amount to pay\n'
        '- subtotal_price: subtotal before tax\n'
        'Return JSON only: {"total_price": "...", "subtotal_price": "..."}\n'
        "Use null if not found. Return the numeric value as it appears (e.g. '12,500' or '12.50').\n\n"
        "Receipt text:\n"
    )

    def _extract_from_markdown(md: str) -> dict:
        """Extract price fields from LlamaParse markdown using GPT-4o-mini."""
        if not md or not md.strip():
            return {"total_price": None, "subtotal_price": None}
        try:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": _llm_extract_prompt + md[:3000]}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 128,
            }
            r = client.post(
                f"{OPENAI_BASE}/chat/completions",
                headers=openai_headers(),
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception:
            return {"total_price": None, "subtotal_price": None}

    for i, item in enumerate(items):
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(items)}] errors={errors}")
        try:
            # Upload image
            t0 = time.perf_counter()
            upload_resp = client.post(
                LLAMA_PARSE_UPLOAD_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("receipt.png", item["image_bytes"], "image/png")},
                data={"language": "en"},
                timeout=60,
            )
            upload_resp.raise_for_status()
            job_id = upload_resp.json()["id"]

            # Poll for result (max 60s)
            result_md = None
            for _ in range(30):
                time.sleep(2)
                result_resp = client.get(
                    LLAMA_PARSE_JOB_URL.format(job_id=job_id),
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                if result_resp.status_code == 200:
                    result_md = result_resp.json().get("markdown", "")
                    break
                elif result_resp.status_code == 404:
                    continue

            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)

            if result_md is None:
                raise RuntimeError(f"LlamaParse job {job_id} did not complete in time")

            time.sleep(0.5)  # avoid 429 on GPT-4o-mini extraction
            pred = _extract_from_markdown(result_md)
            per_item.append({"pred": pred, "gt": item["ground_truth"]})
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{i}] {exc}")
            per_item.append({"pred": {}, "gt": item["ground_truth"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors}


# ---------------------------------------------------------------------------
# Provider: Mawlaia pipeline (OpenAI OCR text → Mawlaia DocParse)
# ---------------------------------------------------------------------------

OCR_PROMPT = (
    "Extract all text from this receipt image exactly as it appears. "
    "Preserve numbers and labels. Output plain text only."
)

MAWLAIA_FIELDS = [
    {"name": "total_price", "description": "Grand total / final amount to pay on the receipt"},
    {"name": "subtotal_price", "description": "Subtotal before tax and discounts"},
]


def run_mawlaia_pipeline(items: list[dict], client: httpx.Client) -> dict:
    """
    Step 1: OpenAI GPT-4o-mini → OCR text from receipt image
    Step 2: Mawlaia DocParse → extract total_price and subtotal_price from text
    """
    print(f"\n  Running Mawlaia pipeline ({len(items)} receipts)...")
    print("    Step 1: OCR via GPT-4o-mini vision...")
    ocr_texts: list[str | None] = []
    ocr_errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 20 == 0:
            print(f"    [OCR {i + 1}/{len(items)}] errors={ocr_errors}")
        time.sleep(2.0)  # avoid 429 rate limit on OpenAI OCR
        try:
            b64 = _image_to_base64(item["image_bytes"])
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 1024,
            }
            r = client.post(
                f"{OPENAI_BASE}/chat/completions",
                headers=openai_headers(),
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            ocr_texts.append(text)
        except Exception as exc:
            ocr_errors += 1
            if ocr_errors <= 5:
                print(f"    WARN OCR [{i}]: {exc}")
            ocr_texts.append(None)

    print(f"    OCR done: {len(ocr_texts) - ocr_errors}/{len(ocr_texts)} successful")
    print("    Step 2: Mawlaia DocParse extraction...")

    latencies: list[float] = []
    per_item: list[dict] = []
    errors = 0

    for i, (item, ocr_text) in enumerate(zip(items, ocr_texts)):
        if (i + 1) % 20 == 0:
            print(f"    [DocParse {i + 1}/{len(items)}] errors={errors}")
        if not ocr_text:
            errors += 1
            per_item.append({"pred": {}, "gt": item["ground_truth"]})
            continue
        time.sleep(1.0)  # avoid overloading Mawlaia's internal OpenAI calls
        try:
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/doc/extract",
                headers=mawlaia_headers(),
                json={
                    "text": ocr_text,
                    "fields": MAWLAIA_FIELDS,
                },
                timeout=60,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            body = r.json()
            # Response: {"fields": {"total_price": {"value": "...", "confidence": 0.9}, ...}}
            fields_raw = body.get("fields", {})
            pred: dict[str, str | None] = {}
            for fname in ["total_price", "subtotal_price"]:
                field_obj = fields_raw.get(fname, {})
                if isinstance(field_obj, dict):
                    pred[fname] = field_obj.get("value")
                elif isinstance(field_obj, str):
                    pred[fname] = field_obj
                else:
                    pred[fname] = None
            per_item.append({"pred": pred, "gt": item["ground_truth"]})
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN DocParse [{i}]: {exc}")
            per_item.append({"pred": {}, "gt": item["ground_truth"]})

    return {
        "per_item": per_item,
        "latency": latency_stats(latencies),
        "errors": errors,
        "ocr_errors": ocr_errors,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def print_table(all_results: dict[str, dict], n_items: int):
    print(f"\n── DocParse — CORD-v2 (N={n_items}) ─────────────────────────────────────")
    print(f"\nTarget field: total_price (grand total)")
    print(f"{'Provider':<25} {'Accuracy':>10} {'[95% CI]':>18} {'p50ms':>8} {'Errors':>8}")
    print("-" * 72)

    for provider, data in all_results.items():
        m = compute_field_metrics(data["per_item"], "total_price")
        lat = data.get("latency", {})
        p50 = lat.get("p50_ms", "-")
        errs = data.get("errors", 0)
        acc_s = _fmt(m["accuracy"])
        ci_s = f"[{_fmt(m['ci95_low'])}, {_fmt(m['ci95_high'])}]" if m["accuracy"] is not None else "N/A"
        print(f"{provider:<25} {acc_s:>10} {ci_s:>18} {str(p50):>8} {errs:>8}")

    print(f"\nTarget field: subtotal_price")
    print(f"{'Provider':<25} {'Accuracy':>10} {'[95% CI]':>18}")
    print("-" * 56)
    for provider, data in all_results.items():
        m = compute_field_metrics(data["per_item"], "subtotal_price")
        acc_s = _fmt(m["accuracy"])
        ci_s = f"[{_fmt(m['ci95_low'])}, {_fmt(m['ci95_high'])}]" if m["accuracy"] is not None else "N/A"
        print(f"{provider:<25} {acc_s:>10} {ci_s:>18}")

    print(f"\nDataset: {DATASET_NAME} | N={n_items} | Seed: {SEED}")


# ---------------------------------------------------------------------------
# Serialize + save
# ---------------------------------------------------------------------------

def _serialize(all_results: dict, items: list[dict]) -> dict:
    output = {
        "product": "docparse_public",
        "dataset": DATASET_NAME,
        "n_samples": len(items),
        "seed": SEED,
        "target_fields": list(TARGET_FIELDS.keys()),
        "providers": {},
    }
    for provider, data in all_results.items():
        field_metrics = {}
        for field in TARGET_FIELDS:
            m = compute_field_metrics(data["per_item"], field)
            field_metrics[field] = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in m.items()}
        output["providers"][provider] = {
            "field_metrics": field_metrics,
            "latency": data.get("latency", {}),
            "errors": data.get("errors", 0),
        }
    return output


# ---------------------------------------------------------------------------
# Providers registry
# ---------------------------------------------------------------------------

def run_mawlaia_llamaparse(items: list[dict], client: httpx.Client) -> dict:
    """
    Mawlaia DocParse with LlamaParse OCR: send image_base64 + ocr_provider=llamaparse.
    Bypasses OpenAI rate limits — LlamaParse uses its own OCR infrastructure.
    """
    print(f"\n  Running Mawlaia + LlamaParse OCR ({len(items)} receipts)...")
    latencies: list[float] = []
    per_item: list[dict] = []
    errors = 0

    for i, item in enumerate(items):
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(items)}] errors={errors}")
        time.sleep(0.5)  # polite pacing for LlamaParse upload API
        try:
            fmt = item.get("image_format", "png")
            b64 = _image_to_base64(item["image_bytes"])
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/doc/extract",
                headers=mawlaia_headers(),
                json={
                    "image_base64": b64,
                    "image_media_type": f"image/{fmt}",
                    "ocr_provider": "llamaparse",
                    "fields": MAWLAIA_FIELDS,
                },
                timeout=120,  # LlamaParse OCR + extraction can take ~30s
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            body = r.json()
            fields_raw = body.get("fields", {})
            pred: dict[str, str | None] = {}
            for fname in ["total_price", "subtotal_price"]:
                field_obj = fields_raw.get(fname, {})
                val = field_obj.get("value") if isinstance(field_obj, dict) else field_obj
                pred[fname] = str(val) if val is not None else None
            per_item.append({"pred": pred, "gt": item["ground_truth"]})
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{i}]: {exc}")
            per_item.append({"pred": {}, "gt": item["ground_truth"]})

    return {
        "per_item": per_item,
        "latency": latency_stats(latencies),
        "errors": errors,
        "n_items": len(items),
    }


PROVIDERS = {
    "mawlaia_pipeline": (run_mawlaia_pipeline, ["MWL_API_KEY", "OPENAI_API_KEY"]),
    "mawlaia_llamaparse": (run_mawlaia_llamaparse, ["MWL_API_KEY", "LLAMA_PARSE_API_KEY"]),
    "openai_vision": (run_openai_vision, ["OPENAI_API_KEY"]),
    "llamaparse": (run_llamaparse, ["LLAMA_PARSE_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DocParse public benchmark (CORD-v2)")
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}. Valid: {list(PROVIDERS)}")

    items = load_cord(n_items=args.n_items)
    print(f"\nDocParse Public Benchmark — {len(items)} CORD receipts\n")

    all_results: dict[str, dict] = {}
    with httpx.Client() as client:
        for name in args.providers:
            fn, required_keys = PROVIDERS[name]
            missing = [k for k in required_keys if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing: {', '.join(missing)})")
                continue
            all_results[name] = fn(items, client)

    if not all_results:
        print("No providers ran. Exiting.")
        return

    print_table(all_results, len(items))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"docparse_public_{date}.json")

    payload = _serialize(all_results, items)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
