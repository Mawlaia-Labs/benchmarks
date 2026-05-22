"""
DocParse Public Benchmark — English Invoice Dataset
Dataset: mychen76/invoices-and-receipts_ocr_v1 (English synthetic invoices with images)
Providers: Mawlaia DocParse + LlamaParse OCR, Mawlaia DocParse + OpenAI OCR

Target fields: invoice_date, total_gross_worth
Metric: field-level accuracy (exact/numeric-tolerant match)

Usage:
  python3 benchmarks/public/bench_docparse_invoices.py [--n-items 100] [--providers mawlaia_llamaparse mawlaia_pipeline]

Required env vars:
  MWL_API_KEY, LLAMA_PARSE_API_KEY (for mawlaia_llamaparse), OPENAI_API_KEY (for mawlaia_pipeline)
"""
from __future__ import annotations

import argparse
import ast
import asyncio
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


from bench_common import MAWLAIA_BASE, OPENAI_BASE, LLAMA_CLOUD_BASE, env, mawlaia_headers, openai_headers, latency_stats
from bench_stats import bootstrap_ci

SEED = 42
RESULTS_DIR = _THIS_DIR / "results"
DATASET_NAME = "mychen76/invoices-and-receipts_ocr_v1"

TARGET_FIELDS = ["invoice_date", "total_gross_worth"]

MAWLAIA_FIELDS = [
    {"name": "invoice_date", "description": "Invoice date (date the invoice was issued)"},
    {"name": "total_gross_worth", "description": "Total gross amount to pay including VAT/tax"},
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_invoices(n_items: int = 100) -> list[dict]:
    """Load invoices with structured ground truth from both train+test splits."""
    print(f"Loading {DATASET_NAME}…")
    from datasets import load_dataset
    import numpy as np

    items: list[dict] = []
    for split in ["test", "train"]:
        ds = load_dataset(DATASET_NAME, split=split)
        for row in ds:
            data_raw = row.get("parsed_data", "")
            if isinstance(data_raw, str):
                try:
                    data = json.loads(data_raw)
                except Exception:
                    continue
            else:
                data = data_raw

            inner_str = data.get("json", "{}")
            try:
                inner = ast.literal_eval(inner_str) if isinstance(inner_str, str) else inner_str
            except Exception:
                continue

            header = inner.get("header", {})
            summary = inner.get("summary", {})

            invoice_date = header.get("invoice_date")
            total_gross = summary.get("total_gross_worth")

            if not invoice_date or not total_gross:
                continue

            img = row.get("image")
            if img is None:
                continue

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            items.append({
                "image_bytes": image_bytes,
                "ground_truth": {
                    "invoice_date": str(invoice_date).strip(),
                    "total_gross_worth": str(total_gross).strip(),
                },
            })
        print(f"  {split}: {len(items)} usable items so far")
        if len(items) >= n_items * 2:
            break

    print(f"  Collected {len(items)} items with structured GT")

    if n_items < len(items):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(items), size=n_items, replace=False)
        items = [items[i] for i in sorted(idx)]
        print(f"  Sampled {len(items)} items")

    return items


# ---------------------------------------------------------------------------
# Field matching
# ---------------------------------------------------------------------------

def _normalize_amount(s: str | None) -> float | None:
    if not s:
        return None
    s = re.sub(r"[^\d.,]", "", str(s).strip())
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s) - s.rfind(",") <= 3 else s.replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _normalize_date(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    # Try to normalise MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD to YYYY-MM-DD
    # Format: MM/DD/YYYY (US)
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", s)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    # Format: YYYY-MM-DD
    m = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return s.lower()


def _field_match(pred: str | None, gt: str | None, field: str) -> bool:
    if pred is None or gt is None:
        return False
    if field == "total_gross_worth":
        pn, gn = _normalize_amount(pred), _normalize_amount(gt)
        if pn is not None and gn is not None:
            return abs(pn - gn) <= 0.02
    if field == "invoice_date":
        return _normalize_date(pred) == _normalize_date(gt)
    return str(pred).strip().lower() == str(gt).strip().lower()


def compute_field_metrics(per_item: list[dict], field: str) -> dict:
    correct = [
        1.0 if _field_match(item["pred"].get(field), item["gt"].get(field), field) else 0.0
        for item in per_item
        if item["gt"].get(field) is not None
    ]
    if not correct:
        return {"accuracy": None, "ci95_low": None, "ci95_high": None, "n": 0}
    ci = bootstrap_ci(correct, seed=SEED)
    return {"accuracy": round(ci["value"], 4), "ci95_low": round(ci["ci95_low"], 4),
            "ci95_high": round(ci["ci95_high"], 4), "n": len(correct)}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def run_mawlaia_llamaparse(items: list[dict], client: httpx.Client) -> dict:
    print(f"\n  Running Mawlaia + LlamaParse OCR ({len(items)} invoices)…")
    latencies, per_item, errors = [], [], 0

    for i, item in enumerate(items):
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(items)}] errors={errors}")
        time.sleep(0.5)
        try:
            b64 = _image_to_base64(item["image_bytes"])
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/doc/extract",
                headers=mawlaia_headers(),
                json={
                    "image_base64": b64,
                    "image_media_type": "image/png",
                    "ocr_provider": "llamaparse",
                    "fields": MAWLAIA_FIELDS,
                },
                timeout=120,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            fields_raw = r.json().get("fields", {})
            pred = {
                f: (fields_raw[f].get("value") if isinstance(fields_raw.get(f), dict) else fields_raw.get(f))
                for f in TARGET_FIELDS
            }
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{i}]: {exc}")
            pred = {f: None for f in TARGET_FIELDS}
        per_item.append({"pred": pred, "gt": item["ground_truth"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors}


def run_mawlaia_pipeline(items: list[dict], client: httpx.Client) -> dict:
    """OCR via GPT-4o-mini vision, then Mawlaia field extraction."""
    print(f"\n  Running Mawlaia pipeline (OpenAI OCR + Mawlaia extract, {len(items)} invoices)…")
    OCR_PROMPT = (
        "Extract all text from this invoice image exactly as it appears. "
        "Preserve numbers, dates, and labels. Output plain text only."
    )
    latencies, per_item, errors = [], [], 0
    ocr_errors = 0

    ocr_texts: list[str | None] = []
    for i, item in enumerate(items):
        if (i + 1) % 20 == 0:
            print(f"    [OCR {i + 1}/{len(items)}] errors={ocr_errors}")
        time.sleep(2.0)
        try:
            b64 = _image_to_base64(item["image_bytes"])
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
                "temperature": 0, "max_tokens": 1024,
            }
            r = client.post(f"{OPENAI_BASE}/chat/completions", headers=openai_headers(), json=payload, timeout=60)
            r.raise_for_status()
            ocr_texts.append(r.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            ocr_errors += 1
            if ocr_errors <= 3:
                print(f"    WARN OCR [{i}]: {exc}")
            ocr_texts.append(None)

    print(f"    OCR done ({len(ocr_texts) - ocr_errors}/{len(ocr_texts)} OK). Extracting…")

    for i, (item, ocr_text) in enumerate(zip(items, ocr_texts)):
        if (i + 1) % 20 == 0:
            print(f"    [Extract {i + 1}/{len(items)}] errors={errors}")
        if not ocr_text:
            errors += 1
            per_item.append({"pred": {f: None for f in TARGET_FIELDS}, "gt": item["ground_truth"]})
            continue
        time.sleep(1.0)
        try:
            t0 = time.perf_counter()
            r = client.post(
                f"{MAWLAIA_BASE}/doc/extract",
                headers=mawlaia_headers(),
                json={"text": ocr_text, "fields": MAWLAIA_FIELDS},
                timeout=60,
            )
            ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(ms)
            fields_raw = r.json().get("fields", {})
            pred = {
                f: (fields_raw[f].get("value") if isinstance(fields_raw.get(f), dict) else fields_raw.get(f))
                for f in TARGET_FIELDS
            }
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN Extract [{i}]: {exc}")
            pred = {f: None for f in TARGET_FIELDS}
        per_item.append({"pred": pred, "gt": item["ground_truth"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors, "ocr_errors": ocr_errors}


def run_llamaparse_standalone(items: list[dict], client: httpx.Client) -> dict:
    """LlamaParse OCR → GPT-4o-mini field extraction (no Mawlaia)."""
    print(f"\n  Running LlamaParse standalone ({len(items)} invoices)…")
    api_key = env("LLAMA_PARSE_API_KEY")
    UPLOAD_URL = f"{LLAMA_CLOUD_BASE}/parsing/upload"
    JOB_URL    = f"{LLAMA_CLOUD_BASE}/parsing/job/{{job_id}}/result/text"
    EXTRACT_PROMPT = (
        "From the following invoice text extract:\n"
        "- invoice_date: the date the invoice was issued (any format)\n"
        "- total_gross_worth: total gross amount including tax/VAT\n"
        'Return JSON only: {"invoice_date": "...", "total_gross_worth": "..."}\n'
        "Use null if not found.\n\nInvoice text:\n"
    )
    latencies, per_item, errors = [], [], 0

    for i, item in enumerate(items):
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(items)}] errors={errors}")
        try:
            t0 = time.perf_counter()
            up = client.post(
                UPLOAD_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("invoice.png", item["image_bytes"], "image/png")},
                data={"language": "en"},
                timeout=60,
            )
            up.raise_for_status()
            job_id = up.json()["id"]

            result_text = None
            for _ in range(30):
                time.sleep(2)
                res = client.get(
                    JOB_URL.format(job_id=job_id),
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                if res.status_code == 200:
                    result_text = res.json().get("text", "")
                    break
                elif res.status_code == 404:
                    continue

            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)

            if not result_text:
                raise RuntimeError(f"LlamaParse job {job_id} returned no text")

            time.sleep(0.5)
            r = client.post(
                f"{OPENAI_BASE}/chat/completions",
                headers=openai_headers(),
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": EXTRACT_PROMPT + result_text[:3000]}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 128,
                },
                timeout=30,
            )
            r.raise_for_status()
            extracted = json.loads(r.json()["choices"][0]["message"]["content"])
            pred = {f: extracted.get(f) for f in TARGET_FIELDS}
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    WARN [{i}]: {exc}")
            pred = {f: None for f in TARGET_FIELDS}
        per_item.append({"pred": pred, "gt": item["ground_truth"]})

    return {"per_item": per_item, "latency": latency_stats(latencies), "errors": errors}


PROVIDERS = {
    "mawlaia_llamaparse":     (run_mawlaia_llamaparse,     ["MWL_API_KEY", "LLAMA_PARSE_API_KEY"]),
    "mawlaia_pipeline":       (run_mawlaia_pipeline,       ["MWL_API_KEY", "OPENAI_API_KEY"]),
    "llamaparse_standalone":  (run_llamaparse_standalone,  ["LLAMA_PARSE_API_KEY", "OPENAI_API_KEY"]),
}


# ---------------------------------------------------------------------------
# Table + main
# ---------------------------------------------------------------------------

def print_table(all_results: dict, n_items: int):
    print(f"\n── DocParse — English Invoices (N={n_items}) ───────────────────────────────")
    print(f"{'Provider':<25} {'invoice_date Acc':>18} {'[95% CI]':>16} {'total_gross_worth Acc':>22} {'[95% CI]':>16} {'Errors':>8}")
    print("-" * 110)
    for name, data in all_results.items():
        pm = compute_field_metrics(data["per_item"], "invoice_date")
        gm = compute_field_metrics(data["per_item"], "total_gross_worth")
        errs = data.get("errors", 0)
        p_s = f"{pm['accuracy']:.3f}" if pm["accuracy"] is not None else "N/A"
        p_ci = f"[{pm['ci95_low']:.3f},{pm['ci95_high']:.3f}]" if pm["accuracy"] is not None else ""
        g_s = f"{gm['accuracy']:.3f}" if gm["accuracy"] is not None else "N/A"
        g_ci = f"[{gm['ci95_low']:.3f},{gm['ci95_high']:.3f}]" if gm["accuracy"] is not None else ""
        print(f"{name:<25} {p_s:>18} {p_ci:>16} {g_s:>22} {g_ci:>16} {errs:>8}")
    print(f"\nDataset: {DATASET_NAME} | N={n_items} | Seed: {SEED}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    invalid = [p for p in args.providers if p not in PROVIDERS]
    if invalid:
        parser.error(f"Unknown providers: {invalid}")

    items = load_invoices(n_items=args.n_items)
    print(f"\nDocParse Invoice Benchmark — {len(items)} invoices\n")

    all_results: dict[str, dict] = {}
    with httpx.Client() as client:
        for name in args.providers:
            fn, required = PROVIDERS[name]
            missing = [k for k in required if not os.getenv(k)]
            if missing:
                print(f"  Skipping {name} (missing: {', '.join(missing)})")
                continue
            all_results[name] = fn(items, client)

    if not all_results:
        print("No providers ran.")
        return

    print_table(all_results, len(items))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        from datetime import datetime, timezone
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = str(RESULTS_DIR / f"docparse_invoices_{date}.json")

    payload = {
        "product": "docparse",
        "dataset": DATASET_NAME,
        "n_samples": len(items),
        "seed": SEED,
        "target_fields": TARGET_FIELDS,
        "providers": {},
    }
    for name, data in all_results.items():
        field_metrics = {f: compute_field_metrics(data["per_item"], f) for f in TARGET_FIELDS}
        payload["providers"][name] = {
            "field_metrics": field_metrics,
            "latency": data.get("latency", {}),
            "errors": data.get("errors", 0),
        }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
