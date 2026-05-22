# Mawlaia Public Benchmarks

Open benchmark suite evaluating production AI safety APIs across four categories.
All datasets are publicly available on HuggingFace. All results are reproducible with the provided scripts and a valid API key.

**Paper**: [benchmarks/paper/main.tex](../paper/main.tex) (arXiv submission pending)

---

## Products benchmarked

| Product | API endpoint | What it does |
|---------|-------------|-------------|
| PII Vault | `/pii-vault/tokenize` | Detect and pseudonymise PII in text |
| Guardrail | `/guardrail/check` | Detect harmful / unsafe content |
| DocParse | `/doc/extract` | Extract structured fields from document images |
| EvalForge | `/eval/score` | LLM pairwise preference evaluation |

---

## Datasets

### Primary (Dataset 1)

| Script | Dataset | N | Competitors |
|--------|---------|---|-------------|
| `bench_pii_public.py` | [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | 1,000 | OpenAI GPT-4o-mini |
| `bench_guardrail_public.py` | DoNotAnswer + ToxiGen + PKU-SafeRLHF | 1,000 | OpenAI Moderation, Mistral Moderation |
| `bench_docparse_public.py` | [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | 95 | LlamaParse, OpenAI vision |
| `bench_eval_public.py` | [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) | 400 | GPT-4o-mini direct judge |

### Cross-validation (Dataset 2)

| Script | Dataset | N | Competitors |
|--------|---------|---|-------------|
| `bench_pii_gretel.py` | [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | 1,000 | OpenAI GPT-4o-mini |
| `bench_guardrail_aegis.py` | [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | 1,000 | Mistral Moderation |
| `bench_docparse_invoices.py` | [mychen76/invoices-and-receipts_ocr_v1](https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1) | 100 | LlamaParse, OpenAI OCR |
| `bench_eval_rewardbench.py` | [allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench) | 500 | GPT-4o-mini direct judge |

---

## Results

Pre-computed results are in `results/`. File naming: `{product}_{dataset}_{date}.json`.

| File | Contents |
|------|----------|
| `pii_public_v2_20260521.json` | PII — AI4Privacy, all providers |
| `guardrail_public_v3_20260521.json` | Guardrail — mixed corpus, all providers |
| `docparse_public_v2_20260521.json` | DocParse — CORD-v2, all providers |
| `eval_public_v3_20260521.json` | Eval — MT-Bench, pairwise_judge + GPT-4o-mini |
| `pii_gretel_20260522_final.json` | PII — Gretel Finance, all providers |
| `guardrail_aegis_20260522_final.json` | Guardrail — Aegis, all providers |
| `docparse_invoices_20260522_final.json` | DocParse — English Invoices, all providers |
| `eval_rewardbench_20260522.json` | Eval — RewardBench, all providers |

---

## Key findings

### PII Detection
- Regex-only: **25 ms p50** — 50× faster than LLM extraction, suitable for high-throughput pipelines
- LLM fallback raises macro F1 from 0.490 → 0.685 on AI4Privacy; gap vs OpenAI closes to 0.011 on finance-domain text (Gretel)
- Mawlaia outperforms OpenAI on SSN (0.850 vs 0.713) and IP_ADDRESS (0.977 vs 0.752)

### Content Moderation
- Regex-only: near-zero FPR (0.6%) but 99.1% FNR — suitable only as a fast pre-filter
- Hybrid regex+LLM: **FNR 18.0%**, below Mistral Moderation (19.3%) on primary benchmark
- On Aegis AI Safety dataset: **beats Mistral** on accuracy (0.827 vs 0.745), F1 (0.866 vs 0.768), and FNR (0.143 vs 0.351)

### Document Understanding
- With LlamaParse OCR backend: **82.1% field accuracy** on Korean receipts (CORD-v2), matching LlamaParse standalone (83.2%)
- **100% accuracy** on English invoices — OCR script complexity is the bottleneck, not extraction logic

### LLM Evaluation
- Absolute score differencing: **23.7% pairwise accuracy** — inverted preference signal (below random baseline)
- Native pairwise judge: **70.1%** on MT-Bench, **81.8%** on RewardBench — matches GPT-4o-mini within statistical CI on both

---

## Installation

```bash
pip install -r requirements.txt
```

## Running a benchmark

Set the required environment variables, then run any script directly:

```bash
export MWL_API_KEY="your_mawlaia_api_key"
export OPENAI_API_KEY="your_openai_key"        # for OpenAI providers
export MISTRAL_API_KEY="your_mistral_key"      # for Mistral providers
export LLAMA_PARSE_API_KEY="your_llamaparse_key"  # for LlamaParse providers

# Example: PII benchmark on Gretel Finance dataset
python3 bench_pii_gretel.py --providers mawlaia mawlaia_llm openai_zeroshot --n-items 1000

# Example: Guardrail on Aegis (Mawlaia only, skip Mistral)
python3 bench_guardrail_aegis.py --providers mawlaia mawlaia_hybrid --n-items 1000
```

All scripts accept `--providers` (subset of available providers), `--n-items`, and `--output` arguments.

Get a free Mawlaia API key at [mawlaia.com](https://mawlaia.com).

---

## Methodology

- **Bootstrap confidence intervals**: percentile bootstrap, B=1,000 resamples, seed=42
- **Significance testing**: McNemar's test with Edwards continuity correction
- **Stratified sampling**: proportional allocation preserving category distributions
- **Position randomisation**: all pairwise eval judges receive A/B order randomised per item

See `bench_stats.py` for the full statistical implementation.

---

## Reproducibility notes

- All results use `seed=42` throughout
- Datasets are loaded from HuggingFace Hub (cached locally after first run)
- API calls are made with per-provider rate limiting to avoid quota errors
- `bench_common.py` must be in the same directory as the benchmark scripts
