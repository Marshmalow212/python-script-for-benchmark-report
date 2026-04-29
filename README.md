# TrueApp Search Report & Scoring Scripts

Scripts for benchmarking TrueApp's Search API using Gemini AI for relevance scoring and computing retrieval metrics (Precision, Recall, NDCG).

## Prerequisites

- **Python 3** with `requests` installed
- **Bash** shell with `curl`, `jq`, and `uuidgen`
- **Gemini API Key** — set via environment variable:
  ```bash
  export GEMINI_API_KEY='your-gemini-api-key'
  ```

## Files Overview

| File | Description |
|------|-------------|
| `trueapp_login.sh` | Authenticates to TrueApp Common API (token → OTP → login) and outputs a bearer token |
| `search_report_score_enhanced_gemini.py` | Main pipeline: searches keywords via the Search API, scores results with Gemini, exports per-category CSVs |
| `benchmark_tool.py` | Calculates performance metrics (Precision@1, Precision@2, Recall@6, Recall@20, NDCG@10) from scored CSV results |
| `script_gen.sh` | Example wrapper that runs `search_report_score_enhanced_gemini.py` with a pre-set token and parameters |
| `trueapp_keywords_by_category.json` | Keywords organized by category (e.g., PackageData, Device, eSIM) used as search input |

## Procedure

### Step 1: Obtain a Bearer Token

Run the login script to authenticate and get a bearer token:

```bash
bash trueapp_login.sh
```

1. The script requests an access token using client credentials.
2. It sends an OTP to the configured MSISDN.
3. You will be prompted to **enter the OTP code** received on the phone.
4. On success, it prints the `accessToken` and `refreshToken`.

**Copy the `accessToken`** — you'll need it for the next step.

> **Note:** To change the phone number or environment, edit the configuration variables at the top of `trueapp_login.sh`.

### Step 2: Run the Search & Gemini Scoring Pipeline

```bash
python3 search_report_score_enhanced_gemini.py token <ACCESS_TOKEN> lang <LANGUAGE>
```

**CLI arguments** (all optional, keyword-value pairs):

| Argument | Description | Default |
|----------|-------------|---------|
| `token` | Bearer token from Step 1 | Auto-fetched via UAT client credentials |
| `brand` | User brand filter (e.g., `TRUE`, `DTAC`) | `guest` |
| `tel_type` | User type (e.g., `postpaid`, `prepaid`) | `guest` |
| `lang` | Language (`th` or `en`) | `th` |

**Example:**

```bash
python3 search_report_score_enhanced_gemini.py token eyJ0eXAi... brand TRUE tel_type postpaid lang th
```

**What it does:**

1. Loads keywords from `trueapp_keywords_by_category.json`.
2. For each keyword, calls the TrueApp Search API (up to 50 results).
3. Scores each result using Gemini AI (relevance score 0–3).
4. Exports three CSVs per category into `category_search_results_v1/`:
   - `<category>_all_scored.csv` — all results in original API order
   - `<category>_all_sorted.csv` — all results sorted by Gemini score
   - `<category>_ranked_top5.csv` — top results ranked by Gemini score

> **Note:** This step takes significant time due to Gemini API rate limiting (~2s per result + 20s between keywords).

### Step 3: Calculate Performance Metrics

After the scored CSVs are generated, run the benchmark tool:

```bash
python3 benchmark_tool.py
```

This reads the `category_search_results_v1/` directory and computes:

| Metric | Description |
|--------|-------------|
| **Precision@1** | Is the top API result the same as the top Gemini-ranked result? |
| **Precision@2** | Proportion of API top-2 results found in Gemini top-2 |
| **Recall@6** | Fraction of Gemini top-6 results found in API top-6 |
| **Recall@20** | Fraction of Gemini top-20 results found in API top-20 |
| **NDCG@10** | Normalized Discounted Cumulative Gain at position 10 |

**Output files:**

- `performance_metrics_v1.csv` — per-keyword metrics
- `performance_metrics_v1_summary.csv` — per-category averages

### Alternative: Use the Wrapper Script

`script_gen.sh` is a convenience wrapper that runs the scoring pipeline with a hardcoded token:

```bash
bash script_gen.sh
```

> Edit `script_gen.sh` to update the token and parameters before running.

## Output Directory Structure

```
category_search_results_v1/
├── PackageData_all_scored.csv
├── PackageData_all_sorted.csv
├── PackageData_ranked_top5.csv
├── Device_all_scored.csv
├── ...
performance_metrics_v1.csv
performance_metrics_v1_summary.csv
```
