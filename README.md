# TriWeave: 15-Year Multi-Market Research Data

> **Repository snapshot (2026-10-20):** This repository now also contains the
> downstream data audits, feature pipelines, models, baselines, and ablation
> runners in `src/triweave/`. The current exploratory experiment protocol uses
> **311 input nodes and 23 commodity targets**; the original 25-target protocol
> retains its research hold. Code, configurations, and tests are tracked in Git.
> The full historical database is backed up separately in the private repository's
> [data snapshot Release](https://github.com/idealism5114/TriWeave/releases/tag/data-snapshot-20261020-v1).
> See [data restore instructions](data_snapshot/README.md) and
> [benchmark instructions](docs/BENCHMARK_TEST_RUN.md). The acquisition guide below
> is retained as the original collection documentation.

**Data acquisition only.** This package is a restartable, auditable ingestion pipeline for Chinese A-share equities, bond curves/indices, and exchange-listed commodity and treasury futures.

- Project period: **2011-10-01 to 2026-09-30**, an exact 15-year calendar interval. These are *requested bounds*, not an assertion that every contract existed or traded throughout this period.
- Equity universe: **exactly 263 stocks**, taken from the user's prior final Clean V1 stock manifest. Their full identifiers are NOT published in the referenced CMGM GitHub repository, so import the real file (do not guess).
- Bond variables: **25** configured series: 18 historical yield-curve tenors and 7 ChinaBond treasury wealth indices.
- Commodity futures: **24 targets + 14 auxiliary varieties**, plus 4 optionally collected listed treasury-futures varieties (not counted among the 25 bond input series).
- Sources: AKShare adapters to Eastmoney, ChinaBond, Sina, and official exchange daily records. Source coverage/availability is audited rather than assumed.

## 1. Installation

```bash
unzip TriWeave.zip
cd TriWeave
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test]'
python -m triweave init
python -m triweave doctor
python -m pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

AKShare and its upstream websites require internet access. If you are offline, `plan`, `init`, `doctor`, and the mocked pytest suite still work (doctor reports that AKShare is missing). Record the installed versions after the first successful network run:

```bash
python -m pip freeze > logs/installed_packages.txt
```

## 2. Freeze the **real** 263-stock universe

The legacy CMGM repository contains source code, but **not all 263 Clean V1 ticker identifiers**. Obtain the actual final `asset_universe.json` or a CSV with the 263 tickers from the earlier Phase A-Fix results. The importer preserves original order, validates codes, verifies exact count, and logs SHA256. It refuses a 248-ticker legacy list and does not scrape a different 263 stocks to make the number match.

```bash
python -m triweave import-stocks --input /PATH/TO/CLEAN_V1/asset_universe.json
# Or a CSV with the column "code", or a price matrix with date and 263 ticker headers:
python -m triweave import-stocks --input /PATH/TO/stocks_263.csv
```

Alternative **only if you have the old 248-stock wide CSV but not the exact new manifest**: the earlier user-supplied Phase A-Fix report explicitly recorded the 15 stock additions. This importer can reconstruct a **candidate** 263-stock universe from the 248 original codes plus those 15 (it cannot guarantee the original frozen Clean V1 ordering):

```bash
python -m triweave import-stocks --input /PATH/TO/legacy_248_wide.csv --add-phase-a-15
```

The resulting metadata flags `requires_comparison_to_frozen_Clean_V1=true`; confirm membership/order against the original Phase A-Fix final asset manifest before claiming an exact reproduction. No unlisted stock codes are invented.

Run just **one** of these alternatives: the command refuses to overwrite an existing manifest. If your JSON is nested, use `--json-key market_assets.stock` to specify a dotted path to its actual list. The output is `data/manifests/stocks_263.csv`. To build a different research universe, create a separately named protocol and do not call it the same Clean V1 cohort.

The three-row file in `examples/stock_manifest_example.csv` illustrates syntax only. It is **not** a production 263-ticker universe.

## 3. Preview requests (no network)

```bash
python -m triweave plan
python -m triweave plan --groups stocks bonds futures_continuous
python -m triweave plan --groups futures_contracts --exchanges DCE SHFE
```

Default `plan` covers 19 ChinaBond curve-date chunks (300 days each), 7 treasury wealth indices, and 38 exploratory commodity continuous futures, 64 jobs in total. Stock requests add 263 jobs after the genuine universe is imported. Listed-contract jobs are **monthly by exchange**, potentially a much larger number of requests.

## 4. First online smoke test

Start with **a single job** in a fresh run-id. A smoke run needs no stock manifest:

```bash
python -m triweave collect --groups bonds --start 2023-10-01 --end 2023-10-31 --max-jobs 1 --run-id smoke_bonds --strict
python -m triweave collect --groups futures_continuous --start 2023-10-01 --end 2023-10-31 --max-jobs 1 --run-id smoke_futures --strict
python -m triweave collect --groups futures_contracts --exchanges SHFE --start 2023-10-01 --end 2023-10-31 --max-jobs 1 --run-id smoke_contracts --strict
```

Inspect `data/manifests/smoke_*/events.jsonl` and the standardized output. Certain providers change schemas and rate limits: failures stay failures and require an explicit fix, never silent substitution or future-fill.

## 5. Collect the research interval

```bash
# Full bond inputs (18 yield series and 7 treasury-index series):
python -m triweave collect --groups bonds --run-id baseline15y --strict

# Full vendor continuous commodity futures (exploration ONLY):
python -m triweave collect --groups futures_continuous --run-id baseline15y --strict

# Exact 263 A-share securities (after --input import-stocks):
python -m triweave collect --groups stocks --run-id baseline15y --strict

# True per-contract exchange histories, one month per request/exchange:
python -m triweave collect --groups futures_contracts --exchanges DCE SHFE CZCE --run-id baseline15y
python -m triweave collect --groups futures_contracts --exchanges INE GFEX CFFEX --run-id baseline15y

# Data coverage audit, with zero-observation series visible:
python -m triweave audit --run-id baseline15y
```

`--strict` returns nonzero if any job failed; a failure **does not delete successful jobs**. Repeating a run skips complete jobs after checksum verification. To refetch data from a changed upstream source, use `--run-id baseline15y_refresh1`; never overwrite a previous snapshot. Inspect job failures and individual market start dates before claiming that collection is complete.

## 6. Research data layout

```text
TriWeave/
  configs/project.json                   # fixed scope and collection policy
  src/triweave/                        # importable Python package
    cli.py                             # commands
    assets.py                          # fixed 25/24/14 specs
    universe.py                        # import EXACT 263 stock codes
    collect.py                         # source adapters + restart/checksums
    normalize.py                       # schema mapping (no imputation)
    audit.py                           # missingness and date coverage
  data/
    raw/<run-id>/                      # untouched source DataFrames, gzip CSV
      stocks/
      bond_yield_curves/
      bond_wealth_indices/
      futures_continuous_exploratory/
      futures_contracts/<EXCHANGE>/
    standardized/<run-id>/             # normalized non-imputed DataFrames
    manifests/<run-id>/jobs/           # source, args, hashes, fetch time, counts
    manifests/stocks_263.csv           # must be imported from Clean V1
    manifests/non_stock_registry.csv
    audit/<run-id>/asset_coverage.csv
    derived/                            # no synthetic label/continuous return yet
    sealed/                             # never loaded in an exploratory experiment
  docs/                                # schema and source limitations
  tests/                               # network-free unit tests using mocks
  logs/                                # package version snapshot
  experiments/
  notebooks/
  checkpoints/
```

## 7. What the data mean (important)

- This pipeline collects **raw historical market observations** with original date and source identity. It does not make them legally/technically point-in-time (PIT) tradeable information. Publication times are unverified.
- `stocks` uses **unadjusted** Eastmoney A-share history (`adjust=""`). A share's cash dividends, splits, delisting and suspensions need a separate valid corporate-action total-return protocol before creating return targets.
- ChinaBond yield curves are **percentage yields**, NOT bond prices; do not use price pct-change feature engineering on those series. Wealth-index levels are a different type.
- `futures_continuous_exploratory` contains **vendor stitched prices with an unaudited roll policy**. Never calculate official tradable 1d/5d/10d/20d return labels by taking their naive price percentage changes.
- `futures_contracts` provides **listed contract OHLC/settlement/volume/OI** when public exchange data are returned. Use only information known at the stated forecast decision time to derive contract selection, rolls, P&L and target labels in a later *separately versioned* protocol.
- The wheat mapping uses `WH` (strong-gluten wheat) **tentatively**. The old study's wheat target identity and sparsity have not been fully verified. No substitute contract is silently inserted.
- Late-listed contracts cannot have 15 years of valid price history. No `bfill`, no fake constant pre-listing history, no full-period `dropna(any)` asset selection.
- The old CMGM TEST region is irrelevant for constructing this *new* research database, but its previously used metrics must not be reused as fresh out-of-sample evidence. For future modeling, establish new immutable chronological splits and embargo up to the longest target horizon.
- Use provider terms and academic licensing rules for all bulk downloads. ChinaBond indices/curves may have separate usage constraints.

## 8. Quality thresholds / next stage

The purpose of collection is to measure actual coverage, not to force all series to look complete. `triweave audit` exposes first/last real observations, counts and missing series. Before any new model training, explicitly decide:

1. fixed point-in-time data availability for all three markets;
2. futures contract identification, month-selection and roll-return logic;
3. corporate-action-adjusted stock history;
4. bond yield *level/change* and index *return* feature families;
5. per-commodity label validity masks and observation coverage;
6. train/VAL/untouched confirmation split with 20-day horizon embargo;
7. source license, provider revisions, and publication/quote timestamps.

A successful `collect` **does not** imply these have been resolved. See `docs/sources_and_scope.md` and `docs/data_protocol.md`.
