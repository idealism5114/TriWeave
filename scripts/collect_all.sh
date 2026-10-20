#!/usr/bin/env bash
set -euo pipefail

# Execute after importing the *real* 263-stock universe.
if [[ ! -f data/manifests/stocks_263.csv ]]; then
  echo "Missing data/manifests/stocks_263.csv. Import the final Clean V1 universe first." >&2
  echo "Example: python -m triweave import-stocks --input /path/to/asset_universe.json" >&2
  exit 2
fi

python -m triweave doctor
python -m triweave collect --groups bonds --run-id baseline15y --strict
python -m triweave collect --groups stocks --run-id baseline15y --strict
python -m triweave collect --groups futures_continuous --run-id baseline15y --strict
python -m triweave collect --groups futures_contracts --exchanges SHFE DCE CZCE --run-id baseline15y
python -m triweave collect --groups futures_contracts --exchanges INE GFEX CFFEX --run-id baseline15y
python -m triweave audit --run-id baseline15y
