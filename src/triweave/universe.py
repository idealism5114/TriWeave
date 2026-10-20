"""Import an *existing* equity universe without inventing missing tickers."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


STOCK_CODE = re.compile(r"^(?:(sh|sz|bj)[.]?)?(\d{6})$", re.IGNORECASE)

# 15 stock additions explicitly reported by the user's Phase A-Fix audit,
# relative to the historical 248-stock legacy asset list. This is an
# optional reconstruction aid, NOT a substitute for the final frozen manifest.
REPORTED_PHASE_A_ADDED_15 = (
    "sh.600038", "sh.600066", "sh.600297", "sh.600373", "sh.600446",
    "sh.600535", "sh.600705", "sh.600718", "sh.600837", "sh.600863",
    "sh.601211", "sh.603160", "sz.002146", "sz.002424", "sz.002736",
)



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for blob in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(blob)
    return digest.hexdigest()


def normalize_stock_code(value: str) -> str:
    value = str(value).strip().lower()
    match = STOCK_CODE.fullmatch(value)
    if match is None:
        raise ValueError(f"Not a recognized A-share code: {value!r}")
    exchange, six = match.groups()
    if six.startswith(("60", "68", "90")):
        inferred = "sh"
    elif six.startswith(("00", "30", "20")):
        inferred = "sz"
    elif six.startswith(("43", "83", "87", "88", "92")):
        inferred = "bj"
    else:
        raise ValueError(f"Cannot infer A-share exchange for {six}")
    if exchange and exchange != inferred:
        raise ValueError(f"Exchange prefix conflicts with security code: {value}")
    exchange = exchange or inferred
    return f"{exchange}.{six}"


def _codes_from_json(obj, explicit_key=None):
    if explicit_key:
        for key in explicit_key.split("."):
            if not isinstance(obj, dict) or key not in obj:
                raise ValueError(f"JSON stock key {explicit_key!r} cannot be found")
            obj = obj[key]
    elif isinstance(obj, dict):
        for key in ("stock", "stocks", "stock_assets", "stock_codes", "stock_universe"):
            if key in obj:
                obj = obj[key]
                break
        else:
            for key in ("assets", "selected_assets", "input_assets", "asset_universe", "market_assets", "instruments", "universe"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise ValueError("Unknown JSON structure: pass --json-key dotted.path.to.stock.list")
    if isinstance(obj, dict):
        for key in ("stock", "stocks", "stock_assets", "stock_codes", "codes", "assets"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError("JSON stock list must be an array or contain a stock/codes/assets list; use --json-key")
    if not isinstance(obj, list):
        raise ValueError("Stock universe must be a JSON list")
    codes = []
    for row in obj:
        if isinstance(row, str):
            codes.append(row)
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Unexpected JSON stock element: {row!r}")
        market = str(row.get("market", row.get("asset_type", ""))).lower()
        if market and market not in ("stock", "stocks", "equity", "a_share", "a-share"):
            continue
        code = next((row.get(k) for k in ("code", "asset_id", "symbol", "ticker", "asset_name", "name") if row.get(k)), None)
        if code is None:
            raise ValueError(f"No stock code in JSON record: {row!r}")
        codes.append(code)
    return codes


def read_stock_candidates(input_path: Path, json_key=None):
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return _codes_from_json(obj, json_key)
    if path.suffix.lower() != ".csv":
        raise ValueError("Stock universe input must be a .csv or .json file")
    # Read only the header first: a historical wide CSV might include a sealed TEST
    # interval. Never parse its numerical price cells just to find stock IDs.
    fields = list(pd.read_csv(path, dtype=str, encoding="utf-8-sig", nrows=0).columns)
    for key in ("code", "asset_id", "symbol", "ticker"):
        if key in fields:
            cols = [key] + (["market"] if "market" in fields else [])
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=cols)
            if "market" in df.columns:
                df = df.loc[df["market"].str.lower().isin(["stock", "stocks", "equity", "a_share"])]
            return df[key].dropna().tolist()
    # Historical CMGM stock price matrix: date, sh.600xxx, sz.000xxx, ...
    # Header-only; do not inspect time series values.
    price_fields = [field for field in fields if field.lower() not in ("date", "datetime", "trading_date")]
    if price_fields and all(STOCK_CODE.fullmatch(str(field).strip()) for field in price_fields):
        return price_fields
    raise ValueError("No code column and not a stock-price wide matrix")


def import_stock_universe(input_path: Path, destination: Path, expected: int = 263, json_key=None, add_phase_a_15=False):
    items = read_stock_candidates(Path(input_path), json_key=json_key)
    codes = [normalize_stock_code(item) for item in items]
    if add_phase_a_15:
        if expected != 263 or len(codes) != 248:
            raise ValueError("--add-phase-a-15 requires exactly 248 legacy input stocks and expected=263")
        if set(REPORTED_PHASE_A_ADDED_15) & set(codes):
            raise ValueError("The 248-stock legacy source already contains one of the reported additions")
        codes.extend(REPORTED_PHASE_A_ADDED_15)
    if len(codes) != len(set(codes)):
        raise ValueError(f"Duplicate stock tickers in source: {len(codes) - len(set(codes))}")
    if len(codes) != expected:
        raise ValueError(
            f"Expected {expected} actual stock codes; source has {len(codes)}. "
            "Provide the final Clean V1 stock asset manifest (263), not a historical 248-stock list."
        )
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing universe: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"order": i, "asset_id": c, "symbol": c.split(".")[1], "exchange": c.split(".")[0]} for i, c in enumerate(codes)]
    pd.DataFrame(rows).to_csv(destination, index=False)
    metadata = {
        "stock_count": len(codes), "source_file": str(input_path),
        "source_sha256": sha256_file(Path(input_path)),
        "universe_sha256": sha256_file(destination),
        "identity_verified": "From user-supplied manifest; not independently verified historical membership",
        "reconstructed_from_legacy_248_plus_reported_15": bool(add_phase_a_15),
        "requires_comparison_to_frozen_Clean_V1": bool(add_phase_a_15),
    }
    destination.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_stock_universe(path: Path, expected: int = 263):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: triweave import-stocks --input /path/to/clean_v1/asset_universe.json"
        )
    df = pd.read_csv(path, dtype={"asset_id": str, "symbol": str, "exchange": str})
    if len(df) != expected or df["asset_id"].nunique() != expected:
        raise ValueError(f"Refusing production stock collection: expected {expected}, found {len(df)}")
    if not all(normalize_stock_code(c) == c for c in df["asset_id"]):
        raise ValueError("Invalid normalized stock universe")
    if not df["symbol"].eq(df["asset_id"].str.split(".").str[-1]).all():
        raise ValueError("Stock request symbol differs from its frozen asset_id")
    if not df["exchange"].eq(df["asset_id"].str.split(".").str[0]).all():
        raise ValueError("Stock exchange differs from its frozen asset_id")
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise ValueError("Stock universe provenance metadata is missing")
    meta = json.loads(meta_path.read_text())
    if meta.get("universe_sha256") != sha256_file(path):
        raise ValueError("Stock universe fingerprint differs from frozen provenance")
    return df
