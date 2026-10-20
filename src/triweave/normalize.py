"""Strict schema conversion; never fill missing market observations."""
import re

import pandas as pd

from triweave.assets import YIELD_SPECS, ALL_CONTRACT_FUTURES


DATE_COLUMNS = ("date", "\u65e5\u671f", "trading_date", "\u4ea4\u6613\u65e5\u671f")


class SchemaError(ValueError):
    pass


def find_col(df, choices, required=True):
    for item in choices:
        if item in df.columns:
            return item
    if required:
        raise SchemaError(f"No expected column among {choices}; observed: {list(df.columns)}")
    return None


def parse_date_series(values):
    s = values.astype("string").str.strip()
    numeric_dates = s.str.fullmatch(r"\d{8}").fillna(False)
    out = pd.to_datetime(s.where(~numeric_dates), format="mixed", errors="coerce")
    out.loc[numeric_dates] = pd.to_datetime(s.loc[numeric_dates], format="%Y%m%d", errors="coerce")
    # Prevent implicit local-time assumptions; daily date is not publication timestamp.
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def date_subset(df, start, end, date_col=None):
    date_col = date_col or find_col(df, DATE_COLUMNS)
    out = df.copy()
    out["date"] = parse_date_series(out[date_col])
    if out["date"].isna().any():
        raise SchemaError(f"Unparseable date in {date_col}: {out.loc[out['date'].isna(), date_col].head().tolist()}")
    out = out.loc[out["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def verify_unique(df, columns):
    if df.duplicated(columns).any():
        examples = df.loc[df.duplicated(columns, keep=False), columns].head(6).to_dict("records")
        raise SchemaError(f"Duplicate {columns}: {examples}")


def numeric(values):
    return pd.to_numeric(values, errors="coerce")


def finite(values):
    return values.notna() & ~values.isin([float("inf"), float("-inf")])


def normalize_stocks(raw, stock_id, start, end):
    x = date_subset(raw, start, end)
    cols = {
        "open": ("open", "\u5f00\u76d8"),
        "high": ("high", "\u6700\u9ad8"),
        "low": ("low", "\u6700\u4f4e"),
        "close": ("close", "\u6536\u76d8"),
        "volume": ("volume", "\u6210\u4ea4\u91cf"),
        "amount": ("amount", "\u6210\u4ea4\u989d"),
        "turnover": ("turnover_rate", "turnover", "\u6362\u624b\u7387"),
        "turnover_rate": ("turnover_rate", "\u6362\u624b\u7387"),
    }
    out = pd.DataFrame({"date": x["date"], "asset_id": stock_id})
    for k, choices in cols.items():
        col = find_col(x, choices, required=(k == "close"))
        out[k] = numeric(x[col]) if col else float("nan")
    out["market"] = "stock"
    out["unit"] = "RMB_per_share_unadjusted"
    out["volume_unit"] = "Eastmoney_lots_100_shares"
    out["amount_unit"] = "RMB"
    out["turnover_unit"] = "percent"
    out["turnover_rate_unit"] = "percent"
    out["stock_code"] = stock_id.split(".")[-1]
    out["adjustment"] = "unadjusted"
    for field, choices in {"security_name": ("security_name", "\u8bc1\u5238\u540d\u79f0", "\u540d\u79f0"),
                           "trading_status": ("trading_status", "\u4ea4\u6613\u72b6\u6001")}.items():
        col = find_col(x, choices, required=False)
        out[field] = x[col] if col else pd.NA
    out["valid_close"] = finite(out["close"]) & (out["close"] > 0)
    verify_unique(out, ["asset_id", "date"])
    return out.sort_values("date").reset_index(drop=True)


def normalize_bond_yields(raw, start, end):
    x = date_subset(raw, start, end)
    name_col = find_col(x, ("\u66f2\u7ebf\u540d\u79f0", "curve_name"))
    chunks = []
    for asset_id, (curve_name, tenor_col) in YIELD_SPECS.items():
        sub = x.loc[x[name_col] == curve_name]
        if sub.empty:
            continue
        if tenor_col not in sub.columns:
            raise SchemaError(f"Curve {curve_name}: required tenor column {tenor_col} missing")
        tmp = pd.DataFrame({
            "date": sub["date"], "asset_id": asset_id,
            "market": "bond", "instrument_type": "yield_curve",
            "tenor": tenor_col, "value": numeric(sub[tenor_col]),
            "unit": "percent_per_annum", "curve_name": curve_name,
        })
        tmp["valid_value"] = finite(tmp["value"])
        chunks.append(tmp)
    if not chunks:
        raise SchemaError("None of the 18 required yield series matched this response; check AKShare curve labels")
    out = pd.concat(chunks, ignore_index=True)
    verify_unique(out, ["asset_id", "date"])
    return out.sort_values(["asset_id", "date"]).reset_index(drop=True)


def normalize_bond_index(raw, asset_id, start, end):
    x = date_subset(raw, start, end)
    value_col = find_col(x, ("value", "\u503c", "\u6570\u503c"))
    out = pd.DataFrame({"date": x["date"], "asset_id": asset_id,
                        "market": "bond", "instrument_type": "treasury_wealth_index",
                        "value": numeric(x[value_col]), "unit": "index_level"})
    out["valid_value"] = finite(out["value"]) & (out["value"] > 0)
    verify_unique(out, ["asset_id", "date"])
    return out.sort_values("date").reset_index(drop=True)


def _ohlcv(raw, start, end):
    x = date_subset(raw, start, end)
    fields = {
        "open": ("open", "\u5f00\u76d8", "\u5f00\u76d8\u4ef7"),
        "high": ("high", "\u6700\u9ad8", "\u6700\u9ad8\u4ef7"),
        "low": ("low", "\u6700\u4f4e", "\u6700\u4f4e\u4ef7"),
        "close": ("close", "\u6536\u76d8", "\u6536\u76d8\u4ef7"),
        "volume": ("volume", "\u6210\u4ea4\u91cf"),
        "open_interest": ("open_interest", "hold", "\u6301\u4ed3\u91cf"),
        "turnover": ("turnover", "\u6210\u4ea4\u989d"),
        "settle": ("settle", "\u52a8\u6001\u7ed3\u7b97\u4ef7", "\u7ed3\u7b97\u4ef7"),
        "pre_settle": ("pre_settle", "\u6628\u7ed3\u7b97"),
    }
    out = pd.DataFrame({"date": x["date"]})
    for k, choices in fields.items():
        col = find_col(x, choices, required=(k == "close"))
        out[k] = numeric(x[col]) if col else float("nan")
    out["valid_close"] = finite(out["close"]) & (out["close"] > 0)
    return x, out


def normalize_continuous(raw, spec, start, end):
    _, out = _ohlcv(raw, start, end)
    out["asset_id"] = f"FUT_{spec.variety}"
    out["variety"] = spec.variety
    out["exchange"] = spec.exchange
    out["role"] = spec.role
    out["market"] = "commodity_future"
    out["series_type"] = "vendor_continuous_unverified_roll"
    out["research_use"] = "EXPLORATORY_CONTINUOUS_SERIES"
    out["tradable_return_verified"] = False
    out["volume_unit"] = "futures_lots"
    verify_unique(out, ["asset_id", "date"])
    return out.sort_values("date").reset_index(drop=True)


def normalize_contracts(raw, exchange, start, end):
    x, out = _ohlcv(raw, start, end)
    sym_col = find_col(x, ("symbol", "contract", "\u5408\u7ea6"))
    variety_col = find_col(x, ("variety", "product"), required=False)
    out["contract"] = x[sym_col].astype("string").str.upper().str.strip()
    extracted = out["contract"].str.extract(r"^([A-Z]+)", expand=False)
    if variety_col:
        declared = x[variety_col].astype("string").str.upper().str.strip()
        out["variety"] = declared.where(declared.str.fullmatch(r"[A-Z]+"), extracted)
    else:
        out["variety"] = extracted
    allowed = {f.variety: f for f in ALL_CONTRACT_FUTURES if f.exchange == exchange}
    out = out.loc[out["variety"].isin(allowed)].copy()
    out["exchange"] = exchange
    out["asset_id"] = out["variety"].map(lambda code: f"FUT_{code}")
    out["role"] = out["variety"].map(lambda code: allowed[code].role)
    out["market"] = out["role"].map(lambda role: "bond_future" if role == "bond_future" else "commodity_future")
    out["series_type"] = "exchange_listed_contract"
    out["tradable_return_verified"] = False  # contract data still needs an execution/roll protocol
    out["volume_unit"] = "futures_lots"
    out["contract_month_code"] = out["contract"].str.extract(r"([0-9]{3,4})$", expand=False)
    month = pd.to_numeric(out["contract_month_code"].str[-2:], errors="coerce")
    out["delivery_month"] = month.where(month.between(1, 12))
    out["delivery_year_verified"] = False  # CZCE three-digit codes need a separate historical rule.
    if not out.empty:
        verify_unique(out, ["exchange", "contract", "date"])
    return out.sort_values(["contract", "date"]).reset_index(drop=True)
