import pandas as pd
import pytest

from triweave.assets import TARGET_FUTURES, TREASURY_CURVE, AAA_NOTE_CURVE
from triweave.normalize import (
    SchemaError, normalize_stocks, normalize_bond_yields,
    normalize_bond_index, normalize_continuous, normalize_contracts,
    parse_date_series,
)


def test_date_parses_compact_and_iso():
    dates = parse_date_series(pd.Series(["20231001", "2023-10-02"]))
    assert dates.dt.strftime("%Y-%m-%d").tolist() == ["2023-10-01", "2023-10-02"]


def test_stock_no_bfill_preserves_missing_and_disallows_future_date():
    df = pd.DataFrame({"\u65e5\u671f": ["2011-10-03", "2011-10-04", "2011-10-05"],
                       "\u6536\u76d8": [None, 11.0, 11.5]})
    normal = normalize_stocks(df, "sh.600038", "2011-10-03", "2011-10-04")
    assert len(normal) == 2
    assert pd.isna(normal.loc[0, "close"])
    assert not bool(normal.loc[0, "valid_close"])
    assert bool(normal.loc[1, "valid_close"])


def test_yield_curves_only_matching_real_historical_lines():
    frame = pd.DataFrame({
        "\u65e5\u671f": ["2023-10-09", "2023-10-09"],
        "\u66f2\u7ebf\u540d\u79f0": [TREASURY_CURVE, AAA_NOTE_CURVE],
        "10\u5e74": [2.5, None],
        "3\u6708": [1.6, 3.2],
        "6\u6708": [1.8, 3.3],
        "1\u5e74": [1.9, 3.4],
        "3\u5e74": [2.1, 3.6],
        "5\u5e74": [2.3, 3.8],
        "7\u5e74": [2.4, 4.1],
        "30\u5e74": [3.0, None],
    })
    out = normalize_bond_yields(frame, "2023-10-01", "2023-10-31")
    assert out.loc[out["asset_id"] == "CN_GOV_YIELD_10Y", "value"].iloc[0] == 2.5
    assert "CN_AAA_NOTE_YIELD_10Y" not in set(out["asset_id"])
    assert len(out.loc[out["asset_id"].str.startswith("CN_GOV")]) == 8
    assert len(out.loc[out["asset_id"].str.startswith("CN_AAA_NOTE")]) == 5
    assert len(out.loc[out["asset_id"].str.startswith("CN_AAA_BANK")]) == 0


def test_missing_yield_series_not_invented():
    with pytest.raises(SchemaError):
        normalize_bond_yields(pd.DataFrame({"\u65e5\u671f": ["2023-10-09"],
            "\u66f2\u7ebf\u540d\u79f0": ["unknown"]}), "2023-10-01", "2023-10-31")


def test_bond_wealth_index():
    data = pd.DataFrame({"date": ["2023-10-09"], "value": [201.0]})
    result = normalize_bond_index(data, "CN_TREASURY_WEALTH_5Y", "2023-10-01", "2023-10-31")
    assert result.loc[0, "unit"] == "index_level"
    assert result.loc[0, "valid_value"]


def test_futures_continuous_not_audited_contract():
    spec = next(x for x in TARGET_FUTURES if x.variety == "RB")
    df = pd.DataFrame({"date": ["2023-10-09"], "open": [3550], "close": [3580],
                       "settle": [3585], "volume": [200]})
    out = normalize_continuous(df, spec, "2023-10-01", "2023-10-31")
    assert out.loc[0, "series_type"] == "vendor_continuous_unverified_roll"
    assert not out.loc[0, "tradable_return_verified"]


def test_contract_filter_is_exchange_specific():
    df = pd.DataFrame({"date": ["2023-10-09", "2023-10-09", "2023-10-09"],
                       "symbol": ["rb2401", "cu2401", "x1234"],
                       "open": [3500, 30000, 10], "close": [3510, 30100, 11],
                       "volume": [1, 1, 1]})
    out = normalize_contracts(df, "SHFE", "2023-10-01", "2023-10-31")
    assert set(out["variety"]) == {"RB", "CU"}
    assert len(out) == 2


def test_duplicated_stock_date_fails():
    df = pd.DataFrame({"date": ["2023-10-09", "2023-10-09"], "close": [1.0, 2.0]})
    with pytest.raises(SchemaError, match="Duplicate"):
        normalize_stocks(df, "sh.600038", "2023-10-01", "2023-10-31")
