import json
from dataclasses import replace
from pathlib import Path
import runpy
import shutil

import pandas as pd
import pytest

from triweave.assets import validate_asset_configuration
from triweave.collect import Collector, make_jobs
from triweave.normalize import normalize_stocks, parse_date_series
from triweave.universe import sha256_file


def test_stock_amount_and_turnover_have_distinct_units_and_dates_are_sorted():
    raw = pd.DataFrame({"日期": ["2023-10-10", "2023-10-09"], "收盘": [11., 10.],
                        "成交额": [300000., 200000.], "换手率": [1.5, 1.2]})
    out = normalize_stocks(raw, "sh.600000", "2023-10-01", "2023-10-31")
    assert out["date"].tolist() == ["2023-10-09", "2023-10-10"]
    assert out["amount"].tolist() == [200000., 300000.]
    assert out["turnover"].tolist() == [1.2, 1.5]
    assert out["amount_unit"].eq("RMB").all()
    assert out["turnover_unit"].eq("percent").all()
    assert out["trading_status"].isna().all()


def test_infinite_price_is_not_a_valid_quote():
    raw = pd.DataFrame({"date": ["2023-10-09"], "close": [float("inf")]})
    out = normalize_stocks(raw, "sh.600000", "2023-10-01", "2023-10-31")
    assert not out["valid_close"].any()


def test_mixed_dates_are_parsed_without_silently_losing_rows():
    values = pd.Series(["2023-10-09", "2023/10/10", "20231011"])
    assert parse_date_series(values).dt.strftime("%Y-%m-%d").tolist() == ["2023-10-09", "2023-10-10", "2023-10-11"]


class StockApi:
    def __init__(self, frame):
        self.frame = frame

    def stock_zh_a_hist(self, **kwargs):
        return self.frame.copy()


def stock_job():
    universe = pd.DataFrame({"asset_id": ["sh.600000"], "symbol": ["600000"]})
    return make_jobs(["stocks"], "2023-10-01", "2023-10-31", stock_universe=universe)[0]


def test_resume_refuses_changed_date_scope(tmp_path):
    job = stock_job()
    collector = Collector(tmp_path, "scope", api=StockApi(pd.DataFrame({"date": ["2023-10-09"], "close": [10.]})), pause=0)
    assert collector.collect_one(job) == "OK"
    with pytest.raises(RuntimeError, match="request differs"):
        collector.collect_one(replace(job, start="2011-10-01"))


@pytest.mark.parametrize("frame", [pd.DataFrame({"unexpected": [1]}), pd.DataFrame({"date": ["2023-10-09"], "close": [None]})])
def test_invalid_response_fails_and_keeps_raw_evidence(tmp_path, frame):
    collector = Collector(tmp_path, "invalid", api=StockApi(frame), retries=2, pause=0)
    job = stock_job()
    assert collector.collect_one(job) == "FAILED"
    raw, clean, meta = collector.paths(job)
    assert not raw.exists() and not clean.exists() and not meta.exists()
    event = json.loads(collector.logs.read_text().splitlines()[0])
    evidence = event["response_evidence"]
    archived = tmp_path / evidence["path"]
    assert sha256_file(archived) == evidence["sha256"]
    assert pd.read_csv(archived).columns.tolist() == frame.columns.tolist()
    assert event["api_kwargs"] == job.kwargs


def test_main_module_propagates_failure_exit_status(monkeypatch):
    import triweave.cli
    monkeypatch.setattr(triweave.cli, "main", lambda: 7)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("triweave.__main__", run_name="__main__")
    assert exc.value.code == 7


def test_asset_config_drift_fails_before_collection(tmp_path):
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "configs", tmp_path / "configs")
    assert validate_asset_configuration(tmp_path)["bond_variables"] == 25
    file = tmp_path / "configs/futures_universe_38_plus4.csv"
    frame = pd.read_csv(file)
    frame.loc[0, "role"] = "auxiliary"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="definition mismatch"):
        validate_asset_configuration(tmp_path)
