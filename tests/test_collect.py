import json

import pandas as pd
import pytest

from triweave.assets import TREASURY_CURVE
from triweave.collect import Collector, Job, make_jobs, normalise_job
from triweave.audit import coverage_audit


class FakeAkshare:
    calls = 0

    def stock_zh_a_hist(self, **kwargs):
        self.calls += 1
        return pd.DataFrame({"\u65e5\u671f": ["2023-10-09", "2023-10-10"],
                             "\u6536\u76d8": [10.0, 10.1]})

    def bond_china_yield(self, **kwargs):
        self.calls += 1
        return pd.DataFrame({
            "\u65e5\u671f": ["2023-10-09"],
            "\u66f2\u7ebf\u540d\u79f0": [TREASURY_CURVE],
            "3\u6708": [1.5], "6\u6708": [1.6],
            "1\u5e74": [1.8], "3\u5e74": [2.0],
            "5\u5e74": [2.1], "7\u5e74": [2.2],
            "10\u5e74": [2.3], "30\u5e74": [2.8],
        })

    def bond_treasury_index_cbond(self, **kwargs):
        self.calls += 1
        return pd.DataFrame({"date": ["2023-10-09"], "value": [212.0]})

    def futures_zh_daily_sina(self, **kwargs):
        self.calls += 1
        return pd.DataFrame({"date": ["2023-10-09"], "open": [3500], "close": [3501]})

    def get_futures_daily(self, **kwargs):
        self.calls += 1
        return pd.DataFrame({"date": ["2023-10-09", "2023-10-09"],
                            "symbol": ["rb2401", "xx2401"],
                            "close": [3501.0, 2.0]})


class EmptyApi:
    calls = 0

    def futures_zh_daily_sina(self, **kwargs):
        self.calls += 1
        return pd.DataFrame()


def test_make_jobs_has_no_network_and_exact_counts():
    jobs = make_jobs(["bonds", "futures_continuous"], "2011-10-01", "2026-09-30")
    assert len([j for j in jobs if j.api == "bond_china_yield"]) == 19
    assert len([j for j in jobs if j.api == "bond_treasury_index_cbond"]) == 7
    assert len([j for j in jobs if j.api == "futures_zh_daily_sina"]) == 38
    assert len(jobs) == 64
    with pytest.raises(ValueError, match="263"):
        make_jobs(["stocks"], "2023-10-01", "2023-10-31")


def test_early_exchange_date_limits():
    jobs = make_jobs(["futures_contracts"], "2011-10-01", "2012-10-01", markets=["GFEX", "INE", "CFFEX"])
    assert len(jobs) == 0
    jobs = make_jobs(["futures_contracts"], "2023-10-01", "2023-11-30", markets=["DCE", "SHFE"])
    assert len(jobs) == 4


def test_collect_raw_and_standardized_cache_and_checksums(tmp_path):
    fake = FakeAkshare()
    collector = Collector(tmp_path, "testrun", api=fake, retries=1, pause=0)
    job = make_jobs(["bonds"], "2023-10-01", "2023-10-31")[0]
    state = collector.collect_one(job)
    assert state == "OK"
    assert fake.calls == 1
    raw, clean, meta = collector.paths(job)
    assert raw.exists() and clean.exists() and meta.exists()
    meta_obj = json.loads(meta.read_text())
    assert meta_obj["status"] == "OK"
    assert meta_obj["valid_rows"] == 8
    assert collector.collect_one(job) == "CACHED"
    assert fake.calls == 1
    frame = pd.read_csv(clean)
    assert set(frame["asset_id"]) == {
        "CN_GOV_YIELD_3M", "CN_GOV_YIELD_6M", "CN_GOV_YIELD_1Y",
        "CN_GOV_YIELD_3Y", "CN_GOV_YIELD_5Y", "CN_GOV_YIELD_7Y",
        "CN_GOV_YIELD_10Y", "CN_GOV_YIELD_30Y",
    }
    # Corrupt data must not be silently overwritten on rerun.
    with clean.open("ab") as fh:
        fh.write(b"tampering")
    with pytest.raises(RuntimeError, match="modified/corrupted"):
        collector.collect_one(job)


def test_collect_stock_and_audit_does_not_compute_labels(tmp_path):
    fake = FakeAkshare()
    stocks = pd.DataFrame({"asset_id": ["sh.600038"], "symbol": ["600038"]})
    job = make_jobs(["stocks"], "2023-10-01", "2023-10-31", stock_universe=stocks)[0]
    collector = Collector(tmp_path, "stocktest", api=fake, retries=1, pause=0)
    assert collector.collect_one(job) == "OK"
    stats = coverage_audit(tmp_path, "stocktest", stocks)
    assert stats["assets_with_any_rows"] == 1
    assert stats["expected_commodity_targets"] == 24
    assert (tmp_path / "data/audit/stocktest/asset_coverage.csv").exists()


def test_empty_api_fails_without_writing_fake_prices(tmp_path):
    fake = EmptyApi()
    job = make_jobs(["futures_continuous"], "2023-10-01", "2023-10-31")[0]
    collector = Collector(tmp_path, "missing", api=fake, retries=2, pause=0)
    assert collector.collect_one(job) == "FAILED"
    raw, clean, meta = collector.paths(job)
    assert not raw.exists() and not clean.exists() and not meta.exists()
    assert (tmp_path / "data/manifests/missing/events.jsonl").exists()


def test_empty_variety_is_recorded_not_filled(tmp_path):
    fake = FakeAkshare()
    job = make_jobs(["futures_contracts"], "2023-10-01", "2023-10-31", markets=["DCE"])[0]
    collector = Collector(tmp_path, "no_variety", api=fake, retries=1, pause=0)
    assert collector.collect_one(job) == "NO_SELECTED_VARIETIES"
    raw, clean, meta = collector.paths(job)
    assert raw.exists() and not clean.exists()
    assert json.loads(meta.read_text())["status"] == "NO_SELECTED_VARIETIES"


def test_invalid_run_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        Collector(tmp_path, "../escape")
