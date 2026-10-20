"""Restartable raw acquisition: explicit source, strict schema, append-only runs."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import os
import re
import time
import uuid
import traceback

import pandas as pd

from triweave.assets import (
    ALL_FUTURES, EXCHANGE_EARLIEST, INDEX_SPECS,
    TREASURY_INDEX_PERIODS,
)
from triweave.normalize import (
    SchemaError, normalize_stocks, normalize_bond_yields,
    normalize_bond_index, normalize_continuous, normalize_contracts,
)
from triweave.universe import sha256_file


@dataclass(frozen=True)
class Job:
    group: str
    name: str
    api: str
    kwargs: dict
    start: str
    end: str
    context: object = None
    upstream: str = "AKShare"


def _date(value):
    return pd.Timestamp(value).normalize()


def _windows(start, end, max_days):
    p, stop = _date(start), _date(end)
    while p <= stop:
        q = min(p + pd.Timedelta(days=max_days - 1), stop)
        yield p, q
        p = q + pd.Timedelta(days=1)


def _months(start, end):
    start, end = _date(start), _date(end)
    for month in pd.period_range(start, end, freq="M"):
        yield max(start, month.start_time.normalize()), min(end, month.end_time.normalize())


def make_jobs(groups, start, end, stock_universe=None, curve_chunk_days=300, markets=None):
    """Build deterministic jobs without loading AKShare or accessing any price data."""
    start, end = _date(start), _date(end)
    if start > end:
        raise ValueError("start must be <= end")
    if start < pd.Timestamp("1990-01-01"):
        raise ValueError("Unexpectedly early start date")
    result = []
    groups = set(groups)
    if "all" in groups:
        groups = {"stocks", "bonds", "futures_continuous", "futures_contracts"}
    if "stocks" in groups:
        if stock_universe is None:
            raise ValueError("Stock collection needs an exact 263-stock CSV universe: use import-stocks")
        for stock in stock_universe.itertuples():
            result.append(Job(
                group="stocks", name=str(stock.asset_id).replace(".", "_"),
                api="stock_zh_a_hist",
                kwargs={"symbol": str(stock.symbol), "period": "daily",
                        "start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d"),
                        "adjust": ""},
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                context=str(stock.asset_id), upstream="Eastmoney via AKShare",
            ))
    if "bonds" in groups:
        for a, b in _windows(start, end, curve_chunk_days):
            result.append(Job(
                group="bond_yield_curves", name=f"curves_{a:%Y%m%d}_{b:%Y%m%d}",
                api="bond_china_yield",
                kwargs={"start_date": a.strftime("%Y%m%d"), "end_date": b.strftime("%Y%m%d")},
                start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                upstream="ChinaBond historical curves via AKShare",
            ))
        for asset_id, period in INDEX_SPECS.items():
            result.append(Job(
                group="bond_wealth_indices", name=asset_id,
                api="bond_treasury_index_cbond",
                kwargs={"indicator": "\u8d22\u5bcc", "period": period},
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                context=asset_id, upstream="ChinaBond treasury indices via AKShare",
            ))
    if "futures_continuous" in groups:
        for asset in ALL_FUTURES:
            result.append(Job(
                group="futures_continuous_exploratory", name=f"{asset.exchange}_{asset.variety}_0",
                api="futures_zh_daily_sina", kwargs={"symbol": f"{asset.variety}0"},
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                context=asset, upstream="Sina vendor continuous series via AKShare; roll unaudited",
            ))
    if "futures_contracts" in groups:
        markets = [m.upper() for m in (markets or ["DCE", "SHFE", "CZCE", "INE", "GFEX", "CFFEX"])]
        valid = {"DCE", "SHFE", "CZCE", "INE", "GFEX", "CFFEX"}
        if any(m not in valid for m in markets):
            raise ValueError(f"Unknown futures exchange: {markets}")
        for market in markets:
            floor = pd.Timestamp(EXCHANGE_EARLIEST.get(market, start))
            market_start = max(start, floor)
            if market_start > end:
                continue
            for a, b in _months(market_start, end):
                result.append(Job(
                    group=f"futures_contracts/{market}", name=f"{market}_{a:%Y%m%d}_{b:%Y%m%d}",
                    api="get_futures_daily",
                    kwargs={"start_date": a.strftime("%Y%m%d"), "end_date": b.strftime("%Y%m%d"), "market": market},
                    start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                    context=market, upstream=f"{market} exchange daily contracts via AKShare",
                ))
    return result


def normalise_job(job, response):
    if not isinstance(response, pd.DataFrame):
        raise SchemaError(f"Expected a pandas DataFrame, got {type(response).__name__}")
    if response.empty:
        raise SchemaError("The API returned an empty DataFrame")
    if job.group == "stocks":
        return normalize_stocks(response, job.context, job.start, job.end)
    if job.group == "bond_yield_curves":
        return normalize_bond_yields(response, job.start, job.end)
    if job.group == "bond_wealth_indices":
        return normalize_bond_index(response, job.context, job.start, job.end)
    if job.group == "futures_continuous_exploratory":
        return normalize_continuous(response, job.context, job.start, job.end)
    if job.group.startswith("futures_contracts/"):
        return normalize_contracts(response, job.context, job.start, job.end)
    raise ValueError(f"Unexpected job group {job.group}")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{uuid.uuid4().hex}.tmp.csv.gz"
    try:
        df.to_csv(tmp, index=False, compression="gzip", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(document, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{uuid.uuid4().hex}.tmp.json"
    try:
        temp.write_text(json.dumps(document, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def append_log(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        if os.name == "posix":
            import fcntl
            fcntl.flock(out, fcntl.LOCK_EX)
        try:
            out.write(json.dumps(document, ensure_ascii=True, default=str) + "\n")
            out.flush()
        finally:
            if os.name == "posix":
                fcntl.flock(out, fcntl.LOCK_UN)


class Collector:
    def __init__(self, project_root, run_id, api=None, retries=3, pause=1.0):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("run_id may contain only letters, digits, underscore and hyphen")
        self.root = Path(project_root)
        self.run_id = run_id
        self.retries = max(1, int(retries))
        self.pause = max(0.0, float(pause))
        self.api = api  # Lazy-load AKShare, allowing no-network --plan and tests.
        self.logs = self.root / "data" / "manifests" / run_id / "events.jsonl"

    def paths(self, job):
        segment = Path(job.group)
        raw = self.root / "data" / "raw" / self.run_id / segment / f"{job.name}.csv.gz"
        clean = self.root / "data" / "standardized" / self.run_id / segment / f"{job.name}.csv.gz"
        meta = self.root / "data" / "manifests" / self.run_id / "jobs" / segment / f"{job.name}.json"
        return raw, clean, meta

    def _get_api(self):
        if self.api is None:
            try:
                import akshare as ak
            except ImportError as exc:
                raise RuntimeError("Install dependencies first: python -m pip install -e .") from exc
            self.api = ak
        return self.api

    def collect_one(self, job):
        raw_path, clean_path, meta_path = self.paths(job)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = {"requested_start": job.start, "requested_end": job.end,
                        "api": job.api, "api_kwargs": job.kwargs}
            if any(meta.get(k) != v for k, v in expected.items()):
                raise RuntimeError("Existing job request differs; use a new run-id, never reuse a different date scope")
            if meta.get("status") == "OK":
                files = ((raw_path, meta.get("raw_sha256")), (clean_path, meta.get("standardized_sha256")))
                if all(path.exists() and sha256_file(path) == digest for path, digest in files):
                    return "CACHED"
                raise RuntimeError(f"Existing job file was modified/corrupted: {meta_path}. Never silently overwrite")
            if meta.get("status") == "NO_SELECTED_VARIETIES":
                if raw_path.exists() and sha256_file(raw_path) == meta.get("raw_sha256"):
                    return "CACHED_EMPTY_SELECTION"
                raise RuntimeError(f"Existing raw archive was modified: {raw_path}")
            raise RuntimeError(f"Unknown previous job state: {meta_path}")
        if raw_path.exists() or clean_path.exists():
            raise FileExistsError("Untracked raw/standardized file exists; refuse to overwrite")
        last_error = None
        for attempt in range(1, self.retries + 1):
            response = None
            try:
                api = self._get_api()
                response = getattr(api, job.api)(**job.kwargs)
                transformed = normalise_job(job, response)
                if transformed.empty and job.group.startswith("futures_contracts/"):
                    atomic_csv(response, raw_path)
                    status = "NO_SELECTED_VARIETIES"
                    entry = {
                        "status": status, "job_group": job.group, "name": job.name,
                        "requested_start": job.start, "requested_end": job.end,
                        "api": job.api, "api_kwargs": job.kwargs,
                        "upstream": job.upstream, "retrieved_utc": utc_now(),
                        "raw_rows": len(response), "standardized_rows": 0,
                        "raw_sha256": sha256_file(raw_path), "standardized_sha256": None,
                        "note": "Selected varieties absent in genuine exchange response; no rows manufactured",
                    }
                else:
                    if transformed.empty:
                        raise SchemaError("No normalized rows inside requested time interval")
                    validity = transformed.get("valid_close", transformed.get("valid_value"))
                    if validity is None or not validity.any():
                        raise SchemaError("Response contains no valid finite observations")
                    atomic_csv(response, raw_path)
                    atomic_csv(transformed, clean_path)
                    entry = {
                        "status": "OK", "job_group": job.group, "name": job.name,
                        "requested_start": job.start, "requested_end": job.end,
                        "api": job.api, "api_kwargs": job.kwargs,
                        "upstream": job.upstream, "retrieved_utc": utc_now(),
                        "raw_rows": len(response), "standardized_rows": len(transformed),
                        "valid_rows": int(transformed.get("valid_close", transformed.get("valid_value", pd.Series([False] * len(transformed)))).sum()),
                        "first_observed_date": str(transformed["date"].min()),
                        "last_observed_date": str(transformed["date"].max()),
                        "raw_sha256": sha256_file(raw_path),
                        "standardized_sha256": sha256_file(clean_path),
                        "warning": "Date is an observation date, not verified publication/availability time",
                    }
                entry["triweave_collector_version"] = "0.1.0"
                entry["raw_columns"] = list(map(str, response.columns))
                entry["raw_dtypes"] = {str(k): str(v) for k, v in response.dtypes.items()}
                entry["akshare_version"] = getattr(api, "__version__", "unknown_or_mock")
                entry["python_version"] = platform.python_version()
                entry["collector_source_sha256"] = sha256_file(Path(__file__))
                entry["normalizer_source_sha256"] = sha256_file(Path(__file__).with_name("normalize.py"))
                entry["asset_registry_source_sha256"] = sha256_file(Path(__file__).with_name("assets.py"))
                config_path = self.root / "configs" / "project.json"
                entry["protocol_config_sha256"] = sha256_file(config_path) if config_path.exists() else None
                atomic_json(entry, meta_path)
                append_log(self.logs, {"event": "JOB", "status": entry["status"], "name": job.name, "group": job.group, "at": utc_now()})
                return entry["status"]
            except Exception as exc:
                last_error = repr(exc)
                evidence = None
                if response is not None:
                    evidence_path = (self.root / "data" / "raw" / self.run_id / "_failed" /
                                     job.group / job.name / f"{uuid.uuid4().hex}.csv.gz")
                    if isinstance(response, pd.DataFrame):
                        atomic_csv(response, evidence_path)
                        evidence = {"path": str(evidence_path.relative_to(self.root)),
                                    "sha256": sha256_file(evidence_path),
                                    "rows": len(response), "columns": list(map(str, response.columns))}
                append_log(self.logs, {
                    "event": "RETRY_FAILED", "attempt": attempt,
                    "name": job.name, "group": job.group,
                    "error": last_error, "at": utc_now(),
                    "api": job.api, "api_kwargs": job.kwargs,
                    "upstream": job.upstream, "requested_start": job.start, "requested_end": job.end,
                    "response_evidence": evidence, "traceback": traceback.format_exc(),
                })
                # Schema errors will not improve on retry; upstream failures might.
                if isinstance(exc, (SchemaError, FileExistsError)) or raw_path.exists() or clean_path.exists():
                    break
                if attempt < self.retries:
                    time.sleep(min(8.0, 2 ** (attempt - 1)))
        append_log(self.logs, {"event": "JOB_FAILED", "name": job.name, "group": job.group,
                              "api": job.api, "api_kwargs": job.kwargs, "upstream": job.upstream,
                              "requested_start": job.start, "requested_end": job.end,
                              "error": last_error, "at": utc_now()})
        return "FAILED"

    def run(self, jobs, max_jobs=None):
        jobs = list(jobs)
        if max_jobs is not None:
            jobs = jobs[:max_jobs]
        counts = {}
        for pos, job in enumerate(jobs, start=1):
            state = self.collect_one(job)
            counts[state] = counts.get(state, 0) + 1
            print(f"[{pos}/{len(jobs)}] {state:>22} {job.group}/{job.name}")
            if self.pause > 0 and state in {"OK", "NO_SELECTED_VARIETIES", "FAILED"}:
                time.sleep(self.pause)
        return counts
