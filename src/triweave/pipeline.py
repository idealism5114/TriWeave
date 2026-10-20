"""Live smoke gates and resumable collection. Never constructs model labels."""
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys

import pandas as pd

from triweave.assets import validate_asset_configuration
from triweave.cli import project_config
from triweave.collect import Collector, atomic_json, make_jobs, utc_now, append_log
from triweave.transport import observed_http
from triweave.universe import load_stock_universe, sha256_file


@contextmanager
def call_deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError(f"Provider API exceeded {seconds}s wall-clock limit")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TimedApi:
    def __init__(self, api, timeout=180):
        self.api, self.timeout = api, timeout

    def __getattr__(self, name):
        value = getattr(self.api, name)
        if not callable(value):
            return value
        def call(**kwargs):
            with call_deadline(self.timeout):
                return value(**kwargs)
        return call


def environment(root):
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    return {"captured_utc": utc_now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": dict(sorted(packages.items())),
            "disk": shutil.disk_usage(root)._asdict(),
            "source_sha256": {str(p.relative_to(root)): sha256_file(p)
                               for p in sorted((root / 'src/triweave').glob('*.py'))},
            "config_sha256": {str(p.relative_to(root)): sha256_file(p)
                               for p in sorted((root / 'configs').glob('*')) if p.is_file()}}


def read_events(root, run_id, filename="events.jsonl"):
    file = root / "data/manifests" / run_id / filename
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()] if file.exists() else []


def frame_schema(frame, job):
    if job.group in ("bond_yield_curves", "bond_wealth_indices"):
        required = ["date", "asset_id", "value", "unit"]
        numeric = ["value"]
    else:
        required = ["date", "asset_id", "open", "high", "low", "close", "volume"]
        numeric = ["open", "high", "low", "close", "volume"]
        if job.group == "stocks":
            required += ["amount", "turnover"]
            numeric += ["amount", "turnover"]
        else:
            required += ["open_interest"]
            numeric += ["open_interest"]
        if job.group.startswith("futures_contracts/"):
            required += ["contract", "settle"]
            numeric += ["settle"]
    absent = [col for col in required if col not in frame]
    empty = [col for col in numeric if col in frame and pd.to_numeric(frame[col], errors="coerce").isna().all()]
    keys = ["asset_id", "date"] + (["contract"] if "contract" in frame else [])
    duplicate = int(frame.duplicated(keys).sum()) if not absent else None
    dates = pd.to_datetime(frame.get("date", pd.Series(dtype=str)), errors="coerce")
    passed = not frame.empty and not absent and not empty and duplicate == 0 and not dates.isna().any()
    return {"pass": bool(passed), "missing_columns": absent, "all_null_numeric_columns": empty,
            "record_count": len(frame), "unique_date_count": int(dates.nunique()),
            "duplicate_key_count": duplicate, "missing_value_count": int(frame.isna().sum().sum()),
            "first_observation_date": str(dates.min().date()) if not dates.empty else None,
            "last_observation_date": str(dates.max().date()) if not dates.empty else None,
            "columns": list(frame.columns)}


def run_smoke(root, api, run_id):
    out = root / "experiments/data_collection/smoke_test"
    out.mkdir(parents=True, exist_ok=True)
    stocks = load_stock_universe(root / "data/manifests/stocks_263.csv")
    dates = ("2023-10-01", "2023-10-31")
    bonds = make_jobs(["bonds"], *dates)
    continuous = next(j for j in make_jobs(["futures_continuous"], *dates) if j.name == "SHFE_RB_0")
    jobs = [bonds[0], bonds[1], make_jobs(["stocks"], *dates, stock_universe=stocks)[0], continuous]
    jobs += make_jobs(["futures_contracts"], *dates, markets=["SHFE", "DCE", "CZCE", "INE", "GFEX", "CFFEX"])
    collector = Collector(root, run_id, api=TimedApi(api), retries=2, pause=1)
    results = []
    for pos, job in enumerate(jobs, 1):
        print(f"SMOKE {pos}/{len(jobs)} {job.api} {job.name}", flush=True)
        state = collector.collect_one(job)
        raw, clean, meta = collector.paths(job)
        item = {"group": job.group, "name": job.name, "api": job.api,
                "api_kwargs": job.kwargs, "status": state, "pass": False}
        if state in ("OK", "CACHED"):
            frame = pd.read_csv(clean)
            item.update(frame_schema(frame, job))
            item["manifest"] = str(meta.relative_to(root))
            raw_frame = pd.read_csv(raw)
            item["raw_columns"] = list(raw_frame.columns)
            item["raw_dtypes"] = {str(k): str(v) for k, v in raw_frame.dtypes.items()}
            item["raw_rows"] = len(raw_frame)
            item["raw_sha256"] = sha256_file(raw)
        else:
            errors = [e for e in read_events(root, run_id) if e.get("name") == job.name and e.get("error")]
            item["errors"] = errors
        results.append(item)
        atomic_json({"run_id": run_id, "requested_start": dates[0], "requested_end": dates[1],
                     "complete": False, "results": results}, out / "results.json")
        print(f"SMOKE RESULT {job.name}: {item['pass']} ({state})", flush=True)
    result = {"run_id": run_id, "requested_start": dates[0], "requested_end": dates[1],
              "complete": True, "results": results}
    atomic_json(result, out / "results.json")
    atomic_json([e for e in read_events(root, run_id) if e.get("error")], out / "api_errors.json")
    atomic_json(environment(root), out / "environment.json")
    lines = ["# Live API smoke test", "", "Requested dates: 2023-10-01 through 2023-10-31.",
             "A successful HTTP response alone does not pass this test.", "",
             "| API / asset | Passed | Rows | Actual first | Actual last |", "|---|---|---:|---|---|"]
    for item in results:
        lines.append(f"| {item['api']} / {item['name']} | {item['pass']} | {item.get('record_count', 0)} | {item.get('first_observation_date', '')} | {item.get('last_observation_date', '')} |")
    lines += ["", "Failures and original exception tracebacks are in api_errors.json and the run manifests.",
              "Only the corresponding passed API family/exchange may proceed to bulk collection."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def partition_jobs(jobs):
    pre = [j for j in jobs if j.group not in ("stocks",) and not j.group.startswith("futures_contracts/")]
    exchanges = {}
    for job in jobs:
        if job.group.startswith("futures_contracts/"):
            exchanges.setdefault(job.context, []).append(job)
    stocks = [j for j in jobs if j.group == "stocks"]
    return pre, exchanges, stocks


def _run_batch(root, run_id, jobs, passed, api=None):
    root = Path(root)
    cfg = project_config(root)
    if api is None:
        import akshare as api
    collector = Collector(root, run_id, api=TimedApi(api), retries=int(cfg['retry_attempts']), pause=1)
    decisions = []
    stopped = set()
    consecutive = Counter()
    with observed_http(root, run_id, timeout=20, pause=1):
        for pos, job in enumerate(jobs, 1):
            gate = (job.api, job.group)
            if gate not in passed or gate in stopped:
                status = "BLOCKED_SMOKE" if gate not in passed else "BLOCKED_ACCESS_FAILURES"
            else:
                try:
                    status = collector.collect_one(job)
                except (RuntimeError, FileExistsError) as exc:
                    # Local integrity/precondition failures must preserve files and
                    # remain visible without cancelling unrelated exchanges/stocks.
                    import traceback
                    status = 'FAILED'
                    append_log(collector.logs, {'event':'JOB_FAILED','name':job.name,'group':job.group,
                        'api':job.api,'api_kwargs':job.kwargs,'requested_start':job.start,'requested_end':job.end,
                        'error':repr(exc),'at':utc_now(),'traceback':traceback.format_exc(),
                        'failure_kind':'LOCAL_PRECONDITION_NO_OVERWRITE'})
                if status == "FAILED":
                    latest = [e for e in read_events(root, run_id) if e.get('name') == job.name and e.get('event') == 'JOB_FAILED']
                    error = latest[-1].get('error', '') if latest else ''
                    access_error = any(text in error for text in ('403', '412', '429', '567', 'ProxyError', 'ConnectionError', 'Timeout'))
                    consecutive[gate] = consecutive[gate] + 1 if access_error else 0
                    if consecutive[gate] >= 3:
                        stopped.add(gate)
                else:
                    consecutive[gate] = 0
            item = {"group": job.group, "name": job.name, "api": job.api, "api_kwargs": job.kwargs,
                    "requested_start": job.start, "requested_end": job.end, "status": status}
            decisions.append(item)
            append_log(root / 'data/manifests' / run_id / 'execution_decisions.jsonl', {**item, 'at': utc_now()})
            if not status.startswith('BLOCKED'):
                print(f"COLLECT {pos}/{len(jobs)} {status} {job.group}/{job.name}", flush=True)
                if status not in ('CACHED', 'CACHED_EMPTY_SELECTION'):
                    import time
                    time.sleep(float(cfg['request_pause_seconds']))
    return decisions


def run_collection(root, api, run_id, exchange_workers=1):
    smoke = json.loads((root / "experiments/data_collection/smoke_test/results.json").read_text())
    if not smoke.get("complete"):
        raise RuntimeError("Smoke tests have not completed")
    cfg = project_config(root)
    stocks = load_stock_universe(root / "data/manifests/stocks_263.csv")
    jobs = make_jobs(["all"], cfg["research_start"], cfg["research_end"], stock_universe=stocks)
    order = {"bond_yield_curves": 0, "bond_wealth_indices": 1, "futures_continuous_exploratory": 2, "stocks": 4}
    jobs.sort(key=lambda j: order.get(j.group, 3))
    passed = {(r['api'], r['group']) for r in smoke['results'] if r['pass']}
    pre, exchanges, stock_jobs = partition_jobs(jobs)
    decisions = _run_batch(root, run_id, pre, passed, api)
    if exchange_workers > 1 and exchanges:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        # One process per exchange; its network requests remain sequential and rate limited.
        with ProcessPoolExecutor(max_workers=min(exchange_workers, len(exchanges)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_run_batch, str(root), run_id, batch, passed) for batch in exchanges.values()]
            for future in as_completed(futures):
                decisions.extend(future.result())
    else:
        for batch in exchanges.values():
            decisions.extend(_run_batch(root, run_id, batch, passed, api))
    decisions.extend(_run_batch(root, run_id, stock_jobs, passed, api))
    assert len(decisions) == len(jobs)
    assert len({(d['group'], d['name']) for d in decisions}) == len(decisions)
    counts = Counter(d['status'] for d in decisions)
    atomic_json({"run_id": run_id, "counts": dict(counts), "decisions": decisions,
                 "exchange_workers": exchange_workers,
                 "smoke_run_id": smoke['run_id'], "finished_utc": utc_now()},
                root / 'data/manifests' / run_id / 'execution_summary.json')
    print(json.dumps(dict(counts)), flush=True)
    return decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'collect'])
    parser.add_argument('--root', default='.')
    parser.add_argument('--run-id')
    parser.add_argument('--exchange-workers',type=int,default=1,choices=range(1,5))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    validate_asset_configuration(root)
    import akshare as ak
    details = root / 'experiments/data_collection/d1'
    details.mkdir(parents=True, exist_ok=True)
    signatures = {}
    for name in ('bond_china_yield', 'bond_treasury_index_cbond', 'stock_zh_a_hist', 'futures_zh_daily_sina', 'get_futures_daily'):
        function = getattr(ak, name)
        signatures[name] = {'signature': str(inspect.signature(function)), 'source_file': inspect.getsourcefile(function)}
        (details / f'{name}.py').write_text(inspect.getsource(function))
    atomic_json(signatures, details / 'api_signatures.json')
    atomic_json(environment(root), details / 'environment.json')
    run_id = args.run_id or ('smoke_202310' if args.stage == 'smoke' else 'baseline15y')
    if args.stage == 'smoke':
        with observed_http(root, run_id, timeout=20, pause=1):
            run_smoke(root, ak, run_id)
    else:
        run_collection(root, ak, run_id, exchange_workers=args.exchange_workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
