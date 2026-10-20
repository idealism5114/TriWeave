"""Command line interface. '--plan' never touches remote market data."""

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd

from triweave.assets import YIELD_SPECS, INDEX_SPECS, TARGET_FUTURES, AUX_FUTURES, validate_asset_configuration
from triweave.audit import coverage_audit, expected_asset_registry
from triweave.collect import Collector, make_jobs
from triweave.universe import import_stock_universe, load_stock_universe


def project_config(root):
    return json.loads((Path(root) / "configs" / "project.json").read_text(encoding="utf-8"))


def project_root(arg):
    root = Path(arg).resolve()
    if not (root / "configs" / "project.json").exists():
        raise FileNotFoundError(f"Not a TriWeave project folder: {root} (missing configs/project.json)")
    return root


def init_project(root):
    for name in (
        "data/raw", "data/standardized", "data/derived", "data/sealed",
        "data/manifests", "data/audit", "experiments", "checkpoints", "notebooks", "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    folder = root / "data" / "manifests"
    reference = folder / "non_stock_registry.csv"
    if not reference.exists():
        expected_asset_registry().to_csv(reference, index=False)
    print("Directory tree ready.")
    print(f"Non-stock source registry: {reference}")
    print("Stock manifest is intentionally NOT invented. Import actual 263 Clean V1 tickers.")


def parser():
    p = argparse.ArgumentParser(prog="triweave", description="TriWeave 15-year market-data collection")
    p.add_argument("--root", default=".", help="Project root with configs/project.json; default cwd")
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("init", help="Create research directories and nonstock registry")

    imp = subs.add_parser("import-stocks", help="Import EXACT 263 ticker IDs from old Clean V1 manifest")
    imp.add_argument("--input", required=True, help="CSV with code column or wide price CSV, or Clean V1 asset JSON")
    imp.add_argument("--json-key", help="Dot-delimited JSON key when original JSON structure is unusual")
    imp.add_argument("--expected", type=int, default=263, help="Use default 263 for production; other counts ONLY for tests")
    imp.add_argument("--add-phase-a-15", action="store_true", help="Reconstruct 248 legacy stocks + 15 Phase A-Fix reported codes (must separately verify)")

    for cmd in ("plan", "collect"):
        c = subs.add_parser(cmd, help="List jobs without network" if cmd == "plan" else "Download auditable raw daily source tables")
        c.add_argument("--groups", nargs="+", default=["bonds", "futures_continuous"],
                       choices=["stocks", "bonds", "futures_continuous", "futures_contracts", "all"])
        c.add_argument("--start", help="YYYY-MM-DD, default frozen 15-year project start")
        c.add_argument("--end", help="YYYY-MM-DD, default frozen 15-year project end")
        c.add_argument("--exchanges", nargs="+", default=["DCE", "SHFE", "CZCE", "INE", "GFEX", "CFFEX"])
        c.add_argument("--max-jobs", type=int, default=None, help="Limit the executed/listed jobs for smoke tests")
        if cmd == "collect":
            c.add_argument("--run-id", default="baseline15y", help="Use new run-id for refresh; do not overwrite old snapshots")
            c.add_argument("--pause", type=float, default=None, help="Seconds between successful requests")
            c.add_argument("--strict", action="store_true", help="Return failure status if any job fails")
    audit = subs.add_parser("audit", help="Per-asset collection coverage report (no labels)")
    audit.add_argument("--run-id", default="baseline15y")
    subs.add_parser("doctor", help="Check local dependencies and universe readiness")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = project_root(args.root)
    cfg = project_config(root)
    validate_asset_configuration(root)
    stock_path = root / "data" / "manifests" / "stocks_263.csv"
    if args.command == "init":
        init_project(root)
        return 0
    if args.command == "import-stocks":
        result = import_stock_universe(Path(args.input), stock_path, expected=args.expected, json_key=args.json_key,
                                       add_phase_a_15=args.add_phase_a_15)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "doctor":
        try:
            import akshare
            ak_version = getattr(akshare, "__version__", "unknown")
        except ImportError:
            ak_version = "NOT_INSTALLED"
        result = {
            "akshare_version": ak_version,
            "research_start": cfg["research_start"],
            "research_end": cfg["research_end"],
            "configured_stocks": cfg["stock_expected_count"],
            "stock_manifest_exists": stock_path.exists(),
            "stock_manifest_count": len(pd.read_csv(stock_path)) if stock_path.exists() else 0,
            "bond_variables": len(YIELD_SPECS) + len(INDEX_SPECS),
            "target_futures": len(TARGET_FUTURES),
            "auxiliary_futures": len(AUX_FUTURES),
            "note": "AKShare installed does not imply API connectivity; use 1-job smoke collection.",
        }
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "audit":
        stocks = load_stock_universe(stock_path) if stock_path.exists() else None
        out = coverage_audit(root, args.run_id, stocks)
        print(json.dumps(out, indent=2))
        return 0
    groups = args.groups
    stocks = load_stock_universe(stock_path) if ("stocks" in groups or "all" in groups) else None
    jobs = make_jobs(groups, args.start or cfg["research_start"], args.end or cfg["research_end"],
                     stock_universe=stocks, curve_chunk_days=int(cfg["curve_chunk_days"]), markets=args.exchanges)
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise ValueError("--max-jobs must be positive")
    if args.command == "plan":
        print(json.dumps({
            "requested_groups": groups, "research_start": args.start or cfg["research_start"],
            "research_end": args.end or cfg["research_end"],
            "jobs_total": len(jobs), "jobs_by_group": dict(Counter(x.group for x in jobs)),
            "first_jobs": [f"{x.group}/{x.name}" for x in jobs[:min(args.max_jobs or 10, 20)]],
            "note": "Job count is not evidence that the data exist historically. No network accessed.",
        }, indent=2))
        return 0
    if args.command == "collect":
        collector = Collector(root, args.run_id, retries=cfg["retry_attempts"],
                              pause=args.pause if args.pause is not None else cfg["request_pause_seconds"])
        statuses = collector.run(jobs, max_jobs=args.max_jobs)
        print(json.dumps({"run_id": args.run_id, "job_statuses": statuses}, indent=2))
        if args.strict and statuses.get("FAILED", 0):
            return 1
        return 0
    raise RuntimeError("Unexpected command")


if __name__ == "__main__":
    raise SystemExit(main())
