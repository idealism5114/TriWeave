"""Coverage report for collected raw/standardized series; never an MAE evaluation."""
from pathlib import Path
from collections import defaultdict
import json

import pandas as pd

from triweave.assets import (
    YIELD_SPECS, INDEX_SPECS, TARGET_FUTURES, AUX_FUTURES, BOND_FUTURES,
)
from triweave.collect import atomic_json


def expected_asset_registry(stock_universe=None):
    records = []
    if stock_universe is not None:
        for item in stock_universe["asset_id"].tolist():
            records.append({"asset_id": item, "market": "stock", "role": "input", "unit": "RMB_per_share_unadjusted"})
    for asset in YIELD_SPECS:
        records.append({"asset_id": asset, "market": "bond", "role": "input", "unit": "percent_per_annum"})
    for asset in INDEX_SPECS:
        records.append({"asset_id": asset, "market": "bond", "role": "input", "unit": "index_level"})
    for spec in TARGET_FUTURES + AUX_FUTURES + BOND_FUTURES:
        records.append({"asset_id": f"FUT_{spec.variety}", "market": "bond_future" if spec.role == "bond_future" else "commodity_future", "role": spec.role, "unit": "exchange_contract_unit", "exchange": spec.exchange, "mapping_review_required": spec.review_required})
    return pd.DataFrame(records)


def coverage_audit(root, run_id, stock_universe=None):
    root = Path(root)
    if (root / "configs/project.json").exists():
        # Production audit separates exploratory continuous quotes from contracts.
        # The small standalone summary below remains for fixture-only roots.
        from triweave.research_audit import comprehensive_audit
        return comprehensive_audit(root, run_id)
    scan = root / "data" / "standardized" / run_id
    report_root = root / "data" / "audit" / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    accum = defaultdict(lambda: {"dates": set(), "rows": 0, "valid_rows": 0, "first": None, "last": None, "groups": set(), "contracts": set()})
    files_scanned = 0
    rows_scanned = 0
    bad_dates = 0
    if scan.exists():
        for file in sorted(scan.rglob("*.csv.gz")):
            frame = pd.read_csv(file, dtype={"asset_id": str})
            files_scanned += 1
            rows_scanned += len(frame)
            if frame.empty:
                continue
            for asset_id, sub in frame.groupby("asset_id"):
                entry = accum[asset_id]
                dates = pd.to_datetime(sub["date"], errors="coerce")
                bad_dates += int(dates.isna().sum())
                dates = dates.dropna()
                entry["dates"].update(dates.dt.strftime("%Y-%m-%d").tolist())
                entry["rows"] += len(sub)
                flag = "valid_close" if "valid_close" in sub.columns else "valid_value"
                if flag in sub:
                    entry["valid_rows"] += int(sub[flag].astype(str).str.lower().eq("true").sum())
                entry["groups"].add(str(file.relative_to(scan).parent))
                if "contract" in sub.columns:
                    entry["contracts"].update(sub["contract"].dropna().astype(str).tolist())
    expected = expected_asset_registry(stock_universe)
    records = []
    for row in expected.to_dict("records"):
        asset_id = row["asset_id"]
        entry = accum[asset_id]
        dates = sorted(entry["dates"])
        records.append({
            **row,
            "unique_observation_dates": len(dates),
            "row_count": entry["rows"],
            "valid_row_count": entry["valid_rows"],
            "first_observed_date": dates[0] if dates else None,
            "last_observed_date": dates[-1] if dates else None,
            "contract_count": len(entry["contracts"]),
            "collection_status": "COLLECTED_PARTIAL_OR_FULL" if entry["rows"] > 0 else "NOT_COLLECTED_OR_NO_DATA",
            "source_groups": ";".join(sorted(entry["groups"])),
        })
    df = pd.DataFrame(records)
    df.to_csv(report_root / "asset_coverage.csv", index=False)
    stats = {
        "run_id": run_id,
        "scanned_standardized_files": files_scanned,
        "scanned_rows": rows_scanned,
        "bad_dates": bad_dates,
        "expected_stocks": 263,
        "stock_universe_supplied": stock_universe is not None,
        "expected_bond_series": 25,
        "expected_commodity_targets": 24,
        "expected_commodity_auxiliary": 14,
        "expected_optional_treasury_futures": 4,
        "assets_with_any_rows": int((df["row_count"] > 0).sum()),
        "assets_without_rows": df.loc[df["row_count"] == 0, "asset_id"].tolist(),
        "partial_coverage_warning": "Missing records are not zero prices. A full 15-year series is not guaranteed by API availability or listing history.",
        "time_availability_warning": "These are historical observation dates; publication times, futures roll and corporate actions have not been audited.",
    }
    atomic_json(stats, report_root / "report.json")
    text = [
        "# TriWeave acquisition audit",
        "",
        f"Run ID: `{run_id}`",
        f"Standardized files scanned: {files_scanned}",
        f"Rows scanned: {rows_scanned}",
        f"Series with any data: {stats['assets_with_any_rows']}",
        f"Series without data: {stats['assets_without_rows'].__len__()}",
        "",
        "This report is about ingestion coverage, not information availability at a trade decision time.",
        "Do not apply bfill or infer model returns from vendor continuous futures.",
        "See asset_coverage.csv for per-asset first and last observed dates.",
    ]
    (report_root / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return stats
