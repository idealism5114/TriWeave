"""Phase E: offline, date-gated TRAIN/VAL preprocessing and masked mmap datasets.

No provider imports, network calls, TEST ingestion, or model code. Timestamps are
integer UTC seconds; all publication times are explicit exploratory assumptions.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.assets import YIELD_SPECS, INDEX_SPECS, validate_asset_configuration
from triweave.universe import load_stock_universe, sha256_file

PROTOCOL = "TRIWEAVE_25_AFTER_CLOSE_V1_EXPLORATORY"
TARGETS = "LU SC CF PG HC J JM FU C MA SR AG OI RM RB M CU AL ZN SN NI A AU I Y".split()
HORIZONS = (1, 5, 10, 20)
SPLITS = {"TRAIN": ("2011-10-01", "2022-12-31"),
          "VAL": ("2023-01-01", "2024-12-31"),
          "TEST": ("2025-01-01", "2026-09-30")}
KINDS = ("STOCK_PRICE", "BOND_YIELD", "BOND_WEALTH_INDEX", "FUTURES_CONTINUOUS_PRICE")
PRICE_KINDS = (KINDS[0], KINDS[2], KINDS[3])
END = "2024-12-31"
DATA_REL = "data/derived/triweave_25_after_close_v1"
REPORT_REL = "experiments/phase_e_after_close"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def array_sha(a):
    """Exact deterministic .npy content digest, including dtype and shape."""
    import io
    f = io.BytesIO()
    np.save(f, a, allow_pickle=False)
    return hashlib.sha256(f.getvalue()).hexdigest()


def safe_csv(path, start="2011-10-01", end=END, log=None):
    """Gate the first ISO-date field BEFORE CSV parsing/converting any quote fields.

    Source files have one physical line per record, with date first. Buffers may
    contain bytes of excluded rows, but their numeric fields are never parsed,
    materialized as records, converted, or used. Excluded dates are not profiled.
    """
    if end > END or start > END:
        raise ValueError("TEST quote ingestion is disabled")
    opener = gzip.open if str(path).endswith(".gz") else open
    records = []
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader([next(f)]))
        if not header or header[0] != "date":
            raise ValueError("Date-first single-line CSV schema required")
        for line in f:
            date = line.partition(",")[0].strip().strip('"')
            # Check date syntax without touching the numeric suffix.
            if len(date) != 10 or date[4] != "-" or date[7] != "-":
                raise ValueError("Invalid date prefix")
            if not start <= date <= end:
                continue
            pd.Timestamp(date)  # Reject impossible in-scope dates.
            row = next(csv.reader([line]))
            if len(row) != len(header):
                raise ValueError("Single-line source schema mismatch")
            records.append(dict(zip(header, row)))
    frame = pd.DataFrame(records, columns=header)
    if log is not None:
        log.append({"path": str(path), "parsed_rows": len(frame),
                    "min_parsed_date": frame.date.min() if len(frame) else None,
                    "max_parsed_date": frame.date.max() if len(frame) else None,
                    "date_gate_before_quote_parser": True, "TEST_numeric_rows_parsed": 0})
    return frame


def asset_definitions(root):
    root = Path(root)
    validate_asset_configuration(root)
    stocks = load_stock_universe(root / "data/manifests/stocks_263.csv").asset_id.tolist()
    config = pd.read_csv(root / "configs/futures_targets_25_v1.csv", keep_default_na=False)
    assert config.variety.tolist() == TARGETS and config.asset_id.tolist() == ["FUT_" + c for c in TARGETS]
    assert config.order.tolist() == list(range(25)) and not config.variety.duplicated().any()
    assets = []
    for ids, kind, market, unit in [
        (stocks, KINDS[0], "stock", "RMB_per_share"),
        (list(YIELD_SPECS), KINDS[1], "bond", "percent_per_annum"),
        (list(INDEX_SPECS), KINDS[2], "bond", "index_level"),
        (config.asset_id.tolist(), KINDS[3], "future", "native_vendor_price_unit_unverified")]:
        for asset in ids:
            assets.append({"index": len(assets), "asset_id": asset, "economic_type": kind,
                           "market": market, "unit": unit, "available_at_status": "AVAILABLE_AT_ASSUMED"})
    assert len(assets) == 313 and len({x["asset_id"] for x in assets}) == 313
    return assets


def load_sources(root, assets):
    """Reuse frozen lineage, recheck bytes; never call the old full-history reader."""
    root = Path(root)
    old = json.loads((root / "experiments/phase_d_continuous/provenance.json").read_text())
    selected = set(TARGETS)
    entries = [e for e in old["source_files"]
               if e["source_id"] != "sina_continuous_exploratory_v1"
               or e["parameters"]["symbol"][:-1] in selected]
    frames = {a["asset_id"]: [] for a in assets}
    logs = []
    for i, entry in enumerate(entries):
        for key, digest in [("manifest", "manifest_sha256"), ("raw_file", "raw_sha256"),
                            ("standardized_file", "standardized_sha256")]:
            if sha256_file(root / entry[key]) != entry[digest]:
                raise ValueError("Source checksum mismatch: " + entry[key])
        frame = safe_csv(root / entry["standardized_file"], log=logs)
        if frame.empty:
            continue
        frame["__source"] = i
        for asset, sub in frame.groupby("asset_id", sort=False):
            if asset not in frames:
                raise ValueError("Unexpected source asset " + asset)
            frames[asset].append(sub)
    result = {}
    for a in assets:
        asset = a["asset_id"]
        if not frames[asset]:
            raise ValueError("No TRAIN/VAL source for " + asset)
        frame = pd.concat(frames[asset], ignore_index=True)
        if a["economic_type"] in KINDS[1:3] and not frame.unit.eq(a["unit"]).all():
            raise ValueError("Bond units changed")
        if a["economic_type"] == KINDS[0]:
            if not (frame.price_unit.eq("RMB_per_share") & frame.volume_unit.eq("shares")
                    & frame.amount_unit.eq("RMB") & frame.turnover_unit.eq("percent")
                    & frame.adjustment.eq("unadjusted")).all():
                raise ValueError("Stock source/units mismatch")
        originally_sorted = bool(frame.date.is_monotonic_increasing)
        result[asset] = frame.sort_values("date", kind="stable").reset_index(drop=True)
        result[asset].attrs["originally_sorted"] = originally_sorted
    return result, entries, logs


def clock_seconds(dates, local_hour, local_minute=0):
    return (np.asarray(dates, dtype="datetime64[s]").astype(np.int64)
            + local_hour * 3600 + local_minute * 60 - 8 * 3600)


def publication_seconds(dates, kind, full_calendar, protocol):
    p = protocol["publication_assumptions"][kind]
    hh, mm, _ = map(int, p["local_time"].split(":"))
    if p["reference_session_delay"] == 0:
        return clock_seconds(dates, hh, mm)
    # First strictly later reference day, including weekend bond observations.
    c = np.asarray(full_calendar, dtype="U10")
    idx = np.searchsorted(c, np.asarray(dates), side="right")
    if (idx >= len(c)).any():
        raise ValueError("Calendar does not extend to assumed publication")
    return clock_seconds(c[idx], hh, mm)


def quote_quality(frame, kind):
    field = "value" if kind in KINDS[1:3] else "close"
    value = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
    good = np.isfinite(value)
    if kind != KINDS[1]:
        good &= value > 0
    if kind in (KINDS[0], KINDS[3]):
        p = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        good &= np.isfinite(p).all(1) & (p > 0).all(1)
        good &= (p[:, 1] >= np.max(p[:, [0, 2, 3]], axis=1)) & (p[:, 2] <= np.min(p[:, [0, 1, 3]], axis=1))
    if kind == KINDS[0]:
        good &= frame.trading_status.eq("1").to_numpy()
    good &= ~frame.date.duplicated(keep=False).to_numpy()
    return value, good


def align_sources(frames, assets, full_calendar, protocol):
    dates = np.array([d for d in full_calendar if SPLITS["TRAIN"][0] <= d <= END], dtype="U10")
    t, a = len(dates), len(assets)
    decision = clock_seconds(dates, 20)
    aligned = {k: np.full((t, a), np.nan) for k in
               ("value", "open", "high", "low", "volume", "amount", "turnover", "open_interest")}
    aligned.update({"valid": np.zeros((t, a), bool), "observation_day": np.full((t, a), -1, np.int32),
                    "available_at": np.full((t, a), -1, np.int64),
                    "source_index": np.full((t, a), -1, np.int16),
                    "age_days": np.full((t, a), -1, np.int16),
                    "new_observation": np.zeros((t, a), bool),
                    "trading_status": np.full((t, a), -1, np.int8)})
    quality = []
    day_numbers = dates.astype("datetime64[D]").astype(int)
    for j, spec in enumerate(assets):
        f = frames[spec["asset_id"]].copy()
        kind = spec["economic_type"]
        value, good = quote_quality(f, kind)
        pub = publication_seconds(f.date.to_numpy(), kind, full_calendar, protocol)
        duplicate = f.date.duplicated(keep=False)
        # Both duplicate records remain invalid; the index is only for provenance.
        unique = ~f.date.duplicated(keep="first").to_numpy()
        oi = np.flatnonzero(unique)
        obs_dates = f.date.to_numpy()[oi]
        if protocol["publication_assumptions"][kind]["carry"]:
            # Equal publication times: latest original observation date wins.
            idx = np.searchsorted(pub[oi], decision, side="right") - 1
            found = idx >= 0
            rows = oi[np.maximum(idx, 0)]
        else:
            idx = np.searchsorted(obs_dates, dates)
            rows = oi[np.minimum(idx, len(oi) - 1)]
            found = (idx < len(oi)) & (f.date.to_numpy()[rows] == dates) & (pub[rows] <= decision)
        selected = np.flatnonzero(found)
        source_rows = rows[found]
        age = day_numbers[found] - f.date.to_numpy().astype("datetime64[D]").astype(int)[source_rows]
        usable = good[source_rows].copy()
        if kind in KINDS[1:3]:
            usable &= age <= protocol["publication_assumptions"][kind]["max_observation_age_calendar_days"]
        aligned["valid"][selected, j] = usable
        aligned["value"][selected, j] = value[source_rows]
        aligned["observation_day"][selected, j] = f.date.to_numpy().astype("datetime64[D]").astype(int)[source_rows]
        aligned["available_at"][selected, j] = pub[source_rows]
        aligned["source_index"][selected, j] = f.__source.to_numpy(int)[source_rows]
        aligned["age_days"][selected, j] = age
        aligned["new_observation"][selected, j] = np.r_[True, np.diff(source_rows) != 0] & usable
        for field in ("open", "high", "low", "volume", "amount", "turnover", "open_interest"):
            if field in f:
                aligned[field][selected, j] = pd.to_numeric(f[field], errors="coerce").to_numpy(float)[source_rows]
        if kind == KINDS[0]:
            aligned["trading_status"][selected, j] = pd.to_numeric(f.trading_status, errors="coerce").fillna(-1).to_numpy(int)[source_rows]
        quality.append({"asset_id": spec["asset_id"], "economic_type": kind,
                        "in_scope_source_rows": len(f), "duplicate_rows_all_invalid": int(duplicate.sum()),
                        "bad_quote_rows_including_suspensions": int((~good).sum()),
                        "off_reference_calendar_rows": int((~f.date.isin(dates)).sum()),
                        "nonfinite_or_nonpositive_close": int((~np.isfinite(value) | ((value <= 0) if kind != KINDS[1] else False)).sum()),
                        "negative_volume_rows": int(pd.to_numeric(f.get("volume", pd.Series(dtype=float)), errors="coerce").lt(0).sum()),
                        "source_needed_sort": not f.attrs.get("originally_sorted", bool(f.date.is_monotonic_increasing))})
    aligned["dates"] = dates
    aligned["decision_time"] = decision
    return aligned, quality


def registry():
    rows = []
    def add(name, kinds, source, unit, transform, lookback, output_unit):
        rows.append(dict(index=len(rows), feature_name=name, market_type=list(kinds), economic_type=list(kinds),
                         source_field=source, original_unit=unit, transformation=transform,
                         lookback=lookback, output_unit=output_unit,
                         availability_rule="All required as-of reference quotes valid; available_at <= decision_time"))
    for h in HORIZONS:
        add(f"return_{h}d", PRICE_KINDS, "close or wealth value", "native price/index", "value[t]/value[t-h]-1; all h+1 valid", h+1, "decimal_change")
    for w in (5, 10, 20):
        add(f"ma_deviation_{w}d", PRICE_KINDS, "close or wealth value", "native price/index", "value[t]/mean(last w)-1", w, "fraction")
    for w in (5, 10, 20):
        add(f"volatility_{w}d", PRICE_KINDS, "close or wealth value", "native price/index", "population std of last w one-session changes; not annualized", w+1, "decimal_change")
    add("momentum_5_20", PRICE_KINDS, "close or wealth value", "native price/index", "return_5d-return_20d; return channels also encode short/medium momentum", 21, "decimal_change")
    add("volume_change_1d", (KINDS[0], KINDS[3]), "volume", "stock shares / futures native vendor lots basis unverified", "volume[t]/volume[t-1]-1; denominator >0, numerator >=0", 2, "fraction")
    add("amount_change_1d", (KINDS[0],), "amount", "RMB", "amount[t]/amount[t-1]-1; denominator >0, numerator >=0", 2, "fraction")
    add("turnover_pct", (KINDS[0],), "turnover", "percent", "identity; finite >=0", 1, "percent")
    add("open_interest_change_1d", (KINDS[3],), "open_interest", "native vendor lots basis unverified", "OI[t]/OI[t-1]-1; denominator >0, numerator >=0", 2, "fraction")
    add("ohlc_range", (KINDS[3],), "high,low,close", "native vendor price", "(high-low)/close", 1, "fraction")
    add("quote_available", KINDS, "quality-valid as-of quote mask", "boolean", "indicator including known absence; valid even when quote missing", 1, "boolean")
    add("quote_age_days", KINDS, "reference_date-observation_date", "calendar days", "age, only valid quotes; no carry for stocks/futures", 1, "calendar_days")
    add("yield_level_pct", (KINDS[1],), "value", "percent_per_annum", "identity; not a price", 1, "percent_per_annum")
    for h in HORIZONS:
        add(f"yield_delta_{h}d_bps", (KINDS[1],), "value", "percent_per_annum", "(value[t]-value[t-h])*100", h+1, "basis_points")
    for tenor in ("1Y", "3Y"):
        add(f"term_10Y_{tenor}_bps", (KINDS[1],), "CN_GOV_YIELD_10Y and " + tenor, "percent_per_annum", "(10Y-short)*100, only 10Y node, identical observation_date", 1, "basis_points")
    add("credit_spread_bps", (KINDS[1],), "AAA and government same tenor", "percent_per_annum", "(AAA-gov)*100 on AAA nodes, identical observation_date", 1, "basis_points")
    return rows


def rolling_sum(x, w):
    z = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=x.dtype), np.cumsum(x, axis=0)], axis=0)
    result = np.zeros_like(x)
    if w <= len(x):
        result[w-1:] = z[w:] - z[:-w]
    return result


def complete_window(valid, w):
    return rolling_sum(valid.astype(np.int32), w) == w


def lag(x, h, fill=0):
    r = np.full_like(x, fill)
    if h < len(x):
        r[h:] = x[:-h]
    return r


def feature_panel(aligned, assets):
    reg = registry()
    t, a = aligned["valid"].shape
    x = np.zeros((t, a, len(reg)), np.float64)
    valid = np.zeros(x.shape, bool)
    applicable = np.zeros((a, len(reg)), bool)
    kinds = np.array([s["economic_type"] for s in assets])
    ids = [s["asset_id"] for s in assets]
    for r in reg:
        applicable[:, r["index"]] = np.isin(kinds, r["economic_type"])
    for r in reg:
        if r["feature_name"].startswith("term_"):
            applicable[:, r["index"]] = np.array(ids) == "CN_GOV_YIELD_10Y"
        elif r["feature_name"] == "credit_spread_bps":
            applicable[:, r["index"]] = np.char.startswith(np.array(ids), "CN_AAA_")
    index = {r["feature_name"]: r["index"] for r in reg}
    def put(name, value, mask):
        k = index[name]
        m = np.broadcast_to(mask, (t, a)) & applicable[:, k][None] & np.isfinite(value)
        valid[:, :, k] = m
        x[:, :, k] = np.where(m, value, 0)
    q = aligned["valid"]
    value = np.where(q, aligned["value"], 0)
    returns = {}
    for h in HORIZONS:
        previous = lag(value, h)
        r = np.divide(value, previous, out=np.zeros_like(value), where=previous > 0) - 1
        returns[h] = r
        put(f"return_{h}d", r, complete_window(q, h+1))
        put(f"yield_delta_{h}d_bps", (value - previous)*100, complete_window(q, h+1))
    daily_mask = complete_window(q, 2)
    daily = np.where(daily_mask, returns[1], 0)
    for w in (5, 10, 20):
        mean = rolling_sum(value, w) / w
        put(f"ma_deviation_{w}d", np.divide(value, mean, out=np.zeros_like(value), where=mean > 0)-1, complete_window(q, w))
        # Causal cumulative sums, no future inputs in numerical cancellation.
        v = np.maximum(rolling_sum(daily*daily, w)/w - (rolling_sum(daily, w)/w)**2, 0)
        put(f"volatility_{w}d", np.sqrt(v), complete_window(daily_mask, w))
    put("momentum_5_20", returns[5]-returns[20], complete_window(q, 21))
    for field, name in [("volume", "volume_change_1d"), ("amount", "amount_change_1d"), ("open_interest", "open_interest_change_1d")]:
        z = aligned[field]
        prev = lag(z, 1, np.nan)
        good = q & lag(q, 1, False) & np.isfinite(z) & np.isfinite(prev) & (z >= 0) & (prev > 0)
        put(name, np.divide(z, prev, out=np.zeros_like(z), where=good)-1, good)
    put("turnover_pct", aligned["turnover"], q & (aligned["turnover"] >= 0))
    put("ohlc_range", np.divide(aligned["high"]-aligned["low"], value, out=np.zeros_like(value), where=value > 0), q)
    put("quote_available", q.astype(float), np.ones_like(q))
    put("quote_age_days", aligned["age_days"], q)
    put("yield_level_pct", value, q)
    for tenor in ("1Y", "3Y"):
        long, short = ids.index("CN_GOV_YIELD_10Y"), ids.index("CN_GOV_YIELD_"+tenor)
        z = np.zeros_like(value)
        m = np.zeros_like(q)
        z[:, long] = (value[:, long]-value[:, short])*100
        m[:, long] = q[:, long] & q[:, short] & (aligned["observation_day"][:, long] == aligned["observation_day"][:, short])
        put(f"term_10Y_{tenor}_bps", z, m)
    z = np.zeros_like(value)
    m = np.zeros_like(q)
    for j, asset in enumerate(ids):
        if asset.startswith("CN_AAA_"):
            other = ids.index("CN_GOV_YIELD_" + asset.rsplit("_", 1)[1])
            z[:, j] = (value[:, j]-value[:, other])*100
            m[:, j] = q[:, j] & q[:, other] & (aligned["observation_day"][:, j] == aligned["observation_day"][:, other])
    put("credit_spread_bps", z, m)
    return x, valid, applicable


def fit_scaler(x, valid, dates, assets, protocol):
    train = (np.asarray(dates) >= SPLITS["TRAIN"][0]) & (np.asarray(dates) <= SPLITS["TRAIN"][1])
    params = []
    rules = protocol["normalization"]
    train_x, train_valid = x[train], valid[train]
    for kind in KINDS:
        js = [i for i, a in enumerate(assets) if a["economic_type"] == kind]
        for r in registry():
            k = r["index"]
            if kind not in r["economic_type"]:
                continue
            z = train_x[:, js, k]
            m = train_valid[:, js, k]
            values = z[m]
            n = int(len(values))
            mean, std = (float(values.mean()), float(values.std(ddof=0))) if n else (0., 0.)
            insufficient = n < rules["minimum_train_count"]
            constant = bool(n and std == 0)
            floor = std < rules["epsilon"]
            params.append(dict(economic_type=kind, feature_name=r["feature_name"], feature_index=k,
                               valid_train_count=n, mean_train=mean, std_train=std,
                               effective_scale=max(std, rules["epsilon"]),
                               constant_flag=constant, scale_floor_flag=bool(floor),
                               insufficient_train_support=insufficient,
                               statuses=(["CONSTANT_FEATURE"] if constant else []) +
                               (["SCALE_FLOOR_TRIGGERED"] if floor else []) +
                               (["INSUFFICIENT_TRAIN_SUPPORT"] if insufficient else [])))
    return params


def transform(x, valid, assets, params):
    z = np.zeros(x.shape, np.float32)
    m = valid.copy()
    for p in params:
        js = [i for i, a in enumerate(assets) if a["economic_type"] == p["economic_type"]]
        k = p["feature_index"]
        if p["insufficient_train_support"]:
            m[:, js, k] = False
            continue
        v = (x[:, js, k]-p["mean_train"]) / p["effective_scale"]
        z[:, js, k] = np.where(m[:, js, k], v, 0).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite normalized features")
    return z, m


def labels(aligned, assets, split):
    if split not in ("TRAIN", "VAL"):
        raise ValueError("TEST labels are disabled in Phase E")
    dates = aligned["dates"]
    if len(dates) and max(dates) > END:
        raise ValueError("TEST numerical panel forbidden")
    js = [next(i for i, a in enumerate(assets) if a["asset_id"] == "FUT_"+c) for c in TARGETS]
    q = aligned["valid"][:, js]
    price = aligned["value"][:, js]
    y = np.zeros((len(dates), len(HORIZONS), len(js)), np.float32)
    masks = np.zeros(y.shape, bool)
    endpoints = np.full((len(dates), len(HORIZONS)), -1, np.int32)
    lo, hi = SPLITS[split]
    own = (dates >= lo) & (dates <= hi)
    for k, h in enumerate(HORIZONS):
        for i in np.flatnonzero(own):
            end = i+h
            if end >= len(dates) or not lo <= dates[end] <= hi:
                continue
            m = q[i:end+1].all(0)
            masks[i, k] = m
            y[i, k, m] = (price[end, m]/price[i, m]-1).astype(np.float32)
            endpoints[i, k] = end
    origins = np.flatnonzero(masks.any((1, 2))).astype(np.int32)
    return y, masks, endpoints, origins


def dataset_arrays(aligned, assets, x, feature_valid, applicable):
    values = np.where(aligned["valid"], aligned["value"], 0.)
    arrays = {"features": x, "feature_valid_mask": feature_valid, "feature_applicable_mask": applicable,
              "input_valid_mask": aligned["valid"], "aligned_values": values,
              "reference_dates": aligned["dates"], "decision_time": aligned["decision_time"]}
    for name in ("observation_day", "available_at", "source_index", "age_days", "new_observation", "trading_status"):
        arrays[name] = aligned[name]
    for split in ("TRAIN", "VAL"):
        y, m, ends, origins = labels(aligned, assets, split)
        prefix = split.lower()
        arrays[prefix+"_targets"] = y[origins]
        arrays[prefix+"_label_valid_mask"] = m[origins]
        arrays[prefix+"_origin_indices"] = origins
        arrays[prefix+"_endpoint_indices"] = ends[origins]
        indices = origins[:, None] + np.arange(-19, 1)[None, :]
        arrays[prefix+"_window_indices"] = np.maximum(indices, -1).astype(np.int32)
    return arrays


class AfterCloseDataset:
    """Logical [N,20,313,F] samples from one nonredundant daily mmap panel.

    Retains all masks, original dates, source indices and time evidence. No torch
    dependency. The static [313,F] applicability mask is broadcast across20.
    """
    def __init__(self, directory, split="TRAIN"):
        if split not in ("TRAIN", "VAL"):
            raise ValueError("TEST dataset access disabled")
        self.path = Path(directory)
        self.split = split.lower()
        self.manifest = json.loads((self.path/"dataset_manifest.json").read_text())
        self.meta = json.loads((self.path/"asset_manifest.json").read_text())
        self.arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
                       for p in self.path.glob("*.npy")}

    def __len__(self):
        return len(self.arrays[self.split+"_origin_indices"])

    def __getitem__(self, i):
        a = self.arrays
        origin = int(a[self.split+"_origin_indices"][i])
        indices = a[self.split+"_window_indices"][i]
        keep = indices >= 0
        safe = np.maximum(indices, 0)
        def window(name):
            result = np.array(a[name][safe], copy=True)
            result[~keep] = -1 if name in ("observation_day", "available_at", "source_index", "age_days", "trading_status") else 0
            return result
        x = window("features")
        im, fm = window("input_valid_mask"), window("feature_valid_mask")
        times = window("available_at")
        input_times = np.where(im, times, -1)
        ends = a[self.split+"_endpoint_indices"][i]
        future_times = np.full((4,25), -1, np.int64)
        for k, end in enumerate(ends):
            if end >= 0:
                future_times[k] = a["available_at"][end, -25:]
        future_times[~a[self.split+"_label_valid_mask"][i]] = -1
        return dict(X=x, input_valid_mask=im, feature_valid_mask=fm,
                    feature_applicable_mask=a["feature_applicable_mask"],
                    Y=a[self.split+"_targets"][i], label_valid_mask=a[self.split+"_label_valid_mask"][i],
                    origin_date=str(a["reference_dates"][origin]),
                    decision_time=int(a["decision_time"][origin]),
                    asset_ids=[s["asset_id"] for s in self.meta], target_ids=["FUT_"+c for c in TARGETS],
                    horizons=list(HORIZONS), reference_indices=indices,
                    input_max_available_time=int(input_times.max()),
                    target_anchor_price_time=a["available_at"][origin, -25:],
                    target_future_price_time=future_times,
                    observation_day=window("observation_day"),
                    available_at=times, source_index=window("source_index"),
                    quote_age_days=window("age_days"), new_observation=window("new_observation"),
                    trading_status=window("trading_status"))
