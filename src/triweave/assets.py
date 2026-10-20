"""Frozen research asset definitions, not claims of 15-year historical availability."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FutureSpec:
    variety: str
    exchange: str
    name_en: str
    role: str
    review_required: bool = False


TARGET_FUTURES = (
    FutureSpec("LU", "INE", "low_sulfur_fuel_oil", "target"),
    FutureSpec("SC", "INE", "crude_oil", "target"),
    FutureSpec("WH", "CZCE", "wheat_strong_gluten_verify_legacy_mapping", "target", True),
    FutureSpec("CF", "CZCE", "cotton", "target"),
    FutureSpec("PG", "DCE", "liquefied_petroleum_gas", "target"),
    FutureSpec("HC", "SHFE", "hot_rolled_coil", "target"),
    FutureSpec("J", "DCE", "metallurgical_coke", "target"),
    FutureSpec("JM", "DCE", "coking_coal", "target"),
    FutureSpec("FU", "SHFE", "fuel_oil", "target"),
    FutureSpec("C", "DCE", "corn", "target"),
    FutureSpec("MA", "CZCE", "methanol", "target"),
    FutureSpec("SR", "CZCE", "white_sugar", "target"),
    FutureSpec("AG", "SHFE", "silver", "target"),
    FutureSpec("OI", "CZCE", "rapeseed_oil", "target"),
    FutureSpec("RM", "CZCE", "rapeseed_meal", "target"),
    FutureSpec("RB", "SHFE", "rebar", "target"),
    FutureSpec("M", "DCE", "soybean_meal", "target"),
    FutureSpec("CU", "SHFE", "copper", "target"),
    FutureSpec("AL", "SHFE", "aluminium", "target"),
    FutureSpec("ZN", "SHFE", "zinc", "target"),
    FutureSpec("SN", "SHFE", "tin", "target"),
    FutureSpec("NI", "SHFE", "nickel", "target"),
    FutureSpec("A", "DCE", "soybean_no1", "target"),
    FutureSpec("AU", "SHFE", "gold", "target"),
)

AUX_FUTURES = (
    FutureSpec("I", "DCE", "iron_ore", "auxiliary"),
    FutureSpec("FG", "CZCE", "glass", "auxiliary"),
    FutureSpec("SA", "CZCE", "soda_ash", "auxiliary"),
    FutureSpec("TA", "CZCE", "pta", "auxiliary"),
    FutureSpec("EG", "DCE", "ethylene_glycol", "auxiliary"),
    FutureSpec("PP", "DCE", "polypropylene", "auxiliary"),
    FutureSpec("L", "DCE", "polyethylene", "auxiliary"),
    FutureSpec("RU", "SHFE", "natural_rubber", "auxiliary"),
    FutureSpec("Y", "DCE", "soybean_oil", "auxiliary"),
    FutureSpec("P", "DCE", "palm_oil", "auxiliary"),
    FutureSpec("SP", "SHFE", "wood_pulp", "auxiliary"),
    FutureSpec("BU", "SHFE", "bitumen", "auxiliary"),
    FutureSpec("SI", "GFEX", "industrial_silicon", "auxiliary"),
    FutureSpec("LC", "GFEX", "lithium_carbonate", "auxiliary"),
)

BOND_FUTURES = (
    FutureSpec("TF", "CFFEX", "five_year_treasury_future", "bond_future"),
    FutureSpec("T", "CFFEX", "ten_year_treasury_future", "bond_future"),
    FutureSpec("TS", "CFFEX", "two_year_treasury_future", "bond_future"),
    FutureSpec("TL", "CFFEX", "thirty_year_treasury_future", "bond_future"),
)

ALL_FUTURES = TARGET_FUTURES + AUX_FUTURES
ALL_CONTRACT_FUTURES = ALL_FUTURES + BOND_FUTURES

# AKShare bond_china_yield returns these three historical curve categories.
TREASURY_CURVE = "\u4e2d\u503a\u56fd\u503a\u6536\u76ca\u7387\u66f2\u7ebf"
AAA_NOTE_CURVE = "\u4e2d\u503a\u4e2d\u77ed\u671f\u7968\u636e\u6536\u76ca\u7387\u66f2\u7ebf(AAA)"
AAA_BANK_CURVE = "\u4e2d\u503a\u5546\u4e1a\u94f6\u884c\u666e\u901a\u503a\u6536\u76ca\u7387\u66f2\u7ebf(AAA)"

# Identifier -> (exact curve name, exact column label). 18 series.
YIELD_SPECS = {}
for short, field in (
    ("3M", "3\u6708"), ("6M", "6\u6708"),
    ("1Y", "1\u5e74"), ("3Y", "3\u5e74"),
    ("5Y", "5\u5e74"), ("7Y", "7\u5e74"),
    ("10Y", "10\u5e74"), ("30Y", "30\u5e74"),
):
    YIELD_SPECS[f"CN_GOV_YIELD_{short}"] = (TREASURY_CURVE, field)

for prefix, curve in (
    ("CN_AAA_NOTE_YIELD", AAA_NOTE_CURVE),
    ("CN_AAA_BANK_YIELD", AAA_BANK_CURVE),
):
    for short, field in (
        ("3M", "3\u6708"), ("6M", "6\u6708"),
        ("1Y", "1\u5e74"), ("3Y", "3\u5e74"),
        ("5Y", "5\u5e74"),
    ):
        YIELD_SPECS[f"{prefix}_{short}"] = (curve, field)

# 7 index wealth/total-return series, not yields.
TREASURY_INDEX_PERIODS = (
    "0-1Y", "1-3Y", "3-5Y", "5Y", "7Y", "7-10Y", "10Y",
)
INDEX_SPECS = {
    f"CN_TREASURY_WEALTH_{p.replace('-', '_')}": p
    for p in TREASURY_INDEX_PERIODS
}

assert len(YIELD_SPECS) == 18
assert len(INDEX_SPECS) == 7
assert len(YIELD_SPECS) + len(INDEX_SPECS) == 25
assert len(TARGET_FUTURES) == 24
assert len(AUX_FUTURES) == 14

# Only exchange-wide query optimizations, not an attempt to fabricate histories.
EXCHANGE_EARLIEST = {
    "INE": "2018-03-26",
    "GFEX": "2022-12-22",
    "CFFEX": "2013-09-06",
}


def validate_asset_configuration(root):
    """Fail if executable definitions diverge from the frozen CSV asset lists."""
    from pathlib import Path
    import pandas as pd
    root = Path(root)
    bonds = pd.read_csv(root / "configs/bond_variables_25.csv", encoding="utf-8-sig", keep_default_na=False)
    if bonds["asset_id"].tolist() != list(YIELD_SPECS) + list(INDEX_SPECS):
        raise ValueError("Bond CSV identifiers/order differ from the frozen 25-series definitions")
    for row in bonds.to_dict("records"):
        asset = row["asset_id"]
        if asset in YIELD_SPECS:
            curve, tenor = YIELD_SPECS[asset]
            expected = ("yield_curve", curve, tenor, "bond_china_yield", "percent_per_annum")
            observed = tuple(row[k] for k in ("kind", "curve_name", "tenor", "api", "unit"))
        else:
            expected = ("treasury_wealth_index", INDEX_SPECS[asset], "bond_treasury_index_cbond", "index_level")
            observed = tuple(row[k] for k in ("kind", "tenor", "api", "unit"))
        if observed != expected:
            raise ValueError(f"Bond CSV economic definition mismatch: {asset}")
    futures = pd.read_csv(root / "configs/futures_universe_38_plus4.csv", encoding="utf-8-sig", keep_default_na=False)
    specs = ALL_CONTRACT_FUTURES
    if futures["variety"].tolist() != [x.variety for x in specs]:
        raise ValueError("Futures CSV identifiers/order differ from the frozen 24+14+4 definitions")
    for row, spec in zip(futures.to_dict("records"), specs):
        observed = (row["exchange"], row["role"], row["name_en"], str(row["review_required"]).lower())
        expected = (spec.exchange, spec.role, spec.name_en, str(spec.review_required).lower())
        if observed != expected:
            raise ValueError(f"Futures CSV definition mismatch: {spec.variety}")
    return {"bond_variables": len(bonds), "commodity_targets": len(TARGET_FUTURES),
            "commodity_auxiliaries": len(AUX_FUTURES), "optional_treasury_futures": len(BOND_FUTURES)}
