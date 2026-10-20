from triweave.assets import (
    ALL_FUTURES, ALL_CONTRACT_FUTURES, TARGET_FUTURES, AUX_FUTURES,
    YIELD_SPECS, INDEX_SPECS, BOND_FUTURES,
)


def test_counts_and_disjoint_varieties():
    assert (len(TARGET_FUTURES), len(AUX_FUTURES), len(BOND_FUTURES)) == (24, 14, 4)
    assert len(YIELD_SPECS) + len(INDEX_SPECS) == 25
    assert len(ALL_FUTURES) == 38
    assert len({x.variety for x in ALL_CONTRACT_FUTURES}) == 42
    assert next(x for x in TARGET_FUTURES if x.variety == "WH").review_required


def test_bond_18_curve_7_index():
    assert len(YIELD_SPECS) == 18
    assert len(INDEX_SPECS) == 7
    assert "CN_GOV_YIELD_10Y" in YIELD_SPECS
    assert "CN_TREASURY_WEALTH_7_10Y" in INDEX_SPECS
