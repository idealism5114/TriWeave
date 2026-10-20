import json
import pandas as pd
import pytest

from triweave.universe import normalize_stock_code, import_stock_universe, load_stock_universe


def test_code_normalization():
    assert normalize_stock_code("600038") == "sh.600038"
    assert normalize_stock_code("SZ.002146") == "sz.002146"
    assert normalize_stock_code("sh600066") == "sh.600066"
    with pytest.raises(ValueError):
        normalize_stock_code("some-other-asset")


def test_manifest_exact_count_and_provenance(tmp_path):
    inp = tmp_path / "input.csv"
    inp.write_text("code\nsh.600038\nsh.600066\nsz.002146\n", encoding="utf-8")
    dest = tmp_path / "stock_manifest.csv"
    with pytest.raises(ValueError, match="Expected 263"):
        import_stock_universe(inp, dest)
    assert not dest.exists()
    metadata = import_stock_universe(inp, dest, expected=3)
    assert metadata["stock_count"] == 3
    assert len(load_stock_universe(dest, expected=3)) == 3
    assert dest.with_suffix(".meta.json").exists()
    with pytest.raises(FileExistsError):
        import_stock_universe(inp, dest, expected=3)


def test_wide_header_and_json_structure(tmp_path):
    wide = tmp_path / "stock_wide.csv"
    wide.write_text("date,sh.600038,sz.002146\n2011-10-10,10,12\n", encoding="utf-8")
    x = tmp_path / "stocks.csv"
    assert import_stock_universe(wide, x, expected=2)["stock_count"] == 2
    obj = {"assets": [
        {"market": "stock", "asset_id": "sh.600038"},
        {"market": "stock", "asset_id": "sz.002146"},
        {"market": "bond", "asset_id": "T"},
    ]}
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps(obj), encoding="utf-8")
    out = tmp_path / "stock_from_json.csv"
    import_stock_universe(manifest, out, expected=2)
    assert pd.read_csv(out).shape[0] == 2


def test_duplicate_universe_is_rejected(tmp_path):
    source = tmp_path / "dupes.csv"
    source.write_text("code\nsh.600038\n600038\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        import_stock_universe(source, tmp_path / "dest.csv", expected=2)


def test_wrong_exchange_prefix_is_rejected():
    with pytest.raises(ValueError, match="prefix conflicts"):
        normalize_stock_code("sz.600000")


def test_stock_manifest_symbol_and_hash_are_checked(tmp_path):
    source=tmp_path/'input.csv'
    source.write_text('code\nsh.600038\nsz.002146\n')
    dest=tmp_path/'stocks.csv'
    import_stock_universe(source,dest,expected=2)
    frame=pd.read_csv(dest,dtype=str)
    frame.loc[0,'symbol']='600066'
    frame.to_csv(dest,index=False)
    with pytest.raises(ValueError,match='symbol differs'):
        load_stock_universe(dest,expected=2)
    frame.loc[0,'symbol']='600038'
    frame.loc[0,'order']='9'
    frame.to_csv(dest,index=False)
    with pytest.raises(ValueError,match='fingerprint differs'):
        load_stock_universe(dest,expected=2)


def test_reconstruct_248_plus_reported_15_without_fabrication(tmp_path):
    legacy = tmp_path / "legacy_wide.csv"
    columns = ["date"] + [f"sz.{i:06d}" for i in range(248)]
    # Header-only data: import must never need price rows.
    legacy.write_text(",".join(columns) + "\n", encoding="utf-8")
    dest = tmp_path / "stocks_263.csv"
    audit = import_stock_universe(legacy, dest, add_phase_a_15=True)
    assert audit["stock_count"] == 263
    assert audit["reconstructed_from_legacy_248_plus_reported_15"] is True
    assert audit["requires_comparison_to_frozen_Clean_V1"] is True
    assert len(load_stock_universe(dest)) == 263
