"""SessionQA regression: reject unsupported shifts and partial-rule releases."""
import copy
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from triweave import after_close as ac
from triweave.sessionqa_match import match_bars, ordinary_mask, OUT
from triweave.sessionqa_checks import mask_minimum_contract, independent_label_masks
from triweave.sourceqa_investigate import raw_official_csv

ROOT=Path(__file__).resolve().parents[1]


def bar(date, **kw):
    r=dict(date=date,variety='AU',asset_id='FUT_AU',open=10.,high=12.,low=9.,close=11.,
           volume=100.,open_interest=20.,settle=10.5)
    r.update(kw)
    return r


def test_joint_unique_plus_one_not_calendar_day_shift():
    cal=['2013-07-05','2013-07-08','2013-07-09']
    v=pd.DataFrame([bar(cal[0])]);o=pd.DataFrame([bar(cal[1],contract='AU1312',official_source_index=1)])
    f,_=match_bars(v,o,cal)
    assert f.unique_active_shift.item() and f.unique_match_offset.item()==1
    assert f['official_date_+1'].item()=='2013-07-08'
    assert f['full7_contract_matches_+0'].item()==0


def test_flat_repeated_matches_are_ambiguous_not_shift_proof():
    cal=['2013-07-05','2013-07-08','2013-07-09']
    b=bar(cal[0],open=11.,high=11.,low=11.,volume=0.)
    v=pd.DataFrame([b]);o=pd.DataFrame([dict(b,date=d,contract='AU1312',official_source_index=0) for d in cal])
    f,_=match_bars(v,o,cal)
    assert f.matched_date_count.item()==3 and f.ambiguous_dates.item()
    assert not f.unique_active_shift.item() and not f.active_nonflat.item()


def test_ohlc_only_and_activity_unit_difference_never_counts_all7():
    cal=['2013-07-05','2013-07-08']
    f,_=match_bars(pd.DataFrame([bar(cal[0])]),pd.DataFrame([bar(cal[1],volume=200.,contract='AU1312',official_source_index=0)]),cal)
    assert f['full7_contract_matches_+1'].item()==0
    assert f['ohlc4_contract_matches_+1'].item()==1
    assert f.no_full7_match.item() and not f.unique_active_shift.item()


def test_equal_date_multiple_contracts_does_not_become_unique_identity():
    cal=['2013-07-05','2013-07-08']
    o=pd.DataFrame([bar(cal[0],contract=c,official_source_index=0) for c in ['AU1312','AU1406']])
    f,_=match_bars(pd.DataFrame([bar(cal[0])]),o,cal)
    assert f.matched_date_count.item()==1 and f.matched_contract_count.item()==2
    assert not f.unique_date_and_contract.item()


def test_off_reference_bar_is_not_arbitrarily_mapped():
    v=pd.DataFrame([bar('2013-07-06')])
    o=pd.DataFrame([bar('2013-07-08',contract='AU1312',official_source_index=0)])
    f,_=match_bars(v,o,['2013-07-05','2013-07-08'])
    assert f.off_reference_calendar.item() and f.no_full7_match.item()


def test_exchange_scope_excludes_shared_ine_payload_and_special_symbols():
    f=pd.DataFrame(dict(contract=['AU1312','LU2209','SC2209','AU1312TAS','AUefp','AU1312'],variety=['AU','LU','SC','AU','AU','AG']))
    assert ordinary_mask(f,'SHFE').tolist()==[True,False,False,False,False,False]
    assert ordinary_mask(f,'INE').tolist()==[False,True,True,False,False,False]


def test_official_date_gate_rejects_poison_before_csv_and_numeric_parser(tmp_path,monkeypatch):
    p=tmp_path/'official.gz'
    with gzip.open(p,'wt') as f:
        f.write('index,symbol,date,close\n0,AU1312,20130709,253.35\n1,AU2501,20250102,POISON,"BAD\n')
    original=csv.reader
    def guard(lines,*args,**kwargs):
        lines=list(lines)
        assert not any('POISON' in line for line in lines)
        return original(lines,*args,**kwargs)
    monkeypatch.setattr(csv,'reader',guard)
    r=raw_official_csv(p,'2011-10-01','2024-12-31')
    assert r.date.tolist()==['2013-07-09']
    with pytest.raises(ValueError,match='TEST'):raw_official_csv(p,'2024-01-01','2025-01-01')


def test_minimum_rule_masks_entire_observation_without_prices_or_future_dependency():
    dates=np.array(['2013-07-05','2013-07-08','2013-07-09'])
    a=dict(dates=dates,observation_day=dates.astype('datetime64[D]').astype(int)[:,None],
           valid=np.ones((3,1),bool),new_observation=np.ones((3,1),bool),value=np.array([[245.45],[253.35],[251.5]]))
    rule=[dict(asset_id='FUT_AU',start_date='2013-07-08',end_date='2013-07-08')]
    original=copy.deepcopy(a)
    q=mask_minimum_contract(a,[dict(asset_id='FUT_AU')],rule)
    a['value'][2]=1e20
    q2=mask_minimum_contract(a,[dict(asset_id='FUT_AU')],rule)
    assert q['valid'][:,0].tolist()==[True,False,True]
    assert np.array_equal(q['valid'],q2['valid'])
    assert np.array_equal(q['value'],original['value']) and a['valid'].all()
    assert not q['new_observation'][1,0]


def test_minimum_rule_uses_source_observation_date():
    a=dict(dates=np.array(['2013-07-08']),observation_day=np.array([[np.datetime64('2013-07-05','D').astype(int)]]),
           valid=np.ones((1,1),bool),new_observation=np.ones((1,1),bool))
    r=[dict(asset_id='FUT_AU',start_date='2013-07-08',end_date='2013-07-08')]
    assert mask_minimum_contract(a,[dict(asset_id='FUT_AU')],r)['valid'].all()


@pytest.mark.parametrize('hk,h',list(enumerate(ac.HORIZONS)))
def test_quarantined_interior_and_both_endpoints_mask_all_horizons(hk,h):
    dates=pd.bdate_range('2021-01-01',periods=61).strftime('%Y-%m-%d').to_numpy()
    q=np.ones((61,25),bool);q[30,0]=False
    m=independent_label_masks(q,dates,'TRAIN')
    assert not m[30-h:31,hk,0].any()
    assert m[29-h,hk,0] and m[31,hk,0] and m[30-h:31,hk,1].all()


def test_label_reference_session_and_split_boundaries():
    dates=np.array(['2022-12-29','2022-12-30','2023-01-03','2023-01-04','2024-12-31'])
    q=np.ones((5,25),bool)
    m=independent_label_masks(q,dates,'TRAIN');v=independent_label_masks(q,dates,'VAL')
    assert m[0,0].all() and not m[1,0].any()
    assert v[2,0].all() and not v[4].any()


def test_actual_au_ag_launch_regime_cannot_support_blanket_shift():
    f=pd.read_csv(ROOT/OUT/'au_ag_date_matching.csv')
    for c in ['AU','AG']:
        t=f.loc[f.variety.eq(c) & f.date.ge('2013-07-05')]
        assert len(t)==41 and t['full7_contract_matches_+1'].gt(0).sum()==1
        assert t['full7_contract_matches_+0'].sum()==0
        assert t.no_full7_match.sum()==40


def test_real_frozen_disposition_and_no_fake_version():
    p=json.loads((ROOT/OUT/'frozen_disposition.json').read_text())
    assert p['disposition']=='RESEARCH_HOLD'
    assert not p['complete_revised_temporal_protocol'] and not p['new_dataset_generation_allowed']
    assert p['mapping_rules']==[] and len(p['minimum_quarantine_rules'])==5
    assert not (ROOT/'data/derived/triweave_25_after_close_v1_1').exists()


def test_real_raw_normalization_date_comparison():
    f=pd.read_csv(ROOT/OUT/'train_raw_standardized_date_audit.csv')
    assert len(f)>300 and f.mismatched_unique_rows.sum()==0
    assert not f.original_exchange_payload_available.any()
    assert f.raw_date_origin.str.contains('REQUEST_DAY').all()


def test_actual_known_session_risk_rejects_false_declared_clock_pass():
    r=json.loads((ROOT/OUT/'v1_replay_checks.json').read_text())
    assert r['V1_declared_available_at_pass']
    assert r['TRADING_SESSION_CAUSALITY_TEST'] is False
    assert len(r['risk_cases'])==4
    for case in r['risk_cases']:
        assert case['risk_available_time_bound']>case['decision_time']
        assert case['minimum_rule_input_rejected'] and case['minimum_rule_anchor_all_horizons_rejected']
        assert not case['entire_history_covered_by_rule']


def test_actual_v1_twice_replay_does_not_claim_v1_1_reproduction():
    r=json.loads((ROOT/OUT/'v1_replay_checks.json').read_text())
    assert r['V1_reproducible_twice'] and r['DATASET_V1_1_GENERATED'] is False
    assert len(r['arrays_sha256'])==25
    assert all(v['old']==v['replay1']==v['replay2'] for v in r['arrays_sha256'].values())
    assert r['V1_horizon_and_split_boundaries_pass'] and r['V1_prefix_causality_pass']
    assert r['TEST_numeric_rows_parsed']==0


def test_actual_all_four_horizons_minimum_mask_contract_not_a_complete_dataset():
    f=pd.read_csv(ROOT/OUT/'input_label_temporal_consistency.csv')
    assert len(f)==20 and f.groupby(['asset_id','vendor_date']).size().eq(4).all()
    assert f.minimum_rule_labels_using_observation.eq(0).all()
    assert f.minimum_rule_inputs_using_record.eq(0).all()
    assert f.complete_candidate_time_consistency.eq('BLOCKED').all()


def test_audit_preservation_guard_rejects_changed_or_missing_original(tmp_path):
    from triweave.sessionqa_report import protected_check
    from triweave.universe import sha256_file
    p=tmp_path/'original.csv';p.write_text('date,close\n2013-07-08,253.35\n')
    out=tmp_path/OUT;out.mkdir(parents=True)
    (out/'preservation_before.json').write_text(json.dumps({'files':{'original.csv':sha256_file(p)}}))
    assert protected_check(tmp_path)['all_original_files_unchanged']
    p.write_text('silently replaced')
    with pytest.raises(RuntimeError,match='Protected files changed'):protected_check(tmp_path)
    p.unlink()
    with pytest.raises(RuntimeError,match='Protected files changed'):protected_check(tmp_path)
