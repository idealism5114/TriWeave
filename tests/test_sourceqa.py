"""SourceQA source evidence, propagation, padding and isolation regressions."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import triweave.after_close as ac
from triweave.sourceqa_investigate import raw_official_csv,OUT_REL
from triweave.sourceqa_impact import apply_sn_rule

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/OUT_REL


def test_official_raw_date_gate_before_quote_parser(tmp_path,monkeypatch):
    import gzip
    p=tmp_path/'month.csv.gz'
    with gzip.open(p,'wt') as f:f.write('index,symbol,date,close\n0,SN1507,20150327,119270\n1,POISON,20250102,DO_NOT_PARSE,"BAD\n')
    original=csv.reader
    def guard(lines,*args,**kwargs):
        lines=list(lines);assert not any('DO_NOT_PARSE' in line for line in lines)
        return original(lines,*args,**kwargs)
    monkeypatch.setattr(csv,'reader',guard)
    f=raw_official_csv(p,'2011-10-01','2024-12-31')
    assert f.date.tolist()==['2015-03-27'] and f.close.tolist()==['119270']
    with pytest.raises(ValueError,match='TEST'):raw_official_csv(p,'2011-01-01','2025-01-01')


def test_sn_rule_does_not_use_next_price():
    dates=np.array(['2015-03-26','2015-03-27','2015-03-30'])
    a=dict(dates=dates,observation_day=dates.astype('datetime64[D]').astype(int)[:,None],
           valid=np.ones((3,1),bool),new_observation=np.ones((3,1),bool),value=np.array([[1.],[15905.],[118080.]]))
    b=copy.deepcopy(a);b['value'][2]=1e30
    specs=[{'asset_id':'FUT_SN'}]
    aa=apply_sn_rule(a,specs);bb=apply_sn_rule(b,specs)
    assert aa['valid'][:,0].tolist()==[True,False,True]
    assert np.array_equal(aa['valid'],bb['valid'])
    assert aa['value'][1,0]==15905.  # No quote replacement.


def test_sn_rule_does_not_backdate_future_absence():
    dates=np.array(['2015-03-25','2015-03-26'])
    a=dict(dates=dates,observation_day=dates.astype('datetime64[D]').astype(int)[:,None],
           valid=np.ones((2,1),bool),new_observation=np.ones((2,1),bool))
    assert apply_sn_rule(a,[{'asset_id':'FUT_SN'}])['valid'].all()


def test_sn_all_fields_and_activity_evidence():
    e=json.loads((OUT/'investigation_results.json').read_text())['sn_joint_evidence']
    assert e['close']==15905 and e['settle']==15970
    assert e['contract_count']==9 and e['total_volume']==1832 and e['total_oi']==620
    assert e['close']<e['low_min'] and e['settle']<e['settle_min']
    assert e['volume']>e['total_volume'] and e['open_interest']>e['total_oi']


def test_raw_standardized_match_is_not_source_truth():
    f=pd.read_csv(OUT/'early_futures_anomaly_audit.csv')
    assert f.raw_standardized_all7_fields_match.all()
    sn=f.loc[f.asset_id.eq('FUT_SN')].iloc[0]
    assert sn.classification=='SOURCE_IDENTITY_MISMATCH' and not sn.root_cause_verified
    assert not f.classification.eq('GENUINE_MARKET_MOVE').any()


def test_forward_timing_conflict_not_confused_with_repeated_flat_bar():
    f=pd.read_csv(OUT/'timing_match_adjudication.csv')
    strong=f.loc[f.adjudication.eq('TIMING_CONFLICT_STRONG_EVIDENCE')]
    assert set(strong.asset_id)=={'FUT_AG','FUT_AU'}
    assert strong.date_vendor.eq('2013-07-08').all() and strong.date_official.eq('2013-07-09').all()
    assert strong.nonflat_price_bar.all() and not strong.also_matches_same_day.any()
    assert not f.loc[f.asset_id.eq('FUT_WH'),'adjudication'].eq('TIMING_CONFLICT_STRONG_EVIDENCE').any()


def test_legacy_alias_quote_match_does_not_assert_economic_equivalence():
    f=pd.read_csv(OUT/'OI_legacy_alias_evidence.csv')
    assert len(f)==59 and f.verified_raw_quote_identity.sum()==58
    assert f.economic_RO_to_OI_equivalence.eq('NOT_VERIFIED').all()


def test_no_automatic_twenty_percent_quarantine():
    f=pd.read_csv(OUT/'source_quarantine_rules.csv')
    assert f.asset_id.tolist()==['FUT_SN'] and f.observation_date.tolist()==['2015-03-27']
    assert f.application_scope.str.contains('COUNTERFACTUAL').all()


def test_full_real_v1_reproduction_and_sn_candidate_determinism():
    e=json.loads((OUT/'impact_results.json').read_text())
    assert all(e['independent_V1_reproduction'].values())
    r=json.loads((OUT/'candidate_reproducibility.json').read_text())
    assert all(v['identical'] for v in r['arrays'].values()) and r['scaler_identical']


def test_source_and_derived_changes_differ():
    e=json.loads((OUT/'impact_results.json').read_text())
    assert e['changed_raw_values']==0 and e['changed_input_masks']==1
    assert e['changed_feature_masks']>1 and e['changed_label_masks']==4
    assert e['unexplained_material_unscaled_changes']==0
    assert e['max_unscaled_roundoff']<=1e-9


def test_train_refit_global_propagation_is_separate():
    e=json.loads((OUT/'impact_results.json').read_text())
    assert e['changed_scaler_groups']>0 and e['TRAIN_only_scaler_pass']
    assert e['local_feature_dependency_windows']['train']<e['pooled_refit_changed_X_windows']['train']
    p=json.loads((OUT/'candidate_sn_only_scaler_params.json').read_text())
    assert len(p['parameters'])==55 and 'NOT_RELEASED' in p['status']


def test_label_propagation_all_four_horizons():
    e=json.loads((OUT/'impact_results.json').read_text())
    before={'1':57889,'5':57249,'10':56504,'20':55017}
    assert e['valid_label_counts_after_SN_only_simulation']['train']=={h:n-1 for h,n in before.items()}
    assert e['valid_label_counts_after_SN_only_simulation']['val']=={'1':12067,'5':11951,'10':11806,'20':11520}
    assert e['TRAIN_VAL_boundary_pass']


def test_dependency_table_preserves_root_and_mask_propagation():
    f=pd.read_csv(OUT/'anomaly_dependency_impact.csv',low_memory=False)
    required={'asset_id','source_date','affected_origin_date','affected_feature','affected_horizon','impact_type','reason_code'}
    assert required<=set(f)
    sn=f.loc[f.asset_id.eq('FUT_SN')]
    assert {'RAW_SOURCE_OBSERVATION','DERIVED_DAILY_FEATURE_DEPENDENCY','INPUT_WINDOW_PROPAGATION',
            'LABEL_WINDOW_DEPENDENCY','TRAIN_SCALER_SAMPLE_DEPENDENCY','POOLED_TRAIN_SCALER_REFIT_PROPAGATION'}<=set(sn.impact_type)
    assert {1.,5.,10.,20.}<=set(sn.loc[sn.impact_type.eq('LABEL_WINDOW_DEPENDENCY'),'affected_horizon'])
    reviewed=f.loc[f.asset_id.eq('FUT_I')&f.source_date.eq('2015-10-08')
                   &f.impact_type.eq('LABEL_WINDOW_DEPENDENCY')&f.affected_horizon.eq(20)]
    assert len(reviewed)==21
    assert reviewed.label_dependency_semantics.eq('ENDPOINT_VALUE_AND_WINDOW_QUALITY').sum()==2


def test_first30_actual_padding_is_not_fake_history():
    f=pd.read_csv(OUT/'early_window_padding_audit.csv')
    assert len(f)==30 and f.iloc[0].left_padding_sessions==19
    assert f.iloc[0].real_reference_sessions==1
    assert f.real_reference_sessions.add(f.left_padding_sessions).eq(20).all()
    assert f.padding_input_and_feature_masks_false.all() and f.padding_values_zero.all()
    assert not f.future_or_first_valid_quote_copied.any()


def test_intrinsic_causality_does_not_verify_source_timestamps():
    e=json.loads((OUT/'impact_results.json').read_text());t=json.loads((OUT/'timing_adjudication.json').read_text())
    assert e['intrinsic_prefix_pass'] and t['after_close_release_blocked']
    assert e['source_timing_validation'].startswith('FAILED')


def test_extreme_scale_audit_no_automatic_deletion():
    f=pd.read_csv(OUT/'normalization_extreme_audit.csv')
    v=f.loc[f.version.eq('V1')]
    assert v.split.eq('TRAIN').sum()==358 and v.split.eq('VAL').sum()==16
    assert np.isfinite(f.z).all() and not f.low_variance_floor_amplification.any()
    assert f.action.eq('RETAIN_FLAG_NO_VAL_REFIT_NO_CLIP').all()


def test_invalid_supervision_cannot_be_selected_without_mask():
    # Selection contract, no model or MAE calculation.
    y=np.array([1.,np.nan]);mask=np.array([True,False])
    assert np.isfinite(y[mask]).all() and len(y[mask])==1
    ds=ac.AfterCloseDataset(ROOT/ac.DATA_REL,'TRAIN');s=ds[0]
    assert s['label_valid_mask'].dtype==bool and np.isfinite(s['Y'][s['label_valid_mask']]).all()


def test_no_fake_v1_1_created_and_original_hold_retained():
    assert not (ROOT/'data/derived/triweave_25_after_close_v1_1').exists()
    hold=json.loads((ROOT/ac.DATA_REL/'RESEARCH_HOLD.json').read_text())
    assert hold['EXPLORATORY_TRAIN_READY'] is False
    e=json.loads((OUT/'impact_results.json').read_text())
    assert e['V1_1_generated'] is False


def test_all_protected_original_files_sha_unchanged():
    files=json.loads((OUT/'preservation_before.json').read_text())['files']
    assert len(files)==13559
    assert all(ac.sha256_file(ROOT/p)==h for p,h in files.items())


def test_original_identity_and_three_split_protocol_unchanged():
    ds=ac.AfterCloseDataset(ROOT/ac.DATA_REL,'TRAIN')
    assert len(ds.meta)==313 and [s['asset_id'] for s in ds.meta][-25:]==['FUT_'+c for c in ac.TARGETS]
    assert len(ac.TARGETS)==25 and len(ac.registry())==26
    assert set(ac.SPLITS)=={'TRAIN','VAL','TEST'}
    with pytest.raises(ValueError,match='TEST'):ac.AfterCloseDataset(ROOT/ac.DATA_REL,'TEST')
