"""Diagnostic adapter: separate identities, time axes, source gate and masks."""
import copy
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from triweave import after_close as ac
from triweave import bridge as b
from triweave.bridge_audit import mask_oracle

ROOT=Path(__file__).resolve().parents[1]


def test_independent_311_23_universe_preserves_main25_globals():
    p,assets=b.definitions(ROOT)
    assert p['protocol_id']==b.PROTOCOL and p['task_id']==b.TASK
    assert len(assets)==311 and len(ac.asset_definitions(ROOT))==313
    assert len(ac.TARGETS)==25 and 'AU' in ac.TARGETS and 'AG' in ac.TARGETS
    assert list(b.TARGETS)==[x for x in ac.TARGETS if x not in ('AU','AG')]
    assert not {'FUT_AU','FUT_AG'} & {a['asset_id'] for a in assets}
    assert p['frozen_before_new_source_numeric_build_at']


def test_filter_au_ag_before_quote_reader(tmp_path,monkeypatch):
    p=tmp_path/'experiments/phase_d_continuous';p.mkdir(parents=True)
    entries=[]
    for c in ('AU','AG','SN'):
        e=dict(source_id='sina_continuous_exploratory_v1',parameters={'symbol':c+'0'},
               manifest=c+'.json',manifest_sha256='expected',raw_file=c+'.raw',raw_sha256='expected',
               standardized_file=c+'.csv',standardized_sha256='expected')
        entries.append(e)
    (p/'provenance.json').write_text(json.dumps({'source_files':entries}))
    monkeypatch.setattr(b,'sha256_file',lambda p:'expected');calls=[]
    def reader(path,log):
        calls.append(Path(path).name)
        assert Path(path).name=='SN.csv'
        return pd.DataFrame([dict(asset_id='FUT_SN',date='2020-01-02',close='120000')])
    monkeypatch.setattr(b.ac,'safe_csv',reader)
    f,s,l=b.load_sources(tmp_path,[dict(asset_id='FUT_SN',economic_type='FUTURES_CONTINUOUS_PRICE')])
    assert calls==['SN.csv'] and list(f)==['FUT_SN'] and len(s)==1


def test_shift_publication_preserves_time_and_skips_weekend():
    cal=['2013-07-05','2013-07-08','2013-07-09','2013-07-10']
    seconds=ac.clock_seconds(np.array(['2013-07-05','2013-07-08']),19)
    result=b.shift_publication(seconds,cal)
    assert np.array_equal(result,ac.clock_seconds(np.array(['2013-07-09','2013-07-10']),19))
    assert b.shift_publication(np.array([-1]),cal).item()==-1


def test_shift_publication_rejects_nonreference_date():
    with pytest.raises(ValueError,match='outside reference calendar'):
        b.shift_publication(ac.clock_seconds(np.array(['2013-07-06']),19),['2013-07-05','2013-07-08','2013-07-09'])


def test_lag_is_not_core_nonzero_delay_one_session_bug():
    cal=pd.bdate_range('2021-06-01',periods=10).strftime('%Y-%m-%d').tolist()
    dates=np.array(cal[:5]);base=dict(dates=dates,decision_time=ac.clock_seconds(dates,20),valid=np.ones((5,1),bool),
       observation_day=dates.astype('datetime64[D]').astype(int)[:,None],available_at=ac.clock_seconds(dates,19)[:,None],
       value=np.arange(5.)[:,None],age_days=np.zeros((5,1),np.int16),new_observation=np.ones((5,1),bool))
    r=b.lag_inputs(base,[dict(economic_type='FUTURES_CONTINUOUS_PRICE')],cal)
    assert not r['valid'][:2].any() and r['valid'][2:].all()
    assert r['value'][2:,0].tolist()==[0.,1.,2.]
    assert np.array_equal(r['available_at'][2:],base['available_at'][2:])
    assert (r['age_days'][2:]>=2).all()


def test_bond_age_cap_applies_at_final_decision_no_fallback():
    cal=pd.bdate_range('2021-06-01',periods=10).strftime('%Y-%m-%d').tolist();dates=np.array(cal[:6])
    source=np.datetime64('2021-05-27','D').astype(int)
    base=dict(dates=dates,decision_time=ac.clock_seconds(dates,20),valid=np.ones((6,1),bool),
       observation_day=np.full((6,1),source),available_at=np.full((6,1),int(ac.clock_seconds(np.array([cal[0]]),8)[0])),
       value=np.ones((6,1)),age_days=np.zeros((6,1),np.int16),new_observation=np.ones((6,1),bool))
    r=b.lag_inputs(base,[dict(economic_type='BOND_YIELD')],cal)
    assert not r['valid'].any() and not r['new_observation'].any()
    assert r['value'][2,0]==1. # value retained internally but not called a valid quote


def test_sn_quarantine_is_whole_observation_without_replacement_or_future_dependency():
    dates=np.array(['2015-03-26','2015-03-27','2015-03-30'])
    a=dict(dates=dates,valid=np.ones((3,1),bool),new_observation=np.ones((3,1),bool),
           observation_day=dates.astype('datetime64[D]').astype(int)[:,None],value=np.array([[120000.],[15905.],[118080.]]))
    q=b.quarantine(a,[dict(asset_id='FUT_SN')]);a['value'][2]=1e30
    q2=b.quarantine(a,[dict(asset_id='FUT_SN')])
    assert q['valid'][:,0].tolist()==[True,False,True]
    assert np.array_equal(q['valid'],q2['valid']) and a['valid'].all()
    assert q['value'][1,0]==15905. and not q['new_observation'][1,0]


def label_fixture(start='2022-10-03',end='2023-02-28'):
    cal=pd.bdate_range(start,end).strftime('%Y-%m-%d').tolist()
    dates=np.array(cal[:-2]);n=len(dates)
    assets=[dict(asset_id='FUT_'+c) for c in b.TARGETS]
    base=dict(dates=dates,valid=np.ones((n,23),bool),value=np.arange(n)[:,None]+np.arange(23)[None,:]+100.,
              available_at=np.repeat(ac.clock_seconds(dates,19)[:,None],23,axis=1))
    return base,assets,cal


@pytest.mark.parametrize('hk,h',list(enumerate(ac.HORIZONS)))
def test_future_window_anchor_is_three_not_delayed_price_or_original_t(hk,h):
    base,assets,cal=label_fixture()
    y,m,anchors,ends=b.label_arrays(base,assets,cal,'TRAIN');i=5
    assert anchors[i]==i+3 and ends[i,hk]==i+3+h and m[i,hk].all()
    assert y[i,hk,0]==np.float32(base['value'][i+3+h,0]/base['value'][i+3,0]-1)
    assert y[i,hk,0]!=np.float32(base['value'][i+h,0]/base['value'][i,0]-1)


@pytest.mark.parametrize('hk,h',list(enumerate(ac.HORIZONS)))
def test_future_window_masks_every_missing_interior_and_endpoint(hk,h):
    base,assets,cal=label_fixture();base['valid'][30,0]=False
    y,m,anchors,ends=b.label_arrays(base,assets,cal,'TRAIN')
    for i in range(30-3-h,30-3+1):
        assert not m[i,hk,0] and y[i,hk,0]==0
        assert m[i,hk,1] # no all-target dropna


def test_label_end_publication_must_not_cross_train_split():
    base,assets,cal=label_fixture()
    y,m,anchors,ends=b.label_arrays(base,assets,cal,'TRAIN')
    train_last=int(np.flatnonzero(base['dates']<='2022-12-31')[-1])
    # At i+3+1=last source quote, its assumed publication is inVAL: reject.
    assert not m[train_last-4,0].any()
    assert not m[train_last-5,0].any()
    assert m[train_last-6,0].all()
    assert not m[train_last:].any()


def test_no_test_labels_or_quote_access(tmp_path,monkeypatch):
    base,assets,cal=label_fixture()
    with pytest.raises(ValueError,match='TEST'):b.label_arrays(base,assets,cal,'TEST')
    p=tmp_path/'poison.gz'
    with gzip.open(p,'wt') as f:f.write('date,close\n2024-12-31,100\n2025-01-02,POISON,"BAD\n')
    original=csv.reader
    def guard(lines,*args,**kwargs):
        lines=list(lines);assert not any('POISON' in x for x in lines)
        return original(lines,*args,**kwargs)
    monkeypatch.setattr(csv,'reader',guard)
    assert ac.safe_csv(p).date.tolist()==['2024-12-31']
    with pytest.raises(ValueError,match='TEST'):b.BridgeDataset(tmp_path,'TEST')


def test_independent_mask_oracle_covers_all26_channels_and_missing_dependency():
    _,assets=b.definitions(ROOT);dates=pd.bdate_range('2021-01-01',periods=45).strftime('%Y-%m-%d').to_numpy()
    shape=(45,311);v=np.broadcast_to(100.+np.arange(311)[None,:]/100,shape).copy()
    a=dict(dates=dates,valid=np.ones(shape,bool),value=v,open=v,high=v+1,low=v-1,
       volume=np.full(shape,10.),amount=np.full(shape,10000.),turnover=np.ones(shape),open_interest=np.full(shape,20.),
       age_days=np.full(shape,2),observation_day=np.repeat(dates.astype('datetime64[D]').astype(int)[:,None],311,axis=1))
    j=next(i for i,s in enumerate(assets) if s['asset_id']=='FUT_SN');a['valid'][20,j]=False
    x,m,app=ac.feature_panel(a,assets)
    assert np.array_equal(mask_oracle(a,assets),m)
    assert not m[20:41,j,3].any() # all21 quotes needed for20d return
    assert m[20,j,16] # absence indicator remains explicitly observable


def test_frozen_diagnostic_assumptions_do_not_claim_real_pit():
    p=json.loads((ROOT/b.OUT/'frozen_protocol.json').read_text())
    assert p['role'].startswith('TECHNICAL_DIAGNOSTIC')
    assert p['normalization']['old_scaler_reused'] is False
    assert p['label_rule']['id']==b.TASK
    assert p['source_lag_rule']['lag_reference_sessions']==2
    assert not p['TEST_numeric_access_allowed'] and not p['run_models'] and not p['compute_MAE']


@pytest.mark.parametrize('split',['TRAIN','VAL'])
def test_real_mmap_interface_separates_inputs_labels_and_time_axes(split):
    ds=b.BridgeDataset(ROOT/b.DATA,split);sample=ds[0]
    assert sample['X'].shape==(20,311,26) and sample['Y'].shape==(4,23)
    assert len(sample['asset_ids'])==311 and sample['target_ids']==['FUT_'+c for c in b.TARGETS]
    assert not {'FUT_AU','FUT_AG'} & set(sample['asset_ids'])
    assert sample['label_anchor_date']>sample['origin_date'] and sample['task_id']==b.TASK
    valid=sample['input_valid_mask'];assert (sample['assumed_available_at'][valid]<=sample['decision_time']).all()
    pad=sample['reference_indices']<0
    assert not sample['feature_valid_mask'][pad].any() and not valid[pad].any()
    assert (sample['X'][pad]==0).all() and (sample['assumed_available_at'][pad]==-1).all()
    assert np.isfinite(sample['X']).all() and np.isfinite(sample['Y']).all()
    assert not ds.a['features'].flags.writeable


def test_real_source_reproduction_and_fresh_scaler():
    r=json.loads((ROOT/b.OUT/'reproducibility_audit.json').read_text())
    assert r['all_identical'] and r['scaler_parameters_identical']
    assert len(r['arrays'])==33 and len(r['metadata'])==6
    assert all(v['first_sha256']==v['second_sha256'] for v in list(r['arrays'].values())+list(r['metadata'].values()))
    p=pd.read_csv(ROOT/b.OUT/'normalization_audit.csv')
    assert len(p)==55 and not p.old_scaler_reused.any()
    assert not p.VAL_used_for_fit.any() and not p.TEST_used_for_fit.any()
    assert p.SN_changes_statistics.sum()==16


def test_real_sn_excluded_from_inputs_and_all_future_label_windows():
    d=ROOT/b.DATA
    dates=np.load(d/'reference_dates.npy',allow_pickle=False)
    bad=int(np.flatnonzero(dates=='2015-03-27')[0]);j=list(b.TARGETS).index('SN')
    q=np.load(d/'target_quote_valid_mask.npy',mmap_mode='r',allow_pickle=False)
    im=np.load(d/'input_valid_mask.npy',mmap_mode='r',allow_pickle=False)
    assert not q[bad,j] and not im[bad+2,288+j]
    for split in ('train','val'):
        origins=np.load(d/(split+'_origin_indices.npy'),allow_pickle=False)
        m=np.load(d/(split+'_label_valid_mask.npy'),mmap_mode='r',allow_pickle=False)
        for hk,h in enumerate(ac.HORIZONS):
            uses=(origins+3<=bad)&(bad<=origins+3+h)
            assert not m[uses,hk,j].any()


def test_real_formal25_remains_held_and_unchanged():
    from triweave.universe import sha256_file
    before=json.loads((ROOT/b.OUT/'preservation_before.json').read_text())['files']
    path='data/derived/triweave_25_after_close_v1/RESEARCH_HOLD.json'
    assert sha256_file(ROOT/path)==before[path]
    assert not (ROOT/'data/derived/triweave_25_after_close_v1_1').exists()
    flags=json.loads((ROOT/b.OUT/'audit_results.json').read_text())
    assert flags['DIAGNOSTIC_DATASET_READY'] is True
    assert flags['EXPLORATORY_25_RESEARCH_READY'] is False
    assert flags['HISTORICAL_PIT_VERIFIED'] is False


def test_new_known_material_shift_beyond_guard_blocks_diagnostic_gate(tmp_path):
    from triweave.bridge_audit import known_risk_audit
    p=tmp_path/'experiments/phase_e_sourceqa';p.mkdir(parents=True)
    (tmp_path/b.OUT).mkdir(parents=True)
    pd.DataFrame([dict(variety='FU',asset_id='FUT_FU',date_vendor='2021-06-01',date_official='2021-06-04',
                       high=12.,low=9.,reference_session_difference=3.)]).to_csv(p/'all_exact_vendor_official_fingerprints.csv',index=False)
    pd.DataFrame(columns=['variety','adjudication','reference_session_difference']).to_csv(p/'timing_match_adjudication.csv',index=False)
    pd.DataFrame([dict(asset_id='FUT_FU',classification='NOT_VERIFIED')]).to_csv(p/'early_futures_anomaly_audit.csv',index=False)
    dates=pd.bdate_range('2021-06-01',periods=8).strftime('%Y-%m-%d').to_numpy()
    with pytest.raises(AssertionError,match='Known substantive session risk'):
        known_risk_audit(tmp_path,dict(arrays={'reference_dates':dates}))


def test_delivery_verifier_rejects_tampered_dataset_file(tmp_path):
    from triweave.bridge_report import verify
    from triweave.universe import sha256_file
    p=tmp_path/b.OUT;p.mkdir(parents=True)
    original=tmp_path/'features.npy';original.write_bytes(b'original-data')
    record=dict(sha256=sha256_file(original),bytes=original.stat().st_size)
    (p/'delivery_manifest.json').write_text(json.dumps(dict(files={'features.npy':record})))
    original.write_bytes(b'replaced-data')
    with pytest.raises(ValueError,match='Changed delivery'):verify(tmp_path)
