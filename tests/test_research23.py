"""Research23 identity, original-anchor, raw-gate and source-quality regressions."""
import copy
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from triweave import after_close as ac
from triweave import research23 as r
from triweave.research23_audit import original_label_oracle
from triweave.research23_quality import longest_missing

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_three_protocols_and_json_split_roundtrip():
    p,a=r.definitions(ROOT)
    assert p['protocol_id']==r.PROTOCOL and len(a)==311 and len(r.TARGETS)==23
    assert p['splits']=={k:list(v) for k,v in ac.SPLITS.items()}
    assert p['label_anchor_offset_reference_sessions']==0 and not p['FWD3_used']
    assert p['additional_input_lag_reference_sessions']==0
    assert [s['asset_id'] for s in a][-23:]==['FUT_'+c for c in r.TARGETS]
    assert len(ac.TARGETS)==25 and len(ac.asset_definitions(ROOT))==313
    assert json.loads((ROOT/'experiments/phase_e_bridge/frozen_protocol.json').read_text())['protocol_id']=='TRIWEAVE_23_DIAGNOSTIC_V0'


@pytest.mark.parametrize('date_column',['date','日期'])
def test_raw_gate_discards_test_before_csv_parser_and_numbers(tmp_path,monkeypatch,date_column):
    p=tmp_path/'raw.gz'
    with gzip.open(p,'wt') as f:
        f.write(f'curve,{date_column},close\ncurve1,2024-12-31,100\ncurve2,2025-01-02,POISON,"BAD\n')
    original=csv.reader
    def guard(lines,*args,**kwargs):
        lines=list(lines);assert not any('POISON' in x for x in lines)
        return original(lines,*args,**kwargs)
    monkeypatch.setattr(csv,'reader',guard)
    frame=r.raw_csv(p)
    assert frame[date_column].tolist()==['2024-12-31'] and frame.close.tolist()==['100']
    with pytest.raises(ValueError,match='TEST'):r.raw_csv(p,end='2025-01-02')


def test_altering_isolated_poison_test_tail_leaves_permitted_raw_artifacts_identical(tmp_path):
    results=[]
    for n,tail in enumerate(['9999999999999999999','POISON,"NOT_A_RECORD']):
        p=tmp_path/f'raw{n}.gz'
        with gzip.open(p,'wt') as f:f.write('date,close\n2023-01-03,100\n2024-12-31,110\n2025-01-02,'+tail+'\n')
        results.append(r.raw_csv(p).to_csv(index=False).encode())
    assert results[0]==results[1]


def fixture():
    dates=pd.bdate_range('2022-10-03','2023-03-01').strftime('%Y-%m-%d').to_numpy()
    p=np.arange(len(dates))[:,None]**2+np.arange(23)[None,:]+100.
    a=dict(dates=dates,valid=np.ones(p.shape,bool),value=p)
    assets=[dict(asset_id='FUT_'+c) for c in r.TARGETS]
    return a,assets


@pytest.mark.parametrize('hk,h',list(enumerate(ac.HORIZONS)))
def test_original_t_anchor_restored_and_decimal_not_fwd3(hk,h,monkeypatch):
    from triweave import bridge
    monkeypatch.setattr(bridge,'label_arrays',lambda *a,**k:(_ for _ in ()).throw(AssertionError('FWD3 called')))
    a,assets=fixture();y,m,ends,origins=r.labels(a,assets,'TRAIN');i=5
    assert ends[i,hk]==i+h and m[i,hk].all()
    assert y[i,hk,0]==np.float32(a['value'][i+h,0]/a['value'][i,0]-1)
    assert y[i,hk,0]!=np.float32(a['value'][i+3+h,0]/a['value'][i+3,0]-1)


@pytest.mark.parametrize('hk,h',list(enumerate(ac.HORIZONS)))
def test_each_asset_horizon_checks_interior_and_both_endpoints(hk,h):
    a,assets=fixture();a['valid'][30,0]=False
    y,m,ends,origins=r.labels(a,assets,'TRAIN')
    assert not m[30-h:31,hk,0].any() and (y[30-h:31,hk,0]==0).all()
    assert m[30-h:31,hk,1].all() # no all-asset dropna


def test_no_cross_split_or_test_label_access():
    a,assets=fixture();last=int(np.flatnonzero(a['dates']<='2022-12-31')[-1])
    y,m,e,o=r.labels(a,assets,'TRAIN');assert m[last-1,0].all() and not m[last].any()
    assert m[last-5,1].all() and not m[last-4: ,1].any()
    with pytest.raises(ValueError,match='TEST'):r.labels(a,assets,'TEST')
    with pytest.raises(ValueError,match='TEST'):r.Research23Dataset(ROOT,'TEST')


def test_whole_source_quarantine_does_not_replace_or_use_future_price():
    dates=np.array(['2015-03-26','2015-03-27','2015-03-30'])
    a=dict(dates=dates,valid=np.ones((3,1),bool),new_observation=np.ones((3,1),bool),
       observation_day=dates.astype('datetime64[D]').astype(int)[:,None],value=np.array([[120000.],[15905.],[118080.]]))
    rules=[dict(asset_id='FUT_SN',date='2015-03-27')]
    q,accepted=r.apply_quarantine(a,[dict(asset_id='FUT_SN')],rules)
    a['value'][2]=1e30;q2,_=r.apply_quarantine(a,[dict(asset_id='FUT_SN')],rules)
    assert q['valid'][:,0].tolist()==[True,False,True] and np.array_equal(q['valid'],q2['valid'])
    assert q['value'][1,0]==15905. and a['valid'].all() and not accepted[1,0]


def test_frozen_source_rule_evidence_no_jump_threshold_delete():
    from triweave.universe import sha256_file
    q=pd.read_csv(ROOT/r.OUT/'source_quarantine_manifest.csv')
    assert set(zip(q.asset_id,q.date))=={('FUT_SN','2015-03-27'),('FUT_FU','2014-12-31'),('FUT_FU','2016-04-11')}
    assert not q.asset_id.isin(['FUT_I','FUT_C','FUT_PG']).any()
    assert q.replacement_price.isna().all()
    for row in q.to_dict('records'):assert sha256_file(ROOT/row['evidence'])==row['evidence_sha256']


def test_independent_original_formula_oracle():
    a,assets=fixture();a['valid'][17,0]=False
    arrays=dict(reference_dates=a['dates'],input_valid_mask=a['valid'],aligned_values=np.where(a['valid'],a['value'],0.))
    y,m,o=original_label_oracle(arrays,'TRAIN');yy,mm,e,oo=r.labels(a,assets,'TRAIN')
    assert np.array_equal(y,yy) and np.array_equal(m,mm) and np.array_equal(o,oo)


def test_missing_gap_distinguishes_contiguous_reference_sessions():
    assert longest_missing([True,False,False,True,False])==2
    assert longest_missing([False]*10)==10 and longest_missing([True]*5)==0


def test_future_source_perturbation_casts_integer_activity_without_touching_prefix():
    import warnings
    from triweave.research23_audit import perturb_future_sources
    f=pd.DataFrame(dict(date=['2022-12-30','2023-07-03'],volume=[10,20],close=[100,110]))
    with warnings.catch_warnings():
        warnings.simplefilter('error',FutureWarning)
        changed=perturb_future_sources({'FUT_C':f},'2023-06-30')['FUT_C']
    np.testing.assert_allclose(changed.volume,[10.,30.61],rtol=1e-15,atol=0)
    assert changed.close.iloc[0]==100.
    assert f.volume.tolist()==[10,20] and changed.volume.dtype.kind=='f'


@pytest.mark.parametrize('split',['TRAIN','VAL'])
def test_actual_research_mmap_shapes_original_anchor_and_padding(split):
    ds=r.Research23Dataset(ROOT/r.DATA,split);sample=ds[0]
    assert sample['X'].shape==(20,311,26) and sample['Y'].shape==(4,23)
    assert sample['label_anchor_date']==sample['origin_date']
    assert sample['target_ids']==['FUT_'+c for c in r.TARGETS]
    assert not {'FUT_AU','FUT_AG'} & set(sample['asset_ids'])
    m=sample['input_valid_mask'];assert (sample['assumed_available_at'][m]<=sample['decision_time']).all()
    pad=sample['reference_indices']<0
    assert not sample['input_valid_mask'][pad].any() and not sample['feature_valid_mask'][pad].any()
    assert (sample['X'][pad]==0).all() and (sample['assumed_available_at'][pad]==-1).all()
    assert not ds.a['features'].flags.writeable


def test_actual_quarantine_all_input_and_original_label_dependencies():
    path=ROOT/r.DATA;dates=np.load(path/'reference_dates.npy',allow_pickle=False)
    im=np.load(path/'input_valid_mask.npy',mmap_mode='r',allow_pickle=False)
    for asset,day in [('SN','2015-03-27'),('FU','2014-12-31'),('FU','2016-04-11')]:
        point=int(np.flatnonzero(dates==day)[0]);tj=list(r.TARGETS).index(asset)
        assert not im[point,288+tj]
        for split in ('train','val'):
            o=np.load(path/(split+'_origin_indices.npy'),allow_pickle=False)
            m=np.load(path/(split+'_label_valid_mask.npy'),mmap_mode='r',allow_pickle=False)
            for hk,h in enumerate(ac.HORIZONS):assert not m[(o<=point)&(point<=o+h),hk,tj].any()


def test_actual_extreme_audit_is_exhaustive_and_has_raw_denominators():
    out=ROOT/r.OUT;e=pd.read_csv(out/'extreme_feature_audit.csv');s=json.loads((out/'normalization_summary.json').read_text())
    for split in ('TRAIN','VAL'):assert e.split.eq(split).sum()==s[split]['abs_z_gt20']
    largest=e.loc[e.normalized_value.abs().idxmax()]
    assert largest.asset_id=='sz.002831' and largest.feature_name=='amount_change_1d'
    raw=r.raw_csv(ROOT/largest.source_raw_file,start='2016-12-22',end='2016-12-23')
    values=pd.to_numeric(raw.amount).to_numpy();assert values.tolist()==[2492755.,1343509376.]
    assert largest.historical_denominator==values[0]
    np.testing.assert_allclose(largest.unscaled_feature,values[1]/values[0]-1,rtol=1e-14)
    assert not e.scale_floor_flag.any() and e.action.str.contains('NO_CLIP').all()


def test_actual_raw_replay_and_determinism_all_arrays_and_scalers():
    out=ROOT/r.OUT;replay=pd.read_csv(out/'raw_normalizer_replay.csv')
    assert len(replay)==312 and replay.exact_numeric_differences.sum()==0
    data=json.loads((out/'reproducibility_audit.json').read_text())
    assert data['all_identical'] and data['scaler_identical']
    assert len(data['arrays'])==28 and len(data['metadata'])==7
    assert all(v['first_sha256']==v['second_sha256'] for v in list(data['arrays'].values())+list(data['metadata'].values()))
    post=json.loads((out/'post_fix_validation.json').read_text())
    assert all(post['published_arrays_still_identical'].values())


def test_research_loader_refuses_held_candidate_by_default(tmp_path):
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(dict(protocol_id=r.PROTOCOL,DATASET_TECHNICALLY_VALID=True,EXPLORATORY_RESEARCH_READY=False)))
    with pytest.raises(ValueError,match='quarantined'):r.Research23Dataset(tmp_path)


def test_original_preservation_guard_rejects_tampering(tmp_path):
    from triweave.research23_audit import preservation
    from triweave.universe import sha256_file
    out=tmp_path/r.OUT;out.mkdir(parents=True);old=tmp_path/'old.npy';old.write_bytes(b'old-artifact')
    (out/'preservation_before.json').write_text(json.dumps(dict(files={'old.npy':sha256_file(old)})))
    assert preservation(tmp_path)['old_artifacts_unchanged']
    old.write_bytes(b'overwritten')
    with pytest.raises(RuntimeError,match='Old artifacts changed'):preservation(tmp_path)
