"""Phase E causal, economic-unit, isolation, mask and real-delivery tests."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import triweave.after_close as ac

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return json.loads((ROOT/ac.REPORT_REL/'frozen_protocol.json').read_text())


@pytest.fixture
def assets():
    return ac.asset_definitions(ROOT)


@pytest.fixture
def synthetic(assets,protocol):
    full=pd.bdate_range('2022-09-01','2023-05-01').strftime('%Y-%m-%d').tolist()
    dates=np.array(full[:110],dtype='U10');t,a=len(dates),len(assets)
    values=np.broadcast_to(100*1.001**np.arange(t)[:,None],(t,a)).copy()
    values[:,263:281]=2+np.arange(t)[:,None]*.001
    obs=np.broadcast_to(dates.astype('datetime64[D]').astype(int)[:,None],(t,a)).copy()
    times=np.column_stack([ac.publication_seconds(dates,s['economic_type'],full,protocol) for s in assets])
    aligned=dict(dates=dates,value=values,valid=np.ones((t,a),bool),open=values*.999,
                 high=values*1.01,low=values*.99,volume=np.full((t,a),100.),
                 amount=np.full((t,a),1e6),turnover=np.full((t,a),1.),open_interest=np.full((t,a),200.),
                 age_days=np.zeros((t,a),np.int16),observation_day=obs,available_at=times,
                 source_index=np.zeros((t,a),np.int16),new_observation=np.ones((t,a),bool),
                 trading_status=np.ones((t,a),np.int8),decision_time=ac.clock_seconds(dates,20))
    return aligned


def test_01_date_sort_and_schema_gate(tmp_path):
    p=tmp_path/'quotes.csv';p.write_text('date,close\n2024-01-03,3\n2024-01-02,2\n')
    f=ac.safe_csv(p)
    assert sorted(f.date)==['2024-01-02','2024-01-03']
    p.write_text('close,date\n2,2024-01-02\n')
    with pytest.raises(ValueError,match='Date-first'):ac.safe_csv(p)


def test_02_asof_bond_weekend_no_backward_shift(protocol):
    dates=['2024-01-05','2024-01-06','2024-01-07']
    calendar=['2024-01-05','2024-01-08','2024-01-09','2024-01-15','2024-01-16']
    spec=[dict(asset_id='yield',economic_type=ac.KINDS[1])]
    frame=pd.DataFrame(dict(date=dates,value=['1','2','3'],__source=[0,0,0]))
    aligned,_=ac.align_sources({'yield':frame},spec,calendar,protocol)
    assert not aligned['valid'][0,0]
    assert aligned['value'][1,0]==3
    assert aligned['observation_day'][1,0]==np.datetime64('2024-01-07','D').astype(int)
    assert not aligned['valid'][3,0]


def test_03_decision_after_stock_future_publication(protocol):
    for kind in [ac.KINDS[0],ac.KINDS[3]]:
        pub=ac.publication_seconds(['2024-01-02'],kind,['2024-01-02','2024-01-03'],protocol)
        assert pub[0]<ac.clock_seconds(['2024-01-02'],20)[0]
    assert protocol['DECISION_TIME_ASSUMED'] and not protocol['HISTORICAL_PIT_VERIFIED']


def test_04_bond_not_same_day(protocol):
    cal=['2024-01-02','2024-01-03']
    for kind in ac.KINDS[1:3]:
        assert ac.publication_seconds(cal[:1],kind,cal,protocol)[0]>ac.clock_seconds(cal[:1],20)[0]


def test_05_anchor_consistency_and_decimal_units(synthetic,assets):
    y,m,ends,origins=ac.labels(synthetic,assets,'TRAIN')
    assert m[0,0].all()
    np.testing.assert_allclose(y[0,0],.001,rtol=1e-5)
    np.testing.assert_allclose(y[0,3],1.001**20-1,rtol=1e-6)
    assert ends[0,3]==20


@pytest.mark.parametrize('split,end',[('TRAIN','2022-12-31'),('VAL','2024-12-31')])
def test_06_07_08_h20_split_boundaries(assets,split,end):
    dates=np.array(pd.bdate_range(end=end,periods=40).strftime('%Y-%m-%d'))
    n=len(dates)
    aligned={'dates':dates,'valid':np.ones((n,313),bool),'value':np.full((n,313),100.)}
    y,m,ends,origins=ac.labels(aligned,assets,split)
    assert m[:-20,3].all() and not m[-20:,3].any()
    assert not m[-1].any() and m[-2,0].all() and not m[-2,3].any()
    assert len(origins)==39
    with pytest.raises(ValueError,match='TEST labels'):ac.labels(aligned,assets,'TEST')


def test_09_future_prefix(synthetic,assets):
    x,m,_=ac.feature_panel(synthetic,assets)
    altered=copy.deepcopy(synthetic)
    for field in ('value','high','low','open','volume','amount','open_interest','turnover'):
        altered[field][60:]*=7
    xx,mm,_=ac.feature_panel(altered,assets)
    assert np.array_equal(x[:60],xx[:60]) and np.array_equal(m[:60],mm[:60])


def test_10_no_bfill_early_or_hole(synthetic,assets):
    synthetic['valid'][:25,-1]=False
    x,m,_=ac.feature_panel(synthetic,assets)
    i=next(r['index'] for r in ac.registry() if r['feature_name']=='return_20d')
    assert not m[:45,-1,i].any() and m[45,-1,i]
    assert np.all(x[~m]==0)


def test_11_train_only_unique_scaler(synthetic,assets,protocol):
    x,m,_=ac.feature_panel(synthetic,assets)
    p=ac.fit_scaler(x,m,synthetic['dates'],assets,protocol)
    changed=x.copy();changed[synthetic['dates']>'2022-12-31']=1e12
    assert p==ac.fit_scaler(changed,m,synthetic['dates'],assets,protocol)
    ret=p[0];k=ret['feature_index']
    train=synthetic['dates']<='2022-12-31'
    assert ret['valid_train_count']==m[train,:263,k].sum()


def test_12_label_return_not_percentage(synthetic,assets):
    synthetic['value'][1,-25:]=102.;synthetic['value'][0,-25:]=100.
    y,m,_,_=ac.labels(synthetic,assets,'TRAIN')
    np.testing.assert_allclose(y[0,0],.02,rtol=1e-7)


def test_13_yield_basis_point_semantics(synthetic,assets):
    synthetic['value'][0,263:281]=2.;synthetic['value'][1,263:281]=3.
    x,m,app=ac.feature_panel(synthetic,assets)
    ks={r['feature_name']:r['index'] for r in ac.registry()}
    assert x[1,263,ks['yield_delta_1d_bps']]==100
    assert x[1,263,ks['yield_level_pct']]==3
    assert not app[263,ks['return_1d']]


def test_14_suspension_and_duplicate_all_invalid():
    f=pd.DataFrame(dict(date=['2024-01-02','2024-01-03','2024-01-04','2024-01-04'],
                        close=['1']*4,open=['1']*4,high=['1']*4,low=['1']*4,trading_status=['1','0','1','1']))
    _,good=ac.quote_quality(f,ac.KINDS[0])
    assert good.tolist()==[True,False,False,False]


def test_15_one_future_hole_does_not_drop_others(synthetic,assets):
    synthetic['valid'][10,-1]=False
    y,m,_,origins=ac.labels(synthetic,assets,'TRAIN')
    assert not m[9,0,-1] and m[9,0,:-1].all() and 9 in origins
    assert not m[0,3,-1] and m[0,3,:-1].all()


def test_16_17_fixed_orders(assets):
    assert len(assets)==313 and len({a['asset_id'] for a in assets})==313
    assert [a['asset_id'] for a in assets][-25:]==['FUT_'+c for c in ac.TARGETS]
    assert len(ac.TARGETS)==len(set(ac.TARGETS))==25 and 'WH' not in ac.TARGETS
    assert all(c in ac.TARGETS for c in ['C','I','Y'])


def test_18_applicable_vs_missing(synthetic,assets):
    synthetic['valid'][30,0]=False
    x,m,app=ac.feature_panel(synthetic,assets)
    idx={r['feature_name']:r['index'] for r in ac.registry()}
    assert app[0,idx['return_1d']] and not m[30,0,idx['return_1d']]
    assert not app[263,idx['return_1d']]
    assert m[30,0,idx['quote_available']] and x[30,0,idx['quote_available']]==0
    assert not np.any(m&~app[None])


def test_19_constant_insufficient_floor_no_nonfinite(synthetic,assets,protocol):
    x,m,_=ac.feature_panel(synthetic,assets)
    p=ac.fit_scaler(x,m,synthetic['dates'],assets,protocol)
    assert any(r['constant_flag'] for r in p)
    z,mask=ac.transform(x,m,assets,p)
    assert np.isfinite(z).all() and np.all(z[~mask]==0)
    short=ac.fit_scaler(x[:1],m[:1],synthetic['dates'][:1],assets,protocol)
    assert any(r['insufficient_train_support'] for r in short)
    zz,mm=ac.transform(x,m,assets,short)
    assert np.isfinite(zz).all()


def test_20_array_serialization_sha(tmp_path):
    x=np.arange(20,dtype=np.float32).reshape(4,5);p=tmp_path/'a.npy'
    np.save(p,x,allow_pickle=False)
    assert ac.array_sha(x)==ac.sha256_file(p)


def test_21_no_overwrite_existing_delivery(tmp_path):
    from triweave.after_close_build import build
    dest=tmp_path/ac.DATA_REL;dest.mkdir(parents=True)
    (dest/'dataset_manifest.json').write_text('{}')
    with pytest.raises(FileExistsError):build(tmp_path)
    assert (dest/'dataset_manifest.json').read_text()=='{}'


def test_22_test_numeric_gate_before_quote_parser(tmp_path,monkeypatch):
    p=tmp_path/'guard.csv'
    p.write_text('date,close\n2024-12-31,100\n2025-01-02,POISON_TEST,"UNFINISHED\n')
    original=csv.reader
    def guarded(lines,*args,**kwargs):
        rows=list(lines)
        assert not any('POISON_TEST' in line for line in rows)
        return original(rows,*args,**kwargs)
    monkeypatch.setattr(ac.csv,'reader',guarded)
    f=ac.safe_csv(p)
    assert f.close.tolist()==['100']
    with pytest.raises(ValueError,match='TEST quote'):ac.safe_csv(p,end='2025-01-01')


def test_cross_variable_spreads_require_same_observation(synthetic,assets):
    ids=[s['asset_id'] for s in assets];long=ids.index('CN_GOV_YIELD_10Y');short=ids.index('CN_GOV_YIELD_1Y')
    synthetic['value'][:,long]=3;synthetic['value'][:,short]=2
    synthetic['observation_day'][30,short]-=1
    x,m,_=ac.feature_panel(synthetic,assets)
    k=next(r['index'] for r in ac.registry() if r['feature_name']=='term_10Y_1Y_bps')
    assert x[29,long,k]==100 and m[29,long,k] and not m[30,long,k]


def test_negative_activity_only_masks_activity(synthetic,assets):
    synthetic['volume'][30,-1]=-1
    x,m,_=ac.feature_panel(synthetic,assets)
    ks={r['feature_name']:r['index'] for r in ac.registry()}
    assert not m[30,-1,ks['volume_change_1d']] and m[30,-1,ks['return_1d']]


def test_nonpositive_and_ohlc_violation():
    f=pd.DataFrame(dict(date=['2024-01-02','2024-01-03','2024-01-04'],
                        open=[1,1,1],high=[1,1,1],low=[1,1,1],close=[0,2,np.inf]))
    _,q=ac.quote_quality(f,ac.KINDS[3])
    assert not q.any()


def test_real_delivery_if_present():
    path=ROOT/ac.DATA_REL
    if not (ROOT/ac.REPORT_REL/'reproducibility_audit.json').exists():
        pytest.skip('Real dataset build must run before final suite')
    checks=json.loads((ROOT/ac.REPORT_REL/'real_data_checks.json').read_text())
    assert all(checks['checks'].values()) and not checks['evidence']['old_file_differences']
    repro=json.loads((ROOT/ac.REPORT_REL/'reproducibility_audit.json').read_text())
    assert all(r['identical'] for r in repro.values())
    for split in ('TRAIN','VAL'):
        ds=ac.AfterCloseDataset(path,split)
        assert len(ds)>0
        for i in (0,len(ds)-1):
            s=ds[i]
            assert s['X'].shape==(20,313,len(ac.registry())) and s['Y'].shape==(4,25)
            assert s['input_max_available_time']<=s['decision_time']
    with pytest.raises(ValueError,match='TEST dataset'):ac.AfterCloseDataset(path,'TEST')


def test_real_source_review_hold_preserves_original_arrays():
    p=ROOT/ac.REPORT_REL/'source_alert.json'
    if not p.exists():pytest.skip('Post-build source evidence must run first')
    alert=json.loads(p.read_text());hold=json.loads((ROOT/ac.DATA_REL/'RESEARCH_HOLD.json').read_text())
    assert alert['blocking_for_exploratory_training']
    assert alert['original_raw_and_standardized_agree']
    assert alert['vendor_close']<alert['existing_SHFE_finite_contract_low_min']
    assert alert['directly_disputed_anchor_label_count']==4
    assert hold['EXPLORATORY_TRAIN_READY'] is False
    assert hold['evidence_sha256']==ac.sha256_file(p)
    manifest=json.loads((ROOT/ac.DATA_REL/'dataset_manifest.json').read_text())
    assert all(ac.sha256_file(ROOT/ac.DATA_REL/n)==v['sha256'] for n,v in manifest['files'].items())


def test_alignment_view_is_typed_and_keeps_unknown_publication():
    if not (ROOT/ac.DATA_REL/'dataset_manifest.json').exists():pytest.skip('Real build required')
    from triweave.after_close_review import alignment_frame
    f=alignment_frame(ROOT/ac.DATA_REL,'2023-01-03')
    assert len(f)==313 and not f.asset_id.duplicated().any()
    assert f.available_at_status.eq('AVAILABLE_AT_ASSUMED').all()
    assert f.loc[f.economic_type.eq('BOND_YIELD'),'unit'].eq('percent_per_annum').all()
    assert f.loc[f.economic_type.eq('BOND_WEALTH_INDEX'),'unit'].eq('index_level').all()
    assert f.loc[~f.availability_mask,'value'].isna().all()
