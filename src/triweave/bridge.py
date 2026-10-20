"""Independent 311-node/23-target technical diagnostic adapter.

The formal25 protocol, globals and artifacts are never modified. Availability
clocks here are declared engineering assumptions, not historical PIT evidence.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave.universe import sha256_file

OUT='experiments/phase_e_bridge'
DATA='data/derived/triweave_23_diagnostic_v0'
PROTOCOL='TRIWEAVE_23_DIAGNOSTIC_V0'
TASK='TRIWEAVE_23_VENDOR_FWD3_WINDOW_RETURN_V0'
TARGETS=tuple(c for c in ac.TARGETS if c not in ('AU','AG'))
LAG=2
ANCHOR=3


def json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def definitions(root):
    frozen=json.loads((root/OUT/'frozen_protocol.json').read_text())
    assert frozen['protocol_id']==PROTOCOL and frozen['task_id']==TASK
    assert frozen['target_order']==list(TARGETS) and len(TARGETS)==23
    assert frozen['source_lag_rule']['lag_reference_sessions']==LAG
    assert frozen['label_rule']['anchor_offset_reference_sessions']==ANCHOR
    assert frozen['label_rule']['split_publication_purge_reference_sessions']==LAG
    assert frozen['TEST_numeric_access_allowed'] is False
    assets=json.loads((root/OUT/'diagnostic_asset_manifest.json').read_text())
    main=ac.asset_definitions(root)
    ids=[a['asset_id'] for a in main if a['asset_id'] not in ('FUT_AU','FUT_AG')]
    assert [a['asset_id'] for a in assets]==ids and len(ids)==311 and len(set(ids))==311
    assert ids[288:]==['FUT_'+c for c in TARGETS]
    assert [a['economic_type'] for a in assets]==[a['economic_type'] for a in main if a['asset_id'] in ids]
    assert sha256_file(root/frozen['reference_calendar'])==frozen['calendar_sha256']
    assert sha256_file(root/frozen['base_protocol'])==frozen['base_protocol_sha256']
    return frozen,assets


def load_sources(root, assets):
    """Filter source identities BEFORE quote ingestion; AU/AG never parsed."""
    prior=json.loads((root/'experiments/phase_d_continuous/provenance.json').read_text())
    entries=[e for e in prior['source_files'] if e['source_id']!='sina_continuous_exploratory_v1'
             or e['parameters']['symbol'][:-1] in TARGETS]
    frames={a['asset_id']:[] for a in assets};logs=[]
    for ix,e in enumerate(entries):
        for p,h in [('manifest','manifest_sha256'),('raw_file','raw_sha256'),('standardized_file','standardized_sha256')]:
            if sha256_file(root/e[p])!=e[h]:raise ValueError('Changed source: '+e[p])
        f=ac.safe_csv(root/e['standardized_file'],log=logs)
        f['__source']=ix
        for asset,sub in f.groupby('asset_id',sort=False):
            if asset not in frames:raise ValueError('Unexpected diagnostic asset '+asset)
            frames[asset].append(sub)
    result={}
    for a in assets:
        asset=a['asset_id'];kind=a['economic_type']
        if not frames[asset]:raise ValueError('Missing TRAIN/VAL source '+asset)
        f=pd.concat(frames[asset],ignore_index=True)
        if kind in ac.KINDS[1:3] and not f.unit.eq(a['unit']).all():raise ValueError('Bond unit/type mismatch')
        if kind==ac.KINDS[0] and not (f.price_unit.eq('RMB_per_share') & f.volume_unit.eq('shares')
                & f.amount_unit.eq('RMB') & f.turnover_unit.eq('percent') & f.adjustment.eq('unadjusted')).all():
            raise ValueError('Stock unit/adjustment mismatch')
        result[asset]=f.sort_values('date',kind='stable').reset_index(drop=True)
    assert not {'FUT_AU','FUT_AG'} & set(result)
    return result,entries,logs


def shift_publication(seconds, calendar, n=LAG):
    """Shift an assumed publication clock, preserving its local time of day."""
    seconds=np.asarray(seconds);out=np.full(seconds.shape,-1,np.int64)
    hit=seconds>=0
    local=seconds[hit]+8*3600
    days=local.astype('datetime64[s]').astype('datetime64[D]').astype('U10')
    cal=np.asarray(calendar,dtype='U10');ix=np.searchsorted(cal,days)
    if (ix>=len(cal)).any() or not np.array_equal(cal[ix],days):
        raise ValueError('Publication clock outside reference calendar')
    if (ix+n>=len(cal)).any():raise ValueError('Insufficient calendar metadata')
    out[hit]=ac.clock_seconds(cal[ix+n],0)+(local%86400)
    return out


def quarantine(base, assets):
    """Fixed whole-observation rule; no replacement and no future-price access."""
    b=copy.deepcopy(base)
    j=next(i for i,a in enumerate(assets) if a['asset_id']=='FUT_SN')
    hit=(b['dates']=='2015-03-27') & (b['observation_day'][:,j]==np.datetime64('2015-03-27','D').astype(int))
    b['valid'][hit,j]=False;b['new_observation'][hit,j]=False
    return b


def lag_inputs(base, assets, calendar):
    """Shift base as-of panel2 sessions, with wall-clock age recomputed."""
    n=len(base['dates']);out={}
    for k,v in base.items():
        if k in ('dates','decision_time'):out[k]=v.copy();continue
        fill=np.nan if v.dtype.kind=='f' else False if v.dtype.kind=='b' else -1
        out[k]=np.full_like(v,fill)
        if n>LAG:out[k][LAG:]=v[:-LAG]
    out['available_at']=shift_publication(out['available_at'],calendar)
    present=out['observation_day']>=0
    days=out['dates'].astype('datetime64[D]').astype(int)[:,None]
    out['age_days']=np.where(present,days-out['observation_day'],-1).astype(np.int16)
    for j,a in enumerate(assets):
        if a['economic_type'] in ac.KINDS[1:3]:out['valid'][:,j]&=out['age_days'][:,j]<=5
    out['new_observation']&=out['valid']
    return out


def label_arrays(base, assets, calendar, split):
    """New future-vendor-window task; never original P(t+h)/P(t) labels."""
    if split not in ('TRAIN','VAL'):raise ValueError('TEST labels disabled')
    dates=base['dates'];n=len(dates)
    if len(dates) and max(dates)>ac.END:raise ValueError('TEST quotes disabled')
    js=[next(j for j,a in enumerate(assets) if a['asset_id']=='FUT_'+c) for c in TARGETS]
    q=base['valid'][:,js];p=base['value'][:,js]
    avail=shift_publication(base['available_at'][:,js],calendar)
    y=np.zeros((n,4,23),np.float32);mask=np.zeros_like(y,bool)
    anchors=np.full(n,-1,np.int32);ends=np.full((n,4),-1,np.int32)
    lo,hi=ac.SPLITS[split];deadline=int(ac.clock_seconds(np.array([hi]),20)[0])
    for i in range(n):
        a=i+ANCHOR
        if a>=n or not lo<=dates[i]<=dates[a]<=hi:continue
        anchors[i]=a
        for hk,h in enumerate(ac.HORIZONS):
            e=a+h
            if e>=n or not lo<=dates[e]<=hi:continue
            good=q[a:e+1].all(0) & (avail[a:e+1]>=0).all(0) & (avail[a:e+1]<=deadline).all(0)
            candidate=np.zeros(23,np.float64)
            np.divide(p[e],p[a],out=candidate,where=good);candidate-=1
            good &= np.isfinite(candidate) & (np.abs(candidate)<=np.finfo(np.float32).max)
            mask[i,hk]=good;y[i,hk,good]=candidate[good].astype(np.float32);ends[i,hk]=e
    return y,mask,anchors,ends


def calculate(root, with_quarantine=True, cached=None):
    root=Path(root);protocol,assets=definitions(root)
    calendar=pd.read_csv(root/protocol['reference_calendar'],dtype=str).date.tolist()
    if cached is None:frames,sources,logs=load_sources(root,assets)
    else:frames,sources,logs=cached
    original=json.loads((root/protocol['base_protocol']).read_text())
    base,quality=ac.align_sources(frames,assets,calendar,original)
    unmasked=base
    if with_quarantine:base=quarantine(base,assets)
    aligned=lag_inputs(base,assets,calendar)
    x,m,app=ac.feature_panel(aligned,assets)
    params=ac.fit_scaler(x,m,aligned['dates'],assets,protocol)
    z,fm=ac.transform(x,m,assets,params)
    arrays=dict(features=z,unscaled_features=x,feature_valid_mask=fm,unscaled_feature_valid_mask=m,
                feature_applicable_mask=app,input_valid_mask=aligned['valid'],
                aligned_values=np.where(aligned['valid'],aligned['value'],0.),
                reference_dates=aligned['dates'],decision_time=aligned['decision_time'])
    for k in ('observation_day','available_at','source_index','age_days','new_observation','trading_status'):arrays[k]=aligned[k]
    arrays['base_reference_indices']=np.maximum(np.arange(len(aligned['dates']))-LAG,-1).astype(np.int32)
    arrays['target_vendor_values']=np.where(base['valid'][:,-23:],base['value'][:,-23:],0.)
    arrays['target_quote_valid_mask']=base['valid'][:,-23:]
    arrays['target_source_index']=base['source_index'][:,-23:]
    arrays['target_vendor_observation_day']=base['observation_day'][:,-23:]
    arrays['target_assumed_available_at']=shift_publication(base['available_at'][:,-23:],calendar)
    for split in ('TRAIN','VAL'):
        y,lm,anchors,ends=label_arrays(base,assets,calendar,split)
        input_present=ac.rolling_sum(aligned['valid'].any(1).astype(np.int32),20)>0
        # rolling_sum intentionally needs20 rows; short prefix windows are valid.
        input_present[:19]=np.cumsum(aligned['valid'].any(1))[:19]>0
        origins=np.flatnonzero(lm.any((1,2)) & input_present).astype(np.int32)
        s=split.lower();arrays[s+'_targets']=y[origins];arrays[s+'_label_valid_mask']=lm[origins]
        arrays[s+'_origin_indices']=origins;arrays[s+'_anchor_indices']=anchors[origins]
        arrays[s+'_endpoint_indices']=ends[origins]
        arrays[s+'_window_indices']=np.maximum(origins[:,None]+np.arange(-19,1),-1).astype(np.int32)
    return dict(arrays=arrays,aligned=aligned,base=base,unmasked_base=unmasked,assets=assets,
                params=params,frames=frames,sources=sources,logs=logs,protocol=protocol,calendar=calendar,quality=quality)


def metadata(root,result):
    reg=copy.deepcopy(ac.registry())
    for r in reg:
        r['availability_rule']='Input panel delayed2 reference sessions; all dependencies valid at decision under explicit engineering assumptions'
        if r['feature_name']=='quote_age_days':r['transformation']='Decision_date-original_observation_date; bonds max5 calendar days at decision'
    files={'protocol.json':result['protocol'],'asset_manifest.json':result['assets'],
           'target_manifest.json':json.loads((root/OUT/'diagnostic_target_manifest.json').read_text()),
           'feature_registry.json':reg,'scaler_params.json':result['params'],
           'source_manifest.json':result['sources']}
    return {k:json_bytes(v) for k,v in files.items()}


class BridgeDataset:
    """Masked mmap technical-test samples. No TEST or formal25 interface."""
    def __init__(self,path,split='TRAIN'):
        if split not in ('TRAIN','VAL'):raise ValueError('TEST dataset access disabled')
        self.path=Path(path);self.split=split.lower()
        self.manifest=json.loads((self.path/'dataset_manifest.json').read_text())
        if self.manifest['protocol_id']!=PROTOCOL or self.manifest['task_id']!=TASK:raise ValueError('Wrong task')
        if self.manifest['DIAGNOSTIC_DATASET_READY'] is not True:raise ValueError('Diagnostic dataset is held')
        self.a={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in self.path.glob('*.npy')}
        self.assets=json.loads((self.path/'asset_manifest.json').read_text())

    def __len__(self):return len(self.a[self.split+'_origin_indices'])

    def __getitem__(self,i):
        a=self.a;s=self.split;origin=int(a[s+'_origin_indices'][i]);idx=a[s+'_window_indices'][i];pad=idx<0
        def window(k):
            v=np.array(a[k][np.maximum(idx,0)],copy=True)
            v[pad]=-1 if k in ('observation_day','available_at','source_index','age_days','trading_status') else 0
            return v
        anchor=int(a[s+'_anchor_indices'][i]);ends=a[s+'_endpoint_indices'][i]
        enddates=np.array([str(a['reference_dates'][e]) if e>=0 else '' for e in ends])
        return dict(X=window('features'),input_valid_mask=window('input_valid_mask'),
                    feature_valid_mask=window('feature_valid_mask'),feature_applicable_mask=a['feature_applicable_mask'],
                    Y=np.array(a[s+'_targets'][i]),label_valid_mask=np.array(a[s+'_label_valid_mask'][i]),
                    asset_ids=[a['asset_id'] for a in self.assets],target_ids=['FUT_'+c for c in TARGETS],
                    origin_date=str(a['reference_dates'][origin]),decision_time=int(a['decision_time'][origin]),
                    vendor_observation_day=window('observation_day'),assumed_available_at=window('available_at'),
                    source_index=window('source_index'),quote_age_days=window('age_days'),
                    label_anchor_date=str(a['reference_dates'][anchor]),label_end_dates=enddates,
                    reference_indices=np.array(idx),task_id=TASK,diagnostic_only=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:
        from triweave.bridge_report import verify
        verify(Path.cwd())
    elif args.run:
        from triweave.bridge_audit import run
        run(Path.cwd())
    else:p.print_help()
