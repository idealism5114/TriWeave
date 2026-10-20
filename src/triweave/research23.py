"""Research23: original after-close returns, new identities and audited masks.

No Bridge lag or FWD3 labels, no old arrays/scalers, no network or TEST quotes.
Sources are re-normalized from raw files after a pre-parser date gate.
"""
from __future__ import annotations
import copy
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave.assets import ALL_FUTURES, INDEX_SPECS
from triweave.normalize import normalize_bond_yields,normalize_bond_index,normalize_continuous
from triweave.rescue_providers import BaoStockStockProvider
from triweave.universe import sha256_file

OUT='experiments/phase_e_23_research'
DATA='data/derived/triweave_23_after_close_research_v1'
PROTOCOL='TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1'
TARGETS=tuple('LU SC CF PG HC J JM FU C MA SR OI RM RB M CU AL ZN SN NI A I Y'.split())


def json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def definitions(root):
    p=json.loads((root/OUT/'frozen_protocol.json').read_text())
    assert p['protocol_id']==PROTOCOL and p['target_order']==list(TARGETS)
    assert p['label_formula']=='P(t+h)/P(t)-1' and p['label_anchor_offset_reference_sessions']==0
    assert p['additional_input_lag_reference_sessions']==0 and not p['FWD3_used']
    assert p['splits']=={k:list(v) for k,v in ac.SPLITS.items()} and not p['test_numeric_access_allowed']
    c=pd.read_csv(root/'configs/futures_targets_23_research_v1.csv',keep_default_na=False)
    assert c.variety.tolist()==list(TARGETS) and c.order.tolist()==list(range(23))
    assert c.asset_id.tolist()==['FUT_'+x for x in TARGETS]
    a=json.loads((root/OUT/'asset_manifest.json').read_text())
    original=ac.asset_definitions(root)
    expected=[x['asset_id'] for x in original if x['asset_id'] not in ('FUT_AU','FUT_AG')]
    assert [x['asset_id'] for x in a]==expected and len(a)==311 and len(set(expected))==311
    assert expected[288:]==c.asset_id.tolist()
    return p,a


def raw_csv(path,start='2011-10-01',end=ac.END,log=None):
    """Gate ASCII/Chinese date columns before CSV/quote parsing.

    Collection archives are single-line CSV. Preceding curve-name fields have
    no commas; unexpected date token layouts fail rather than infer a schema.
    """
    if start>ac.END or end>ac.END:raise ValueError('TEST numeric ingestion prohibited')
    rows=[];opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig') as f:
        header=next(csv.reader([next(f)]))
        datefield=next((k for k in ('date','日期') if k in header),None)
        if datefield is None:raise ValueError('No raw date field')
        idx=header.index(datefield)
        for line in f:
            fields=line.split(',',idx+1)
            if len(fields)<=idx:raise ValueError('Date prefix schema mismatch')
            token=fields[idx].strip().strip('"')
            if len(token)>=10 and token[4]=='-' and token[7]=='-':day=token[:10]
            elif len(token)==8 and token.isdigit():day=token[:4]+'-'+token[4:6]+'-'+token[6:]
            else:raise ValueError('Invalid raw date prefix')
            if not start<=day<=end:continue
            pd.Timestamp(day)
            record=next(csv.reader([line]))
            if len(record)!=len(header):raise ValueError('Raw single-line record mismatch')
            row=dict(zip(header,record));row[datefield]=day;rows.append(row)
    frame=pd.DataFrame(rows,columns=header)
    if log is not None:log.append(dict(path=str(path),parsed_rows=len(frame),date_gate_before_quote_parser=True,
       min_parsed_date=frame[datefield].min() if len(frame) else None,max_parsed_date=frame[datefield].max() if len(frame) else None,
       TEST_numeric_rows_parsed=0,reader='RAW_RENORMALIZATION_DATE_GATE'))
    return frame


def load_sources(root,assets):
    old=json.loads((root/'experiments/phase_d_continuous/provenance.json').read_text())
    entries=[e for e in old['source_files'] if e['source_id']!='sina_continuous_exploratory_v1' or e['parameters']['symbol'][:-1] in TARGETS]
    frames={a['asset_id']:[] for a in assets};logs=[];replay=[]
    specs={s.variety:s for s in ALL_FUTURES}
    numeric_fields=('open','high','low','close','preclose','volume','amount','turn','pctChg','turnover','open_interest','settle','value')
    for ix,e in enumerate(entries):
        for k,h in [('manifest','manifest_sha256'),('raw_file','raw_sha256'),('standardized_file','standardized_sha256')]:
            if sha256_file(root/e[k])!=e[h]:raise ValueError('Source fingerprint changed: '+e[k])
        raw=raw_csv(root/e['raw_file'],log=logs)
        archived=ac.safe_csv(root/e['standardized_file'],log=logs)
        if raw.empty:
            assert archived.empty
            replay.append(dict(source_index=ix,source_id=e['source_id'],raw_rows=0,standardized_rows=0,
                               exact_numeric_differences=0,roundtrip_only_differences=0))
            continue
        kind=e['source_id']
        if kind=='baostock_unadjusted_v1':clean=BaoStockStockProvider.normalize(raw,e['parameters']['code'],'2011-10-01',ac.END)
        elif kind=='sina_continuous_exploratory_v1':clean=normalize_continuous(raw,specs[e['parameters']['symbol'][:-1]],'2011-10-01',ac.END)
        elif kind=='chinabond_bond_yield_curves_v1':clean=normalize_bond_yields(raw,'2011-10-01',ac.END)
        elif kind=='chinabond_bond_wealth_indices_v1':
            asset=next(k for k,v in INDEX_SPECS.items() if v==e['parameters']['period'])
            clean=normalize_bond_index(raw,asset,'2011-10-01',ac.END)
        else:raise ValueError('Unknown source '+kind)
        keys=['asset_id','date'];left=clean.sort_values(keys).reset_index(drop=True);right=archived.sort_values(keys).reset_index(drop=True)
        assert left[keys].equals(right[keys]) and len(left)==len(right),'Raw/standardized identities or dates differ'
        differences=0;roundtrip=0;maxdiff=0.
        for field in numeric_fields:
            if field not in clean or field not in archived:continue
            x=pd.to_numeric(left[field],errors='coerce').to_numpy(float);y=pd.to_numeric(right[field],errors='coerce').to_numpy(float)
            same=(x==y)|(np.isnan(x)&np.isnan(y));diff=~same;differences+=int(diff.sum())
            if diff.any():
                # Fixed IEEE representational check, not an economic error tolerance.
                tol=2*np.spacing(np.maximum(np.abs(x),np.abs(y)))
                ok=np.isfinite(x)&np.isfinite(y)&(np.abs(x-y)<=tol)
                if not (ok|same).all():raise ValueError(f'Raw/standardized substantive numeric discrepancy {e["raw_file"]}:{field}')
                roundtrip+=int(diff.sum());maxdiff=max(maxdiff,float(np.abs(x[diff]-y[diff]).max()))
        for field in ('unit','price_unit','volume_unit','amount_unit','turnover_unit','adjustment','trading_status'):
            if field in left and field in right:assert left[field].astype(str).equals(right[field].astype(str)),field
        clean['__source']=ix
        for asset,sub in clean.groupby('asset_id',sort=False):
            if asset not in frames:raise ValueError('Unexpected raw source identity '+asset)
            frames[asset].append(sub)
        replay.append(dict(source_index=ix,source_id=kind,raw_rows=len(raw),standardized_rows=len(archived),
                           exact_numeric_differences=differences,roundtrip_only_differences=roundtrip,max_difference=maxdiff))
    result={}
    for spec in assets:
        asset=spec['asset_id']
        if not frames[asset]:raise ValueError('No permitted raw observations '+asset)
        result[asset]=pd.concat(frames[asset],ignore_index=True).sort_values('date',kind='stable').reset_index(drop=True)
    return result,entries,logs,replay


def apply_quarantine(aligned,assets,rules):
    a=copy.deepcopy(aligned);accepted=np.ones(a['valid'].shape,bool);ids=[x['asset_id'] for x in assets]
    for r in rules:
        j=ids.index(r['asset_id']);hit=(a['dates']==r['date'])&(a['observation_day'][:,j]==np.datetime64(r['date'],'D').astype(int))
        a['valid'][hit,j]=False;a['new_observation'][hit,j]=False;accepted[hit,j]=False
    return a,accepted


def labels(aligned,assets,split):
    if split not in ('TRAIN','VAL'):raise ValueError('TEST labels disabled')
    dates=aligned['dates'];n=len(dates)
    if n and max(dates)>ac.END:raise ValueError('TEST numerical panel forbidden')
    js=[next(j for j,a in enumerate(assets) if a['asset_id']=='FUT_'+c) for c in TARGETS]
    q=aligned['valid'][:,js];p=aligned['value'][:,js]
    y=np.zeros((n,4,23),np.float32);m=np.zeros_like(y,bool);ends=np.full((n,4),-1,np.int32)
    lo,hi=ac.SPLITS[split]
    for i in np.flatnonzero((dates>=lo)&(dates<=hi)):
        for hk,h in enumerate(ac.HORIZONS):
            e=i+h
            if e>=n or not lo<=dates[e]<=hi:continue
            good=q[i:e+1].all(0)
            values=np.zeros(23);np.divide(p[e],p[i],out=values,where=good);values-=1
            good&=np.isfinite(values)&(np.abs(values)<=np.finfo(np.float32).max)
            y[i,hk,good]=values[good].astype(np.float32);m[i,hk]=good;ends[i,hk]=e
    origins=np.flatnonzero(m.any((1,2))).astype(np.int32)
    return y,m,ends,origins


def calculate(root,with_quarantine=True,cached=None):
    root=Path(root);protocol,assets=definitions(root)
    frames,sources,logs,replay=load_sources(root,assets) if cached is None else cached
    calendar=pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    original,quality=ac.align_sources(frames,assets,calendar,protocol)
    rules=pd.read_csv(root/OUT/'source_quarantine_manifest.csv',keep_default_na=False).to_dict('records')
    aligned,accepted=apply_quarantine(original,assets,rules) if with_quarantine else (original,np.ones(original['valid'].shape,bool))
    x,m,app=ac.feature_panel(aligned,assets);params=ac.fit_scaler(x,m,aligned['dates'],assets,protocol)
    z,fm=ac.transform(x,m,assets,params)
    arrays=dict(features=z,unscaled_features=x,feature_valid_mask=fm,unscaled_feature_valid_mask=m,
      feature_applicable_mask=app,input_valid_mask=aligned['valid'],aligned_values=np.where(aligned['valid'],aligned['value'],0.),
      reference_dates=aligned['dates'],decision_time=aligned['decision_time'],source_observation_acceptance_mask=accepted)
    for k in ('observation_day','available_at','source_index','age_days','new_observation','trading_status'):arrays[k]=aligned[k]
    for split in ('TRAIN','VAL'):
        y,lm,ends,origins=labels(aligned,assets,split);s=split.lower()
        arrays[s+'_targets']=y[origins];arrays[s+'_label_valid_mask']=lm[origins]
        arrays[s+'_origin_indices']=origins;arrays[s+'_anchor_indices']=origins.copy();arrays[s+'_endpoint_indices']=ends[origins]
        arrays[s+'_window_indices']=np.maximum(origins[:,None]+np.arange(-19,1),-1).astype(np.int32)
    return dict(arrays=arrays,aligned=aligned,original=original,assets=assets,frames=frames,sources=sources,logs=logs,
                replay=replay,params=params,protocol=protocol,rules=rules,calendar=calendar,quality=quality)


def metadata(root,r):
    docs={'protocol.json':r['protocol'],'asset_manifest.json':r['assets'],
          'target_manifest.json':json.loads((root/OUT/'target_manifest.json').read_text()),
          'feature_registry.json':ac.registry(),'scaler_params.json':r['params'],
          'source_manifest.json':r['sources'],'source_quarantine_manifest.json':r['rules']}
    return {name:json_bytes(v) for name,v in docs.items()}


class Research23Dataset:
    def __init__(self,path,split='TRAIN',allow_quarantined=False):
        if split not in ('TRAIN','VAL'):raise ValueError('TEST dataset disabled')
        self.path=Path(path);self.split=split.lower();self.manifest=json.loads((self.path/'dataset_manifest.json').read_text())
        if self.manifest['protocol_id']!=PROTOCOL:raise ValueError('Wrong research task')
        if not self.manifest['DATASET_TECHNICALLY_VALID']:raise ValueError('Technically invalid dataset')
        if not self.manifest['EXPLORATORY_RESEARCH_READY'] and not allow_quarantined:raise ValueError('Research candidate quarantined')
        self.a={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in self.path.glob('*.npy')}
        self.assets=json.loads((self.path/'asset_manifest.json').read_text())

    def __len__(self):return len(self.a[self.split+'_origin_indices'])

    def __getitem__(self,index):
        a=self.a;s=self.split;i=int(a[s+'_origin_indices'][index]);idx=a[s+'_window_indices'][index];pad=idx<0
        def window(k):
            v=np.array(a[k][np.maximum(idx,0)],copy=True)
            v[pad]=-1 if k in ('observation_day','available_at','source_index','age_days','trading_status') else 0
            return v
        return dict(X=window('features'),Y=np.array(a[s+'_targets'][index]),input_valid_mask=window('input_valid_mask'),
          feature_valid_mask=window('feature_valid_mask'),feature_applicable_mask=a['feature_applicable_mask'],
          label_valid_mask=np.array(a[s+'_label_valid_mask'][index]),origin_date=str(a['reference_dates'][i]),
          decision_time=int(a['decision_time'][i]),observation_day=window('observation_day'),assumed_available_at=window('available_at'),
          source_index=window('source_index'),quote_age_days=window('age_days'),reference_indices=np.array(idx),
          label_anchor_date=str(a['reference_dates'][i]),
          label_end_dates=[str(a['reference_dates'][e]) if e>=0 else '' for e in a[s+'_endpoint_indices'][index]],
          asset_ids=[x['asset_id'] for x in self.assets],target_ids=['FUT_'+c for c in TARGETS],protocol_id=PROTOCOL)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.run:
        from triweave.research23_audit import run
        run(Path.cwd())
    elif args.verify:
        from triweave.research23_report import verify
        verify(Path.cwd())
    else:p.print_help()
