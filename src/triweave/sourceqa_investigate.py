"""Offline source evidence. No quote is corrected and no TEST number is parsed."""
from __future__ import annotations
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import safe_csv, write_json, TARGETS, END
from triweave.universe import sha256_file

OUT_REL='experiments/phase_e_sourceqa'
FIELDS=['open','high','low','close','volume','open_interest','settle']
EVENTS=[('SN','2015-03-27'),('I','2015-10-08'),('I','2015-10-09'),
        ('C','2016-04-05'),('FU','2014-08-29'),('PG','2022-02-24'),('FU','2019-08-06')]


def raw_official_csv(path,start,end,log=None):
    """Monthly exchange raw: gate date column2 before interpreting any numbers."""
    if end>END:raise ValueError('TEST quote access disabled')
    records=[]
    with gzip.open(path,'rt',encoding='utf-8-sig') as f:
        header=next(csv.reader([next(f)]));idx=header.index('date')
        for line in f:
            token=line.split(',',idx+1)[idx].strip().strip('"')
            date=token if '-' in token else token[:4]+'-'+token[4:6]+'-'+token[6:8]
            if not start<=date<=end:continue
            pd.Timestamp(date)
            row=next(csv.reader([line]));r=dict(zip(header,row));r['date']=date;records.append(r)
    frame=pd.DataFrame(records,columns=header)
    if log is not None:log.append(dict(path=str(path),parsed_rows=len(frame),max_parsed_date=frame.date.max() if len(frame) else None,TEST_numeric_rows_parsed=0,date_gate_before_quote_parser=True))
    return frame


def numeric(f):
    f=f.copy()
    for field in FIELDS:
        if field in f:f[field]=pd.to_numeric(f[field],errors='coerce')
    return f


def load_official(root):
    chunks=[];sources=[];logs=[]
    for exchange in ('SHFE','CZCE','INE'):
        for p in sorted((root/'data/manifests/baseline15y/jobs/futures_contracts'/exchange).glob('*.json')):
            m=json.loads(p.read_text());args=m.get('api_kwargs',{})
            if m.get('status')!='OK' or args.get('start_date','')>'20241231':continue
            std=root/'data/standardized/baseline15y/futures_contracts'/exchange/(m['name']+'.csv.gz')
            raw=root/'data/raw/baseline15y/futures_contracts'/exchange/(m['name']+'.csv.gz')
            assert sha256_file(std)==m['standardized_sha256']
            assert sha256_file(raw)==m['raw_sha256']
            e=dict(manifest=str(p.relative_to(root)),manifest_sha256=sha256_file(p),
                   standardized_file=str(std.relative_to(root)),standardized_sha256=m['standardized_sha256'],
                   raw_file=str(raw.relative_to(root)),raw_sha256=m['raw_sha256'],api=m.get('api'),parameters=args)
            f=safe_csv(std,log=logs);f['__source']=len(sources);sources.append(e)
            chunks.append(numeric(f))
    return pd.concat(chunks,ignore_index=True),sources,logs


def investigate(root):
    root=Path(root);out=root/OUT_REL
    if (out/'investigation_results.json').exists():raise FileExistsError('Investigation is append-only')
    prior=json.loads((root/'experiments/phase_d_continuous/provenance.json').read_text())
    vendors=[];sources=[];logs=[]
    for e in prior['source_files']:
        if e['source_id']!='sina_continuous_exploratory_v1':continue
        assert sha256_file(root/e['raw_file'])==e['raw_sha256']
        assert sha256_file(root/e['standardized_file'])==e['standardized_sha256']
        f=safe_csv(root/e['raw_file'],log=logs).rename(columns={'hold':'open_interest'})
        f['asset_id']='FUT_'+e['parameters']['symbol'][:-1];f['variety']=e['parameters']['symbol'][:-1]
        vendors.append(numeric(f));sources.append(e)
    vendor=pd.concat(vendors,ignore_index=True)
    official,exchange_sources,exchange_logs=load_official(root)
    logs+=exchange_logs
    fields=['date']+FIELDS
    selected=vendor.loc[vendor.variety.isin(TARGETS)]
    # Expand source checks across all25, not just the initially disputed point.
    o=official.loc[official.variety.isin(TARGETS)].copy()
    envelope=o.groupby(['variety','date'],sort=True).agg(
        contract_count=('contract','size'),low_min=('low','min'),high_max=('high','max'),
        close_min=('close','min'),close_max=('close','max'),settle_min=('settle','min'),
        settle_max=('settle','max'),total_volume=('volume','sum'),total_oi=('open_interest','sum')).reset_index()
    joined=selected.merge(envelope,on=['variety','date'],how='inner',validate='one_to_one')
    joined['outside_all_traded_price_envelope']=(joined.close<joined.low_min)|(joined.close>joined.high_max)
    joined['outside_all_settlement_envelope']=(joined.settle<joined.settle_min)|(joined.settle>joined.settle_max)
    joined['volume_exceeds_observed_contract_sum']=joined.volume>joined.total_volume
    joined['oi_exceeds_observed_contract_sum']=joined.open_interest>joined.total_oi
    joined['joint_identity_contradiction']=joined[[
        'outside_all_traded_price_envelope','outside_all_settlement_envelope',
        'volume_exceeds_observed_contract_sum','oi_exceeds_observed_contract_sum']].all(axis=1)
    joined.to_csv(out/'all25_existing_official_consistency_screen.csv',index=False)
    severe=joined.loc[joined.joint_identity_contradiction]
    severe.to_csv(out/'joint_source_identity_contradictions.csv',index=False)
    # Exact cross-variety fingerprints, never infer identity from just a price.
    complete=vendor[FIELDS].apply(np.isfinite).all(1)
    matched=vendor.loc[complete&vendor.duplicated(['date']+FIELDS,keep=False)]
    matched.to_csv(out/'all38_exact_cross_variety_fingerprints.csv',index=False)
    events=[];event_sources=[]
    calendar=pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    for variety,date in EVENTS:
        f=vendor.loc[vendor.variety.eq(variety)]
        pos=calendar.index(date);previous=calendar[pos-1];end=calendar[min(pos+20,len(calendar)-1)]
        raw=f.loc[f.date.between(previous,end)]
        original=next(e for e in sources if e['parameters']['symbol']==variety+'0')
        std=numeric(safe_csv(root/original['standardized_file'],start=previous,end=end,log=logs))
        raw[fields+['asset_id']].to_csv(out/f'evidence_{variety}_{date}_vendor_raw.csv',index=False)
        std.to_csv(out/f'evidence_{variety}_{date}_vendor_standardized.csv',index=False)
        observed=o.loc[o.variety.eq(variety)&o.date.between(previous,end)]
        observed.to_csv(out/f'evidence_{variety}_{date}_all_existing_contracts.csv',index=False)
        current=raw.loc[raw.date.eq(date)].iloc[0];same_std=std.loc[std.date.eq(date)].iloc[0]
        assert all(current[c]==same_std[c] or (pd.isna(current[c]) and pd.isna(same_std[c])) for c in FIELDS)
        candidates=[]
        for day in (previous,date):
            v=f.loc[f.date.eq(day)]
            if v.empty:continue
            sub=observed.loc[observed.date.eq(day)]
            # Exact OHLC matches identify a candidate contract only for this day.
            for _,row in sub.iterrows():
                if all(pd.notna(row[c]) and row[c]==v.iloc[0][c] for c in ['open','high','low','close']):
                    candidates.append(dict(date=day,contract=row.contract,
                                           volume_match=bool(row.volume==v.iloc[0].volume),
                                           oi_match=bool(row.open_interest==v.iloc[0].open_interest),
                                           settle_match=bool(row.settle==v.iloc[0].settle)))
        prev=f.loc[f.date.eq(previous)]
        change=float(current.close/prev.iloc[0].close-1) if len(prev) else None
        if variety=='SN':
            status='SOURCE_IDENTITY_MISMATCH'
            action='QUARANTINE_SUBJECT_TO_JOINT_EVIDENCE'
            explanation='Price AND settlement outside observed SN contract envelopes; volume and OI exceed sum of all observed SN contracts. Unknown source construction/root cause; not a demonstrated replacement price.'
        elif variety=='FU':
            status='POSSIBLE_ROLL_ADJUSTMENT';action='RETAIN_FLAG_UNVERIFIED_ROLL'
            explanation='Candidate same-day contract OHLC evidence; different delivery prices may be stitched. Never executable cross-contract return.'
        else:
            status='NOT_VERIFIED';action='RETAIN_FLAG_NO_INDEPENDENT_ERROR_EVIDENCE'
            explanation='No existing DCE contract coverage for independent identity check. Cannot call a genuine market move or confirmed source error from a jump.'
        events.append(dict(asset_id='FUT_'+variety,source_date=date,source_id=original['source_id'],
                           close=float(current.close),settle=float(current.settle),volume=float(current.volume),
                           open_interest=float(current.open_interest),decimal_change=change,classification=status,
                           root_cause_verified=False,continuous_adjustment_verified=False,
                           raw_standardized_all7_fields_match=True,
                           same_day_official_contracts=int(observed.date.eq(date).sum()),
                           candidate_contract_matches=json.dumps(candidates),action=action,reason=explanation,
                           evidence_reference=f'evidence_{variety}_{date}_all_existing_contracts.csv'))
        # Independently inspect original official raw fields for event dates.
        for idx in observed.__source.unique():
            e=exchange_sources[int(idx)]
            r=raw_official_csv(root/e['raw_file'],previous,end,log=logs)
            r=r.loc[r.variety.astype(str).str.upper().eq(variety)]
            r.to_csv(out/f'evidence_{variety}_{date}_official_raw_{int(idx)}.csv',index=False)
            event_sources.append(e)
    pd.DataFrame(events).to_csv(out/'early_futures_anomaly_audit.csv',index=False)
    sn=joined.loc[joined.variety.eq('SN')&joined.date.eq('2015-03-27')].iloc[0]
    sn_contracts=o.loc[o.variety.eq('SN')&o.date.isin(['2015-03-27','2015-03-30'])]
    sn_contracts.to_csv(out/'SN_two_days_all_contracts.csv',index=False)
    result=dict(source_status='SOURCE_IDENTITY_MISMATCH_ACCEPTANCE_FAILURE_NOT_UNIQUE_ROOT_CAUSE_PROOF',
                adjusted_continuous_definition='NOT_VERIFIED',sn_joint_evidence={k:sn[k].item() if isinstance(sn[k],np.generic) else sn[k]
                    for k in ['close','settle','volume','open_interest','contract_count','low_min','high_max','settle_min','settle_max','total_volume','total_oi','joint_identity_contradiction']},
                official_comparable_asset_dates=len(joined),joint_contradictions=severe[['asset_id','date']].to_dict('records'),
                exact_cross_variety_matches=len(matched),vendor_sources=sources,official_sources=exchange_sources,
                quote_read_log=logs,TEST_numeric_rows_parsed=0,network_requests=0,
                residual_DCE_identity_uncertainty='I/C/PG remain NOT_VERIFIED, not automatically excluded based on change size')
    write_json(out/'investigation_results.json',result)
    print(json.dumps({k:result[k] for k in ['sn_joint_evidence','official_comparable_asset_dates','joint_contradictions','exact_cross_variety_matches']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':investigate(Path.cwd())
