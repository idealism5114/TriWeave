"""Extend the offline investigation when date/session evidence disagrees."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from triweave.after_close import safe_csv,write_json
from triweave.sourceqa_investigate import FIELDS,numeric,raw_official_csv,OUT_REL


def expand(root):
    root=Path(root);out=root/OUT_REL
    inv=json.loads((out/'investigation_results.json').read_text())
    if (out/'timing_expansion.json').exists():raise FileExistsError('Timing evidence already exists')
    chunks=[];vendors=[];logs=[]
    for e in inv['official_sources']:
        f=numeric(safe_csv(root/e['standardized_file'],log=logs));chunks.append(f)
    official=pd.concat(chunks,ignore_index=True)
    for e in inv['vendor_sources']:
        f=numeric(safe_csv(root/e['raw_file'],log=logs).rename(columns={'hold':'open_interest'}))
        f['variety']=e['parameters']['symbol'][:-1];vendors.append(f)
    vendor=pd.concat(vendors,ignore_index=True)
    positive=official[['open','high','low','close']].gt(0).all(axis=1)&official[FIELDS].apply(np.isfinite).all(axis=1)
    ov=official.loc[positive&official.volume.gt(0)&official.open_interest.gt(0)]
    vv=vendor.loc[vendor[FIELDS].apply(np.isfinite).all(axis=1)&vendor.volume.gt(0)&vendor.open_interest.gt(0)]
    matched=vv.merge(ov[['date','variety','contract']+FIELDS],on=['variety']+FIELDS,suffixes=('_vendor','_official'))
    calendar=pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    index={d:i for i,d in enumerate(calendar)}
    matched['reference_session_difference']=matched.date_official.map(index)-matched.date_vendor.map(index)
    matched['asset_id']='FUT_'+matched.variety
    matched.to_csv(out/'all_exact_vendor_official_fingerprints.csv',index=False)
    forward=matched.loc[matched.reference_session_difference.gt(0)]
    forward.to_csv(out/'forward_session_exact_matches.csv',index=False)
    forward.groupby(['asset_id',forward.date_vendor.str[:4]],sort=True).agg(
        matches=('contract','size'),source_dates=('date_vendor','nunique'),first_date=('date_vendor','min'),last_date=('date_vendor','max')).to_csv(out/'forward_session_match_summary.csv')
    # Zero OHLC inactive rows cannot define a traded price envelope.
    screen=pd.read_csv(out/'all25_existing_official_consistency_screen.csv')
    screen['traded_price_comparison_supported']=screen.low_min.gt(0)&screen.high_max.gt(0)
    screen['screen_only_not_confirmed_error']=True
    screen.to_csv(out/'qualified_consistency_screen.csv',index=False)
    # Check all59 OI flags against original raw RO/OI, no identity substitution.
    oi=screen.loc[screen.asset_id.eq('FUT_OI')&screen.joint_identity_contradiction]
    alias=[]
    for e in inv['official_sources']:
        if e['parameters'].get('market')!='CZCE':continue
        lo=e['parameters']['start_date'];hi=e['parameters']['end_date']
        event_dates=oi.loc[oi.date.str.replace('-','').between(lo,hi),'date'].tolist()
        if not event_dates:continue
        r=numeric(raw_official_csv(root/e['raw_file'],min(event_dates),max(event_dates),log=logs))
        for date in event_dates:
            v=vendor.loc[vendor.variety.eq('OI')&vendor.date.eq(date)].iloc[0]
            sub=r.loc[r.date.eq(date)&r.variety.eq('RO')]
            same=sub.loc[sub[FIELDS].eq(v[FIELDS]).all(axis=1)]
            alias.append(dict(asset_id='FUT_OI',source_date=date,
                              raw_RO_exact_7field_match_contracts=';'.join(same.symbol.tolist()),
                              verified_raw_quote_identity=bool(len(same)),
                              economic_RO_to_OI_equivalence='NOT_VERIFIED',
                              reason='Standardized OI subset omits raw RO; preliminary envelope is not complete economic-universe evidence'))
    pd.DataFrame(alias).to_csv(out/'OI_legacy_alias_evidence.csv',index=False)
    result=dict(forward_unique_vendor_dates=int(forward[['variety','date_vendor']].drop_duplicates().shape[0]),
                forward_exact_7field_contract_matches=len(forward),
                forward_assets=sorted(forward.asset_id.unique()),
                first_forward_source_date=forward.date_vendor.min() if len(forward) else None,
                last_forward_source_date=forward.date_vendor.max() if len(forward) else None,
                source_date_shift_ROOT_CAUSE='NOT_VERIFIED: vendor session labeling / endpoint vintage / aggregator date mapping cannot be distinguished from these files',
                evidence_consequence='Existing 19:00 same-source-date availability assumption is not supported for these observations. Snapshot exact future-session matches require historical session/publication evidence before release.',
                full_after_close_release_blocked=bool(len(forward)),
                OI_preliminary_flags=len(oi),OI_raw_RO_exact_matches=sum(x['verified_raw_quote_identity'] for x in alias),
                OI_economic_continuity_verified=False,TEST_numeric_rows_parsed=0,network_requests=0,quote_read_log=logs)
    write_json(out/'timing_expansion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='quote_read_log'},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':expand(Path.cwd())
