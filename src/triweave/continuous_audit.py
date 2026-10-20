"""Offline continuous-series usability audit; masks/counts only, no model labels."""
import argparse
import json
from pathlib import Path

import pandas as pd

from triweave.assets import ALL_FUTURES,TARGET_FUTURES,YIELD_SPECS,INDEX_SPECS,validate_asset_configuration
from triweave.collect import atomic_json,utc_now
from triweave.normalize import SchemaError
from triweave.universe import sha256_file,load_stock_universe

HORIZONS=(1,5,10,20)
SPLITS={'TRAIN':('2017-01-01','2022-12-31'),'VAL':('2023-01-01','2024-12-31')}
ANALYSIS_END='2024-12-31'


def finite(s):
    x=pd.to_numeric(s,errors='coerce')
    return x.notna()&~x.isin([float('inf'),float('-inf')])


def availability(frame,calendar,kind):
    """Exclude every duplicate date, never choose one price or impute a quote."""
    f=frame.copy()
    if 'date' not in f:raise SchemaError('Observation dates required')
    dates=pd.to_datetime(f.date,format='%Y-%m-%d',errors='coerce')
    if dates.isna().any():raise SchemaError('Invalid observation date')
    f['date']=dates.dt.strftime('%Y-%m-%d')
    field='value' if kind in ('yield_curve','treasury_wealth_index') else 'close'
    if field not in f:raise SchemaError('Required field missing: '+field)
    value=pd.to_numeric(f[field],errors='coerce');good=finite(value)
    if kind!='yield_curve':good&=value.gt(0)
    if kind in ('stock','future'):
        required=['open','high','low','close']
        if set(required)-set(f):raise SchemaError('Complete OHLC schema required')
        p=f[required].apply(pd.to_numeric,errors='coerce')
        good&=p.apply(finite).all(axis=1)&p.gt(0).all(axis=1)
        good&=p.high.ge(p[['open','low','close']].max(axis=1))&p.low.le(p[['open','high','close']].min(axis=1))
    if kind=='stock':
        if 'trading_status' not in f:raise SchemaError('Stock trading status required')
        good&=f.trading_status.astype(str).eq('1')
    good&=~f.date.duplicated(keep=False)
    usable=set(f.loc[good,'date'])
    return pd.Series([d in usable for d in calendar],index=pd.Index(calendar,name='date'),dtype=bool)


def candidate_masks(mask,horizons=HORIZONS):
    """Boolean eligibility only. No returns are calculated, especially in SEALED."""
    if not mask.index.is_monotonic_increasing or mask.index.has_duplicates:
        raise SchemaError('Unique ordered reference calendar required')
    if len(mask) and max(mask.index)>ANALYSIS_END:raise SchemaError('SEALED candidate-label access prohibited')
    dates=pd.Series(mask.index,index=mask.index);result={}
    for h in horizons:
        valid=mask.copy()
        for offset in range(1,h+1):valid&=mask.shift(-offset,fill_value=False)
        endpoint=dates.shift(-h);inside=pd.Series(False,index=mask.index)
        for start,end in SPLITS.values():
            inside|=dates.between(start,end)&endpoint.between(start,end)
        result[h]=valid&inside
    return pd.DataFrame(result,index=mask.index)


def longest_gap(mask):
    longest=current=0
    for good in mask:
        current=0 if good else current+1;longest=max(longest,current)
    return longest


def quote_quality_counts(frame,calendar):
    """Off-calendar is a date-status question, not automatically a bad price."""
    own=availability(frame,sorted(frame.date.unique()),'future')
    bad=~frame.date.map(own).fillna(False).astype(bool)
    return {'valid_source_dates':int(own.sum()),'invalid_OHLC_or_duplicate_rows':int(bad.sum()),
            'off_reference_calendar_dates':int(frame.loc[~frame.date.isin(calendar),'date'].nunique())}


def differing_cells(a,b):
    return int((a.ne(b)&~(a.isna()&b.isna())).sum())


def _verified_frame(root,manifest,source_id):
    m=json.loads(manifest.read_text())
    if m.get('status')=='OK':
        relative=manifest.relative_to(root/'data/manifests').parts
        run=relative[0];group=m['job_group'];name=m['name']+'.csv.gz'
        raw=root/'data/raw'/run/group/name;std=root/'data/standardized'/run/group/name
        retrieved=m['retrieved_utc']
    elif m.get('status','').startswith('OBSERVATIONS_SAVED'):
        raw=root/m['raw_file'];std=root/m['standardized_file'];retrieved=m['retrieved_at']
    else:raise SchemaError('Unsuccessful job cannot provide observations')
    for p,key in ((raw,'raw_sha256'),(std,'standardized_sha256')):
        if sha256_file(p)!=m[key]:raise SchemaError('Immutable source archive checksum failure')
    f=pd.read_csv(std,dtype={'date':str,'asset_id':str,'trading_status':str})
    f['__manifest']=str(manifest.relative_to(root))
    entry={'source_id':source_id,'manifest':str(manifest.relative_to(root)),
        'manifest_sha256':sha256_file(manifest),'raw_file':str(raw.relative_to(root)),'raw_sha256':m['raw_sha256'],
        'standardized_file':str(std.relative_to(root)),'standardized_sha256':m['standardized_sha256'],
        'retrieved_at':retrieved,'api':m.get('api',m.get('params',{}).get('api')),
        'parameters':m.get('api_kwargs',m.get('params',{})),'raw_columns':m.get('raw_columns',[])}
    return f,entry


def read_assets(root):
    frames={};sources=[]
    stocks=load_stock_universe(root/'data/manifests/stocks_263.csv').asset_id.tolist()
    stock_dir=root/'data/manifests/rescue_20261015/baostock_unadjusted_v1/stocks'
    for p in sorted(stock_dir.glob('*.json')):
        f,e=_verified_frame(root,p,'baostock_unadjusted_v1')
        code=json.loads(p.read_text())['params']['code']
        if not f.asset_id.eq(code).all() or code not in stocks or code in frames:raise SchemaError('Stock universe/source mismatch')
        frames[code]=f;sources.append(e)
    if set(frames)!=set(stocks):raise SchemaError('Exact 263 stock sources required')
    for spec in ALL_FUTURES:
        p=root/'data/manifests/baseline15y/jobs/futures_continuous_exploratory'/f'{spec.exchange}_{spec.variety}_0.json'
        f,e=_verified_frame(root,p,'sina_continuous_exploratory_v1')
        if e['parameters']!={'symbol':spec.variety+'0'} or not f.asset_id.eq('FUT_'+spec.variety).all():
            raise SchemaError('Vendor continuous identity/parameters mismatch')
        raw=pd.read_csv(root/e['raw_file'],dtype={'date':str})
        # Original vendor fields have no dated underlying contract identifier or adjustment factor.
        e['original_raw_columns_read']=list(raw)
        e['raw_dates_outside_requested_window']=int((raw.date.lt('2011-10-01')|raw.date.gt('2026-09-30')).sum())
        frames['FUT_'+spec.variety]=f;sources.append(e)
    bond_chunks={k:[] for k in list(YIELD_SPECS)+list(INDEX_SPECS)}
    for group in ('bond_yield_curves','bond_wealth_indices'):
        for p in sorted((root/'data/manifests/baseline15y/jobs'/group).glob('*.json')):
            f,e=_verified_frame(root,p,'chinabond_'+group+'_v1');sources.append(e)
            for asset,sub in f.groupby('asset_id'):
                if asset not in bond_chunks:raise SchemaError('Bond universe mismatch')
                bond_chunks[asset].append(sub)
    for asset,chunks in bond_chunks.items():
        if not chunks:raise SchemaError('Missing configured bond variable')
        f=pd.concat(chunks,ignore_index=True)
        expected='percent_per_annum' if asset in YIELD_SPECS else 'index_level'
        if not f.unit.eq(expected).all():raise SchemaError('Bond economic type/unit mismatch')
        frames[asset]=f
    return frames,sources,stocks


def existing_vendor_vintages(root):
    rows=[]
    for p in (root/'data/manifests').glob('*/jobs/futures_continuous_exploratory/*.json'):
        if 'baseline15y' in p.parts:continue
        m=json.loads(p.read_text())
        if m.get('status')!='OK':continue
        current=root/'data/manifests/baseline15y/jobs/futures_continuous_exploratory'/p.name
        if not current.exists():continue
        a,ea=_verified_frame(root,p,'sina_continuous_exploratory_v1')
        b,eb=_verified_frame(root,current,'sina_continuous_exploratory_v1')
        # Raw API requests return all history even when normalized smoke is one month.
        a=pd.read_csv(root/ea['raw_file']);b=pd.read_csv(root/eb['raw_file'])
        a=a.loc[a.date.between('2011-10-01',ANALYSIS_END)];b=b.loc[b.date.between('2011-10-01',ANALYSIS_END)]
        common=a.merge(b,on='date',suffixes=('_earlier','_later'),validate='one_to_one')
        fields=['open','high','low','close','volume','hold','settle']
        differences={c:differing_cells(common[c+'_earlier'],common[c+'_later']) for c in fields}
        rows.append({'symbol':m['api_kwargs']['symbol'],'earlier_manifest':str(p.relative_to(root)),
            'later_manifest':str(current.relative_to(root)),'common_nonsealed_dates':len(common),
            'differing_cells':sum(differences.values()),'differences_by_field':json.dumps(differences),
            'conclusion':'Short capture-interval agreement/disagreement only; not historical PIT or no-back-adjustment proof'})
    return pd.DataFrame(rows,columns=['symbol','earlier_manifest','later_manifest','common_nonsealed_dates','differing_cells','differences_by_field','conclusion'])


def audit(root):
    root=Path(root).resolve();out=root/'experiments/phase_d_continuous'
    if (out/'delivery_manifest.json').exists():raise FileExistsError('Existing audit sealed; create a new version')
    rules=json.loads((out/'frozen_audit_rules.json').read_text());validate_asset_configuration(root)
    frames,sources,stocks=read_assets(root)
    calpath=root/'data/manifests/baseline15y/reference_trading_calendar.csv'
    full=pd.read_csv(calpath,dtype=str).date
    calendar=sorted(full.loc[full.between('2011-10-01','2026-09-30')].unique())
    analysis=[d for d in calendar if d<=ANALYSIS_END]
    ids=stocks+list(YIELD_SPECS)+list(INDEX_SPECS)+['FUT_'+s.variety for s in ALL_FUTURES]
    mask_columns={};cross_assets=[]
    for asset in ids:
        kind='future' if asset.startswith('FUT_') else 'yield_curve' if asset in YIELD_SPECS else 'treasury_wealth_index' if asset in INDEX_SPECS else 'stock'
        frame=frames[asset];nonsealed=frame.loc[frame.date.le(ANALYSIS_END)]
        mask=availability(nonsealed,analysis,kind);mask_columns[asset]=mask
        row={'asset_id':asset,'market':'future' if kind=='future' else 'bond' if asset in YIELD_SPECS or asset in INDEX_SPECS else 'stock',
            'economic_type':kind,'original_unit':frame.unit.iloc[0] if 'unit' in frame else frame.price_unit.iloc[0] if 'price_unit' in frame else 'NATIVE_VENDOR_QUOTE_UNIT_NOT_VERIFIED',
            'available_at_status':'UNKNOWN_NO_SAME_DAY_OPEN_ASSUMPTION',
            'off_reference_calendar_dates':int(nonsealed.loc[~nonsealed.date.isin(analysis),'date'].nunique()),
            'duplicate_date_count':int(nonsealed.date.duplicated().sum())}
        for split,(start,end) in SPLITS.items():
            x=mask.loc[(mask.index>=start)&(mask.index<=end)]
            row[split+'_valid_dates']=int(x.sum());row[split+'_coverage_rate']=float(x.mean()) if len(x) else None
        cross_assets.append(row)
    masks=pd.DataFrame(mask_columns,index=pd.Index(analysis,name='date'))
    masks.to_csv(out/'per_asset_availability_mask.csv.gz',compression='gzip')
    pd.DataFrame(cross_assets).to_csv(out/'cross_market_asset_coverage.csv',index=False)
    availability_summary={m:pd.DataFrame(cross_assets).loc[lambda f:f.market.eq(m)].copy() for m in ('stock','bond','future')}
    future_rows=[];target_rows=[];roll=[];anomalies=[];labels={};full_label_wide={}
    quote_issues=[];off_calendar=[];missing_dates=[]
    listings=json.loads((root/'data/manifests/listing_dates_verified.json').read_text())
    source_map={e['parameters'].get('symbol'):e for e in sources if e['source_id']=='sina_continuous_exploratory_v1'}
    for spec in ALL_FUTURES:
        asset='FUT_'+spec.variety;f=frames[asset];m=availability(f,calendar,'future')
        observed=set(f.date);mask=masks[asset];candidates=candidate_masks(mask)
        labels[asset]=candidates
        row={'asset_id':asset,'asset_name':spec.name_en,'target_or_auxiliary':spec.role,
            'source':'Sina InnerFuturesNewService.getDailyKLine via AKShare','vendor_symbol':spec.variety+'0',
            'first_valid_date':m.index[m].min() if m.any() else None,'last_valid_date':m.index[m].max() if m.any() else None,
            'record_count':len(f),'valid_reference_dates':int(m.sum()),'missing_count':int((~m).sum()),
            'longest_missing_gap':longest_gap(m),'nonpositive_price_count':int(pd.to_numeric(f.close,errors='coerce').le(0).sum()),
            'nonfinite_close_count':int((~finite(f.close)).sum()),'duplicate_date_count':int(f.date.duplicated().sum()),
            **quote_quality_counts(f,calendar),
            'missing_observation_dates':int(sum(d not in observed for d in calendar)),
            'calendar_basis':'FROZEN_BASELINE_REFERENCE_NOT_INDEPENDENT_FUTURES_VERIFICATION',
            'source_manifest':source_map[spec.variety+'0']['manifest']}
        listing=listings.get(asset,{}).get('date')
        row['verified_listing_date']=listing
        row['prelisting_reference_dates']=int(sum(d<listing for d in calendar)) if listing else None
        row['prelisting_observations']=int(f.date.lt(listing).sum()) if listing else None
        row['post_verified_listing_missing_dates']=int((~m.loc[m.index>=listing]).sum()) if listing else None
        for split,(start,end) in SPLITS.items():
            x=mask.loc[(mask.index>=start)&(mask.index<=end)]
            row[split+'_valid_dates']=int(x.sum());row[split+'_coverage_rate']=float(x.mean())
            row[split+'_longest_missing_gap']=longest_gap(x)
            for h in HORIZONS:row[f'{split}_{h}d_valid_labels']=int(candidates.loc[x.index,h].sum())
        row['TRAIN_at_least_252_valid_quotes']=row['TRAIN_valid_dates']>=252
        row['TRAIN_at_least_756_valid_quotes']=row['TRAIN_valid_dates']>=756
        row['VAL_2024_valid_dates']=int(mask.loc[(mask.index>='2024-01-01')&(mask.index<=ANALYSIS_END)].sum())
        row['VAL_2024_status']='CANDIDATE_EVALUABLE' if row['VAL_2024_valid_dates'] else 'NOT_EVALUABLE'
        row['candidate_label_definition']='BOOLEAN_GAP_FREE_SAME_SPLIT_VENDOR_CLOSE_WINDOW_NOT_FORMAL_LABEL'
        for h in HORIZONS:row[f'{h}d_valid_labels']=row[f'TRAIN_{h}d_valid_labels']+row[f'VAL_{h}d_valid_labels']
        row['label_count_scope']='TRAIN_PLUS_VAL_ONLY_NO_SEALED_LABELS'
        future_rows.append(row)
        own=availability(f,sorted(f.date.unique()),'future')
        invalid_dates=set(own.index[~own])
        for r in f.loc[f.date.le(ANALYSIS_END)&f.date.isin(invalid_dates)].to_dict('records'):
            quote_issues.append({'asset_id':asset,'date':r['date'],
                **{c:r[c] for c in ('open','high','low','close')},
                'reason':'OHLC_NONFINITE_NONPOSITIVE_INCONSISTENT_OR_DUPLICATE',
                'action':'RAW_PRESERVED_AVAILABILITY_FALSE','manifest':row['source_manifest']})
        for date in f.loc[f.date.le(ANALYSIS_END)&~f.date.isin(calendar),'date']:
            off_calendar.append({'asset_id':asset,'date':date,'status':'OFF_REFERENCE_CALENDAR_REQUIRES_REVIEW',
                'action':'Preserved raw; not inserted as an extra reference session','manifest':row['source_manifest']})
        for date in mask.index[~mask]:
            reason='INVALID_OR_DUPLICATE_QUOTE' if date in observed else 'MISSING_SOURCE_OBSERVATION'
            if listing and date<listing:reason='BEFORE_VERIFIED_LISTING'
            elif date<f.date.min():reason='BEFORE_FIRST_SOURCE_OBSERVATION_LISTING_NOT_VERIFIED'
            elif date>f.date.max():reason='AFTER_LAST_SOURCE_OBSERVATION_NOT_FILLED'
            missing_dates.append({'asset_id':asset,'date':date,'reason':reason})
        if spec.role=='target':
            t={k:row[k] for k in ('asset_id','asset_name','vendor_symbol','TRAIN_valid_dates','VAL_valid_dates','VAL_2024_status')}
            for split,(start,end) in SPLITS.items():
                for h in HORIZONS:t[f'{split}_{h}d_candidate_labels']=row[f'{split}_{h}d_valid_labels']
            for year in (2023,2024):
                index=candidates.index.str.startswith(str(year))
                for h in HORIZONS:t[f'VAL_{year}_{h}d_candidate_labels']=int(candidates.loc[index,h].sum())
            target_rows.append(t)
            for h in HORIZONS:full_label_wide[f'{asset}__{h}d']=candidates[h]
        # Adjacent-session diagnostic changes only within TRAIN or VAL, no SEALED.
        q=f.loc[f.date.le(ANALYSIS_END)&~f.date.duplicated(keep=False)].set_index('date').close.reindex(analysis)
        previous=q.shift(1);valid_pair=mask&mask.shift(1,fill_value=False)
        same_split=pd.Series(False,index=analysis)
        for start,end in SPLITS.values():same_split|=pd.Series(analysis,index=analysis).between(start,end)&pd.Series(analysis,index=analysis).shift(1).between(start,end)
        changes=(q/previous-1).where(valid_pair&same_split)
        flagged=changes.abs().gt(rules['price_jump_flag_fraction'])
        for date in changes.index[flagged]:
            anomalies.append({'asset_id':asset,'date':date,'candidate_adjacent_price_change':changes.loc[date],
                'anomaly':'LARGE_VENDOR_PRICE_CHANGE_REVIEW','threshold_fraction':rules['price_jump_flag_fraction'],
                'roll_confirmed':False,'action':'PRESERVE_PRICE_NO_CLIPPING_OR_AUTOMATIC_EXCLUSION'})
        roll.append({'asset_id':asset,'asset_name':spec.name_en,'role':spec.role,'source':'Sina via AKShare',
            'vendor_symbol':spec.variety+'0','continuous_definition_status':'ROLL_METHOD_NOT_VERIFIED',
            'main_continuous_status':'SYMBOL_ZERO_CANDIDATE_NO_DATED_UNDERLYING_MAPPING',
            'adjustment_status':'NOT_VERIFIED_NO_ADJUSTMENT_PARAMETERS_OR_FACTORS',
            'historical_roll_dates_identifiable':False,'cross_contract_gap_possible':True,
            'future_back_adjustment_or_history_revision':'NOT_VERIFIED',
            'large_train_val_adjacent_change_count':int(flagged.sum()),
            'historical_point_in_time_vintage':'NOT_VERIFIED_CURRENT_CAPTURE',
            'descriptive_research_ready':bool(m.any()),
            'exploratory_vendor_series_prediction':'CANDIDATE_ONLY_FIXED_SNAPSHOT_WITH_MASKS',
            'formal_economic_return_label_ready':False,
            'economic_meaning':'Operational vendor price series only; not verified tradable or constant-maturity return',
            'evidence':source_map[spec.variety+'0']['manifest']})
    futures=pd.DataFrame(future_rows);targets=pd.DataFrame(target_rows)
    futures.to_csv(out/'futures_38_coverage.csv',index=False);targets.to_csv(out/'target_24_label_coverage.csv',index=False)
    targets.loc[targets.asset_id.ne('FUT_WH')].to_csv(out/'target_23_auxiliary_report.csv',index=False)
    pd.DataFrame(roll).to_csv(out/'continuous_roll_audit.csv',index=False)
    pd.DataFrame(quote_issues,columns=['asset_id','date','open','high','low','close','reason','action','manifest']).to_csv(out/'continuous_quote_quality_issues.csv',index=False)
    pd.DataFrame(off_calendar,columns=['asset_id','date','status','action','manifest']).to_csv(out/'continuous_off_calendar_dates.csv',index=False)
    pd.DataFrame(missing_dates,columns=['asset_id','date','reason']).to_csv(out/'continuous_missing_dates.csv.gz',index=False,compression='gzip')
    pd.DataFrame(anomalies,columns=['asset_id','date','candidate_adjacent_price_change','anomaly','threshold_fraction','roll_confirmed','action']).to_csv(out/'continuous_price_anomalies.csv',index=False)
    labels_wide=pd.DataFrame(full_label_wide)
    audit_dates=[d for d in analysis if d>='2017-01-01']
    labels_wide.loc[audit_dates].to_csv(out/'per_target_label_mask.csv.gz',compression='gzip',index_label='date')
    daily=pd.DataFrame(index=pd.Index(audit_dates,name='date'))
    daily['split']=['TRAIN' if d<'2023-01-01' else 'VAL' for d in audit_dates]
    for name,columns in [('stocks_263',stocks),('bond_yields_18',list(YIELD_SPECS)),('bond_wealth_7',list(INDEX_SPECS)),
                         ('futures_targets_24',['FUT_'+s.variety for s in TARGET_FUTURES]),
                         ('futures_auxiliary_14',['FUT_'+s.variety for s in ALL_FUTURES if s.role=='auxiliary'])]:
        counts=masks[columns].sum(axis=1);daily[name+'_observed']=counts.reindex(audit_dates)
        daily[name+'_lag2_assumption']=counts.shift(2).reindex(audit_dates)
    for h in HORIZONS:
        columns=[f'FUT_{s.variety}__{h}d' for s in TARGET_FUTURES]
        daily[f'{h}d_candidate_evaluable_targets_24']=labels_wide[columns].sum(axis=1).reindex(audit_dates)
        daily[f'{h}d_candidate_evaluable_targets_23_ex_WH']=labels_wide[[c for c in columns if not c.startswith('FUT_WH__')]].sum(axis=1).reindex(audit_dates)
    daily.to_csv(out/'cross_market_date_coverage.csv')
    existing_vendor_vintages(root).to_csv(out/'existing_vendor_vintage_comparison.csv',index=False)
    atomic_json({'at':utc_now(),'calendar_file':str(calpath.relative_to(root)),'calendar_sha256':sha256_file(calpath),
        'rules':rules,'source_files':sources,'stock_order':stocks,'bond_order':list(YIELD_SPECS)+list(INDEX_SPECS),
        'futures_order':['FUT_'+s.variety for s in ALL_FUTURES],'target_order':['FUT_'+s.variety for s in TARGET_FUTURES],
        'network_calls':0,'sealed_candidate_labels_accessed':False,'numeric_forward_return_labels_created':False,
        'all_sources_sha256_verified':True},out/'provenance.json')
    from triweave.continuous_report import write_report
    return write_report(root,out,futures,targets,daily,pd.DataFrame(roll),pd.DataFrame(cross_assets),anomalies)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    a=p.parse_args(argv);result=audit(a.root);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
