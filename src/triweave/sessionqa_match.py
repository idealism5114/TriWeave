"""Offline SessionQA fingerprints. Numeric matches are evidence, not PIT proof.

All date gates precede quote parsing, including the non-date-first SDK raw CSVs.
TRAIN is completed and a disposition frozen before the separate VAL invocation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from triweave import after_close as ac
from triweave.sourceqa_investigate import FIELDS, numeric, raw_official_csv
from triweave.universe import sha256_file
from triweave.assets import ALL_CONTRACT_FUTURES

OUT = 'experiments/phase_e_sessionqa'
OFFSETS = (-2, -1, 0, 1, 2)
OHLC = ['open', 'high', 'low', 'close']


def ordinary_mask(t, exchange):
    # SHFE SDK responses may include INE contracts. Apply the declared exchange
    # scope before comparing with the existing standardized exchange partition.
    allowed = {s.variety for s in ALL_CONTRACT_FUTURES if s.exchange == exchange} & set(ac.TARGETS)
    return (t.contract.str.fullmatch(r'[A-Z]+[0-9]{3,4}').fillna(False)
            & t.variety.isin(allowed)
            & t.contract.str.extract(r'^([A-Z]+)',expand=False).eq(t.variety))


def valid_ohlc(f):
    return (np.isfinite(f[OHLC]).all(axis=1) & f[OHLC].gt(0).all(axis=1)
            & f.high.ge(f[OHLC].max(axis=1)) & f.low.le(f[OHLC].min(axis=1)))


def read_sources(root, split):
    lo, hi = ac.SPLITS[split]
    if split not in ('TRAIN', 'VAL'):
        raise ValueError('TEST numeric access prohibited')
    logs, vendors, official, sources, comparisons = [], [], [], [], []
    prior = json.loads((root/'experiments/phase_d_continuous/provenance.json').read_text())
    for e in prior['source_files']:
        if e['source_id'] != 'sina_continuous_exploratory_v1':
            continue
        c = e['parameters']['symbol'][:-1]
        if c not in ac.TARGETS:
            continue
        for p, h in [('raw_file','raw_sha256'), ('standardized_file','standardized_sha256')]:
            assert sha256_file(root/e[p]) == e[h]
        f = numeric(ac.safe_csv(root/e['raw_file'], start=lo, end=hi, log=logs)
                    .rename(columns={'hold':'open_interest'}))
        f['variety'] = c
        f['asset_id'] = 'FUT_'+c
        vendors.append(f)
        sources.append(e)
    inv = json.loads((root/'experiments/phase_e_sourceqa/investigation_results.json').read_text())
    for ix, e in enumerate(inv['official_sources']):
        p = e['parameters']
        if p['start_date'] > hi.replace('-','') or p['end_date'] < lo.replace('-',''):
            continue
        assert sha256_file(root/e['manifest']) == e['manifest_sha256']
        for k in ('raw', 'standardized'):
            assert sha256_file(root/e[k+'_file']) == e[k+'_sha256']
        f = numeric(ac.safe_csv(root/e['standardized_file'], start=lo, end=hi, log=logs))
        raw = numeric(raw_official_csv(root/e['raw_file'], lo, hi, log=logs))
        raw['contract'] = raw.symbol.str.upper()
        for t in (f, raw):
            t['variety'] = t.variety.str.upper()
        exchange = p['market']
        f, raw = f.loc[ordinary_mask(f,exchange)].copy(), raw.loc[ordinary_mask(raw,exchange)].copy()
        keys = ['date','contract','variety'] + FIELDS
        a, b = f[keys].drop_duplicates(), raw[keys].drop_duplicates()
        delta = a.merge(b, on=keys, how='outer', indicator=True)
        unmatched = int(delta['_merge'].ne('both').sum())
        if unmatched:
            print('Raw/standardized difference',e['raw_file'],unmatched,
                  delta.loc[delta['_merge'].ne('both')].head(3).to_dict('records'),flush=True)
        comparisons.append(dict(manifest=e['manifest'], ordinary_standardized_rows=len(f),
                                ordinary_raw_rows=len(raw), mismatched_unique_rows=unmatched,
                                raw_date_origin='SDK_REQUEST_DAY_NOT_ORIGINAL_HTTP_BUSINESS_DATE',
                                original_exchange_payload_available=False,
                                mismatch_examples=delta.loc[delta['_merge'].ne('both')].head(3).to_json(orient='records')))
        f['official_source_index'] = ix
        official.append(f)
        sources.append(e)
        # Small raw evidence extracts retain original requested archive date.
        if split == 'TRAIN' and p['start_date'] <= '20130709' <= p['end_date']:
            raw.loc[raw.variety.isin(['AU','AG']) & raw.date.between('2013-07-04','2013-07-10')].to_csv(
                root/OUT/'evidence'/f'raw_AU_AG_{Path(e["raw_file"]).stem}.csv',index=False)
    return pd.concat(vendors,ignore_index=True), pd.concat(official,ignore_index=True), sources, logs, comparisons


def match_bars(vendor, official, calendar):
    """One row per vendor bar, with five offset denominators and matches.

    Strict all7 match requires positive coherent OHLC and finite seven fields.
    No tolerance/unit scaling is inferred. OHLC-only matches remain diagnostics.
    """
    v = vendor.sort_values(['variety','date']).reset_index(drop=True).copy()
    v['vendor_row'] = np.arange(len(v))
    v['vendor_date'] = v.date
    v['valid_ohlc'] = valid_ohlc(v)
    v['finite_all7'] = np.isfinite(v[FIELDS]).all(axis=1)
    v['zero_volume'] = v.volume.eq(0)
    v['flat_ohlc'] = v[OHLC].eq(v.close,axis=0).all(axis=1)
    v['active_nonflat'] = v.valid_ohlc & v.volume.gt(0) & v.open_interest.gt(0) & v.high.gt(v.low)
    v['joint_bar_repeated_in_split'] = v.duplicated(['variety']+FIELDS, keep=False)
    v['adjacent_joint_bar_repeated'] = (v[FIELDS].eq(v.groupby('variety')[FIELDS].shift()).all(axis=1)
                                              | v[FIELDS].eq(v.groupby('variety')[FIELDS].shift(-1)).all(axis=1))
    position = {d:i for i,d in enumerate(calendar)}
    v['off_reference_calendar'] = ~v.date.isin(position)
    o = official.loc[valid_ohlc(official)].copy()
    o = o.drop_duplicates(['variety','date','contract']+FIELDS)
    available = o.groupby(['variety','date']).contract.nunique()
    matches = []
    for offset in OFFSETS:
        suffix = f'{offset:+d}'
        v['official_date_'+suffix] = [calendar[position[d]+offset] if d in position and 0 <= position[d]+offset < len(calendar) else '' for d in v.date]
        col = 'official_date_'+suffix
        ix = pd.MultiIndex.from_arrays([v.variety,v[col]])
        v['comparable_contracts_'+suffix] = available.reindex(ix,fill_value=0).to_numpy()
        common = v.loc[v.valid_ohlc & v.finite_all7, ['vendor_row','variety',col]+FIELDS].rename(columns={col:'date'})
        both = common.merge(o.loc[np.isfinite(o[FIELDS]).all(axis=1), ['variety','date','contract','official_source_index']+FIELDS], on=['variety','date']+FIELDS, how='inner')
        unique = both.groupby('vendor_row').contract.nunique()
        v['full7_contract_matches_'+suffix] = v.vendor_row.map(unique).fillna(0).astype(int)
        contracts = both.groupby('vendor_row').contract.agg(lambda s:'|'.join(sorted(set(s))))
        v['contracts_'+suffix] = v.vendor_row.map(contracts).fillna('')
        diagnostic = common[['vendor_row','variety','date']+OHLC].merge(o[['variety','date','contract']+OHLC],on=['variety','date']+OHLC)
        v['ohlc4_contract_matches_'+suffix] = v.vendor_row.map(diagnostic.groupby('vendor_row').contract.nunique()).fillna(0).astype(int)
        if len(both):
            both['offset_sessions'] = offset
            matches.append(both)
    counts = v[[f'full7_contract_matches_{k:+d}' for k in OFFSETS]]
    v['matched_date_count'] = counts.gt(0).sum(axis=1)
    v['matched_contract_count'] = counts.sum(axis=1)
    v['no_full7_match'] = v.matched_date_count.eq(0)
    v['ambiguous_dates'] = v.matched_date_count.gt(1)
    v['unique_date_and_contract'] = v.matched_date_count.eq(1) & v.matched_contract_count.eq(1)
    v['unique_match_offset'] = [next((k for k in OFFSETS if r[f'full7_contract_matches_{k:+d}']>0),None) if r['unique_date_and_contract'] else None for r in v.to_dict('records')]
    v['unique_active_shift'] = v.unique_date_and_contract & v.active_nonflat & ~v.joint_bar_repeated_in_split & v.unique_match_offset.ne(0)
    v['source_id'] = 'sina_continuous_exploratory_v1'
    v['identity_scope'] = 'EXACT_NATIVE_NUMERIC_FINGERPRINT_ONLY_UNIT_AND_HISTORICAL_PIT_NOT_INDEPENDENTLY_VERIFIED'
    return v, pd.concat(matches,ignore_index=True) if matches else pd.DataFrame()


def summarize(f, group):
    rows = []
    for keys, v in f.groupby(group,dropna=False):
        if len(group)==1: keys = (keys[0] if isinstance(keys,tuple) else keys,)
        row = dict(zip(group,keys))
        row.update(vendor_rows=len(v), active_nonflat=int(v.active_nonflat.sum()),
                   finite_all7=int(v.finite_all7.sum()), zero_volume=int(v.zero_volume.sum()),
                   flat_ohlc=int(v.flat_ohlc.sum()), repeated_joint_bars=int(v.joint_bar_repeated_in_split.sum()),
                   unique_match_rows=int(v.unique_date_and_contract.sum()), ambiguous_date_rows=int(v.ambiguous_dates.sum()),
                   no_full7_match_rows=int(v.no_full7_match.sum()), unique_active_shift_rows=int(v.unique_active_shift.sum()))
        for k in OFFSETS:
            s=f'{k:+d}'
            n=int(v['comparable_contracts_'+s].gt(0).sum())
            m=int(v['full7_contract_matches_'+s].gt(0).sum())
            u=int((v.unique_date_and_contract & v.unique_match_offset.eq(k)).sum())
            row.update({f'comparable_{s}':n,f'full7_matches_{s}':m,f'unique_matches_{s}':u,
                        f'ohlc4_matches_{s}':int(v['ohlc4_contract_matches_'+s].gt(0).sum()),
                        f'full7_rate_comparable_{s}':m/n if n else None,
                        f'full7_rate_all_vendor_{s}':m/len(v)})
        rows.append(row)
    return pd.DataFrame(rows)


def run(root, split):
    root=Path(root); out=root/OUT
    if split not in ('TRAIN','VAL'): raise ValueError('TEST prohibited')
    if split=='VAL' and not (out/'frozen_disposition.json').exists():
        raise RuntimeError('Freeze TRAIN disposition before VAL quotes')
    resultfile=out/f'{split.lower()}_matching_results.json'
    if resultfile.exists(): raise FileExistsError(resultfile)
    vendor, official, sources, logs, comparisons = read_sources(root,split)
    calendar=pd.read_csv(root/'data/manifests/baseline15y/reference_trading_calendar.csv',dtype=str).date.tolist()
    f,matches=match_bars(vendor,official,calendar)
    f['year']=f.date.str[:4]
    f['period']=np.where(f.date<'2013-07-05','PRE_AU_AG_NIGHT_LAUNCH','POST_AU_AG_NIGHT_LAUNCH')
    f.to_csv(out/f'{split.lower()}_all25_date_matching.csv',index=False)
    matches.to_csv(out/f'{split.lower()}_all25_full7_matches.csv',index=False)
    summarize(f,['variety','year']).to_csv(out/f'{split.lower()}_year_matching_summary.csv',index=False)
    summarize(f,['variety','period']).to_csv(out/f'{split.lower()}_launch_period_summary.csv',index=False)
    summary=summarize(f,['variety'])
    summary.to_csv(out/f'{split.lower()}_asset_matching_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/f'{split.lower()}_raw_standardized_date_audit.csv',index=False)
    if split=='TRAIN':
        f.loc[f.variety.isin(['AU','AG']) & f.date.between('2013-05-01','2013-08-31')].to_csv(out/'au_ag_date_matching.csv',index=False)
        f.loc[f.unique_active_shift].to_csv(out/'train_unique_active_date_conflicts.csv',index=False)
    result=dict(split=split,vendor_rows=len(vendor),official_ordinary_rows=len(official),
                official_source_files=len(comparisons), sources=sources,quote_read_log=logs,
                TEST_numeric_rows_parsed=0, raw_standardized_mismatches=sum(c['mismatched_unique_rows'] for c in comparisons),
                unit_scope='Forensic joint identity in untransformed native numbers. Units and historical counting convention not independently certified; nonmatches not errors.',
                summary=summary.replace({np.nan:None}).to_dict('records'))
    ac.write_json(resultfile,result)
    print(summary[['variety','vendor_rows','comparable_+0','full7_matches_+0','full7_matches_+1','unique_active_shift_rows','ambiguous_date_rows','no_full7_match_rows']].to_string(index=False),flush=True)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['TRAIN','VAL'],required=True)
    run(Path.cwd(),p.parse_args().split)
