"""Read-only baseline audit plus separately versioned rescue observations."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import shutil

import pandas as pd

from triweave.assets import ALL_CONTRACT_FUTURES,TARGET_FUTURES,AUX_FUTURES
from triweave.collect import atomic_json,utc_now
from triweave.pipeline import environment
from triweave.universe import sha256_file,load_stock_universe


def classify_anomaly(kind, contract, volume, oi, close, ohlc_invalid=False, missing_fields=False):
    """Same-day evidence only. Activity never means retrospectively chosen main contract."""
    if '_TAS' in str(contract) or kind=='SPECIAL_TRADING_MODE_NOT_OUTRIGHT_QUOTE':
        return 'REQUIRES_REVIEW','Known special trading mode; preserve separately, not an outright quote'
    if kind=='LONG_ZERO_VOLUME_RUN_REVIEW':
        return 'LIKELY_INACTIVE_CONTRACT','Original fixed-rule flag establishes a zero-volume run; listing/tradability still unverified'
    if kind in ('INVALID_VALUE_OR_OHLC_REVIEW','UNRECOGNIZED_CONTRACT_CODE_REVIEW'):
        if pd.notna(volume) and volume<0 or pd.notna(oi) and oi<0:
            return 'CONFIRMED_INVALID','Negative activity is invalid as a numerical activity observation'
        if pd.notna(volume) and volume==0:
            return 'LIKELY_INACTIVE_CONTRACT','No observed trading on this date; zero/missing quotes preserved'
        if missing_fields:
            return 'POSSIBLE_SOURCE_SCHEMA_ISSUE','Required source fields absent; inactivity versus parser/source issue not established'
        if ohlc_invalid or pd.notna(close) and close<=0:
            return 'CONFIRMED_INVALID','Invalid as an ordinary OHLC quote, not evidence that the provider fabricated data'
        return 'REQUIRES_REVIEW','Original flag requires source-level confirmation'
    return 'REQUIRES_REVIEW','Price jump/constant sequence can be economic, a roll, inactivity or a source issue; no automatic repair'


def source_id(group):
    if group=='stocks': return 'eastmoney_unadjusted_v1'
    if group=='futures_continuous_exploratory': return 'sina_continuous_exploratory_v1'
    if group.startswith('futures_contracts/'): return group.split('/')[1].lower()+'_official_akshare_v1'
    return 'chinabond_'+group+'_v1'


def read_observations(root,run_id):
    baseline=[];new=[];metadata=[]
    for p in sorted((root/'data/manifests/baseline15y/jobs').rglob('*.json')):
        m=json.loads(p.read_text())
        if m['status']!='OK': continue
        group=m['job_group'];path=root/'data/standardized/baseline15y'/group/(m['name']+'.csv.gz')
        raw=root/'data/raw/baseline15y'/group/(m['name']+'.csv.gz')
        if sha256_file(path)!=m['standardized_sha256'] or sha256_file(raw)!=m['raw_sha256']:
            raise ValueError('Baseline integrity failure: '+str(p))
        f=pd.read_csv(path)
        f['source_id']=source_id(group)
        if group=='stocks': f['series_type']='stock'
        elif group.startswith('bond_'): f['series_type']=f['instrument_type']
        f['raw_file']=str(raw.relative_to(root));f['manifest']=str(p.relative_to(root));f['retrieved_at']=m['retrieved_utc']
        baseline.append(f);metadata.append({**m,'source_id':source_id(group),'manifest':str(p.relative_to(root))})
    for p in sorted((root/'data/manifests'/run_id).rglob('*.json')):
        m=json.loads(p.read_text())
        if not m.get('standardized_file') or m.get('category') not in ('stocks','futures_contracts','futures_contracts_daily_partial_v3'): continue
        if not m['status'].startswith('OBSERVATIONS_SAVED'): continue
        raw=root/m['raw_file'];path=root/m['standardized_file']
        if sha256_file(raw)!=m['raw_sha256'] or sha256_file(path)!=m['standardized_sha256']: raise ValueError('Rescue integrity failure: '+str(p))
        f=pd.read_csv(path,dtype={'asset_id':str,'trading_status':str})
        if m['category']=='stocks': f['series_type']='stock';f['exchange']=f.asset_id.str[:2].str.upper()
        f['source_id']=m['source_id'];f['raw_file']=m['raw_file'];f['manifest']=str(p.relative_to(root))
        f['retrieved_at']=m['params'].get('upstream_retrieved_at',m['retrieved_at'])
        new.append(f);metadata.append({**m,'manifest':str(p.relative_to(root))})
    return pd.concat(baseline,ignore_index=True),pd.concat(new,ignore_index=True) if new else pd.DataFrame(),metadata


def anomaly_triage(old,frames):
    issues=old.copy().reset_index(names='original_anomaly_id')
    issues['date']=issues['date'].fillna('').astype(str).str[:10]
    issues['contract']=issues['contract'].fillna('')
    rows=frames.copy();rows['date']=rows.date.astype(str).str[:10]
    if 'contract' not in rows: rows['contract']=''
    rows['contract']=rows.contract.fillna('')
    columns=['asset_id','series_type','date','contract','source_id','raw_file','manifest','volume','open_interest','open','high','low','close','settle']
    for col in columns:
        if col not in rows: rows[col]=float('nan')
    keys=['asset_id','series_type','date','contract']
    unique=rows[columns].drop_duplicates(keys)
    issues=issues.merge(unique,on=keys,how='left',validate='many_to_one')
    source_map=rows.groupby(['asset_id','series_type']).source_id.first().to_dict()
    issues['source']=issues.source_id
    status=[];reasons=[];active=[];bad_ohlc=[]
    for row in issues.to_dict('records'):
        v,oi,c=row['volume'],row['open_interest'],row['close']
        required=[row[x] for x in ('open','high','low','close')]
        missing=any(pd.isna(x) for x in required) or pd.isna(oi)
        bad=not any(pd.isna(x) for x in required) and (row['high']<max(row['open'],row['close'],row['low']) or row['low']>min(row['open'],row['close'],row['high']))
        category,reason=classify_anomaly(row['anomaly'],row['contract'],v,oi,c,bad,missing)
        status.append(category);reasons.append(reason);bad_ohlc.append(bad)
        active.append(bool(v>0) if pd.notna(v) else None)
    issues['review_status']=status;issues['evidence_reason']=reasons;issues['is_active_contract']=active
    issues['activity_definition']='Same-date volume > 0; never an ex-post main-contract choice'
    issues['source']=issues.apply(lambda r: r['source'] if pd.notna(r['source']) else source_map.get((r.asset_id,r.series_type),'NOT_VERIFIED'),axis=1)
    issues['contract_id']=issues.contract.replace('',pd.NA);issues['anomaly_type']=issues.anomaly
    issues['trading_volume']=issues.volume;issues['ohlc_relation_invalid']=bad_ohlc
    issues['row_resolution_status']=issues.raw_file.notna().map({True:'EXACT_OBSERVATION_LINKED',False:'AGGREGATED_EVENT_NOT_SINGLE_OBSERVATION'})
    issues['evidence_interval_start']=issues['date'];issues['evidence_interval_end']=issues['date']
    issues['evidence_raw_files']=issues.raw_file;issues['evidence_manifests']=issues.manifest
    groups={(asset,kind):group for (asset,kind),group in rows.groupby(['asset_id','series_type'],sort=False)}
    for index in issues.index[issues.raw_file.isna()]:
        event=issues.loc[index];f=groups.get((event.asset_id,event.series_type),pd.DataFrame())
        if not f.empty and event.contract:
            f=f.loc[f.contract.eq(event.contract)]
        elif not f.empty and event.anomaly=='SPECIAL_TRADING_MODE_NOT_OUTRIGHT_QUOTE':
            f=f.loc[f.contract.str.contains('_TAS',na=False)]
        if not f.empty:
            issues.loc[index,'evidence_interval_start']=f.date.min();issues.loc[index,'evidence_interval_end']=f.date.max()
            issues.loc[index,'evidence_raw_files']=';'.join(sorted(f.raw_file.dropna().unique()))
            issues.loc[index,'evidence_manifests']=';'.join(sorted(f.manifest.dropna().unique()))
    issues['aggregation_note']='Range is the supporting contract/source window; blank point date is not an invented observation date'
    issues['action']='PRESERVE_RAW_NO_AUTOMATIC_PRICE_REPLACEMENT'
    return issues


def stock_overlap(old,new):
    columns=['asset_id','date','field','source_a','source_b','raw_file_a','raw_file_b','value_a_original','value_b_original',
             'comparison_value_a','comparison_value_b','unit','difference','comparison_status','economic_identity_status']
    rows=[]
    if new.empty: return pd.DataFrame(columns=columns)
    a=old.loc[old.series_type.eq('stock')];b=new.loc[new.series_type.eq('stock')]
    if a.empty or b.empty:return pd.DataFrame(columns=columns)
    joined=a.merge(b,on=['asset_id','date'],suffixes=('_a','_b'),validate='one_to_one')
    for field in ('close','volume','amount','turnover'):
        x=joined[field+'_a'];y=joined[field+'_b'];factor=100 if field=='volume' else 1
        for i in joined.index:
            av,bv=x.loc[i],y.loc[i];ca=av*factor
            row=joined.loc[i]
            rows.append({'asset_id':row.asset_id,'date':row.date,'field':field,'source_a':row.source_id_a,'source_b':row.source_id_b,
              'raw_file_a':row.raw_file_a,'raw_file_b':row.raw_file_b,'value_a_original':av,'value_b_original':bv,
              'comparison_value_a':ca,'comparison_value_b':bv,'unit':{'close':'RMB_per_share','volume':'shares','amount':'RMB','turnover':'percent'}[field],
              'difference':bv-ca,'comparison_status':'EXACT_NUMERIC_EQUAL' if ca==bv else 'DIFFERENCE_REQUIRES_REVIEW',
              'economic_identity_status':'Unadjusted same stock/date; volume lots explicitly multiplied by 100; no claim that numeric agreement proves PIT or suspension equivalence'})
    return pd.DataFrame(rows,columns=columns)


def wh_audit(root,frames,out):
    f=frames.loc[frames.asset_id.eq('FUT_WH') & frames.series_type.eq('exchange_listed_contract')].copy()
    f['year']=pd.to_datetime(f.date).dt.year
    yearly=f.groupby('year').agg(record_count=('date','size'),positive_volume_records=('volume',lambda x:int(x.gt(0).sum())),
        positive_close_records=('close',lambda x:int(x.gt(0).sum())),first_date=('date','min'),last_date=('date','max'))
    yearly.to_csv(out/'wh_official_yearly.csv')
    trades=f.loc[f.volume.gt(0),['date','contract','close','settle','volume','open_interest','source_id','raw_file','manifest']]
    trades.loc[trades.date.ge('2023-01-01')].to_csv(out/'wh_post2023_observed_trades.csv',index=False)
    legacy=root.parent/'Commedities/fetch data/futures_crawler.py'
    legacy_data=root.parent/'Commedities/Data/futures_data/农产品/小麦.csv'
    legacy_frame=pd.read_csv(legacy_data)
    text=f'''# WH 专项核查\n\nCONFIRMED：原 CMGM 源码将“小麦”映射至 CZCE 的 WH，并调用 AKShare futures_zh_daily_sina(symbol="WH0")。旧文件合约代码也是 WH0，属于第三方连续序列，最后观测为 {legacy_frame.date.max()}，不是逐合约标签。\n\n官方逐合约数据按年统计见 wh_official_yearly.csv；2023年有 {int(yearly.loc[2023,'positive_volume_records'])} 条正成交量记录，2024年为 {int(yearly.loc[2024,'positive_volume_records'])} 条。最后一条已取得的正成交量记录为 {trades.date.max()}；2025年的孤立交易仍需核验，不得概括成2023年以后绝无交易。\n\nVAL 2023—2024 的覆盖远不足以确认最长20步的有效收益标签，标记 WH = NOT_EVALUABLE。当前没有构造标签、没有根据VAL预测结果选择协议。\n\n官方历史规章同时列出旧 WS 和新 WH，证明代码/制度需要分期核验，不能直接拼接，更不能替换为 PM 普麦。主连 WH0 早于普通 WH 首次观测的历史很可能包含前身或供应商拼接；这是待验证推断，不能视为经济等价证明。\n\n官方证据：[2012年交易规则](https://www.czce.com.cn/cn/rootfiles/2012/11/28/1349588556736465-1349588556738184.pdf)、[2023年交易所报告](https://www.czce.com.cn/cn/rootfiles/2023/07/12/1689459177722861-1689459177739293.pdf)。完整历史换码公告及合约规格比较尚未取得独立完整证据。\n\nWH_TARGET_MAPPING_VERIFIED = NOT_VERIFIED。原24个目标及其顺序保留。后续协议候选：A 保留24目标、使用有效标签mask；B 将小麦作为特殊目标并另行冻结可评价集合。本轮不选定方案、不改目标。\n\n只读原始证据：\n- 源码 SHA256：{sha256_file(legacy)}\n- 旧小麦文件 SHA256：{sha256_file(legacy_data)}\n'''
    raw_history=out/'wh_raw_code_history.csv'
    if raw_history.exists():
        text+='\n## 已有官方raw中的代码历史\n\n'+raw_history.read_text()+'\n这是逐合约代码出现范围，不是经济等价或上市日期证明；WS/WH/WT/PM保持分开。上述PDF的搜索索引可核对相关代码，但本环境未成功完整下载文档，未声称已核验所有条款。\n'
    (out/'wh_mapping_audit.md').write_text(text)
    return yearly


def write_coverage(root,old,new,out,run_id='rescue_20261015'):
    stocks=load_stock_universe(root/'data/manifests/stocks_263.csv')
    rows=[];provider_rows=[]
    combined=pd.concat([old,new],ignore_index=True)
    groups={key:f for key,f in combined.groupby(['asset_id','series_type'],sort=False)}
    profiles={}
    for p in (root/'data/standardized'/run_id/'baostock_security_metadata_v1/security_metadata').glob('*.csv.gz'):
        profiles=pd.read_csv(p,dtype=str).set_index('asset_id').to_dict('index')
    for code in stocks.asset_id:
        for source in ('eastmoney_unadjusted_v1','baostock_unadjusted_v1'):
            f=groups.get((code,'stock'),pd.DataFrame(columns=combined.columns))
            f=f.loc[f.source_id.eq(source)]
            valid=f.close.notna()&f.close.gt(0) if len(f) else pd.Series(dtype=bool)
            if len(f) and source.startswith('baostock'): valid &= f.trading_status.eq('1')
            profile=profiles.get(code,{})
            listing=profile.get('ipoDate',profile.get('IPODate'))
            row={'asset_id':code,'source_id':source,'record_count':len(f),'valid_trade_quote_count':int(valid.sum()),
                 'security_name':profile.get('code_name'),'name_time_status':'Current provider metadata, not historical PIT names',
                 'first_observation_date':f.date.min() if len(f) else None,'last_observation_date':f.date.max() if len(f) else None,
                 'unique_date_count':int(f.date.nunique()) if len(f) else 0,'duplicate_date_count':int(f.date.duplicated().sum()) if len(f) else 0,
                 'suspended_rows':int(f.trading_status.eq('0').sum()) if len(f) and 'trading_status' in f else None,
                 'provider_listing_date':listing,'listing_date_status':'PROVIDER_METADATA_NOT_INDEPENDENTLY_CERTIFIED' if listing else 'NOT_VERIFIED',
                 'prelisting_rows':int(f.date.lt(listing).sum()) if listing and len(f) else None,
                 'collection_status':'OBSERVATIONS_SAVED_HISTORY_QA_PENDING' if valid.sum() else 'BLOCKED_OR_NO_VERIFIED_OBSERVATIONS'}
            row.update(stock_quote_quality(f,source))
            provider_rows.append(row)
            if source.startswith('baostock'): rows.append(row)
    pd.DataFrame(rows).to_csv(out/'stock_collection_results.csv',index=False)
    pd.DataFrame(provider_rows).to_csv(out/'stock_provider_comparison.csv',index=False)
    shutil.copyfile(root/'experiments/data_collection/baseline15y/bond_coverage.csv',out/'bond_coverage_reused.csv')
    contracts=[];identity=[];exchange=[]
    old_cov=pd.read_csv(root/'experiments/data_collection/baseline15y/exchange_contract_coverage.csv').set_index('asset_id')
    for spec in ALL_CONTRACT_FUTURES:
        asset='FUT_'+spec.variety
        f=groups.get((asset,'exchange_listed_contract'),pd.DataFrame(columns=combined.columns))
        from triweave.research_audit import valid_quote_mask
        good=valid_quote_mask(f,'exchange_listed_contract') if len(f) else pd.Series(dtype=bool)
        row={'asset_id':asset,'exchange':spec.exchange,'role':spec.role,'record_count':len(f),
             'valid_outright_quote_count':int(good.sum()),'contract_count':int(f.contract.nunique()) if len(f) else 0,
             'first_observation_date':f.loc[good,'date'].min() if good.any() else None,
             'last_observation_date':f.loc[good,'date'].max() if good.any() else None,
             'has_continuous_series':spec.role!='bond_future','has_contract_observations':bool(good.any()),
             'contract_history_complete':bool(old_cov.loc[asset,'collection_status']=='SUCCESS'),
             'causal_continuous_returns_constructible':'BLOCKED_UNFROZEN_ROLL_AND_IDENTITY_PROTOCOL',
             'official_labels_available':False,'collection_status':'PARTIAL' if len(f) else 'BLOCKED',
             'source_ids':';'.join(sorted(f.source_id.unique())) if len(f) else None}
        contracts.append(row)
        if len(f):
            for (source,code),sub in f.groupby(['source_id','contract'],sort=False):
                ordinary=bool(pd.Series([code]).str.fullmatch(r'[A-Z]+[0-9]{3,4}').iloc[0])
                special='_TAS' in code
                prefix=str(pd.Series([code]).str.extract(r'^([A-Z]+)',expand=False).iloc[0])
                matched=prefix==spec.variety and sub.exchange.eq(spec.exchange).all()
                identity.append({'asset_id':asset,'exchange':spec.exchange,'contract_id':code,'source':source,
                    'trading_mode':'TAS_SPECIAL' if special else 'OUTRIGHT' if ordinary else 'NOT_VERIFIED',
                    'record_count':len(sub),'first_date':sub.date.min(),'last_date':sub.date.max(),
                    'delivery_month':code[-2:] if code[-2:].isdigit() else None,'delivery_year_status':'NOT_VERIFIED',
                    'identity_status':'SPECIAL_MODE_EXCLUDED_FROM_OUTRIGHT_QUOTES' if special else 'CODE_AND_EXCHANGE_MATCHED_CONTRACT_METADATA_PENDING' if matched and ordinary else 'MISMATCH_REQUIRES_REVIEW',
                    'manifest_references':';'.join(sorted(sub.manifest.unique()))})
    contract_table=pd.DataFrame(contracts);contract_table.to_csv(out/'futures_contract_coverage.csv',index=False)
    pd.DataFrame(identity).to_csv(out/'futures_contract_identity_audit.csv',index=False)
    for market in ('SHFE','DCE','CZCE','INE','GFEX','CFFEX'):
        f=contract_table.loc[contract_table.exchange.eq(market)]
        exchange.append({'exchange':market,'configured_varieties':len(f),'varieties_with_contract_observations':int(f.has_contract_observations.sum()),
            'baseline_rows':int(old_cov.loc[old_cov.exchange.eq(market),'record_count'].sum()),
            'rows_after_rescue':int(f.record_count.sum()),'history_complete':bool(f.contract_history_complete.all()),
            'official_source_status':'BLOCKED_HTTP_412' if market=='DCE' else 'PARTIAL_HTTP_567' if market=='GFEX' else 'READ_ONLY_REUSED',
            'tushare_status':'DATA_PROVIDER_AUTH_REQUIRED_NOT_CALLED' if market in ('DCE','GFEX') else 'NOT_NEEDED',
            'cross_provider_units_and_prices':'NOT_VERIFIED_NO_SECOND_AUTHORIZED_CONTRACT_SOURCE'})
    pd.DataFrame(exchange).to_csv(out/'exchange_provider_comparison.csv',index=False)
    contract_table.loc[contract_table.exchange.eq('DCE')].to_csv(out/'dce_collection_results.csv',index=False)
    contract_table.loc[contract_table.exchange.eq('GFEX')].to_csv(out/'gfex_collection_results.csv',index=False)
    coverage=[{'asset_id':r['asset_id'],'market':'stock','has_observations':r['valid_trade_quote_count']>0,'record_count':r['record_count'],'collection_status':r['collection_status']} for r in rows]
    bonds=pd.read_csv(out/'bond_coverage_reused.csv')
    coverage += [{'asset_id':r['asset_id'],'market':'bond','has_observations':r['valid_observation_count']>0,'record_count':r['record_count'],'collection_status':r['collection_status']} for r in bonds.to_dict('records')]
    coverage += [{'market':'bond_future' if r['role']=='bond_future' else 'commodity_future',**r} for r in contracts]
    pd.DataFrame(coverage).to_csv(out/'data_rescue_coverage.csv',index=False)
    return pd.DataFrame(rows),contract_table,pd.DataFrame(exchange)


def stock_quote_quality(frame,source):
    if frame.empty:
        return {'active_ohlc_invalid_rows':0,'active_required_numeric_missing_cells':0,'negative_activity_rows':0}
    active=frame.loc[frame.trading_status.eq('1')] if source.startswith('baostock') else frame
    cols=['open','high','low','close','volume','amount','turnover']
    values=active.reindex(columns=cols).apply(pd.to_numeric,errors='coerce')
    bad=(values.high<values[['open','low','close']].max(axis=1))|(values.low>values[['open','high','close']].min(axis=1))
    bad |= values[['open','high','low','close']].le(0).any(axis=1) | values[cols].isin([float('inf'),float('-inf')]).any(axis=1)
    negative=frame.reindex(columns=['volume','amount','turnover']).apply(pd.to_numeric,errors='coerce').lt(0).any(axis=1)
    return {'active_ohlc_invalid_rows':int(bad.sum()),'active_required_numeric_missing_cells':int(values.isna().sum().sum()),
            'negative_activity_rows':int(negative.sum())}


def report(root,run_id='rescue_20261015'):
    root=Path(root);out=root/'experiments/data_rescue';out.mkdir(parents=True,exist_ok=True)
    if not (out/'stock_incremental_results.json').exists():
        raise RuntimeError('Stock acquisition is still running or interrupted; finish/resume before final report')
    old,new,metadata=read_observations(root,run_id)
    stocks,contracts,exchanges=write_coverage(root,old,new,out,run_id)
    original=pd.read_csv(root/'experiments/data_collection/baseline15y/anomaly_report.csv')
    triage=anomaly_triage(original,old);triage.to_csv(out/'anomaly_triage.csv',index=False)
    assert len(triage)==74172 and triage.original_anomaly_id.nunique()==74172
    overlap=stock_overlap(old,new);overlap.to_csv(out/'source_overlap_audit.csv',index=False)
    wh_audit(root,old,out)
    all_rows=pd.concat([old,new],ignore_index=True)
    fields=['asset_id','series_type','date','exchange','contract','source_id','raw_file','manifest','retrieved_at','value',
            'close','settle','volume','open_interest','volume_unit','unit','price_unit','amount_unit','turnover']
    lineage=all_rows.reindex(columns=fields).copy()
    lineage['source']=lineage.source_id
    lineage['price']=lineage.close;lineage['settlement']=lineage.settle
    lineage['trading_mode']=lineage.series_type
    is_contract=lineage.series_type.eq('exchange_listed_contract')
    is_tas=lineage.contract.fillna('').astype(str).str.contains('_TAS')
    lineage.loc[is_contract,'trading_mode']='OUTRIGHT'
    lineage.loc[is_tas,'trading_mode']='TAS_SPECIAL'
    lineage['excluded_from_ordinary_contract_quotes']=is_tas
    lineage['source_priority_rank']=lineage.source_id.map(lambda s:20 if s in ('eastmoney_unadjusted_v1','tushare_fut_daily_contract_v1') else 10)
    lineage.loc[lineage.series_type.eq('vendor_continuous_unverified_roll'),'source_priority_rank']=None
    lineage['price_unit']=lineage.price_unit.fillna(lineage.unit).fillna('ORIGINAL_EXCHANGE_QUOTE_UNIT_NOT_VERIFIED')
    lineage['unit']=lineage.unit.fillna(lineage.price_unit)
    lineage['settlement_unit']=lineage.price_unit;lineage['open_interest_unit']='provider_reported_lots_or_NOT_APPLICABLE'
    lineage['unit_verification_status']='Source unit retained; cross-provider historical statistical basis NOT_VERIFIED'
    lineage['source_selection_status']='PRESERVED_SEPARATELY_PROVISIONAL_PRIORITY_NO_PRICE_FUSION'
    lineage['eligible_for_official_labels']=False
    lineage.to_csv(out/'source_lineage.csv.gz',index=False,compression='gzip')
    special=all_rows.loc[all_rows.contract.fillna('').astype(str).str.contains('_TAS')]
    special.reindex(columns=fields).to_csv(out/'special_trading_mode_records.csv.gz',index=False,compression='gzip')
    stats=[]
    for (asset,kind,source),f in all_rows.loc[all_rows.series_type.str.contains('contract|continuous',na=False)].groupby(['asset_id','series_type','source_id']):
        stats.append({'asset_id':asset,'series_type':kind,'source':source,'record_count':len(f),
            'zero_volume_rows':int(f.volume.eq(0).sum()),'missing_open_interest_rows':int(f.open_interest.isna().sum()),
            'zero_OHLC_rows':int(f[['open','high','low','close']].eq(0).any(axis=1).sum()),
            'settle_differs_from_close_rows':int((f.settle.notna()&f.close.notna()&f.settle.ne(f.close)).sum()),
            'settle_close_note':'Different economic fields; a nonzero difference alone is not invalid',
            'far_month_status':'NOT_VERIFIED_PENDING_DELIVERY_YEAR_AND_CONTRACT_METADATA'})
    pd.DataFrame(stats).to_csv(out/'supplemental_quality_statistics.csv',index=False)
    identities=[]
    for source,f in all_rows.groupby('source_id'):
        identities.append({'source_id':source,'record_count':len(f),'series_types':';'.join(sorted(f.series_type.unique())),
             'raw_files':f.raw_file.nunique(),'retrieved_first':f.retrieved_at.min(),'retrieved_last':f.retrieved_at.max(),
             'source_priority_rank':None if source=='sina_continuous_exploratory_v1' else 20 if source in ('eastmoney_unadjusted_v1','tushare_fut_daily_contract_v1') else 10,
             'license_status':'PROVIDER_TERMS_AND_REDISTRIBUTION_SCOPE_NOT_VERIFIED',
             'normalization':'No imputation or statistical fitting; native source fields retained'})
    pd.DataFrame(identities).to_csv(out/'source_identity.csv',index=False)
    pd.DataFrame(identities)[['source_id','source_priority_rank','series_types']].to_csv(out/'source_priority.csv',index=False)
    policy=root/'data/manifests'/run_id/'source_policy.json'
    shutil.copyfile(policy,out/'source_priority.json')
    errors=[]
    for filename in ('rescue_events.jsonl','http_events.jsonl'):
        p=root/'data/manifests'/run_id/filename
        for line in p.read_text().splitlines() if p.exists() else []:
            row=json.loads(line)
            if row.get('status') in ('BLOCKED','FAILED','DATA_PROVIDER_AUTH_REQUIRED') or row.get('error') or row.get('status_code',200)>=400:
                errors.append({**row,'evidence_file':str(p.relative_to(root))})
    for p in (root/'data/manifests'/run_id).rglob('*.json'):
        m=json.loads(p.read_text())
        if m.get('status')=='SCHEMA_FAILED': errors.append({**m,'evidence_file':str(p.relative_to(root))})
    pd.DataFrame(errors).to_csv(out/'failed_requests.csv',index=False)
    preserved=json.loads((out/'baseline_preservation.json').read_text())
    mismatches=[name for name,e in preserved['files'].items() if not (root/name).exists() or sha256_file(root/name)!=e['sha256']]
    atomic_json({'at':utc_now(),'baseline_files_checked':len(preserved['files']),'mismatches':mismatches,
                 'baseline_standardized_rows':len(old)},out/'baseline_preservation_verified.json')
    counts={key:int(triage.review_status.eq(key).sum()) for key in ('CONFIRMED_INVALID','LIKELY_INACTIVE_CONTRACT','POSSIBLE_SOURCE_SCHEMA_ISSUE','REQUIRES_REVIEW')}
    recovered=int(stocks.valid_trade_quote_count.gt(0).sum())
    gfex_rows=int(contracts.loc[contracts.exchange.eq('GFEX'),'record_count'].sum())
    flags={'STOCK_263_RECOVERED':'CONFIRMED' if recovered==263 else 'PARTIAL' if recovered else 'BLOCKED',
           'BOND_25_PRESERVED':'CONFIRMED' if not mismatches else 'BLOCKED',
           'DCE_CONTRACT_DATA_RECOVERED':'BLOCKED','GFEX_CONTRACT_DATA_RECOVERED':'PARTIAL' if gfex_rows else 'BLOCKED',
           'FUTURES_SOURCE_LINEAGE_PASS':'CONFIRMED' if lineage.raw_file.notna().all() and lineage.manifest.notna().all() else 'PARTIAL',
           'FUTURES_ANOMALY_TRIAGE_PASS':'CONFIRMED','WH_TARGET_MAPPING_VERIFIED':'NOT_VERIFIED',
           'DATA_CAUSALITY_PASS':'NOT_VERIFIED','READY_FOR_CAUSAL_PREPROCESSING':'BLOCKED'}
    results={'at':utc_now(),'run_id':run_id,'flags':flags,'baseline_rows_preserved':len(old),
             'new_stock_observation_rows':int(new.series_type.eq('stock').sum()),'stocks_with_valid_quotes':recovered,
             'new_gfex_rows':gfex_rows,'new_gfex_dates':int(new.loc[new.series_type.eq('exchange_listed_contract'),'date'].nunique()),
             'anomaly_triage_counts':counts,'anomaly_original_events':len(original),'anomaly_triage_events':len(triage),
             'anomaly_exact_observation_links':int(triage.row_resolution_status.eq('EXACT_OBSERVATION_LINKED').sum()),
             'anomaly_aggregate_events':int(triage.raw_file.isna().sum()),'baseline_integrity_mismatches':mismatches,
             'continuous_series_available':38,'contract_varieties_with_observations':int(contracts.has_contract_observations.sum()),
             'dce_failure':'Official same-endpoint single retest HTTP 412; no authorized alternate provider token',
             'gfex_failure':'Single-day worked, monthly smoke failed with HTTP 567 on 2023-10-24; partial success bodies replayed without new requests',
             'tushare_live_status':'DATA_PROVIDER_AUTH_REQUIRED_NOT_CALLED; user confirmed no authorization this round',
             'lineage_pass_meaning':'Every saved source row has raw file, source ID, manifest, units/status and timestamp; does not establish cross-provider economic/PIT compatibility',
             'triage_pass_meaning':'All original events classified with evidence or explicit aggregate scope; not a claim that review/repairs are finished',
             'no_model_training':True,'no_official_labels':True,'no_price_filling':True,'no_automatic_asset_or_target_changes':True}
    atomic_json(results,out/'audit_results.json')
    provenance={'at':utc_now(),'baseline_inventory':'baseline_preservation.json','baseline_verification':'baseline_preservation_verified.json',
                'baseline_inventory_sha256':sha256_file(out/'baseline_preservation.json'),
                'excluded_source_versions':json.loads((root/'data/manifests'/run_id/'excluded_source_versions.json').read_text()),
                'source_priority_policy':json.loads(policy.read_text()),'jobs':metadata,'environment':environment(root),
                'source_sha256':{str(p.relative_to(root)):sha256_file(p) for p in (root/'src/triweave').glob('*.py')},
                'config_sha256':{str(p.relative_to(root)):sha256_file(p) for p in (root/'configs').glob('*') if p.is_file()},
                'docs':{'BaoStock':'https://www.baostock.com/','user_reference':'https://github.com/zxygithub/baostock',
                        'Tushare_daily':'https://tushare.pro/document/2?doc_id=138','Tushare_contracts':'https://tushare.pro/document/2?doc_id=135'},
                'credentials':'No token values recorded; no Tushare requests made without authorization',
                'network_evidence':'probe_results.json and data/manifests/'+run_id+'/http_events.jsonl',
                'frozen_research_windows':{'WARMUP':['2011-10-01','2016-12-31'],'TRAIN':['2017-01-01','2022-12-31'],
                                         'VAL':['2023-01-01','2024-12-31'],'SEALED':['2025-01-01','2026-09-30']}}
    atomic_json(provenance,out/'provenance.json');atomic_json(environment(root),out/'environment.json')
    def table(f,cols):
        return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join('---' for c in cols)+' |']+
                         ['| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in f.to_dict('records')])
    text=['# Phase D-Rescue：股票与期货数据源恢复审计','',f'生成时间 {utc_now()}；窗口 2011-10-01—2026-09-30。',
      '',f'原 {len(old):,} 行、{len(preserved["files"]):,} 个历史文件逐一复核，发现 {len(mismatches)} 个指纹差异。旧成功产物没有覆盖；债券25条、SHFE/CZCE/INE/CFFEX只读复用。',
      '',table(pd.DataFrame([{'field':k,'status':v} for k,v in flags.items()]),['field','status']),
      '',f'股票当前 {recovered}/263 具有有效交易报价；新BaoStock原生记录 {results["new_stock_observation_rows"]:,} 行，停牌行保留但不计为可交易报价。真实起止日期、有效数量、停牌和上市证据见 stock_collection_results.csv。',
      '263只名单是原Clean V1固定研究cohort，本轮未重新选股；它不是2011年逐时点可投资成分股清单，原历史选择偏差限制仍然保留。',
      'BaoStock 3只2023-10月测试及600000全区间测试均真实通过，adjustflag=3。volume为股、amount为元、turn为百分比；旧Eastmoney volume为手，重叠审核显式乘100比较，原文件不转换覆盖。',
      'BaoStock使用公开SDK默认匿名TCP服务；未改HTTP代理、服务器地址、认证或路由。原Eastmoney仍连接中断。DNS均解析、既定代理TCP连接及一般HTTPS正常；不能据此唯一归因于代理故障或上游访问限制，具体原因保持NOT_VERIFIED。',
      '',table(exchanges,['exchange','baseline_rows','rows_after_rescue','varieties_with_contract_observations','official_source_status']),
      'DCE只做一次同接口2023-10-09复测，HTTP412，未再请求。Tushare无Token且用户确认本轮无授权，fut_basic/fut_daily没有伪调用；适配器仅完成离线测试，真实授权与GFEX支持情况仍须上线验证。未找到额外获授权的本地DCE/GFEX逐合约文件（检索范围为myresearch工程目录）。',
      f'GFEX单日接口恢复，完整月测试在10月24日返回567，未启动全历史下载。成功的11个日期、{gfex_rows}条SI/LC记录从已归档JSON的param.trade_date回显恢复，不重新请求、不猜日期。它们不能证明46个月完整覆盖。',
      'GFEX原始JSON同时含中文variety和varietyOrder，初版发生同名列冲突；v2又发现22条小计行未排除。两版raw/standardized/manifest完整保留并隔离，新v3仅接纳真实交割月份，补充小计/中文名回归测试，未覆盖旧版本。',
      '', '## 来源与融合边界','',
      'source_priority.json在重叠价格分析之前冻结来源顺序；source_identity.csv、source_lineage.csv.gz记录每条来源记录的raw文件、manifest、下载时间、价格/结算/成交量/持仓量及单位状态。不同来源不合并覆盖；source_overlap_audit.csv保留原值、转换规则与差异，没有通过VAL/SEALED收益决定来源。',
      'source_overlap_audit.csv当前能比较的独立供应商重叠是Eastmoney与BaoStock的600000；期货没有第二个获授权逐合约供应商，跨供应商价格/持仓统计口径保持NOT_VERIFIED。Sina连续行情不会用来补逐合约缺口。',
      '价格接近不证明经济身份、发布时点、单位或复权可比。旧交易所统计口径仍需按年份核验；Tushare文档注明2020年前双边、之后单边，适配器保留原值并注记，不擅自除以2。',
      '', '## 74,172条既有异常分诊','',table(pd.DataFrame([{'review_status':k,'count':v} for k,v in counts.items()]),['review_status','count']),
      'CONFIRMED_INVALID仅指作为普通报价在数学/数值上无效，不指认上游造假。零成交和零/缺失OHLC主要归LIKELY_INACTIVE_CONTRACT；跳变、常量、换月和特殊方式不能凭价格大小定性。',
      f'其中 {results["anomaly_exact_observation_links"]:,} 条定位到确切观测；{results["anomaly_aggregate_events"]} 条原始长区间/特殊方式聚合标记保留范围语义，不捏造单一日期。evidence_interval与manifest列提供支持窗口。SC_TAS共3996条另外逐条展开于special_trading_mode_records.csv.gz，不能用于普通SC报价。',
      '缺失持仓、零OHLC、结算/收盘差异另见supplemental_quality_statistics.csv；结算与收盘本来不同，差异本身不判错。非活跃远月仍需合约年份/挂牌信息，不能事后选择主力。',
      '', '## WH与剩余阻断','',
      '旧源码及旧CSV确认“小麦”使用WH0第三方连续；当前官方WH首条观测2012-07-24，旧WS与新WH存在制度/代码迁移，完整经济口径尚未核实。2023有12条正成交量合约日记录，2024没有；2025官方返回仍有孤立正成交记录。不能把“主连停止更新”写成“交易所品种消失”。',
      'WH在2023—2024标记NOT_EVALUABLE，24个目标及顺序不变。A有效标签mask与B特殊目标方案仅作为以后协议候选，本轮不选定、不建标签。详见wh_mapping_audit.md。',
      '', '## 环境、修复与可重现性','',
      'Python3.13.13，AKShare1.19.1，pandas2.3.3；新增可选rescue依赖BaoStock0.9.4、Tushare1.4.29，父环境未改动。git status已执行；当前项目不是Git仓库，原始输出保留git_status_before.txt，使用源码SHA256归档，不声称Git工作区干净。',
      '新增独立BaoStockStockProvider、TushareFuturesContractProvider和RescueStore，没有替换旧采集器。有限重试、串行限速、失败保留、同名拒绝覆盖、scope与SHA256校验均有回归测试；tests.txt记录最终实际结果，不能将测试通过等同于行情有效。',
      '', '## 下一步与安全续跑','',
      '优先：1）获取DCE官方授权导出或具有所需权限的Tushare；2）补齐GFEX缺失历史及合约身份、上市/交割信息；3）核实WH/旧代码与交易所统计口径；4）处理待复核源异常、股票公司行为/PIT及最长20步标签边界后，再评估因果预处理。',
      '公开数据接口不可用并不证明必须付费。可以先走官方合法导出；Tushare期货接口官方文档当前要求至少2000积分，是否购买取决于可取得的授权，Token不应写入聊天、日志或版本库。',
      '已有股票来源可断点续跑：`python -m triweave.rescue collect-stocks --run-id '+run_id+'`；然后`python -m triweave.rescue report --run-id '+run_id+'`。已完成请求按manifest复用。重新探测来源须建立新版本并保留本轮失败证据；GFEX月度gate失败时collect-gfex会拒绝大批量请求。',
      '本轮仅采集与固定规则审核，未训练、拟合标准化、生成正式收益标签、评价模型或访问SEALED预测指标；未修改目标、原CMGM、旧缓存或旧manifest，未提交/推送。',
      '', '来源文档：[BaoStock官网](https://www.baostock.com/)、[Tushare日线](https://tushare.pro/document/2?doc_id=138)、[Tushare合约身份](https://tushare.pro/document/2?doc_id=135)。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results
