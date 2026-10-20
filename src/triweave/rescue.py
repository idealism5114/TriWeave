"""Phase D-Rescue: explicit new sources, bounded live gates, immutable baseline."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit

import pandas as pd

from triweave.collect import atomic_json, append_log, utc_now
from triweave.pipeline import call_deadline, environment, frame_schema
from triweave.collect import Job
from triweave.rescue_providers import (BaoStockStockProvider, TushareFuturesContractProvider,
    RescueStore, ProviderBlocked, BAO_FIELDS, TOKEN_KEYS, safe_error)
from triweave.transport import observed_http
from triweave.universe import load_stock_universe

RUN='rescue_20261015'


def stock_params(code,start,end):
    return dict(code=code,start_date=start,end_date=end,frequency='d',adjustflag='3',fields=BAO_FIELDS)


def stock_smoke_schema(frame,start,end):
    job=Job('stocks','rescue','query_history_k_data_plus',{},start,end)
    result=frame_schema(frame,job)
    active=frame.loc[frame.trading_status.eq('1')]
    invalid=((active.high<active[['open','low','close']].max(axis=1))|
             (active.low>active[['open','high','close']].min(axis=1)))
    result.update(active_rows=len(active),suspended_rows=int(frame.trading_status.eq('0').sum()),
                  invalid_ohlc_rows=int(invalid.sum()),valid_trade_quotes=int(frame.valid_trade_quote.sum()))
    result['pass']=result['pass'] and not invalid.any() and frame.valid_trade_quote.any()
    result['pass']=bool(result['pass'])
    return result


def probe(root,run_id=RUN):
    import akshare as ak
    import baostock.common.contants as bc
    out=root/'experiments/data_rescue';out.mkdir(parents=True,exist_ok=True)
    destination=out/'probe_results.json'
    if destination.exists(): raise FileExistsError('Probe already recorded; choose a new output version before a new live test')
    store=RescueStore(root,run_id)
    cfg=json.loads((root/'configs/project.json').read_text())
    diagnostics={'at':utc_now(),'proxy':{},'dns':{},'tcp':{},'http':{},'routing_changes':False}
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy'):
        value=os.environ.get(key)
        if value:
            u=urlsplit(value)
            diagnostics['proxy'][key]={'set':True,'host':u.hostname,'port':u.port,'credentials_present':bool(u.username or u.password)} if key!='NO_PROXY' else {'set':True}
    for host in ('push2his.eastmoney.com','www.dce.com.cn','www.gfex.com.cn',bc.BAOSTOCK_SERVER_IP):
        try:
            with call_deadline(8): ips=sorted({r[4][0] for r in socket.getaddrinfo(host,None)})
            diagnostics['dns'][host]={'status':'RESOLVED','addresses':ips}
        except Exception as exc: diagnostics['dns'][host]={'status':'FAILED','error':safe_error(exc)}
    targets=[('baostock_documented_native_transport',bc.BAOSTOCK_SERVER_IP,bc.BAOSTOCK_SERVER_PORT)]
    proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    if proxy:
        u=urlsplit(proxy);targets.append(('configured_http_proxy',u.hostname,u.port or 80))
    for label,host,port in targets:
        try:
            with call_deadline(10), socket.create_connection((host,port),timeout=8): pass
            diagnostics['tcp'][label]={'status':'CONNECTED','host':host,'port':port}
        except Exception as exc: diagnostics['tcp'][label]={'status':'FAILED','host':host,'port':port,'error':safe_error(exc)}
    results=[]
    def event(source,name,status,**extra):
        row={'source_id':source,'name':name,'status':status,'at':utc_now(),**extra}
        results.append(row);append_log(root/'data/manifests'/run_id/'rescue_events.jsonl',row)
        print(source,name,status,flush=True)
    with observed_http(root,run_id,timeout=12,pause=1):
        import requests
        try:
            response=requests.get('https://pypi.org/simple/baostock/')
            diagnostics['http']['general_https']={'status_code':response.status_code,'nonempty':bool(response.content)}
        except Exception as exc: diagnostics['http']['general_https']={'error':safe_error(exc)}
        try:
            frame=ak.stock_zh_a_hist(symbol='600004',period='daily',start_date='20231001',end_date='20231031',adjust='')
            from triweave.normalize import normalize_stocks
            doc=store.save('eastmoney_unadjusted_v1','diagnostic','sh_600004_202310',
                {'symbol':'600004','start_date':'20231001','end_date':'20231031','adjust':''},frame,
                lambda raw:normalize_stocks(raw,'sh.600004','2023-10-01','2023-10-31'))
            event('eastmoney_unadjusted_v1','sh.600004','OBSERVATIONS',manifest=doc)
        except Exception as exc: event('eastmoney_unadjusted_v1','sh.600004','BLOCKED',error=safe_error(exc))
        # Exactly one fresh official API call for each failed exchange.
        for exchange,fn in (('DCE',ak.get_dce_daily),('GFEX',ak.get_gfex_daily)):
            try:
                frame=fn(date='20231009')
                from triweave.normalize import normalize_contracts
                doc=store.save(exchange.lower()+'_official_akshare_v1','smoke','20231009',{'date':'20231009'},frame,
                    lambda raw:normalize_contracts(raw,exchange,'2023-10-09','2023-10-09'))
                event(exchange.lower()+'_official_akshare_v1',exchange,'OBSERVATIONS',manifest=doc)
            except Exception as exc: event(exchange.lower()+'_official_akshare_v1',exchange,'BLOCKED',error=safe_error(exc))
    provider=BaoStockStockProvider()
    codes=['sh.600000','sh.600038','sz.002146']
    cases=[(code,'2023-10-01','2023-10-31','smoke') for code in codes]
    cases.append((codes[0],cfg['research_start'],cfg['research_end'],'stocks'))
    try:
        provider.login();login_error=None
    except Exception as exc: login_error=safe_error(exc)
    for code,start,end,category in cases:
        name=code.replace('.','_')+'_'+start.replace('-','')+'_'+end.replace('-','')
        if login_error:
            event(provider.source_id,name,'BLOCKED',error=login_error,query_attempted=False)
            continue
        try:
            raw=provider.fetch(code,start,end)
            doc=store.save(provider.source_id,category,name,stock_params(code,start,end),raw,
                           lambda raw:provider.normalize(raw,code,start,end))
            r,s,m=store.files(provider.source_id,category,name)
            schema=stock_smoke_schema(pd.read_csv(s,dtype={'trading_status':str}),start,end)
            event(provider.source_id,name,'CONFIRMED' if schema['pass'] else 'FAILED_SCHEMA',schema=schema,manifest=doc,query_attempted=True)
        except Exception as exc: event(provider.source_id,name,'BLOCKED',error=safe_error(exc),query_attempted=True)
    try: provider.logout()
    except Exception as exc: event(provider.source_id,'logout','FAILED',error=safe_error(exc))
    auth=any(os.environ.get(k) for k in TOKEN_KEYS)
    for exchange in ('DCE','GFEX'):
        event(TushareFuturesContractProvider.source_id,exchange,'NOT_VERIFIED' if auth else 'DATA_PROVIDER_AUTH_REQUIRED',credential_present=auth,api_attempted=False)
    bao=[r for r in results if r['source_id']==provider.source_id and r['name']!='logout']
    doc={'at':utc_now(),'run_id':run_id,'results':results,'network':diagnostics,
         'bao_bulk_gate':len(bao)==4 and all(r['status']=='CONFIRMED' for r in bao),
         'tushare_credential_present':auth,'tushare_gate':False}
    atomic_json(doc,destination);atomic_json(environment(root),out/'environment.json')
    return doc


def collect_stocks(root,run_id=RUN):
    gate=json.loads((root/'experiments/data_rescue/probe_results.json').read_text())
    if not gate['bao_bulk_gate']: raise ProviderBlocked('BaoStock live small/full smoke gate failed; no bulk requests')
    cfg=json.loads((root/'configs/project.json').read_text());start,end=cfg['research_start'],cfg['research_end']
    stocks=load_stock_universe(root/'data/manifests/stocks_263.csv');provider=BaoStockStockProvider();store=RescueStore(root,run_id)
    rows=[];failures=0
    provider.login()
    try:
        for code in stocks.asset_id:
            name=code.replace('.','_')+'_'+start.replace('-','')+'_'+end.replace('-','');params=stock_params(code,start,end)
            cached=store.cached(provider.source_id,'stocks',name,params)
            if cached:
                rows.append(cached);continue
            if failures>=3:
                rows.append({'asset_id':code,'status':'BLOCKED_AFTER_CONSECUTIVE_FAILURES'});continue
            for attempt in range(1,4):
                try:
                    raw=provider.fetch(code,start,end)
                    doc=store.save(provider.source_id,'stocks',name,params,raw,lambda r:provider.normalize(r,code,start,end))
                    rows.append({**doc,'asset_id':code});failures=0;break
                except Exception as exc:
                    append_log(root/'data/manifests'/run_id/'rescue_events.jsonl',{'at':utc_now(),'source_id':provider.source_id,'name':code,'attempt':attempt,'status':'FAILED','error':safe_error(exc)})
                    if isinstance(exc,ValueError) or attempt==3:
                        rows.append({'asset_id':code,'status':'FAILED','error':safe_error(exc)});failures+=1;break
                    time.sleep(2**attempt)
            print('STOCK',code,rows[-1]['status'],flush=True)
    finally: provider.logout()
    atomic_json({'rows':rows,'at':utc_now()},root/'experiments/data_rescue/stock_incremental_results.json')


def gfex_batch(root,run_id=RUN,smoke=False):
    import akshare as ak
    from triweave.collect import make_jobs
    from triweave.normalize import normalize_contracts
    source='gfex_official_akshare_v1';store=RescueStore(root,run_id)
    out=root/'experiments/data_rescue'
    if not smoke:
        gate=json.loads((out/'gfex_month_smoke.json').read_text())
        if not gate['pass']: raise ProviderBlocked('GFEX monthly live schema gate failed')
    cfg=json.loads((root/'configs/project.json').read_text())
    start,end=('2023-10-01','2023-10-31') if smoke else (cfg['research_start'],cfg['research_end'])
    jobs=make_jobs(['futures_contracts'],start,end,markets=['GFEX'])
    rows=[];consecutive=0
    with observed_http(root,run_id,timeout=15,pause=1):
        for job in jobs:
            category='smoke_month' if smoke else 'futures_contracts'
            cached=store.cached(source,category,job.name,job.kwargs)
            if cached:
                rows.append(cached);continue
            if consecutive>=3:
                rows.append({'name':job.name,'status':'BLOCKED_AFTER_CONSECUTIVE_FAILURES'});continue
            for attempt in range(1,4):
                try:
                    with call_deadline(180): raw=ak.get_futures_daily(**job.kwargs)
                    def normalize(raw):
                        f=normalize_contracts(raw,'GFEX',job.start,job.end)
                        if f.empty: raise ValueError('GFEX returned no selected real contracts')
                        f['source_id']=source;f['price_unit']='GFEX_quoted_unit_PENDING_OFFICIAL_SPEC_CHECK'
                        f['amount_unit']='GFEX_original_unit_PENDING_OFFICIAL_REPORT_CHECK'
                        return f
                    doc=store.save(source,category,job.name,job.kwargs,raw,normalize)
                    r,s,m=store.files(source,category,job.name)
                    check=frame_schema(pd.read_csv(s),job)
                    rows.append({**doc,'schema':check});consecutive=0;break
                except Exception as exc:
                    append_log(root/'data/manifests'/run_id/'rescue_events.jsonl',{'at':utc_now(),'source_id':source,'name':job.name,
                         'attempt':attempt,'status':'FAILED','error':safe_error(exc)})
                    if isinstance(exc,ValueError) or attempt==3 or any(s in str(exc) for s in ('412','403','567','429')):
                        rows.append({'name':job.name,'status':'FAILED','error':safe_error(exc)});consecutive+=1;break
                    time.sleep(2**attempt)
            print('GFEX',job.name,rows[-1]['status'],flush=True)
    if smoke:
        passed=len(rows)==1 and rows[0].get('schema',{}).get('pass',False)
        if passed:
            r,s,m=store.files(source,'smoke_month',jobs[0].name)
            f=pd.read_csv(s);passed=set(f.variety)=={'SI','LC'}
        atomic_json({'at':utc_now(),'pass':bool(passed),'rows':rows},out/'gfex_month_smoke.json')
    else: atomic_json({'at':utc_now(),'rows':rows},out/'gfex_incremental_results.json')


def gfex_archived_normalize(raw, echo):
    """Replay authentic archived success bodies using their echoed query date."""
    from triweave.normalize import normalize_contracts,SchemaError
    dates=echo.get('trade_date',[])
    if len(dates)!=1 or echo.get('trade_type')!=['0']:
        raise SchemaError('GFEX response must identify a single futures query date')
    date=dates[0]
    if len(date)!=8 or not date.isdigit(): raise SchemaError('Invalid GFEX echoed date')
    f=raw.copy()
    f=f.loc[f['varietyOrder'].astype(str).str.lower().isin(['si','lc'])].copy()
    # Product subtotals carry the same varietyOrder. A variety filter alone
    # would admit "小计" as a fictional delivery month.
    f=f.loc[f['delivMonth'].astype(str).str.fullmatch(r'[0-9]{3,4}')].copy()
    aliases={'volumn':'volume','openInterest':'open_interest',
             'clearPrice':'settle','lastClear':'pre_settle'}
    f=f.rename(columns=aliases)
    f['variety']=f['varietyOrder'].str.upper()
    f['symbol']=f.variety.str.upper()+f.delivMonth.astype(str)
    f['variety']=f.variety.str.upper();f['date']=date
    iso=pd.Timestamp(date).strftime('%Y-%m-%d')
    clean=normalize_contracts(f,'GFEX',iso,iso)
    clean['source_id']='gfex_official_akshare_v1'
    clean['price_unit']='GFEX_original_quote_unit_NOT_INDEPENDENTLY_VERIFIED'
    return clean


def recover_gfex_partial(root,run_id=RUN):
    """No network: retain successful days from the failed monthly smoke."""
    import gzip
    from triweave.universe import sha256_file
    store=RescueStore(root,run_id);rows=[];seen=set()
    events=root/'data/manifests'/run_id/'http_events.jsonl'
    for line in events.read_text().splitlines():
        event=json.loads(line)
        if 'gfex.com.cn/u/interfacesWebTiDayQuotes/loadList' not in event.get('url','') or event.get('status_code')!=200:
            continue
        archive=root/event['archive']
        if sha256_file(archive)!=event['archive_sha256']: raise ValueError('HTTP evidence checksum mismatch')
        body=json.loads(gzip.decompress(archive.read_bytes()))
        echo=body.get('param',{});dates=echo.get('trade_date',[])
        if len(dates)!=1 or dates[0] in seen: continue
        seen.add(dates[0]);raw=pd.DataFrame(body['data']);name=dates[0]
        params={'echoed_request':echo,'upstream_retrieved_at':event['at'],'response_archive':event['archive'],
                'response_archive_sha256':event['archive_sha256'],'raw_representation':'Original GFEX JSON data array; response envelope separately preserved'}
        params['normalizer_source_sha256']=sha256_file(Path(__file__))
        cached=store.cached('gfex_official_akshare_v1','futures_contracts_daily_partial_v3',name,params)
        doc=cached or store.save('gfex_official_akshare_v1','futures_contracts_daily_partial_v3',name,params,raw,
                                 lambda r:gfex_archived_normalize(r,echo))
        rows.append(doc)
    atomic_json({'at':utc_now(),'network_requests':0,'status':'PARTIAL','rows':rows},root/'experiments/data_rescue/gfex_partial_recovery_v3.json')


def stock_metadata(root,run_id=RUN):
    if not (root/'experiments/data_rescue/stock_incremental_results.json').exists():
        raise ProviderBlocked('Finish the sequential stock session before requesting metadata')
    provider=BaoStockStockProvider();store=RescueStore(root,run_id)
    params={'api':'query_stock_basic','code':'','scope':'Full provider directory; fixed 263 identities only in standardized metadata'}
    source='baostock_security_metadata_v1'
    if store.cached(source,'security_metadata','directory',params): return
    stocks=load_stock_universe(root/'data/manifests/stocks_263.csv')
    provider.login()
    try:
        with call_deadline(90):
            rs=provider.sdk.query_stock_basic()
            if rs.error_code!='0': raise ProviderBlocked('BaoStock basic error '+rs.error_code)
            rows=[]
            while rs.next(): rows.append(rs.get_row_data())
            if rs.error_code!='0': raise ProviderBlocked('BaoStock partial directory '+rs.error_code)
        raw=pd.DataFrame(rows,columns=rs.fields)
        if raw.empty: raise ValueError('Empty security directory')
        def normalize(raw):
            f=raw.loc[raw.code.isin(stocks.asset_id)].copy()
            f['asset_id']=f.code;f['source_id']=source
            f['metadata_time_status']='Current directory; historical naming/publication vintages NOT_VERIFIED'
            return f
        store.save(source,'security_metadata','directory',params,raw,normalize)
    finally: provider.logout()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['probe','collect-stocks','stock-metadata','smoke-gfex','collect-gfex','recover-gfex-partial','report'])
    parser.add_argument('--root',default='.')
    parser.add_argument('--run-id',default=RUN)
    args=parser.parse_args(argv);root=Path(args.root).resolve()
    if args.stage=='probe': probe(root,args.run_id)
    elif args.stage=='collect-stocks': collect_stocks(root,args.run_id)
    elif args.stage=='smoke-gfex': gfex_batch(root,args.run_id,smoke=True)
    elif args.stage=='collect-gfex': gfex_batch(root,args.run_id)
    elif args.stage=='recover-gfex-partial': recover_gfex_partial(root,args.run_id)
    elif args.stage=='stock-metadata': stock_metadata(root,args.run_id)
    else:
        from triweave.rescue_audit import report
        report(root,args.run_id)


if __name__=='__main__': main()
