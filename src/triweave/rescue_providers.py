"""Independent rescue providers. No source substitution, price filling or labels."""
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import time

import pandas as pd

from triweave.collect import atomic_csv, atomic_json, utc_now
from triweave.normalize import SchemaError, parse_date_series, verify_unique, finite
from triweave.pipeline import call_deadline
from triweave.universe import normalize_stock_code, sha256_file

BAO_FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,adjustflag'
FUT_FIELDS = 'ts_code,trade_date,open,high,low,close,settle,pre_settle,vol,oi,amount'
TOKEN_KEYS = ('TUSHARE_TOKEN','TUSHARE_PRO_TOKEN','TS_TOKEN')


class ProviderBlocked(RuntimeError):
    pass


class TruncationError(SchemaError):
    pass


def safe_error(error):
    message = str(error)
    for key in TOKEN_KEYS:
        value = os.environ.get(key)
        if value:
            message = message.replace(value,'REDACTED')
    return message


def validate_dates(frame, column, start, end):
    dates = parse_date_series(frame[column])
    if dates.isna().any() or not dates.between(pd.Timestamp(start),pd.Timestamp(end)).all():
        raise SchemaError('Invalid or out-of-request source observation dates')
    return dates.dt.strftime('%Y-%m-%d')


class BaoStockStockProvider:
    source_id = 'baostock_unadjusted_v1'

    def __init__(self, sdk=None, pause=1):
        if sdk is None:
            import baostock as sdk
        self.sdk,self.pause,self.connected = sdk,pause,False

    def login(self):
        # Documented native SDK transport; no alternate proxy, host or authentication.
        with call_deadline(40):
            previous=socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(15)
                result=self.sdk.login()
            finally:
                socket.setdefaulttimeout(previous)
        if str(result.error_code)!='0':
            raise ProviderBlocked('BaoStock login '+str(result.error_code)+': '+str(result.error_msg))
        self.connected=True

    def logout(self):
        if self.connected:
            with call_deadline(15): self.sdk.logout()
            self.connected=False

    def fetch(self, code, start, end):
        if not self.connected: raise ProviderBlocked('BaoStock login required')
        code=normalize_stock_code(code)
        time.sleep(self.pause)
        with call_deadline(90):
            result=self.sdk.query_history_k_data_plus(code,BAO_FIELDS,start_date=start,end_date=end,
                                                      frequency='d',adjustflag='3')
            if str(result.error_code)!='0':
                raise ProviderBlocked('BaoStock query '+str(result.error_code)+': '+str(result.error_msg))
            rows=[]
            while result.next(): rows.append(result.get_row_data())
            if str(result.error_code)!='0':
                raise ProviderBlocked('BaoStock incomplete result '+str(result.error_code))
        frame=pd.DataFrame(rows,columns=result.fields)
        if frame.empty: raise SchemaError('BaoStock returned empty data; not collection success')
        return frame

    @staticmethod
    def normalize(raw, code, start, end):
        required=set(BAO_FIELDS.split(','))
        if required-set(raw): raise SchemaError('BaoStock missing fields: '+str(sorted(required-set(raw))))
        if not raw['code'].eq(code).all(): raise SchemaError('BaoStock stock identity mismatch')
        if not raw['adjustflag'].astype(str).eq('3').all(): raise SchemaError('Unadjusted BaoStock flag required')
        out=pd.DataFrame({'date':validate_dates(raw,'date',start,end),'asset_id':code})
        for col in ('open','high','low','close','preclose','volume','amount','turn','pctChg'):
            out[col]=pd.to_numeric(raw[col],errors='coerce')
        out['turnover']=out['turn'];out['trading_status']=raw['tradestatus'].astype(str)
        out['isST']=raw['isST'].astype(str);out['adjustment']='unadjusted'
        out['source_id']=BaoStockStockProvider.source_id
        out['price_unit']='RMB_per_share';out['volume_unit']='shares';out['amount_unit']='RMB';out['turnover_unit']='percent'
        out['valid_trade_quote']=finite(out['close']) & out['close'].gt(0) & out['trading_status'].eq('1')
        out['suspension_rule']='Provider rows preserved; no synthetic prices; status=0 excluded from tradable quotes'
        verify_unique(out,['asset_id','date'])
        return out.sort_values('date').reset_index(drop=True)


class TushareFuturesContractProvider:
    source_id='tushare_fut_daily_contract_v1'

    def __init__(self, client=None, pause=1, cap=2000):
        if client is None:
            token=next((os.environ[k] for k in TOKEN_KEYS if os.environ.get(k)),None)
            if not token: raise ProviderBlocked('DATA_PROVIDER_AUTH_REQUIRED: Tushare token absent')
            from tushare.pro.client import DataApi
            client=DataApi(token,timeout=20)
        self.client,self.pause,self.cap=client,pause,cap

    def _query(self, api, **params):
        time.sleep(self.pause)
        # Do not archive request tokens or credential-bearing exception messages.
        try:
            with call_deadline(40): return self.client.query(api,**params)
        except Exception as exc:
            raise ProviderBlocked(safe_error(exc)) from None

    def basic(self, exchange, variety):
        f=self._query('fut_basic',exchange=exchange,fut_type='1',fut_code=variety,
                      fields='ts_code,symbol,exchange,name,fut_code,trade_unit,per_unit,quote_unit,list_date,delist_date,d_month')
        if len(f)>=10000: raise TruncationError('fut_basic reached documented 10000-row cap')
        if f.empty: raise SchemaError('Empty fut_basic: identities not verified')
        if not f['exchange'].eq(exchange).all(): raise SchemaError('fut_basic exchange mismatch')
        if not f['fut_code'].astype(str).str.upper().eq(variety.upper()).all():
            raise SchemaError('fut_basic returned a different configured variety')
        if not f['symbol'].str.fullmatch(r'[A-Za-z]+[0-9]{3,4}').all():
            raise SchemaError('Non-listed/continuous code in fut_type=1 identity table')
        verify_unique(f,['ts_code'])
        return f

    def daily(self, exchange, date, basic):
        raw=self._query('fut_daily',exchange=exchange,trade_date=date,fields=FUT_FIELDS)
        if len(raw)>=self.cap:
            # A cap hit is never assumed complete. Resolve to one real contract/date.
            eligible=basic.loc[(basic.list_date<=date)&(basic.delist_date.isna()|basic.delist_date.eq('')|(basic.delist_date>=date))]
            if eligible.empty: raise TruncationError('No verified contract identities for capped daily response')
            chunks=[]
            for code in eligible.ts_code:
                part=self._query('fut_daily',ts_code=code,trade_date=date,fields=FUT_FIELDS)
                if len(part)>1: raise TruncationError('More than one row for a single contract/date')
                chunks.append(part)
            return pd.concat(chunks,ignore_index=True)
        return raw

    @staticmethod
    def normalize(raw, exchange, basic, start, end):
        if set(FUT_FIELDS.split(','))-set(raw): raise SchemaError('fut_daily required fields absent')
        if raw.empty: raise SchemaError('Empty fut_daily is not success')
        # Exchange-wide requests also return varieties outside the fixed universe.
        # Keep their original response in raw; normalize only configured varieties.
        # Unknown/continuous contracts within a configured variety must still fail.
        configured=basic.fut_code.astype(str).str.upper().unique()
        prefixes=raw.ts_code.astype(str).str.extract(r'^([A-Za-z]+)',expand=False).str.upper()
        selected=raw.loc[prefixes.isin(configured)].copy()
        if selected.empty: raise SchemaError('No configured ordinary contracts in fut_daily')
        if not selected.ts_code.isin(basic.ts_code).all(): raise SchemaError('Daily rows lack verified ordinary contract identity')
        out=selected.merge(basic,on='ts_code',how='left',validate='many_to_one')
        out['date']=validate_dates(out,'trade_date',start,end)
        if not out.exchange.eq(exchange).all(): raise SchemaError('Daily/basic exchange mismatch')
        dates=out.trade_date.astype(str)
        if (dates<out.list_date).any() or ((out.delist_date.notna()&out.delist_date.ne(''))&(dates>out.delist_date)).any():
            raise SchemaError('Quote outside documented contract listing/delisting dates')
        for col in ('open','high','low','close','settle','pre_settle','vol','oi','amount'):
            out[col]=pd.to_numeric(out[col],errors='coerce')
        out['contract']=out.symbol.str.upper();out['asset_id']='FUT_'+out.fut_code.str.upper()
        out['volume']=out.vol;out['open_interest']=out.oi
        out['volume_unit']='lots';out['amount_unit']='10000_RMB';out['open_interest_unit']='lots'
        out['price_unit']=out['quote_unit']
        out['statistical_basis']=out['date'].map(lambda d:'two_sided_PER_PROVIDER_DOC' if d<'2020-01-01' else 'single_sided_PER_PROVIDER_DOC')
        out['source_id']=TushareFuturesContractProvider.source_id
        out['series_type']='exchange_listed_contract';out['tradable_return_verified']=False
        verify_unique(out,['exchange','contract','date'])
        return out.sort_values(['contract','date']).reset_index(drop=True)


class RescueStore:
    """Source-specific immutable files, including failed-schema response evidence."""
    def __init__(self, root, run_id='rescue_20261015'):
        self.root,self.run_id=Path(root),run_id
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id): raise ValueError('Unsafe run ID')

    def paths(self, source, category, name):
        if not all(re.fullmatch(r'[A-Za-z0-9_-]+',s) for s in (source,category,name)):
            raise ValueError('Unsafe source path')
        stem=Path(self.run_id)/source/category/name
        return tuple(self.root/'data'/folder/str(stem).replace('\\','/') for folder in ('raw','standardized','manifests'))

    def files(self, source, category, name):
        raw,std,meta=self.paths(source,category,name)
        return Path(str(raw)+'.csv.gz'),Path(str(std)+'.csv.gz'),Path(str(meta)+'.json')

    def cached(self, source, category, name, params):
        raw,std,meta=self.files(source,category,name)
        if not meta.exists():
            if raw.exists() or std.exists(): raise FileExistsError('Untracked file preserved; use a new version')
            return None
        m=json.loads(meta.read_text())
        if m['params']!=params: raise ValueError('Request scope changed; new version required')
        for path,key in ((raw,'raw_sha256'),(std,'standardized_sha256')):
            if m.get(key) and (not path.exists() or sha256_file(path)!=m[key]): raise ValueError('Source archive integrity failure')
        return m

    def save(self, source, category, name, params, raw, normalize):
        if self.cached(source,category,name,params): raise FileExistsError('Existing source version cannot be overwritten')
        r,s,m=self.files(source,category,name)
        atomic_csv(raw,r)
        doc={'source_id':source,'category':category,'name':name,'params':params,'retrieved_at':utc_now(),
             'raw_file':str(r.relative_to(self.root)),'raw_sha256':sha256_file(r),'raw_rows':len(raw),
             'raw_columns':list(raw),'adapter_sha256':sha256_file(Path(__file__)),
             'packages':{p:importlib.metadata.version(p) for p in ('baostock','tushare','pandas')}}
        try:
            clean=normalize(raw)
            if clean.empty: raise SchemaError('No standardized observations')
            atomic_csv(clean,s)
            doc.update(status='OBSERVATIONS_SAVED_NOT_FULL_HISTORY_VERIFIED',standardized_file=str(s.relative_to(self.root)),
                       standardized_sha256=sha256_file(s),standardized_rows=len(clean))
        except Exception as exc:
            doc.update(status='SCHEMA_FAILED',error=safe_error(exc),standardized_sha256=None)
            atomic_json(doc,m)
            raise
        atomic_json(doc,m)
        return doc
