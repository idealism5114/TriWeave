import json
from types import SimpleNamespace

import pandas as pd
import pytest

from triweave.rescue_providers import (BaoStockStockProvider, TushareFuturesContractProvider,
    RescueStore, ProviderBlocked, SchemaError, TruncationError, TOKEN_KEYS, safe_error)


def stock_raw():
    return pd.DataFrame({'date':['2023-10-09','2023-10-10'],'code':['sh.600000']*2,
        'open':['10','10'],'high':['11','10'],'low':['9','10'],'close':['10','10'],
        'preclose':['9','10'],'volume':['100','0'],'amount':['1000','0'],'turn':['1','0'],
        'tradestatus':['1','0'],'pctChg':['1','0'],'isST':['0','0'],'adjustflag':['3','3']})


def test_baostock_preserves_suspension_and_native_units():
    raw=stock_raw();before=raw.copy(deep=True)
    f=BaoStockStockProvider.normalize(raw,'sh.600000','2023-10-01','2023-10-31')
    assert f.volume.tolist()==[100,0] and f.volume_unit.eq('shares').all()
    assert f.valid_trade_quote.tolist()==[True,False]
    assert f.close.tolist()==[10,10]  # Original provider suspension value, no replacement.
    assert f.source_id.eq('baostock_unadjusted_v1').all()
    pd.testing.assert_frame_equal(raw,before)


@pytest.mark.parametrize('column,value',[('code','sh.600038'),('adjustflag','2'),('date','2025-01-01')])
def test_baostock_rejects_identity_adjustment_or_date_drift(column,value):
    raw=stock_raw();raw.loc[0,column]=value
    with pytest.raises(SchemaError): BaoStockStockProvider.normalize(raw,'sh.600000','2023-10-01','2023-10-31')


def test_baostock_failed_login_cannot_query():
    sdk=SimpleNamespace(login=lambda:SimpleNamespace(error_code='10002003',error_msg='network unavailable'))
    p=BaoStockStockProvider(sdk,pause=0)
    with pytest.raises(ProviderBlocked): p.login()
    with pytest.raises(ProviderBlocked): p.fetch('sh.600000','2023-10-01','2023-10-31')


def test_tushare_missing_token_is_not_a_success(monkeypatch):
    for key in TOKEN_KEYS: monkeypatch.delenv(key,raising=False)
    with pytest.raises(ProviderBlocked,match='DATA_PROVIDER_AUTH_REQUIRED'): TushareFuturesContractProvider()


def test_credential_values_never_appear_in_logged_errors(monkeypatch):
    monkeypatch.setenv('TUSHARE_TOKEN','private-fixture-token')
    assert 'private-fixture-token' not in safe_error('server echoed private-fixture-token')


def basic():
    return pd.DataFrame({'ts_code':['M2401.DCE','M2405.DCE','M2409.DCE'],
        'symbol':['M2401','M2405','M2409'],'exchange':['DCE']*3,'fut_code':['M']*3,
        'list_date':['20230101']*3,'delist_date':['20241231']*3,'quote_unit':['RMB/tonne']*3})


def daily():
    return pd.DataFrame({'ts_code':['M2401.DCE'],'trade_date':['20231009'],
        'open':[100.],'high':[101.],'low':[99.],'close':[100.],'settle':[100.],
        'pre_settle':[99.],'vol':[10.],'oi':[20.],'amount':[30.]})


def test_tushare_cap_is_resolved_by_verified_contract_not_silently_truncated():
    calls=[]
    class Client:
        def query(self,api,**kwargs):
            calls.append(kwargs)
            if 'exchange' in kwargs: return pd.concat([daily(),daily()],ignore_index=True)
            f=daily();f['ts_code']=kwargs['ts_code'];return f
    p=TushareFuturesContractProvider(Client(),pause=0,cap=2)
    f=p.daily('DCE','20231009',basic())
    assert len(f)==3 and len(calls)==4
    assert f.ts_code.nunique()==3


def test_tushare_identity_check_and_original_units():
    f=TushareFuturesContractProvider.normalize(daily(),'DCE',basic(),'2023-10-01','2023-10-31')
    assert f.volume.iloc[0]==10 and f.amount.iloc[0]==30
    assert f.amount_unit.iloc[0]=='10000_RMB'
    assert f.price_unit.iloc[0]=='RMB/tonne'
    raw=daily();raw['ts_code']='M9999.DCE'
    with pytest.raises(SchemaError): TushareFuturesContractProvider.normalize(raw,'DCE',basic(),'2023-10-01','2023-10-31')


def test_tushare_exchange_response_preserved_but_unconfigured_varieties_not_normalized():
    extra=daily();extra['ts_code']='B2401.DCE'
    raw=pd.concat([daily(),extra],ignore_index=True);before=raw.copy(deep=True)
    normalized=TushareFuturesContractProvider.normalize(raw,'DCE',basic(),'2023-10-01','2023-10-31')
    assert normalized.ts_code.tolist()==['M2401.DCE']
    pd.testing.assert_frame_equal(raw,before)


def test_tushare_basic_rejects_ignored_variety_filter():
    class Client:
        def query(self,api,**kwargs): return basic()
    provider=TushareFuturesContractProvider(Client(),pause=0)
    with pytest.raises(SchemaError,match='different configured variety'):
        provider.basic('DCE','C')


def test_immutable_source_store_refuses_overwrite_and_detects_modified_file(tmp_path):
    store=RescueStore(tmp_path,'test');params={'code':'sh.600000'}
    normalize=lambda r:BaoStockStockProvider.normalize(r,'sh.600000','2023-10-01','2023-10-31')
    store.save('bao','stocks','sample',params,stock_raw(),normalize)
    assert store.cached('bao','stocks','sample',params)['raw_rows']==2
    with pytest.raises(FileExistsError): store.save('bao','stocks','sample',params,stock_raw(),normalize)
    with pytest.raises(ValueError): store.cached('bao','stocks','sample',{'code':'other'})
    raw,std,meta=store.files('bao','stocks','sample');raw.write_bytes(b'modified fixture')
    with pytest.raises(ValueError,match='integrity'): store.cached('bao','stocks','sample',params)


def test_schema_failure_preserves_original_response(tmp_path):
    store=RescueStore(tmp_path,'test')
    with pytest.raises(SchemaError):
        store.save('bao','stocks','bad',{},stock_raw().drop(columns='code'),
            lambda r:BaoStockStockProvider.normalize(r,'sh.600000','2023-10-01','2023-10-31'))
    raw,std,meta=store.files('bao','stocks','bad')
    assert raw.exists() and not std.exists()
    assert json.loads(meta.read_text())['status']=='SCHEMA_FAILED'
