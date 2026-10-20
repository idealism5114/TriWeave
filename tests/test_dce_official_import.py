import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from triweave.providers.dce_official_import import (IDENTITY_VERIFIED, import_file,
    load_identities, normalize_table, read_table, SchemaError)


@pytest.fixture
def files(tmp_path):
    evidence = tmp_path/'directory_evidence.txt'
    evidence.write_text('SYNTHETIC TEST DIRECTORY; NOT REAL MARKET DATA')
    ids = pd.DataFrame([{'exchange':'DCE','variety':'M','contract_code':'M2001',
        'delivery_year':'2020','delivery_month':'1','listing_date':'2019-01-02','last_trade_date':'2020-01-15',
        'verification_status':IDENTITY_VERIFIED,'identity_evidence_file':evidence.name,
        'identity_evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}])
    identity=tmp_path/'identity.csv';ids.to_csv(identity,index=False)
    source=tmp_path/'source.json'
    sidecar={'source_id':'dce_official_file_test_v1','exchange':'DCE','source_url':'https://www.dce.com.cn/',
        'acquired_at':'2026-10-15T21:00:00+08:00','permission_status':'AUTHORIZED_RESEARCH_USE',
        'data_origin':'DCE_OFFICIAL_EXPORT','price_unit':'RMB_per_tonne','volume_unit':'lots',
        'open_interest_unit':'lots','amount_unit':'10000_RMB','statistics_basis':'NOT_VERIFIED'}
    source.write_text(json.dumps(sidecar))
    raw=pd.DataFrame({'date':['2019-10-08','2019-10-09'],'contract':['m2001','M2001'],
        'open':['100','101'],'high':['102','103'],'low':['99','100'],'close':['101','102'],
        'settle':['100','101'],'volume':['10','20'],'open_interest':['20','30']})
    data=tmp_path/'sample.csv';raw.to_csv(data,index=False)
    return tmp_path,data,identity,source,raw,ids,sidecar


def test_dry_run_writes_no_archive_and_import_is_exact_immutable_resume(files):
    tmp,data,identity,source,raw,ids,side=files
    before=data.read_bytes()
    d=import_file(data,identity,source,tmp,'sample',dry_run=True)
    assert d['standardized_rows']==2 and not (tmp/'data').exists()
    d=import_file(data,identity,source,tmp,'sample')
    assert data.read_bytes()==before and d['standardized_rows']==2
    result=pd.read_csv(tmp/d['standardized_file'])
    assert result.delivery_year.eq(2020).all() and not result.eligible_for_official_labels.any()
    assert all((tmp/p).stat().st_mode & 0o222 == 0 for p in d['archive_fingerprints'])
    assert import_file(data,identity,source,tmp,'sample')['resume_status']=='VERIFIED_CACHE_REUSED'
    side['statistics_basis']='different';source.write_text(json.dumps(side))
    with pytest.raises(FileExistsError):import_file(data,identity,source,tmp,'sample')


@pytest.mark.parametrize('col,value',[
    ('contract','M0'),('contract','M1999'),('contract','M2005'),('exchange','SHFE'),
    ('date','10/09/2019'),('date',''),('date','2018-01-01'),('open','-1'),('high','90'),
    ('close','0'),('settle',''),('volume','-1'),('volume','1.5'),('open_interest','inf'),
    ('open_interest','missing'),('trading_status','mystery')])
def test_rejects_unknown_continuous_contracts_dates_units_and_invalid_activity(files,col,value):
    tmp,data,identity,source,raw,ids,side=files
    raw.loc[0,col]=value
    with pytest.raises(SchemaError):normalize_table(raw,load_identities(identity)[0],side)


def test_explicit_inactive_rows_preserved_not_treated_as_quotes(files):
    tmp,data,identity,source,raw,ids,side=files
    raw.loc[0,['open','high','low','close','settle']]=''
    raw.loc[0,'volume']='0';raw['trading_status']=['INACTIVE','ACTIVE']
    result=normalize_table(raw,load_identities(identity)[0],side)
    assert len(result)==2 and result.valid_trade_quote.tolist()==[False,True]
    assert pd.isna(result.close.iloc[0]) and result.close.iloc[1]==102


def test_duplicate_rows_or_unverified_year_rejected(files):
    tmp,data,identity,source,raw,ids,side=files
    with pytest.raises(SchemaError):normalize_table(pd.concat([raw,raw]),load_identities(identity)[0],side)
    ids['verification_status']='NOT_VERIFIED';ids.to_csv(identity,index=False)
    with pytest.raises(SchemaError):load_identities(identity)
    ids['verification_status']=IDENTITY_VERIFIED;ids['delivery_year']='2120';ids.to_csv(identity,index=False)
    with pytest.raises(SchemaError):load_identities(identity)


def test_failed_real_import_keeps_exact_bytes_and_manifest(files):
    tmp,data,identity,source,raw,ids,side=files
    raw['contract']='M0';raw.to_csv(data,index=False)
    with pytest.raises(SchemaError):import_file(data,identity,source,tmp,'failure')
    manifests=list((tmp/'data/manifests/failure').rglob('*.json'))
    d=json.loads(manifests[0].read_text());assert d['status']=='REJECTED_PRESERVED'
    archived=list((tmp/'data/raw/failure').rglob('original.csv'))
    assert archived[0].read_bytes()==data.read_bytes() and not (tmp/'data/standardized').exists()


def test_generic_excel_chinese_aliases_and_duplicate_headers(files):
    tmp,data,identity,source,raw,ids,side=files
    raw=raw.rename(columns={'date':'交易日期','contract':'合约','open':'开盘价','high':'最高价',
        'low':'最低价','close':'收盘价','settle':'结算价','volume':'成交量','open_interest':'持仓量'})
    p=tmp/'synthetic.xlsx';raw.to_excel(p,index=False)
    assert len(normalize_table(read_table(p),load_identities(identity)[0],side))==2
    p=tmp/'duplicate.csv';p.write_text('date,date,contract\n20191008,20191008,M2001\n')
    with pytest.raises(SchemaError,match='Duplicate raw headers'):read_table(p)


def test_synthetic_demo_cannot_write_research_data_and_overlap_is_refused(files):
    tmp,data,identity,source,raw,ids,side=files
    side['data_origin']='SYNTHETIC_TEST';source.write_text(json.dumps(side))
    with pytest.raises(SchemaError):import_file(data,identity,source,tmp,'x',synthetic=True)
    assert import_file(data,identity,source,tmp,'x',dry_run=True,synthetic=True)['status']=='SYNTHETIC_DRY_RUN_ONLY'
    side['data_origin']='DCE_OFFICIAL_EXPORT';source.write_text(json.dumps(side))
    import_file(data,identity,source,tmp,'overlap')
    raw.loc[0,'volume']='11';raw.to_csv(data,index=False)
    with pytest.raises(FileExistsError,match='Overlapping'):import_file(data,identity,source,tmp,'overlap')


def test_native_status_flags_and_credential_sidecars_need_explicit_review(files):
    from triweave.providers.dce_official_import import validate_sidecar
    tmp,data,identity,source,raw,ids,side=files
    raw['trading_status']='0'
    with pytest.raises(SchemaError,match='Unknown explicit'):normalize_table(raw,load_identities(identity)[0],side)
    side['token']='synthetic-secret'
    with pytest.raises(SchemaError,match='must not contain credentials'):validate_sidecar(side)


def test_quote_cannot_postdate_acquisition(files):
    tmp,data,identity,source,raw,ids,side=files
    side['acquired_at']='2019-10-01T12:00:00+08:00';source.write_text(json.dumps(side))
    with pytest.raises(SchemaError,match='later than'):import_file(data,identity,source,tmp,'future',dry_run=True)
