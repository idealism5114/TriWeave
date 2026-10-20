import pandas as pd

from triweave.research_audit import inspect_series


def record(kind='stock', market='stock'):
    return {'asset_id':'sample','asset_name':'sample','market':market,'role':'input','series_type':kind,
            'calendar_basis':'TEST_ONLY'}


def test_audit_never_calls_missing_data_unavailable():
    row, anomalies=inspect_series(pd.DataFrame(),record(),pd.date_range('2023-10-09','2023-10-11'),
                                 '2023-10-09','2023-10-11')
    assert row['collection_status']=='NOT_VERIFIED'
    assert row['prelisting_observation_count'] is None
    assert row['record_count']==0 and not anomalies


def test_yield_percent_is_not_a_price_and_is_never_transformed():
    frame=pd.DataFrame({'date':['2023-10-09','2023-10-10'],'value':[-0.1,1.5]})
    original=frame.copy(deep=True)
    row, anomalies=inspect_series(frame,record('yield_curve','bond'),pd.date_range('2023-10-09','2023-10-10'),
                                 '2023-10-09','2023-10-10')
    assert row['invalid_value_count']==0
    assert row['collection_status']=='SUCCESS'
    assert anomalies[0]['anomaly']=='LEVEL_JUMP_REVIEW'
    pd.testing.assert_frame_equal(frame,original)


def test_contract_dates_and_jumps_are_checked_within_contract_only():
    frame=pd.DataFrame({'date':['2023-10-09','2023-10-09'],'contract':['RB2401','RB2405'],
                        'open':[100.,200.],'high':[100.,200.],'low':[100.,200.],'close':[100.,200.],
                        'volume':[1,1],'open_interest':[2,2],'settle':[100.,200.]})
    row, anomalies=inspect_series(frame,record('exchange_listed_contract','commodity_future'),
                                 pd.to_datetime(['2023-10-09']),'2023-10-09','2023-10-09')
    assert row['duplicate_date_count']==0
    assert row['unique_date_count']==1 and row['record_count']==2
    assert not anomalies


def test_verified_late_listing_does_not_create_prelisting_gap():
    frame=pd.DataFrame({'date':['2023-10-10','2023-10-11'],'value':[100.,101.]})
    row,_=inspect_series(frame,record('treasury_wealth_index','bond'),pd.date_range('2023-10-09','2023-10-11'),
                         '2023-10-09','2023-10-11',{'date':'2023-10-10','source':'TEST_ONLY'})
    assert row['coverage_ratio']==1 and row['prelisting_observation_count']==0
    assert row['collection_status']=='SUCCESS'


def test_full_report_has_all_configured_assets_and_does_not_pass_one_day_as_15_years(tmp_path):
    # Synthetic fixture identity/quotes stay exclusively inside pytest's temporary root.
    import json
    import shutil
    from pathlib import Path
    from triweave.universe import import_stock_universe
    from triweave.collect import Collector,make_jobs
    from triweave.assets import TREASURY_CURVE
    from triweave.research_audit import comprehensive_audit
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source/'configs',tmp_path/'configs')
    stocks=tmp_path/'fixture_stock_ids.csv'
    stocks.write_text('code\n'+''.join(f'sh.{600000+i}\n' for i in range(263)))
    import_stock_universe(stocks,tmp_path/'data/manifests/stocks_263.csv')
    class FixtureApi:
        def bond_china_yield(self,**kwargs):
            return pd.DataFrame({'日期':['2023-10-09'],'曲线名称':[TREASURY_CURVE],
                                 **{tenor:[2.] for tenor in ('3月','6月','1年','3年','5年','7年','10年','30年')}})
    job=make_jobs(['bonds'],'2023-10-01','2023-10-31')[0]
    assert Collector(tmp_path,'fixture',api=FixtureApi(),pause=0).collect_one(job)=='OK'
    import gzip
    from triweave.universe import sha256_file
    orphan=tmp_path/'data/raw/fixture/_interrupted/original.csv.gz'
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(gzip.compress(b'date,close\n2023-10-09,100\n'))
    recovery={'archive_path':str(orphan.relative_to(tmp_path)),'sha256':sha256_file(orphan)}
    (tmp_path/'data/manifests/fixture/interrupted_write_recovery.json').write_text(json.dumps(recovery))
    result=comprehensive_audit(tmp_path,'fixture')
    out=tmp_path/'experiments/data_collection/fixture'
    assert len(pd.read_csv(out/'stock_coverage.csv'))==263
    assert len(pd.read_csv(out/'bond_coverage.csv'))==25
    assert len(pd.read_csv(out/'futures_target_coverage.csv'))==24
    assert len(pd.read_csv(out/'futures_auxiliary_coverage.csv'))==14
    assert len(pd.read_csv(out/'exchange_contract_coverage.csv'))==42
    exchanges=pd.read_csv(out/'exchange_collection_summary.csv')
    assert len(exchanges)==6 and exchanges['planned_months'].sum()==846
    assert exchanges['saved_response_months'].sum()==0
    assert exchanges['not_attempted_months'].sum()==846
    assert all(type(v) is bool or v in ('NOT_VERIFIED','BLOCKED') for v in result['flags'].values())
    bonds=pd.read_csv(out/'bond_coverage.csv')
    assert not bonds['collection_status'].eq('SUCCESS').any()
    assert not result['flags']['BOND_25_COLLECTION_PASS']
    assert result['flags']['READY_FOR_CAUSAL_PREPROCESSING']=='BLOCKED'
    assert (out/'REPORT.md').exists() and (out/'missing_intervals.csv').exists()
    fingerprints=json.loads((out/'data_fingerprints.json').read_text())
    assert fingerprints['files'][recovery['archive_path']]['matches_manifest'] is True
    assert 'NOT_A_VERIFIED_JOB' in fingerprints['files'][recovery['archive_path']]['purpose']


def test_csv_fingerprint_ignores_gzip_container_time(tmp_path):
    import gzip
    from triweave.research_audit import csv_content_fingerprint
    from triweave.universe import sha256_file
    a,b=tmp_path/'a.csv.gz',tmp_path/'b.csv.gz'
    payload=b'date,close\n2023-10-09,10.0\n'
    a.write_bytes(gzip.compress(payload,mtime=1))
    b.write_bytes(gzip.compress(payload,mtime=2))
    assert sha256_file(a)!=sha256_file(b)
    assert csv_content_fingerprint(a)==csv_content_fingerprint(b)


def test_zero_index_level_is_not_a_real_observation_date():
    frame=pd.DataFrame({'date':['2023-10-09','2023-10-10'],'value':[0.,float('inf')]})
    row,_=inspect_series(frame,record('treasury_wealth_index','bond'),pd.date_range('2023-10-09','2023-10-10'),
                         '2023-10-09','2023-10-10')
    assert row['record_count']==2 and row['unique_date_count']==2
    assert row['valid_observation_count']==row['valid_unique_date_count']==0
    assert row['coverage_ratio']==0 and row['longest_missing_gap']==2
    assert row['first_observation_date'] is None
    assert row['collection_status']=='FAILED'


def test_exchange_request_counts_preserve_saved_months_and_latest_failures():
    from types import SimpleNamespace
    from triweave.research_audit import exchange_summary
    jobs=[SimpleNamespace(group='futures_contracts/DCE',name=n) for n in ('one','two','three')]
    manifests=[{'job_group':'futures_contracts/DCE','name':'one'}]
    decisions=[{'group':'futures_contracts/DCE','name':n,'status':s}
               for n,s in [('one','BLOCKED_SMOKE'),('two','FAILED'),('two','BLOCKED_ACCESS_FAILURES')]]
    coverage=pd.DataFrame([{'series_type':'exchange_listed_contract','exchange':'DCE',
                           'first_observation_date':None,'last_observation_date':None,
                           'valid_observation_count':0,'record_count':0}])
    row=exchange_summary(jobs,manifests,decisions,coverage).set_index('exchange').loc['DCE']
    assert row['saved_response_months']==1
    assert row['failed_months']==0 and row['blocked_months']==1
    assert row['not_attempted_months']==1


def test_nonfinite_activity_and_invalid_settlement_are_flagged_without_repair():
    frame=pd.DataFrame({'date':['2023-10-09','2023-10-10','2023-10-11'],
                        'contract':['RB2401']*3, 'open':['100']*3,'high':['101']*3,
                        'low':['99']*3,'close':['100']*3,'volume':[float('inf'),1.,1.],
                        'open_interest':[2.,2.,2.],'settle':[100.,0.,'bad']})
    original=frame.copy(deep=True)
    row,issues=inspect_series(frame,record('exchange_listed_contract','commodity_future'),
                             pd.date_range('2023-10-09','2023-10-11'),'2023-10-09','2023-10-11')
    assert row['invalid_value_count']==2
    assert row['required_field_missing_count']==1
    assert row['collection_status']=='PARTIAL'
    assert sum(i['anomaly']=='INVALID_VALUE_OR_OHLC_REVIEW' for i in issues)==2
    pd.testing.assert_frame_equal(frame,original)


def test_tas_record_cannot_fill_outright_contract_history():
    frame=pd.DataFrame({'date':['2023-10-09'],'contract':['SC_TAS2311'],'variety':['SC'],
                        'open':[100.],'high':[100.],'low':[100.],'close':[100.],
                        'volume':[1.],'open_interest':[2.],'settle':[100.]})
    row,issues=inspect_series(frame,record('exchange_listed_contract','commodity_future'),
                             pd.to_datetime(['2023-10-09']),'2023-10-09','2023-10-09')
    assert row['record_count']==1 and row['special_trading_mode_row_count']==1
    assert row['unrecognized_contract_code_count']==0
    assert row['valid_observation_count']==0 and row['coverage_ratio']==0
    assert any(i['anomaly']=='SPECIAL_TRADING_MODE_NOT_OUTRIGHT_QUOTE' for i in issues)
