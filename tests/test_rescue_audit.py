import pandas as pd
from triweave.rescue_audit import classify_anomaly,stock_overlap,anomaly_triage


def test_triage_does_not_treat_tas_or_zero_activity_as_fabrication():
    assert classify_anomaly('INVALID_VALUE_OR_OHLC_REVIEW','SC_TAS2401',0,0,0)[0]=='REQUIRES_REVIEW'
    assert classify_anomaly('INVALID_VALUE_OR_OHLC_REVIEW','RB2401',0,20,0)[0]=='LIKELY_INACTIVE_CONTRACT'
    assert classify_anomaly('INVALID_VALUE_OR_OHLC_REVIEW','RB2401',10,20,0)[0]=='CONFIRMED_INVALID'
    assert classify_anomaly('INVALID_VALUE_OR_OHLC_REVIEW','RB2401',10,float('nan'),100,missing_fields=True)[0]=='POSSIBLE_SOURCE_SCHEMA_ISSUE'
    assert classify_anomaly('LEVEL_JUMP_REVIEW','RB2401',10,20,200)[0]=='REQUIRES_REVIEW'


def test_overlap_keeps_original_values_and_explicit_volume_conversion():
    old=pd.DataFrame({'asset_id':['sh.600000'],'date':['2023-10-09'],'series_type':['stock'],
        'source_id':['eastmoney'],'raw_file':['old'],'close':[10.],'volume':[2.], 'amount':[2000.],'turnover':[1.]})
    new=old.copy();new['source_id']='baostock';new['raw_file']='new';new['volume']=200.
    result=stock_overlap(old,new)
    v=result.loc[result.field.eq('volume')].iloc[0]
    assert v.value_a_original==2 and v.comparison_value_a==200 and v.difference==0
    assert old.volume.iloc[0]==2 and result.source_a.eq('eastmoney').all()


def test_triage_links_each_original_event_without_dropping_aggregates():
    old=pd.DataFrame([{'asset_id':'FUT_RB','series_type':'exchange_listed_contract',
        'date':'2023-10-09 00:00:00','contract':'RB2401','anomaly':'INVALID_VALUE_OR_OHLC_REVIEW'},
        {'asset_id':'FUT_RB','series_type':'exchange_listed_contract','date':None,'contract':'RB2401','anomaly':'LONG_ZERO_VOLUME_RUN_REVIEW'}])
    frame=pd.DataFrame({'asset_id':['FUT_RB'],'series_type':['exchange_listed_contract'],'date':['2023-10-09'],
        'contract':['RB2401'],'source_id':['official'],'raw_file':['raw'],'manifest':['meta'],
        'volume':[0.],'open_interest':[0.],'open':[0.],'high':[0.],'low':[0.],'close':[0.],'settle':[100.]})
    result=anomaly_triage(old,frame)
    assert len(result)==2 and result.original_anomaly_id.nunique()==2
    assert result.review_status.eq('LIKELY_INACTIVE_CONTRACT').all()
    assert result.iloc[0].raw_file=='raw'
    assert result.iloc[1].row_resolution_status=='AGGREGATED_EVENT_NOT_SINGLE_OBSERVATION'


def test_stock_quality_does_not_turn_suspension_placeholders_into_active_quotes():
    from triweave.rescue_audit import stock_quote_quality
    f=pd.DataFrame({'trading_status':['1','0'],'open':[10.,0.],'high':[11.,0.],
        'low':[9.,0.],'close':[10.,0.],'volume':[100.,0.],'amount':[1000.,0.],'turnover':[1.,0.]})
    result=stock_quote_quality(f,'baostock_unadjusted_v1')
    assert result['active_ohlc_invalid_rows']==0 and result['active_required_numeric_missing_cells']==0
    f.loc[0,'high']=8.
    assert stock_quote_quality(f,'baostock_unadjusted_v1')['active_ohlc_invalid_rows']==1


def test_final_report_refuses_unfinished_stock_acquisition(tmp_path):
    import pytest
    from triweave.rescue_audit import report
    with pytest.raises(RuntimeError,match='still running or interrupted'):
        report(tmp_path)


def test_metadata_cannot_open_concurrent_baostock_session(tmp_path):
    import pytest
    from triweave.rescue import stock_metadata
    from triweave.rescue_providers import ProviderBlocked
    with pytest.raises(ProviderBlocked,match='Finish the sequential stock session'):
        stock_metadata(tmp_path)
