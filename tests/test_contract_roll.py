import pandas as pd
import pytest

from triweave.contract_identity import parse_code,build_identity_manifest,shfe_metadata,VERIFIED
from triweave.roll_diagnostics import (fixed_expiry_selection,lagged_activity_selection,
    same_contract_checks,adjacent_contract_price_relations,SchemaError)


def sample():
    days=pd.bdate_range('2019-10-01',periods=14).strftime('%Y-%m-%d').tolist()
    registry=pd.DataFrame({'contract_code':['M1910','M1911'],
        'listing_date_if_verified':['2019-01-01']*2,
        'last_trade_date_if_verified':[days[8],days[-1]],
        'metadata_available_date_for_diagnostic':[days[0]]*2,'verification_status':[VERIFIED]*2})
    quotes=pd.DataFrame([{'date':d,'contract':c,'close':100+i+(50 if c=='M1911' else 0),
        'volume':(10 if i<4 else 20) if c=='M1911' else (20 if i<4 else 10),
        'open_interest':50} for i,d in enumerate(days) for c in registry.contract_code])
    return days,registry,quotes


@pytest.mark.parametrize('exchange,code,status',[
    ('DCE','M0','INVALID_OR_CONTINUOUS_CODE'),('SHFE','CU1913','INVALID_MONTH_OR_EXCHANGE_FORMAT'),
    ('INE','SC_TAS2001','SPECIAL_MODE_EXCLUDED'),('CZCE','CF001','ORDINARY_CODE_PATTERN_ONLY'),
    ('DCE','M001','INVALID_MONTH_OR_EXCHANGE_FORMAT')])
def test_code_shape_does_not_certify_delivery_year(exchange,code,status):
    result=parse_code(code,exchange)
    assert result['code_status']==status and 'delivery_year' not in result


def test_both_rules_are_prefix_causal_and_cannot_use_future_identity_publications():
    days,registry,q=sample();cut=days[6]
    before=lagged_activity_selection(q,registry,days)
    future=q.copy();mask=future.date.gt(cut)
    future.loc[mask,['volume','open_interest','close']]=[999999,999999,999999]
    after=lagged_activity_selection(future,registry,days)
    pd.testing.assert_frame_equal(before.loc[before.decision_date<=cut],after.loc[after.decision_date<=cut])
    prefix=lagged_activity_selection(q.loc[q.date<=cut],registry,days[:7])
    pd.testing.assert_frame_equal(before.iloc[:7],prefix)
    a=fixed_expiry_selection(registry,days,n=2,calendar_verified=True)
    changed=registry.copy();changed.loc[1,'metadata_available_date_for_diagnostic']=days[9]
    newer=lagged_activity_selection(q,changed,days)
    assert not newer.loc[newer.decision_date<days[9],'contract'].eq('M1911').any()
    pd.testing.assert_frame_equal(a,fixed_expiry_selection(registry,days,n=2,calendar_verified=True))
    future_a=fixed_expiry_selection(changed,days,n=2,calendar_verified=True)
    changed.loc[1,'listing_date_if_verified']='2018-01-01'
    mutated_a=fixed_expiry_selection(changed,days,n=2,calendar_verified=True)
    pd.testing.assert_frame_equal(future_a.loc[future_a.decision_date<days[9]],mutated_a.loc[mutated_a.decision_date<days[9]])


def test_fixed_rule_requires_ex_ante_calendar_and_verified_identity():
    days,registry,q=sample()
    with pytest.raises(SchemaError,match='calendar NOT_VERIFIED'):fixed_expiry_selection(registry,days)
    registry['verification_status']='NOT_VERIFIED'
    with pytest.raises(SchemaError):lagged_activity_selection(q,registry,days)
    with pytest.raises(SchemaError):fixed_expiry_selection(registry,days,calendar_verified=True)


def test_same_contract_arithmetic_never_crosses_contract_or_missing_session():
    days,registry,q=sample()
    q=q.loc[~(q.contract.eq('M1910')&q.date.eq(days[2]))]
    out=same_contract_checks(q,days)
    assert pd.isna(out.loc[out.contract.eq('M1910')&out.date.eq(days[3]),'diagnostic_same_contract_price_change'].iloc[0])
    assert out.groupby('contract').first()['previous_close'].notna().all()  # Later same-code pairs exist.
    for contract,f in out.groupby('contract'):
        assert pd.isna(f.iloc[0].diagnostic_same_contract_price_change)
    value=out.loc[out.contract.eq('M1911')&out.date.eq(days[1]),'diagnostic_same_contract_price_change'].iloc[0]
    assert value==pytest.approx(151/150-1)


def test_czce_repeated_decade_codes_are_separate_unverified_episodes():
    f=pd.DataFrame({'exchange':['CZCE']*2,'variety':['CF']*2,'contract':['CF501']*2,
        'source_id':['official']*2,'date':['2014-11-03','2024-11-01'],'manifest':['a','b']})
    result=build_identity_manifest(f,pd.DataFrame())
    assert len(result)==2 and result.contract_identity_key.nunique()==2
    assert result.delivery_year.isna().all() and result.verification_status.eq('NOT_VERIFIED').all()


def test_real_shfe_metadata_shape_does_not_mislabel_sc_or_certify_date_mismatch(tmp_path):
    import json
    rows=[]
    for code,start,end in [('cu1910','20191016','20191022'),('fu2001','20191223','20191227'),('sc2001','20191223','20191227')]:
        rows.append({'INSTRUMENTID':code,'OPENDATE':'20190102','EXPIREDATE':'20191015',
            'STARTDELIVDATE':start,'ENDDELIVDATE':end,'TRADINGDAY':'20191008','UPDATE_DATE':'2019-10-08 19:57:30'})
    p=tmp_path/'snapshot.json';p.write_text(json.dumps({'report_date':'20191008','ContractBaseInfo':rows}))
    result=shfe_metadata(tmp_path,p).set_index('contract_code')
    assert 'SC2001' not in result.index
    assert result.loc['CU1910','delivery_year']==2019 and result.loc['CU1910','verification_status']==VERIFIED
    assert pd.isna(result.loc['FU2001','delivery_year']) and result.loc['FU2001','verification_status']=='NOT_VERIFIED'


def test_vendor_diagnostic_reads_actual_job_naming_and_only_fixed_train_window(tmp_path):
    import json,hashlib
    from triweave.contracts import vendor_train_diagnostic
    p=tmp_path/'data/standardized/baseline15y/futures_continuous_exploratory/SHFE_CU_0.csv.gz'
    p.parent.mkdir(parents=True)
    pd.DataFrame({'date':['2019-10-09','2025-01-02'],'close':[100,99999]}).to_csv(p,index=False)
    m=tmp_path/'data/manifests/baseline15y/jobs/futures_continuous_exploratory/SHFE_CU_0.json'
    m.parent.mkdir(parents=True);m.write_text(json.dumps({'status':'OK','standardized_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
    f=vendor_train_diagnostic(tmp_path,'SHFE','CU')
    assert len(f)==1 and f.date.iloc[0]=='2019-10-09'
    p.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='checksum'):vendor_train_diagnostic(tmp_path,'SHFE','CU')


def test_forced_synthetic_roll_keeps_term_basis_out_of_same_contract_changes():
    days,registry,q=sample()
    selections=lagged_activity_selection(q,registry,days)
    assert selections.loc[selections.decision_date.eq(days[3]),'contract'].iloc[0]=='M1910'
    assert selections.loc[selections.decision_date.eq(days[7]),'contract'].iloc[0]=='M1911'
    gaps=adjacent_contract_price_relations(q,registry)
    assert gaps.same_day_term_gap.eq(50).all() and gaps.date.max()==days[8]
    pairs=same_contract_checks(q,days)
    assert pairs.diagnostic_same_contract_price_change.dropna().lt(0.02).all()
    assert not gaps.official_label.any() and not pairs.official_label.any()
