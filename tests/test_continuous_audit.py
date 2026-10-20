import pandas as pd
import pytest

from triweave.continuous_audit import availability,candidate_masks,longest_gap,quote_quality_counts,differing_cells,SchemaError


def prices(days):
    return pd.DataFrame({'date':days,'open':[100.]*len(days),'high':[102.]*len(days),
        'low':[99.]*len(days),'close':[101.]*len(days)})


def test_missing_session_cannot_be_compressed_into_h_step_candidate():
    days=pd.bdate_range('2020-01-02',periods=8).strftime('%Y-%m-%d').tolist()
    f=prices(days);f=f.loc[f.date.ne(days[2])]
    m=availability(f,days,'future');c=candidate_masks(m,horizons=(1,5))
    assert not c.loc[days[0],5] and not c.loc[days[1],1]
    assert c.loc[days[3],1] and not m.loc[days[2]]


def test_split_ends_are_masked_without_using_val_or_sealed_prices():
    days=['2022-12-29','2022-12-30','2023-01-03','2023-01-04','2024-12-30','2024-12-31']
    m=pd.Series(True,index=days)
    c=candidate_masks(m,horizons=(1,))
    assert c.loc['2022-12-29',1] and not c.loc['2022-12-30',1]
    assert not c.loc['2024-12-31',1]
    with pytest.raises(SchemaError,match='SEALED'):
        candidate_masks(pd.Series(True,index=days+['2025-01-02']))


def test_nonpositive_infinite_or_bad_ohlc_and_all_duplicate_dates_unavailable():
    days=pd.bdate_range('2020-01-02',periods=5).strftime('%Y-%m-%d').tolist()
    f=prices(days);f.loc[0,'close']=0.;f.loc[1,'close']=float('inf');f.loc[2,'high']=90.
    f=pd.concat([f,f.iloc[[3]]],ignore_index=True)
    m=availability(f,days,'future')
    assert m.tolist()==[False,False,False,False,True]


def test_stock_suspension_and_bond_economic_types_remain_distinct():
    days=['2020-01-02','2020-01-03'];f=prices(days);f['trading_status']=['1','0']
    assert availability(f,days,'stock').tolist()==[True,False]
    b=pd.DataFrame({'date':days,'value':[-0.1,0.]})
    assert availability(b,days,'yield_curve').all()
    assert not availability(b,days,'treasury_wealth_index').any()


def test_no_all_asset_dropna_or_prelisting_backfill():
    days=pd.bdate_range('2020-01-02',periods=5).strftime('%Y-%m-%d').tolist()
    old=availability(prices(days),days,'future')
    late=availability(prices(days[3:]),days,'future')
    masks=pd.DataFrame({'old':old,'late':late})
    assert len(masks)==5 and not masks.late.iloc[:3].any()
    one=candidate_masks(old,horizons=(1,))[1]
    two=candidate_masks(late,horizons=(1,))[1]
    assert one.iloc[:3].all() and not two.iloc[:3].any()
    assert longest_gap(late)==3


def test_future_perturbation_cannot_change_past_inputs_or_completed_label_windows():
    days=pd.bdate_range('2020-01-02',periods=40).strftime('%Y-%m-%d').tolist()
    f=prices(days);before=availability(f,days,'future');c=candidate_masks(before)
    changed=f.copy();changed.loc[changed.date.gt(days[25]),'close']=-1
    after=availability(changed,days,'future');d=candidate_masks(after)
    pd.testing.assert_series_equal(before.iloc[:26],after.iloc[:26])
    for h in (1,5,10,20):pd.testing.assert_series_equal(c[h].iloc[:26-h],d[h].iloc[:26-h])


def test_all_four_horizons_have_independent_end_masks():
    days=pd.bdate_range('2020-01-02',periods=30).strftime('%Y-%m-%d').tolist()
    c=candidate_masks(pd.Series(True,index=days))
    assert {h:int(c[h].sum()) for h in c}=={1:29,5:25,10:20,20:10}
    assert all(c.dtypes==bool)


def test_off_calendar_quote_is_not_mislabeled_as_numeric_ohlc_failure():
    f=prices(['2020-01-02','2020-01-03'])
    quality=quote_quality_counts(f,['2020-01-02'])
    assert quality=={'valid_source_dates':2,'invalid_OHLC_or_duplicate_rows':0,'off_reference_calendar_dates':1}


def test_vendor_revision_comparison_treats_two_missing_cells_as_equal():
    assert differing_cells(pd.Series([1.,float('nan'),2.]),pd.Series([1.,float('nan'),3.]))==1
