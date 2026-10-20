import pandas as pd
import pytest
from triweave.rescue import gfex_archived_normalize
from triweave.normalize import SchemaError


def test_gfex_replay_uses_echoed_date_and_preserves_original_fields():
    raw=pd.DataFrame({'variety':['工业硅','碳酸锂','工业硅小计'],'varietyOrder':['si','lc','si'],'delivMonth':['2311','2401','小计'],
        'open':[100,200,0],'high':[101,201,0],'low':[99,199,0],'close':[100,200,0],
        'volumn':[10,20,0],'openInterest':[30,40,0],'clearPrice':[100,200,0],
        'lastClear':[99,199,0]})
    original=raw.copy(deep=True)
    f=gfex_archived_normalize(raw,{'trade_date':['20231009'],'trade_type':['0']})
    assert set(f.contract)=={'SI2311','LC2401'} and f.date.eq('2023-10-09').all()
    assert set(f.asset_id)=={'FUT_SI','FUT_LC'}
    pd.testing.assert_frame_equal(raw,original)
    with pytest.raises(SchemaError): gfex_archived_normalize(raw,{})
