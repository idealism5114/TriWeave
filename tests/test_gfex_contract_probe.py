import json
from types import SimpleNamespace
import pytest

from triweave.providers.gfex_incremental import probe


def test_one_access_failure_preserved_and_never_retried(tmp_path):
    calls=[]
    class Session:
        def post(self,*args,**kwargs):
            calls.append(kwargs);return SimpleNamespace(status_code=567,content=b'blocked synthetic fixture')
    d=probe(tmp_path,Session())
    assert len(calls)==1 and d['new_standardized_rows']==0 and d['status']=='BLOCKED'
    assert (tmp_path/d['response_archive']).read_bytes()==b'blocked synthetic fixture'
    with pytest.raises(FileExistsError):probe(tmp_path,Session())
    assert len(calls)==1


def test_successful_old_day_is_never_requested(tmp_path):
    p=tmp_path/'data/manifests/rescue_20261015/gfex_official_akshare_v1/futures_contracts_daily_partial_v3/20231024.json'
    p.parent.mkdir(parents=True);p.write_text('{}')
    with pytest.raises(FileExistsError,match='Already successful'):probe(tmp_path)


def test_http_200_html_is_a_failed_market_response(tmp_path):
    class Response:
        status_code=200
        content=b'<html><body>synthetic access challenge</body></html>'
        def json(self):raise ValueError('not JSON')
    class Session:
        def post(self,*args,**kwargs):return Response()
    d=probe(tmp_path,Session())
    assert d['status']=='BLOCKED' and d['status_code']==200 and d['new_standardized_rows']==0
    assert d['error_type']=='ValueError' and not (tmp_path/'data/standardized').exists()
