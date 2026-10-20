"""One bounded GFEX missing-day probe; never re-request a successful old day."""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from triweave.collect import atomic_json,utc_now
from triweave.rescue import gfex_archived_normalize
from triweave.rescue_providers import RescueStore
from triweave.universe import sha256_file

URL='http://www.gfex.com.cn/u/interfacesWebTiDayQuotes/loadList'


def probe(root,session=None,date='20231024'):
    root=Path(root).resolve();out=root/'experiments/phase_d_contracts/gfex_missing_day_probe.json'
    if out.exists():raise FileExistsError('Bounded probe already recorded; no repeated failed request')
    previous=root/'data/manifests/rescue_20261015/gfex_official_akshare_v1/futures_contracts_daily_partial_v3'/(date+'.json')
    if previous.exists():raise FileExistsError('Already successful v3 day must not be downloaded again')
    session=session or requests.Session()
    doc={'at':utc_now(),'date':date,'url':URL,'attempts':1,'source_id':'gfex_official_akshare_v1',
         'proxy_or_auth_changes':False,'new_standardized_rows':0,'status':'BLOCKED'}
    raw=root/'data/raw/contracts_20261015/gfex_http'/(date+'.response')
    try:
        response=session.post(URL,data={'trade_date':date,'trade_type':'0'},timeout=(5,15))
        raw.parent.mkdir(parents=True,exist_ok=True)
        if raw.exists():raise FileExistsError('Untracked HTTP evidence preserved')
        raw.write_bytes(response.content)
        doc.update(status_code=response.status_code,response_archive=str(raw.relative_to(root)),
                   response_sha256=sha256_file(raw),bytes=len(response.content))
        if response.status_code!=200:raise RuntimeError('HTTP '+str(response.status_code)+'; stop, no retries')
        body=response.json();echo=body.get('param',{})
        if echo.get('trade_date')!=[date] or echo.get('trade_type')!=['0']:
            raise ValueError('Response date/type echo mismatch')
        original=pd.DataFrame(body.get('data',[]))
        if original.empty:raise ValueError('Empty official response is not success')
        clean=gfex_archived_normalize(original,echo)
        if clean.empty or set(clean.variety)!={'SI','LC'}:raise ValueError('Both expected ordinary varieties must pass schema')
        store=RescueStore(root,'contracts_20261015')
        params={'date':date,'upstream_retrieved_at':doc['at'],'response_archive':str(raw.relative_to(root)),
                'response_sha256':doc['response_sha256'],'normalizer_source_sha256':sha256_file(root/'src/triweave/rescue.py')}
        m=store.save(doc['source_id'],'futures_contracts_daily_v1',date,params,original,
                     lambda f:gfex_archived_normalize(f,echo))
        doc.update(status='PARTIAL_ONE_NEW_DATE_ONLY',new_standardized_rows=m['standardized_rows'],manifest=m)
    except Exception as exc:
        doc.update(error_type=type(exc).__name__,error=str(exc))
    atomic_json(doc,out)
    print(json.dumps(doc,ensure_ascii=False,indent=2))
    return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    a=p.parse_args(argv);probe(a.root)


if __name__=='__main__':main()
