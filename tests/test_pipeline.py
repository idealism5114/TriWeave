import json
import pandas as pd

from triweave.collect import Job
from triweave.pipeline import frame_schema, run_collection


def test_smoke_schema_does_not_accept_close_only_success():
    job = Job('stocks', 'sample', 'stock_zh_a_hist', {}, '2023-10-01', '2023-10-31')
    frame = pd.DataFrame({'date': ['2023-10-09'], 'asset_id': ['sh.600000'], 'close': [10.]})
    assert not frame_schema(frame, job)['pass']


def test_failed_smoke_blocks_actual_market_calls(tmp_path, monkeypatch):
    import triweave.pipeline as pipeline
    output = tmp_path / 'experiments/data_collection/smoke_test'
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps({'complete': True, 'run_id': 'smoke', 'results': []}))
    monkeypatch.setattr(pipeline, 'project_config', lambda root: {'research_start': '2011-10-01', 'research_end': '2026-09-30', 'retry_attempts': 1, 'request_pause_seconds': 0})
    monkeypatch.setattr(pipeline, 'load_stock_universe', lambda path: pd.DataFrame())
    job = Job('stocks', 'sample', 'stock_zh_a_hist', {}, '2011-10-01', '2026-09-30')
    monkeypatch.setattr(pipeline, 'make_jobs', lambda *a, **k: [job])
    class ForbiddenApi:
        def __getattr__(self, name):
            raise AssertionError('A failed smoke gate must not make provider calls')
    decisions = run_collection(tmp_path, ForbiddenApi(), 'baseline')
    assert decisions[0]['status'] == 'BLOCKED_SMOKE'
    assert not (tmp_path / 'data/standardized').exists()


def test_exchange_partition_preserves_every_job_and_stock_stage_order():
    from triweave.pipeline import partition_jobs
    jobs=[Job('stocks','stock','stock_zh_a_hist',{},'2023-10-01','2023-10-31'),
          Job('futures_contracts/SHFE','shfe','get_futures_daily',{},'2023-10-01','2023-10-31',context='SHFE'),
          Job('bond_yield_curves','bond','bond_china_yield',{},'2023-10-01','2023-10-31'),
          Job('futures_contracts/CZCE','czce','get_futures_daily',{},'2023-10-01','2023-10-31',context='CZCE')]
    pre,exchanges,stocks=partition_jobs(jobs)
    assert [j.name for j in pre]==['bond']
    assert list(exchanges)==['SHFE','CZCE']
    assert [j.name for j in stocks]==['stock']
    assert {j.name for j in pre+sum(exchanges.values(),[])+stocks}=={j.name for j in jobs}


def test_independent_exchange_processes_preserve_complete_logs(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import shutil
    from pathlib import Path
    from triweave.pipeline import _run_batch
    shutil.copytree(Path(__file__).resolve().parents[1]/'configs',tmp_path/'configs')
    jobs=[Job('futures_contracts/'+market,market,'get_futures_daily',{},'2023-10-01','2023-10-31',context=market)
          for market in ('SHFE','CZCE')]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_run_batch,str(tmp_path),'parallel',[job],set()) for job in jobs]
        results=[future.result(timeout=30) for future in futures]
    assert all(result[0]['status']=='BLOCKED_SMOKE' for result in results)
    events=[json.loads(line) for line in (tmp_path/'data/manifests/parallel/execution_decisions.jsonl').read_text().splitlines()]
    assert {e['name'] for e in events}=={'SHFE','CZCE'}
    assert not (tmp_path/'data/standardized').exists()


def test_local_interrupted_write_does_not_cancel_other_jobs(tmp_path,monkeypatch):
    import triweave.pipeline as pipeline
    monkeypatch.setattr(pipeline,'project_config',lambda root: {'retry_attempts':1,'request_pause_seconds':0})
    jobs=[Job('stocks',name,'stock_zh_a_hist',{},'2023-10-01','2023-10-31') for name in ('interrupted','next')]
    calls=[]
    def collect(self,job):
        calls.append(job.name)
        if job.name=='interrupted':
            raise FileExistsError('Untracked raw archive: refuse to overwrite')
        return 'CACHED'
    monkeypatch.setattr(pipeline.Collector,'collect_one',collect)
    decisions=pipeline._run_batch(tmp_path,'local',jobs,{('stock_zh_a_hist','stocks')},object())
    assert calls==['interrupted','next']
    assert [d['status'] for d in decisions]==['FAILED','CACHED']
    events=pipeline.read_events(tmp_path,'local')
    assert events[0]['failure_kind']=='LOCAL_PRECONDITION_NO_OVERWRITE'
    assert 'FileExistsError' in events[0]['traceback']
