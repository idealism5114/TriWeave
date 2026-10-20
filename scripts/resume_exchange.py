"""Resume a single smoke-approved exchange while other disjoint exchanges run."""
import argparse
import json
from pathlib import Path

from triweave.assets import validate_asset_configuration
from triweave.cli import project_config
from triweave.collect import make_jobs, atomic_json, utc_now
from triweave.pipeline import _run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exchange',required=True,choices=['SHFE','DCE','CZCE','INE','GFEX','CFFEX'])
    parser.add_argument('--run-id',default='baseline15y')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_asset_configuration(root)
    cfg = project_config(root)
    smoke = json.loads((root/'experiments/data_collection/smoke_test/results.json').read_text())
    if not smoke.get('complete'):
        raise RuntimeError('Complete real smoke tests are required')
    passed = {(r['api'],r['group']) for r in smoke['results'] if r['pass']}
    jobs = make_jobs(['futures_contracts'],cfg['research_start'],cfg['research_end'],markets=[args.exchange])
    decisions = _run_batch(root,args.run_id,jobs,passed)
    atomic_json({'at':utc_now(),'exchange':args.exchange,'decisions':decisions},
                root/'data/manifests'/args.run_id/('supplemental_'+args.exchange+'_summary.json'))
    return 0 if all(d['status'] in ('OK','CACHED','NO_SELECTED_VARIETIES','CACHED_EMPTY_SELECTION') for d in decisions) else 1


if __name__=='__main__':
    raise SystemExit(main())
