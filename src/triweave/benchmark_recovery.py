"""Versioned two-model repair, reusing35 immutable fits, then evaluating all37.

This entry point never changes an original source file or previous run. The
new suite inherits successful checkpoint paths with verified SHA256 records.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import torch

from . import benchmark as bm, experiment_suite as es, test_release as release
from . import slbr_phase as phase
from .models.benchmark import make_model as original_make_model
from .models.benchmark_recovery import make_model, REPAIRED, IMPLEMENTATIONS, NORM_FLOOR
from .models.slbr_triweave import masked_huber
from .slbr_data import DATA, INPUT_KEYS, manifests, to_device

PARENT = 'bench_all37_seed42_v1'
RUN = 'bench_all37_seed42_recovery_v1'
AUDIT = es.OUT / 'recovery_implementation_v1'
_original_hashes = bm.hashes
_original_training_protocol = bm.training_protocol


def model_hashes(root):
    return _original_hashes(root) | {p: phase.sha(root / p) for p in (
        'src/triweave/models/benchmark_recovery/__init__.py', 'src/triweave/benchmark_recovery.py')}


def gpu_probe(root, device):
    """Failure batch replay only: TRAIN samples, no optimizer fit or TEST."""
    dev = torch.device(device)
    if dev.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This gate requires a real CUDA device, no CPU fallback')
    assets, targets = manifests(root)
    out = root / AUDIT
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id, epoch in zip(REPAIRED, (1, 25)):
        filename = root / es.OUT / PARENT / 'runs' / (model_id.replace('/', '__') + '__attempt001') / f'nonfinite_epoch{epoch}.pt'
        dump = torch.load(filename, map_location='cpu', weights_only=False)
        b = to_device(dump['batch'], dev)
        row = dict(model_id=model_id, failure_snapshot_sha256=phase.sha(filename))
        for repaired, factory in ((False, original_make_model), (True, make_model)):
            phase.seed42()
            model = factory(model_id, assets, targets).to(dev)
            model.load_state_dict(dump['model_state_dict'], strict=True)
            model.train(); torch.cuda.reset_peak_memory_stats(dev)
            try:
                prediction = model(**{k: b[k] for k in INPUT_KEYS})
                loss, _ = masked_huber(prediction, b['Y'], b['label_valid_mask'])
                loss.backward()
                nonfinite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if nonfinite:
                    raise FloatingPointError('Nonfinite gradients: ' + ','.join(nonfinite))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                result = dict(PASS=True, loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                    prediction_abs_max=float(prediction.detach().abs().max()),
                    allocated_peak_MiB=torch.cuda.max_memory_allocated(dev) / 2**20)
            except (RuntimeError, FloatingPointError) as error:
                result = dict(PASS=False, error=str(error))
                if repaired:
                    release.write_json(out / 'gpu_probe_failure.json', dict(model_id=model_id, **result))
                    raise
            row['repaired' if repaired else 'original'] = result
            del model
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    value = dict(PASS=True, source_hashes=model_hashes(root), models=rows,
        deterministic_algorithms=True, GPU=torch.cuda.get_device_name(dev), environment=phase.environment(),
        TEST_numeric_rows_parsed=0, optimizer_updates=0, norm_floor=NORM_FLOOR)
    release.write_json(out / 'gpu_probe.json', value)
    return value


def launch(root, suite, task, gpu, testing):
    es.enough_disk(root)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    command = [sys.executable, '-B', '-u', '-m', 'triweave.benchmark_recovery',
        '--eval-worker' if testing else '--worker', '--run-id', suite.name,
        '--model', task['model_id'], '--attempt', task['attempt'], '--device', 'cpu' if gpu is None else 'cuda:0']
    handle = (root / task['log_path']).open('a')
    process = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
    task.update(status='RUNNING', pid=process.pid, process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                device='cpu' if gpu is None else f'physical_GPU_{gpu}')
    print('START ' + task['model_id'] + ' -> ' + task['device'] + ' log=' + task['log_path'], flush=True)
    return process, task, handle


def configure():
    bm.hashes = model_hashes
    bm.make_model = make_model
    es.make_model = make_model
    es.launch = launch
    bm.training_protocol = lambda: _original_training_protocol() | dict(
        numerical_repair_version='BENCHMARK_RECOVERY_V1', numerical_implementations=IMPLEMENTATIONS,
        TCN_weightnorm_floor=NORM_FLOOR, deterministic_algorithms=True,
        nonfinite_checks='error_if_nonfinite=True, no invalid gradient replacement or batch skipping')


def initialize(root, run_id=RUN, parent_id=PARENT, gpus=(0, 1), min_free_gib=4.):
    root = Path(root)
    parent = es.suite_path(root, parent_id)
    dest = es.suite_path(root, run_id)
    if dest.exists():
        raise FileExistsError('Recovery already exists; use --resume')
    cfg = release.read_json(parent / 'suite_config.json')
    release.check_files(root, cfg['source_hashes'])
    preflight = es.validate_preflight(root)
    probe = release.read_json(root / AUDIT / 'gpu_probe.json')
    if not probe['PASS'] or probe['source_hashes'] != model_hashes(root):
        raise RuntimeError('Current repair has not passed the CUDA failure-replay gate')
    if ((parent / 'models_frozen.json').exists() or (parent / 'test_dataset').exists() or
        list((parent / 'runs').glob('*/test_complete.json'))):
        raise RuntimeError('Parent TEST already opened; cannot refit after TEST access')
    free = es.gpu_free_mib()
    if not gpus or any(g not in free for g in gpus):
        raise ValueError('Requested GPU unavailable')
    es.enough_disk(root, 2.)
    tasks = release.read_json(parent / 'tasks.json')
    inherited = {}
    with es.suite_lock(parent):
        for task in tasks:
            if es.pid_alive(task):
                raise RuntimeError('Parent worker still alive; no overlapping recovery')
            complete = es.completed(root, task)
            if task['model_id'] in REPAIRED:
                if complete is not None:
                    raise RuntimeError('Repair would overwrite an already completed model')
            else:
                if complete is None:
                    raise RuntimeError('Unexpected unfinished model: ' + task['model_id'])
                inherited[task['model_id']] = complete
        dest.mkdir()
        (dest / 'logs').mkdir()
        new_tasks = []
        for task in tasks:
            if task['model_id'] in REPAIRED:
                new_tasks.append(es.new_task(root, dest, task['model_id'], number=2))
            else:
                old_run = root / task['run_path']
                new_run = dest / 'runs' / task['attempt']
                new_run.mkdir(parents=True)
                # Immutable checkpoints are referenced, not copied or fitted.
                # All TEST outputs will be local to the recovery run, keeping
                # the original run's files and semantics completely unchanged.
                checkpoint = root / inherited[task['model_id']]['checkpoint_path']
                (new_run / checkpoint.name).symlink_to(os.path.relpath(checkpoint, new_run))
                for name in ('config.json', 'training_complete.json', 'checkpoint_audit.json',
                    'validation_metrics.csv', 'per_horizon_metrics.csv', 'per_target_horizon_metrics.csv',
                    'training_history.csv', 'validation_predictions.npz', 'dataset_gate.txt'):
                    if (old_run / name).exists():
                        shutil.copy2(old_run / name, new_run / name)
                inherited_task = dict(task, status='COMPLETE', inherited_from=parent_id,
                    original_run_path=task['run_path'], run_path=str(new_run.relative_to(root)),
                    log_path=str((dest / 'logs' / (task['attempt'] + '.log')).relative_to(root)))
                inherited_task.pop('pid', None); inherited_task.pop('process_start_ticks', None)
                new_tasks.append(inherited_task)
        config = copy.deepcopy(cfg)
        config.update(run_id=run_id, gpus=list(gpus), min_free_gib=min_free_gib,
            source_hashes=es.workflow_hashes(root), training_protocol=_original_training_protocol(),
            recovery_parent_run=parent_id, inherited_models=list(inherited), retrain_models=list(REPAIRED),
            implementation_by_model={m: IMPLEMENTATIONS.get(m, 'ORIGINAL_BENCHMARK_V1') for m in cfg['models']},
            numerical_repairs=dict(TCN_norm_floor=NORM_FLOOR, normalization_accumulator_dtype='float64',
                CrossGNN_threshold='sort(values)[2], exact kth-value threshold; deterministic mode retained'),
            test_authorization=cfg['test_authorization'] + ' 用户追加授权：解决两个问题后重跑这两项，然后进入test。')
        release.write_json(dest / 'suite_config.json', config)
        release.write_json(dest / 'tasks.json', new_tasks)
        release.write_json(dest / 'inherited_checkpoints.json', inherited)
        release.write_json(dest / 'recovery_provenance.json', dict(
            parent_suite_config_sha256=phase.sha(parent / 'suite_config.json'),
            parent_tasks_sha256=phase.sha(parent / 'tasks.json'),
            original_preflight_sha256=phase.sha(root / bm.OUT / 'preflight.json'),
            repaired_preflight_gpu_sha256=phase.sha(root / AUDIT / 'gpu_probe.json'),
            parent_files_preserved=True, inherited_checkpoint_count=len(inherited), retrained_count=2,
            TEST_opened_before_retraining=False))
        release.write_json(dest / 'preflight.json', preflight | dict(source_hashes=model_hashes(root),
            repair_gate=probe, repaired_implementations=IMPLEMENTATIONS))
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('probe', 'run', 'status', 'results', 'worker', 'eval-worker'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--run-id', default=RUN)
    parser.add_argument('--parent-run-id', default=PARENT)
    parser.add_argument('--with-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--min-free-gib', type=float, default=4.)
    parser.add_argument('--model', choices=es.MODEL_IDS)
    parser.add_argument('--attempt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); root = Path.cwd()
    gpus = [int(g) for g in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus) or args.min_free_gib <= 0:
        parser.error('Invalid GPU admission settings')
    if args.probe:
        gpu_probe(root, args.device)
        return
    dest = es.suite_path(root, args.run_id)
    if args.run:
        if not args.with_test:
            parser.error('Explicit --with-test required')
        if not args.resume:
            initialize(root, args.run_id, args.parent_run_id, gpus, args.min_free_gib)
    configure()
    if args.run:
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Recovery interrupted; artifacts preserved')
        signal.signal(signal.SIGTERM, interrupted)
        es.run(root, SimpleNamespace(run_id=args.run_id, with_test=True, resume=True, retry_failed=args.retry_failed))
    elif args.status:
        es.status(root, dest)
    elif args.results:
        if not es.summarize(root, dest):
            raise SystemExit(1)
    else:
        if not args.model or not args.attempt:
            parser.error('Worker requires model and attempt')
        if args.worker:
            es.worker(root, dest, args.model, args.attempt, args.device)
        else:
            es.evaluate_test(root, dest, args.model, args.device)


if __name__ == '__main__':
    main()
