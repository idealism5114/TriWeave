"""Bounded parallel baselines/ablations, then fixed-checkpoint TEST evaluation.

--run requires --with-test: this is an explicit new authorization workflow.
The frozen legacy trainer continues to see only TRAIN/VAL. TEST is opened only
after every prespecified model has completed and its checkpoint SHA is frozen.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import benchmark as bm, slbr_phase as phase, test_release as release
from .models.benchmark import MODEL_IDS, make_model
from .models.benchmark.classical import CLASSICAL, MaskedClassical
from .slbr_data import DATA, INPUT_KEYS, SLBRDataset, collate, manifests

OUT = Path('experiments/phase_g_benchmark_test')
AUTHORIZATION = '用户明确要求：同时启动消融实验和baseline，最终都在测试集评估并读取最终test结果。'


def suite_path(root, run_id):
    if not run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Use a simple unique --run-id (letters/digits/_/-)')
    return Path(root) / OUT / run_id


def workflow_hashes(root):
    paths = sorted((Path(root) / 'src/triweave').rglob('*.py'))
    paths += [Path(root) / p for p in (
        'configs/triweave_benchmark_v1.json',
        'data/manifests/baseline15y/reference_trading_calendar.csv',
        'experiments/phase_e_23_research/frozen_protocol.json',
        'configs/futures_targets_23_research_v1.csv')]
    return {str(p.relative_to(root)): phase.sha(p) for p in paths}


def is_classical(model_id):
    return model_id.split('/')[1] in CLASSICAL


def validate_preflight(root):
    value = release.read_json(Path(root) / bm.OUT / 'preflight.json')
    if not value['PASS'] or value['source_hashes'] != bm.hashes(Path(root)):
        raise ValueError('Frozen model migration preflight mismatch')
    return value


@contextmanager
def suite_lock(suite):
    with (suite / 'launcher.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another launcher owns this suite') from error
        yield


def gpu_free_mib():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True)
    return {int(line.split(',')[0]): int(line.split(',')[1]) for line in result.stdout.splitlines() if line.strip()}


def enough_disk(root, minimum_gib=1.):
    free = shutil.disk_usage(root).free / 2**30
    if free < minimum_gib:
        raise RuntimeError(f'Only {free:.2f} GiB free; need {minimum_gib:.2f} GiB. No old files were deleted.')
    return free


def completed(root, task):
    run = Path(root) / task['run_path']
    marker = run / 'training_complete.json'
    if not marker.exists():
        return None
    value = release.read_json(marker)
    config = release.read_json(run / 'config.json')
    if value['model_id'] != task['model_id'] or config['model_id'] != task['model_id']:
        raise ValueError('Task identity mismatch')
    checkpoint = run / ('estimator.joblib' if is_classical(task['model_id']) else 'best.pt')
    expected = value['estimator_sha256'] if is_classical(task['model_id']) else value['checkpoint_sha256']
    if phase.sha(checkpoint) != expected:
        raise ValueError('Completed checkpoint changed: ' + task['model_id'])
    if not is_classical(task['model_id']):
        if not release.read_json(run / 'checkpoint_audit.json')['PASS']:
            raise ValueError('Checkpoint selection/reload audit failed')
    return dict(checkpoint_path=str(checkpoint.relative_to(root)), checkpoint_sha256=expected,
        run_path=task['run_path'], training_complete_sha256=phase.sha(marker),
        config_sha256=phase.sha(run / 'config.json'), best_epoch=value.get('best_epoch'),
        val_selection_score=float(pd.read_csv(run / 'validation_metrics.csv').iloc[0]['VAL_SELECTION_SCORE']))


def freeze_models(root, suite, config, tasks):
    records = {}
    for task in tasks:
        value = completed(root, task)
        if value is None:
            raise RuntimeError('All-model barrier: unfinished ' + task['model_id'])
        records[task['model_id']] = value
    value = dict(models=config['models'], checkpoints=records, source_hashes=config['source_hashes'],
        dataset_manifest_sha256=config['dataset_manifest_sha256'],
        experiment_config_sha256=phase.sha(suite / 'suite_config.json'),
        selection='All neural checkpoints selected by VAL only; classical hyperparameters fixed before fitting',
        TEST_used_for_selection=False, all_training_finished_before_TEST_open=True)
    path = suite / 'models_frozen.json'
    if path.exists():
        if release.read_json(path) != value:
            raise ValueError('Existing freeze differs; cannot change models after TEST release')
    else:
        release.write_json(path, value)
    return value


def shared_train_design(root, suite, model):
    """One immutable TRAIN mmap shared by the three CPU estimators."""
    dest = suite / 'classical_cache'
    dest.mkdir(exist_ok=True)
    marker = dest / 'manifest.json'
    identity = dict(dataset_manifest_sha256=phase.sha(root / DATA / 'dataset_manifest.json'),
                    shape=[len(SLBRDataset(root, 'TRAIN')), 161720], dtype='float32')
    if marker.exists():
        info = release.read_json(marker)
        path = dest / info['file']
        if any(info[k] != v for k, v in identity.items()) or phase.sha(path) != info['sha256']:
            raise ValueError('Shared classical cache changed')
        return np.load(path, mmap_mode='r')
    attempt = 1
    while (dest / f'train_design_attempt{attempt:03d}.npy').exists():
        attempt += 1
    path = dest / f'train_design_attempt{attempt:03d}.npy'
    enough_disk(root, np.prod(identity['shape']) * 4 / 2**30 + 1.)
    ds = SLBRDataset(root, 'TRAIN')
    matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(identity['shape']))
    for start in range(0, len(ds), 32):
        batch = collate([ds[i] for i in range(start, min(start + 32, len(ds)))])
        matrix[start:start + len(batch['X'])] = model.design(**{k: batch[k] for k in INPUT_KEYS})
    matrix.flush()
    release.write_json(marker, dict(identity, file=path.name, sha256=phase.sha(path)))
    return np.load(path, mmap_mode='r')


class ClassicalPredictor(torch.nn.Module):
    def __init__(self, estimator):
        super().__init__()
        self.estimator, self.targets = estimator, estimator.targets
        self.config = SimpleNamespace(horizons=(1, 5, 10, 20))

    def forward(self, return_diagnostics=False, **inputs):
        # The legacy evaluator always passes return_diagnostics=False. It must
        # not leak into sklearn's predict signature (legacy wrapper omitted it).
        if return_diagnostics:
            raise ValueError('Classical diagnostics unavailable')
        return torch.from_numpy(self.estimator.predict(**inputs).astype(np.float32))


def train_classical_shared(root, suite, out, model, metadata):
    import joblib
    design = shared_train_design(root, suite, model)
    ds = SLBRDataset(root, 'TRAIN').base
    print('Fitting fixed ' + model.name + ' with per-head TRAIN masks; this can take hours.', flush=True)
    labels_mask = ds.a['train_label_valid_mask'].reshape(len(ds), -1)
    total_groups = len({labels_mask[:, i].tobytes() for i in range(labels_mask.shape[1]) if labels_mask[:, i].any()})
    original_constructor = model.make_estimator
    count = 0
    def progress_constructor():
        nonlocal count
        count += 1
        print(f'{pd.Timestamp.now(tz="UTC").isoformat()} fitting TRAIN mask group {count}/{total_groups}', flush=True)
        return original_constructor()
    model.make_estimator = progress_constructor
    try:
        model.fit_design(design, ds.a['train_targets'], ds.a['train_label_valid_mask'], split='TRAIN')
    finally:
        del model.make_estimator
    print('Fixed classical fit finished; saving checkpoint and evaluating VAL.', flush=True)
    joblib.dump(dict(estimator=model, metadata=metadata), out / 'estimator.joblib', compress=3)
    result = phase.evaluate(ClassicalPredictor(model), phase.loaders(root, 32)['VAL'], torch.device('cpu'))
    bm.save_validation(out, result)
    release.write_json(out / 'training_complete.json', dict(model_id=metadata['model_id'], TRAIN_only=True,
        TEST_numeric_rows_parsed=0, fitted_outputs=model.fitted_outputs.tolist(), estimator_groups=len(model.groups),
        estimator_sha256=phase.sha(out / 'estimator.joblib'), shared_design_cache=str((suite / 'classical_cache').relative_to(root))))


def worker(root, suite, model_id, attempt, device):
    config = release.read_json(suite / 'suite_config.json')
    release.check_files(root, config['source_hashes'])
    if (suite / 'models_frozen.json').exists():
        raise RuntimeError('Models already frozen; further training forbidden in this suite')
    if not any(t['model_id'] == model_id and t['attempt'] == attempt for t in release.read_json(suite / 'tasks.json')):
        raise ValueError('Worker is not a prespecified active attempt')
    # Routing and the classical cache are process-local adapters. No sealed
    # legacy source/config/experiment is edited and the neural loop is identical.
    bm.OUT = suite.relative_to(root)
    bm.train_classical = lambda root, out, model, metadata: train_classical_shared(root, suite, out, model, metadata)
    bm.train(root, model_id, device, attempt)


def evaluate_test(root, suite, model_id, device):
    import joblib
    config, freeze = release.verify_freeze(root, suite, model_id=model_id)
    dataset = suite / 'test_dataset'
    release.verify_dataset(dataset, phase.sha(suite / 'models_frozen.json'))
    record = freeze['checkpoints'][model_id]
    run = root / record['run_path']
    marker = run / 'test_complete.json'
    if marker.exists():
        info = release.read_json(marker)
        if (info['checkpoint_sha256'] != record['checkpoint_sha256'] or
            info['models_frozen_sha256'] != phase.sha(suite / 'models_frozen.json') or
            info['test_dataset_manifest_sha256'] != phase.sha(dataset / 'dataset_manifest.json')):
            raise ValueError('TEST completion checkpoint changed')
        release.check_files(root, info['result_files'])
        return
    attempt = 1
    while (run / f'test_attempt_{attempt:03d}').exists():
        attempt += 1
    out = run / f'test_attempt_{attempt:03d}'
    out.mkdir()
    phase.seed42()
    assets, targets = manifests(root)
    dev = torch.device('cpu' if is_classical(model_id) else device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent fallback')
    checkpoint = root / record['checkpoint_path']
    if is_classical(model_id):
        fitted = joblib.load(checkpoint)['estimator']
        if not fitted.fitted_outputs.all():
            raise ValueError('Unfitted classical output heads; cannot silently evaluate zero predictions')
        model = ClassicalPredictor(fitted)
    else:
        cp = torch.load(checkpoint, map_location=dev, weights_only=False)
        if cp['metadata']['model_id'] != model_id:
            raise ValueError('Checkpoint model mismatch')
        model = make_model(model_id, assets, targets).to(dev)
        model.load_state_dict(cp['model_state_dict'], strict=True)
        model.set_epoch(cp['best_epoch'])
    loader = DataLoader(release.TestDataset(dataset), batch_size=32, shuffle=False, drop_last=False,
                        num_workers=0, collate_fn=collate)
    with torch.no_grad():
        result = phase.evaluate(model, loader, dev)
    result['metrics']['TEST_FOUR_HORIZON_MACRO_MAE'] = result['metrics'].pop('VAL_SELECTION_SCORE')
    phase.write_csv(out / 'test_metrics.csv', [result['metrics']])
    phase.write_csv(out / 'test_per_horizon_metrics.csv', result['horizons'])
    phase.write_csv(out / 'test_per_target_horizon_metrics.csv', result['targets'])
    np.savez_compressed(out / 'test_predictions.npz', prediction=result['prediction'], target=result['Y'],
        label_valid_mask=result['mask'], origin_dates=np.array(result['origin_dates']))
    if phase.sha(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during TEST evaluation')
    release.write_json(marker, dict(model_id=model_id, split='TEST', checkpoint_sha256=record['checkpoint_sha256'],
        models_frozen_sha256=phase.sha(suite / 'models_frozen.json'),
        test_dataset_manifest_sha256=phase.sha(dataset / 'dataset_manifest.json'),
        result_path=str(out.relative_to(root)), TEST_used_for_selection=False,
        result_files={str(p.relative_to(root)): phase.sha(p) for p in sorted(out.iterdir())}))


def summarize(root, suite):
    if not (suite / 'models_frozen.json').exists() or not (suite / 'test_dataset/dataset_manifest.json').exists():
        print('TEST results are not ready: training freeze or TEST dataset preparation is unfinished. Use --status.', flush=True)
        return False
    config, freeze = release.verify_freeze(root, suite)
    release.verify_dataset(suite / 'test_dataset', phase.sha(suite / 'models_frozen.json'))
    rows, details = [], []
    missing = []
    for model_id in config['models']:
        record = freeze['checkpoints'][model_id]
        run = root / record['run_path']
        marker = run / 'test_complete.json'
        if not marker.exists():
            missing.append(model_id)
            continue
        info = release.read_json(marker)
        if (info['checkpoint_sha256'] != record['checkpoint_sha256'] or
            info['models_frozen_sha256'] != phase.sha(suite / 'models_frozen.json') or
            info['test_dataset_manifest_sha256'] != phase.sha(suite / 'test_dataset/dataset_manifest.json')):
            raise ValueError('TEST result provenance mismatch')
        release.check_files(root, info['result_files'])
        out = root / info['result_path']
        horizon = pd.read_csv(out / 'test_per_horizon_metrics.csv')
        metrics = pd.read_csv(out / 'test_metrics.csv').iloc[0]
        row = dict(model_id=model_id, family=model_id.split('/')[0], seed=42,
            best_epoch=record['best_epoch'], val_selection_score=record['val_selection_score'],
            test_four_horizon_macro_MAE=float(metrics['TEST_FOUR_HORIZON_MACRO_MAE']),
            test_micro_MAE=float(metrics['micro_MAE']), test_origins=int(metrics['origin_count']),
            test_valid_labels=int(metrics['valid_labels']), status='COMPLETE')
        for r in horizon.to_dict('records'):
            row[f"test_{int(r['horizon'])}d_macro_MAE"] = r['macro_MAE']
            row[f"test_{int(r['horizon'])}d_valid_labels"] = r['valid_label_count']
        rows.append(row)
        details.append(pd.read_csv(out / 'test_per_target_horizon_metrics.csv').assign(model_id=model_id))
    if rows:
        phase.write_csv(suite / 'test_summary.csv', rows)
        pd.concat(details, ignore_index=True).to_csv(suite / 'test_per_target_horizon.csv', index=False)
    release.write_json(suite / 'result_status.json', dict(complete=len(rows), expected=len(config['models']),
        all_complete=not missing, missing_models=missing, unit='decimal_return',
        TEST_used_for_selection=False, CONFIRMATORY_RESEARCH_READY=False))
    if rows:
        cols = ['model_id'] + [f'test_{h}d_macro_MAE' for h in (1, 5, 10, 20)]
        print(pd.DataFrame(rows)[cols].to_string(index=False), flush=True)
    if missing:
        print('TEST NOT COMPLETE: ' + ', '.join(missing), flush=True)
    return not missing


def status(root, suite):
    config = release.read_json(suite / 'suite_config.json')
    tasks = release.read_json(suite / 'tasks.json')
    print(f"{suite}\nPrespecified models: {len(config['models'])}; TEST authorized: {config['test_authorized']}")
    for task in tasks:
        print(f"{task['model_id']:42} {task['status']:12} {task.get('pid', '')} {task['log_path']}")
    if (suite / 'test_tasks.json').exists():
        print('TEST evaluation:')
        for task in release.read_json(suite / 'test_tasks.json'):
            print(f"{task['model_id']:42} {task['status']:12} {task.get('pid', '')} {task['log_path']}")
    if (suite / 'result_status.json').exists():
        print(json.dumps(release.read_json(suite / 'result_status.json'), ensure_ascii=False))


def choose_pending(pending, preferred):
    return next((t for t in pending if t['model_id'].startswith(preferred + '/')), pending[0] if pending else None)


def pid_alive(task):
    if not task.get('pid'):
        return False
    try:
        return Path(f"/proc/{task['pid']}/stat").read_text().split()[21] == task.get('process_start_ticks')
    except FileNotFoundError:
        return False


def schedule(root, suite, tasks, config, *, testing=False):
    """One task per selected GPU, one classical task on CPU. Bounded queues."""
    active = {}
    def save():
        release.write_json(suite / ('test_tasks.json' if testing else 'tasks.json'), tasks)
    save()
    try:
        while True:
            for key, (process, task, handle) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                handle.close()
                task['exit_code'] = rc
                ok = (root / task['run_path'] / 'test_complete.json').exists() if testing else completed(root, task) is not None
                task['status'] = 'COMPLETE' if rc == 0 and ok else 'FAILED'
                del active[key]
                save()
                print(f"{'TEST' if testing else 'TRAIN'} {task['status']} {task['model_id']} (exit {rc})", flush=True)
            pending = [t for t in tasks if t['status'] == 'PENDING']
            if not pending and not active:
                break
            available = gpu_free_mib() if any(not is_classical(t['model_id']) for t in pending) else {}
            for index, gpu in enumerate(config['gpus']):
                if gpu in active or available.get(gpu, 0) < config['min_free_gib'] * 1024:
                    continue
                queue = [t for t in pending if not is_classical(t['model_id'])]
                task = choose_pending(queue, 'baseline' if index % 2 == 0 else 'ablation')
                if task:
                    active[gpu] = launch(root, suite, task, gpu, testing)
                    pending.remove(task)
                    save()
            if 'cpu' not in active:
                task = next((t for t in pending if is_classical(t['model_id'])), None)
                if task:
                    active['cpu'] = launch(root, suite, task, None, testing)
                    save()
            print(f"{'TEST' if testing else 'TRAIN'} queue: running={len(active)}, pending={sum(t['status']=='PENDING' for t in tasks)}, complete={sum(t['status']=='COMPLETE' for t in tasks)}, free_gpu_MiB={available}", flush=True)
            time.sleep(30)
    except BaseException:
        # Preserve all evidence. Interrupt children, wait for exit, and mark the
        # attempts interrupted; explicit --retry-failed starts fresh new folders.
        for process, task, handle in active.values():
            process.terminate()
        for process, task, handle in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
            task['status'] = 'INTERRUPTED'
        save()
        raise
    if any(t['status'] != 'COMPLETE' for t in tasks):
        raise RuntimeError('Some jobs failed. Successful artifacts preserved; inspect logs before --resume --retry-failed.')


def launch(root, suite, task, gpu, testing):
    enough_disk(root)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    command = [sys.executable, '-B', '-u', '-m', 'triweave.experiment_suite',
               '--eval-worker' if testing else '--worker', '--run-id', suite.name,
               '--model', task['model_id'], '--attempt', task['attempt'],
               '--device', 'cpu' if gpu is None else 'cuda:0']
    handle = (root / task['log_path']).open('a')
    process = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
    task.update(status='RUNNING', pid=process.pid,
        process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
        device='cpu' if gpu is None else f'physical_GPU_{gpu}')
    print('START ' + task['model_id'] + ' -> ' + task['device'] + ' log=' + task['log_path'], flush=True)
    return process, task, handle


def new_task(root, suite, model_id, number=1, testing=False):
    attempt = model_id.replace('/', '__') + f'__attempt{number:03d}'
    return dict(model_id=model_id, attempt=attempt, status='PENDING',
        run_path=str((suite / 'runs' / attempt).relative_to(root)),
        log_path=str((suite / 'logs' / (attempt + ('__test' if testing else '') + '.log')).relative_to(root)))


def run(root, args):
    suite = suite_path(root, args.run_id)
    if not args.with_test:
        raise ValueError('--run requires explicit --with-test')
    if not args.resume:
        if suite.exists():
            raise FileExistsError('Existing run; use --resume or a new --run-id')
        preflight = validate_preflight(root)
        model_ids = list(args.models or MODEL_IDS)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError('Duplicate model IDs')
        free = gpu_free_mib() if any(not is_classical(m) for m in model_ids) else {}
        if any(not is_classical(m) for m in model_ids) and any(g not in free for g in args.gpus):
            raise ValueError('Requested GPU index unavailable')
        enough_disk(root, 4. if any(is_classical(m) for m in model_ids) else 2.)
        suite.mkdir(parents=True)
        (suite / 'logs').mkdir()
        gate = bm.dataset_gate(root, suite)
        config = dict(run_id=args.run_id, models=model_ids, gpus=args.gpus, min_free_gib=args.min_free_gib,
            seed=42, test_authorized=True, test_authorization=AUTHORIZATION,
            source_hashes=workflow_hashes(root), dataset_manifest_sha256=phase.sha(root / DATA / 'dataset_manifest.json'),
            training_protocol=bm.training_protocol(), test_split=['2025-01-01', '2026-09-30'],
            TEST_policy='Legacy TEST=PROHIBITED applies only to fitting/selection; this outer workflow explicitly authorizes fixed-checkpoint final evaluation',
            evaluate_only_after_all_models_frozen=True, dataset_gate=gate,
            retry_policy='Explicit retries start fresh attempts and preserve interrupted files',
            limitations=['Exploratory Research23, not original25', 'Historical PIT and roll method unverified'])
        release.write_json(suite / 'suite_config.json', config)
        release.write_json(suite / 'preflight.json', preflight)
        tasks = [new_task(root, suite, m) for m in model_ids]
        release.write_json(suite / 'tasks.json', tasks)
    with suite_lock(suite):
        config = release.read_json(suite / 'suite_config.json')
        release.check_files(root, config['source_hashes'])
        if phase.sha(root / DATA / 'dataset_manifest.json') != config['dataset_manifest_sha256']:
            raise ValueError('Frozen TRAIN/VAL dataset changed')
        tasks = release.read_json(suite / 'tasks.json')
        for ix, task in enumerate(tasks):
            if pid_alive(task):
                raise RuntimeError('Previous worker still alive: ' + str(task['pid']))
            if completed(root, task) is not None:
                task['status'] = 'COMPLETE'
            elif task['status'] != 'PENDING' or (root / task['run_path']).exists():
                if not args.retry_failed or (suite / 'models_frozen.json').exists():
                    raise RuntimeError('Incomplete attempt preserved. Inspect log, then explicitly --resume --retry-failed: ' + task['model_id'])
                number = int(task['attempt'].rsplit('attempt', 1)[1]) + 1
                with (suite / 'attempt_history.jsonl').open('a') as history:
                    history.write(json.dumps(task, ensure_ascii=False) + '\n')
                tasks[ix] = new_task(root, suite, task['model_id'], number)
        release.write_json(suite / 'tasks.json', tasks)
        if not (suite / 'models_frozen.json').exists():
            schedule(root, suite, tasks, config)
        freeze_models(root, suite, config, tasks)
        print('ALL MODELS FROZEN. Preparing separately versioned TEST data using frozen TRAIN scalers.', flush=True)
        release.prepare(root, suite)
        test_tasks = []
        for task in tasks:
            t = dict(task, status='PENDING', log_path=task['log_path'].replace('.log', '__test.log'))
            t.pop('pid', None)
            t.pop('process_start_ticks', None)
            if (root / task['run_path'] / 'test_complete.json').exists():
                t['status'] = 'COMPLETE'
            test_tasks.append(t)
        # Interrupted evaluation children are never silently duplicated.
        if (suite / 'test_tasks.json').exists():
            if any(pid_alive(t) for t in release.read_json(suite / 'test_tasks.json')):
                raise RuntimeError('Previous TEST evaluator is still alive')
        schedule(root, suite, test_tasks, config, testing=True)
        if not summarize(root, suite):
            raise RuntimeError('Incomplete TEST results')
        print('ALL COMPLETE: ' + str(suite / 'test_summary.csv'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('run', 'status', 'results', 'preflight', 'worker', 'eval-worker'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--with-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--gpus', type=lambda value: [int(v) for v in value.split(',')], default=[0, 1])
    parser.add_argument('--min-free-gib', type=float, default=6.)
    parser.add_argument('--models', nargs='+', choices=MODEL_IDS)
    parser.add_argument('--model', choices=MODEL_IDS)
    parser.add_argument('--attempt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path.cwd()
    if args.min_free_gib <= 0 or not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Use unique nonnegative GPU indices and positive minimum free memory')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    if args.preflight:
        validate_preflight(root)
        print('Frozen model source gate PASS; no TEST quotes read.')
        print('Available GPUs (MiB): ' + json.dumps(gpu_free_mib()))
        print(f'Free disk: {shutil.disk_usage(root).free / 2**30:.2f} GiB')
        with tempfile.TemporaryDirectory(prefix='triweave_suite_preflight_') as temporary:
            bm.dataset_gate(root, Path(temporary))
        return
    suite = suite_path(root, args.run_id)
    if args.run:
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Launcher interrupted; artifacts preserved')
        signal.signal(signal.SIGTERM, interrupted)
        run(root, args)
    elif args.status:
        status(root, suite)
    elif args.results:
        if not summarize(root, suite):
            raise SystemExit(1)
    else:
        if not args.model or not args.attempt:
            parser.error('Worker requires --model and --attempt')
        if args.worker:
            worker(root, suite, args.model, args.attempt, args.device)
        else:
            evaluate_test(root, suite, args.model, args.device)


if __name__ == '__main__':
    main()
