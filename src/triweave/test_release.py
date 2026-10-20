"""Explicit, versioned TEST release for already frozen Research23 models.

The historical TRAIN/VAL builders and their seals remain unchanged.  Range
adapters execute the exact frozen functions with private globals (never mutate
after_close.END or research23).  No market requests or scaler fitting occurs.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

from . import after_close as ac, research23 as r23
from .slbr_data import DATA, INPUT_KEYS
from .universe import sha256_file as sha

RELEASE = 'TRIWEAVE_23_AFTER_CLOSE_RESEARCH_V1_TEST_RELEASE_V1'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(r23.json_bytes(value))
    temporary.replace(path)


def check_files(root, files):
    for name, expected in files.items():
        digest = expected['sha256'] if isinstance(expected, dict) else expected
        if sha(Path(root) / name) != digest:
            raise ValueError('Fingerprint mismatch: ' + name)


def range_csv(path, start, end, log=None):
    """Date gate before parsing quote columns, including authorized TEST rows."""
    if start < ac.SPLITS['TRAIN'][0] or end > ac.SPLITS['TEST'][1] or start > end:
        raise ValueError('Outside frozen research range')
    records = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        header = next(csv.reader([next(handle)]))
        field = next((k for k in ('date', '日期') if k in header), None)
        if field is None:
            raise ValueError('Missing source date')
        idx = header.index(field)
        for line in handle:
            prefix = line.split(',', idx + 1)
            if len(prefix) <= idx:
                raise ValueError('Invalid date prefix schema')
            token = prefix[idx].strip().strip('"')
            if len(token) >= 10 and token[4] == '-' and token[7] == '-':
                day = token[:10]
            elif len(token) == 8 and token.isdigit():
                day = token[:4] + '-' + token[4:6] + '-' + token[6:]
            else:
                raise ValueError('Invalid source date')
            if not start <= day <= end:
                continue
            pd.Timestamp(day)
            values = next(csv.reader([line]))
            if len(values) != len(header):
                raise ValueError('Invalid single-line CSV')
            row = dict(zip(header, values))
            row[field] = day
            records.append(row)
    frame = pd.DataFrame(records, columns=header)
    if log is not None:
        log.append(dict(path=str(path), parsed_rows=len(frame), date_gate_before_quote_parser=True,
                        TEST_numeric_rows_parsed=int(frame[field].ge(ac.SPLITS['TEST'][0]).sum()),
                        min_parsed_date=frame[field].min() if len(frame) else None,
                        max_parsed_date=frame[field].max() if len(frame) else None))
    return frame


def private_function(fn, **overrides):
    """Reuse frozen code with isolated bindings; no global monkeypatching."""
    return FunctionType(fn.__code__, dict(fn.__globals__, **overrides), fn.__name__, fn.__defaults__, fn.__closure__)


def load_sources(root, assets, end, *, authorized=False):
    if end > ac.END and not authorized:
        raise PermissionError('Explicit TEST release authorization required')
    def reader(path, start=ac.SPLITS['TRAIN'][0], end=end, log=None):
        return range_csv(path, start, end, log)
    # Original raw replay, raw/standardized SHA and numeric/unit comparison are
    # retained verbatim; only the date range and the two CSV readers differ.
    local_ac = SimpleNamespace(END=end, safe_csv=reader)
    return private_function(r23.load_sources, ac=local_ac, raw_csv=reader)(root, assets)


def align_sources(frames, assets, calendar, protocol, end):
    if end > ac.SPLITS['TEST'][1]:
        raise ValueError('Beyond authorized TEST endpoint')
    return private_function(ac.align_sources, END=end)(frames, assets, calendar, protocol)


def labels(aligned, assets, split):
    """Same frozen anchor/horizons/masks; a separate explicit TEST range."""
    if split not in ac.SPLITS:
        raise ValueError(split)
    dates = aligned['dates']
    js = [next(j for j, a in enumerate(assets) if a['asset_id'] == 'FUT_' + c) for c in r23.TARGETS]
    q, p = aligned['valid'][:, js], aligned['value'][:, js]
    y = np.zeros((len(dates), 4, 23), np.float32)
    mask = np.zeros_like(y, bool)
    ends = np.full((len(dates), 4), -1, np.int32)
    lo, hi = ac.SPLITS[split]
    for i in np.flatnonzero((dates >= lo) & (dates <= hi)):
        for hk, horizon in enumerate(ac.HORIZONS):
            endpoint = i + horizon
            if endpoint >= len(dates) or not lo <= dates[endpoint] <= hi:
                continue
            good = q[i:endpoint + 1].all(0)
            values = np.zeros(23)
            np.divide(p[endpoint], p[i], out=values, where=good)
            values -= 1
            good &= np.isfinite(values) & (np.abs(values) <= np.finfo(np.float32).max)
            y[i, hk, good] = values[good].astype(np.float32)
            mask[i, hk] = good
            ends[i, hk] = endpoint
    origins = np.flatnonzero(mask.any((1, 2))).astype(np.int32)
    return y, mask, ends, origins


def verify_freeze(root, suite, model_id=None):
    root, suite = Path(root), Path(suite)
    config = read_json(suite / 'suite_config.json')
    freeze = read_json(suite / 'models_frozen.json')
    if not config['test_authorized'] or freeze['models'] != config['models']:
        raise PermissionError('TEST requires the full prespecified model freeze')
    if freeze['experiment_config_sha256'] != sha(suite / 'suite_config.json'):
        raise ValueError('Frozen experiment configuration changed')
    if freeze['source_hashes'] != config['source_hashes']:
        raise ValueError('Freeze/source mismatch')
    check_files(root, config['source_hashes'])
    if freeze['dataset_manifest_sha256'] != sha(root / DATA / 'dataset_manifest.json'):
        raise ValueError('Training dataset changed')
    if set(freeze['checkpoints']) != set(config['models']):
        raise ValueError('Not all models frozen')
    for selected_id, item in freeze['checkpoints'].items():
        if model_id is not None and selected_id != model_id:
            continue
        run = root / item['run_path']
        check_files(root, {item['checkpoint_path']: item['checkpoint_sha256']})
        if sha(run / 'training_complete.json') != item['training_complete_sha256']:
            raise ValueError('Training completion record changed: ' + selected_id)
        if sha(run / 'config.json') != item['config_sha256']:
            raise ValueError('Training configuration changed: ' + selected_id)
    return config, freeze


def verify_dataset(path, freeze_sha=None):
    path = Path(path)
    manifest = read_json(path / 'dataset_manifest.json')
    if manifest['release_id'] != RELEASE or not manifest['audit_pass']:
        raise ValueError('Invalid TEST release')
    if freeze_sha is not None and manifest['models_frozen_sha256'] != freeze_sha:
        raise ValueError('Different frozen experiment')
    check_files(path, manifest['files'])
    return manifest


def prepare(root, suite):
    """Only called after all prespecified checkpoints have been frozen."""
    root, suite = Path(root), Path(suite)
    config, freeze = verify_freeze(root, suite)
    freeze_sha = sha(suite / 'models_frozen.json')
    dest = suite / 'test_dataset'
    if (dest / 'dataset_manifest.json').exists():
        return verify_dataset(dest, freeze_sha)
    if dest.exists():
        suffix = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')
        preserved = dest.with_name('test_dataset_incomplete_' + suffix)
        dest.rename(preserved)
        print('Preserved incomplete TEST build: ' + str(preserved), flush=True)
    dest.mkdir()
    write_json(dest / 'authorization.json', dict(user_request=config['test_authorization'],
        models_frozen_sha256=freeze_sha, opened_at=pd.Timestamp.now(tz='UTC').isoformat(),
        scope='Fixed-checkpoint descriptive TEST evaluation; no selection, fitting or tuning'))
    protocol, assets = r23.definitions(root)
    frames, sources, logs, replay = load_sources(root, assets, ac.SPLITS['TEST'][1], authorized=True)
    calendar = pd.read_csv(root / 'data/manifests/baseline15y/reference_trading_calendar.csv', dtype=str).date.tolist()
    # Next-session bond availability on the final raw date may fall beyond TEST.
    # Use only an existing archived calendar; never invent additional sessions.
    aligned, quality = align_sources(frames, assets, calendar, protocol, ac.SPLITS['TEST'][1])
    rules = read_json(root / DATA / 'source_quarantine_manifest.json')
    aligned, accepted = r23.apply_quarantine(aligned, assets, rules)
    raw_x, raw_mask, app = ac.feature_panel(aligned, assets)
    params = read_json(root / DATA / 'scaler_params.json')
    z, mask = ac.transform(raw_x, raw_mask, assets, params)  # NEVER fit_scaler
    arrays = dict(features=z, feature_valid_mask=mask, feature_applicable_mask=app,
        input_valid_mask=aligned['valid'], reference_dates=aligned['dates'],
        decision_time=aligned['decision_time'], source_observation_acceptance_mask=accepted)
    for key in ('observation_day', 'available_at', 'source_index', 'age_days', 'new_observation', 'trading_status'):
        arrays[key] = aligned[key]
    # Rebuilding a longer panel cannot revise ANY historical model input or
    # label. Exact prefix comparison is a release gate, not a numeric tolerance.
    historical = np.load(root / DATA / 'reference_dates.npy', mmap_mode='r')
    n = len(historical)
    prefix = {}
    for key, value in arrays.items():
        old = np.load(root / DATA / (key + '.npy'), mmap_mode='r')
        current = value if key == 'feature_applicable_mask' else value[:n]
        prefix[key] = bool(np.array_equal(current, old))
    prefix['unscaled_features'] = bool(np.array_equal(raw_x[:n], np.load(root / DATA / 'unscaled_features.npy', mmap_mode='r')))
    prefix['unscaled_feature_valid_mask'] = bool(np.array_equal(raw_mask[:n], np.load(root / DATA / 'unscaled_feature_valid_mask.npy', mmap_mode='r')))
    for split in ('TRAIN', 'VAL'):
        y, m, ends, origins = labels(aligned, assets, split)
        s = split.lower()
        for key, value in dict(targets=y[origins], label_valid_mask=m[origins], origin_indices=origins,
                               anchor_indices=origins, endpoint_indices=ends[origins],
                               window_indices=np.maximum(origins[:, None] + np.arange(-19, 1), -1).astype(np.int32)).items():
            prefix[s + '_' + key] = bool(np.array_equal(value, np.load(root / DATA / (s + '_' + key + '.npy'), mmap_mode='r')))
    if not all(prefix.values()):
        write_json(dest / 'failed_prefix_audit.json', prefix)
        raise ValueError('TEST extension changed frozen TRAIN/VAL: ' + str([k for k, ok in prefix.items() if not ok]))
    y, m, ends, origins = labels(aligned, assets, 'TEST')
    if not len(origins) or not m[origins].sum((0, 2)).all():
        raise ValueError('Empty TEST split or empty evaluation horizon')
    if not np.isfinite(z).all() or not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite TEST data')
    q = aligned['valid']
    if (aligned['available_at'][q] > np.broadcast_to(aligned['decision_time'][:, None], q.shape)[q]).any():
        raise ValueError('Software availability clock violation')
    # Persist only the last 19 VAL context sessions plus TEST; no second copy of
    # the entire TRAIN feature panel is needed.
    offset = int(np.flatnonzero(aligned['dates'] >= ac.SPLITS['TEST'][0])[0]) - 19
    if offset < 0:
        raise ValueError('Insufficient historical context')
    arrays = {k: v if k == 'feature_applicable_mask' else v[offset:] for k, v in arrays.items()}
    arrays.update(test_targets=y[origins], test_label_valid_mask=m[origins], test_origin_indices=origins - offset,
        test_anchor_indices=origins - offset,
        test_endpoint_indices=np.where(ends[origins] >= 0, ends[origins] - offset, -1).astype(np.int32),
        test_window_indices=(origins[:, None] + np.arange(-19, 1) - offset).astype(np.int32))
    for key, value in arrays.items():
        np.save(dest / (key + '.npy'), value, allow_pickle=False)
    for name in ('asset_manifest.json', 'target_manifest.json', 'feature_registry.json', 'scaler_params.json', 'source_quarantine_manifest.json'):
        (dest / name).write_bytes((root / DATA / name).read_bytes())
    write_json(dest / 'source_manifest.json', sources)
    write_json(dest / 'audit.json', dict(prefix_equal=prefix, scaler_refitted=False,
        scaler_sha256=sha(root / DATA / 'scaler_params.json'), replay=replay, read_log=logs,
        source_quality=quality, TEST_numeric_rows_parsed=sum(x['TEST_numeric_rows_parsed'] for x in logs),
        input_nodes=311, targets=list(r23.TARGETS), horizons=list(ac.HORIZONS),
        label_formula='P(t+h)/P(t)-1', publication_assumptions=protocol['publication_assumptions'],
        HISTORICAL_PIT_VERIFIED=False, FUTURES_ROLL_METHOD_VERIFIED=False, CONFIRMATORY_RESEARCH_READY=False))
    manifest = dict(release_id=RELEASE, parent_protocol_id=r23.PROTOCOL, audit_pass=True,
        models_frozen_sha256=freeze_sha, origin_count=len(origins),
        first_origin=str(aligned['dates'][origins[0]]), last_origin=str(aligned['dates'][origins[-1]]),
        label_counts={str(h): int(m[origins, k].sum()) for k, h in enumerate(ac.HORIZONS)},
        split=list(ac.SPLITS['TEST']), scaler_refitted=False, frozen_dataset_manifest_sha256=freeze['dataset_manifest_sha256'],
        files={p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
    write_json(dest / 'dataset_manifest.json', manifest)
    return verify_dataset(dest, freeze_sha)


class TestDataset:
    """No TRAIN/VAL fitting interface; callers must first verify the release."""
    __test__ = False

    def __init__(self, path):
        self.path = Path(path)
        manifest = read_json(self.path / 'dataset_manifest.json')
        if manifest.get('release_id') != RELEASE or not manifest.get('audit_pass'):
            raise ValueError('TEST dataset not released')
        self.a = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in self.path.glob('*.npy')}

    def __len__(self):
        return len(self.a['test_origin_indices'])

    def __getitem__(self, index):
        a = self.a
        idx = a['test_window_indices'][index]
        if (idx < 0).any():
            raise ValueError('Invalid TEST history window')
        i = int(a['test_origin_indices'][index])
        if not ac.SPLITS['TEST'][0] <= str(a['reference_dates'][i]) <= ac.SPLITS['TEST'][1]:
            raise ValueError('Out-of-split TEST origin')
        sample = dict(X=np.array(a['features'][idx]), input_valid_mask=np.array(a['input_valid_mask'][idx]),
            feature_valid_mask=np.array(a['feature_valid_mask'][idx]), feature_applicable_mask=a['feature_applicable_mask'],
            Y=np.array(a['test_targets'][index]), label_valid_mask=np.array(a['test_label_valid_mask'][index]),
            origin_date=str(a['reference_dates'][i]))
        return sample
