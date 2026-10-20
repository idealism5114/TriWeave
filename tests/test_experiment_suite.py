"""TEST release/orchestration tests use synthetic quotes, never archived TEST."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from triweave import after_close as ac, research23 as r23
from triweave import experiment_suite as suite, test_release as release


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    release.write_json(path, value)


def fixture_panel():
    dates = np.array(['2024-12-27', '2024-12-30', '2024-12-31', '2025-01-02', '2025-01-03', '2025-01-06'], dtype='U10')
    assets = [dict(asset_id='FUT_' + c) for c in r23.TARGETS]
    p = np.broadcast_to(np.arange(100., 106.)[:, None], (6, 23)).copy()
    return dict(dates=dates, value=p, valid=np.ones_like(p, bool)), assets


def test_labels_do_not_cross_val_test_boundary_and_keep_per_target_missingness():
    panel, assets = fixture_panel()
    panel['valid'][4, 0] = False
    for split in ('VAL', 'TEST'):
        y, mask, ends, origins = release.labels(panel, assets, split)
        lo, hi = ac.SPLITS[split]
        for i in origins:
            for h, horizon in enumerate(ac.HORIZONS):
                if mask[i, h].any():
                    assert lo <= panel['dates'][i] <= panel['dates'][ends[i, h]] <= hi
                    assert ends[i, h] == i + horizon
                    expected = panel['value'][i + horizon] / panel['value'][i] - 1
                    np.testing.assert_array_equal(y[i, h][mask[i, h]], expected[mask[i, h]].astype(np.float32))
    y, mask, ends, origins = release.labels(panel, assets, 'TEST')
    assert not mask[3, 0, 0] and mask[3, 0, 1:].all()
    assert 3 in origins and not mask[2].any() and not mask[-1].any()
    assert not mask[:, 1:].any()


@pytest.mark.parametrize('split', ['TRAIN', 'VAL'])
def test_label_adapter_exactly_matches_frozen_label_definition(split):
    panel, assets = fixture_panel()
    panel['dates'] = np.array(pd.bdate_range('2022-12-01' if split == 'TRAIN' else '2024-01-02', periods=6).strftime('%Y-%m-%d'))
    panel['valid'][2, 4] = False
    original = r23.labels(panel, assets, split)
    adapted = release.labels(panel, assets, split)
    for old, new in zip(original, adapted):
        np.testing.assert_array_equal(old, new)


def test_date_reader_excludes_out_of_range_before_quote_parser(tmp_path):
    path = tmp_path / 'raw.csv'
    path.write_text('date,close\n2024-12-31,100\n2025-01-02,"broken,unclosed\n')
    log = []
    frame = release.range_csv(path, '2011-10-01', ac.END, log)
    assert frame.close.tolist() == ['100'] and log[0]['TEST_numeric_rows_parsed'] == 0
    with pytest.raises(ValueError):
        release.range_csv(path, '2011-10-01', '2026-10-01')
    path.write_text('名称,日期,价格\n曲线,20250102,123\n')
    log = []
    frame = release.range_csv(path, '2025-01-01', '2025-01-03', log)
    assert frame['日期'].tolist() == ['2025-01-02'] and log[0]['TEST_numeric_rows_parsed'] == 1


def test_extension_requires_authorization_before_any_reader(tmp_path):
    with pytest.raises(PermissionError):
        release.load_sources(tmp_path, [], '2026-09-30')
    with pytest.raises(ValueError):
        r23.raw_csv(tmp_path / 'does_not_exist', end='2025-01-02')


def test_private_alignment_does_not_mutate_frozen_globals_and_future_quotes_do_not_change_prefix():
    assets = [dict(asset_id='fixture', economic_type=ac.KINDS[3])]
    dates = ['2024-12-30', '2024-12-31', '2025-01-02', '2025-01-03']
    frame = pd.DataFrame(dict(date=dates, close=[100., 101., 102., 103.],
        open=[100., 101., 102., 103.], high=[101., 102., 103., 104.],
        low=[99., 100., 101., 102.], __source=[0]*4))
    protocol = dict(publication_assumptions={ac.KINDS[3]: dict(local_time='19:00:00', reference_session_delay=0, carry=False)})
    frames = dict(fixture=frame)
    old, _ = ac.align_sources(frames, assets, dates, protocol)
    extended, _ = release.align_sources(frames, assets, dates, protocol, '2025-01-03')
    assert ac.END == '2024-12-31' and r23.ac is ac
    assert len(old['dates']) == 2 and len(extended['dates']) == 4
    for key in old:
        np.testing.assert_array_equal(old[key], extended[key][:2])
    changed = frame.copy()
    changed.loc[2:, ['open', 'high', 'low', 'close']] *= 100
    perturbed, _ = release.align_sources(dict(fixture=changed), assets, dates, protocol, '2025-01-03')
    for key in old:
        np.testing.assert_array_equal(extended[key][:2], perturbed[key][:2])


def test_all_model_freeze_requires_every_complete_checkpoint(tmp_path, monkeypatch):
    destination = tmp_path / 'suite'
    destination.mkdir()
    config = dict(models=['baseline/GRU', 'ablation/full'], source_hashes={}, dataset_manifest_sha256='dataset')
    save(destination / 'suite_config.json', config)
    tasks = [dict(model_id=m) for m in config['models']]
    monkeypatch.setattr(suite, 'completed', lambda root, task: None if task['model_id'] == 'ablation/full' else {'sha': 'fixture'})
    with pytest.raises(RuntimeError, match='All-model barrier'):
        suite.freeze_models(tmp_path, destination, config, tasks)
    assert not (destination / 'models_frozen.json').exists()
    monkeypatch.setattr(suite, 'completed', lambda root, task: dict(sha=task['model_id']))
    frozen = suite.freeze_models(tmp_path, destination, config, tasks)
    assert frozen['TEST_used_for_selection'] is False
    assert frozen['all_training_finished_before_TEST_open']
    suite.freeze_models(tmp_path, destination, config, tasks)
    monkeypatch.setattr(suite, 'completed', lambda root, task: dict(sha='changed'))
    with pytest.raises(ValueError, match='cannot change models'):
        suite.freeze_models(tmp_path, destination, config, tasks)


def test_verify_freeze_rejects_permission_and_tampered_checkpoint(tmp_path):
    destination = tmp_path / 'suite'
    data = tmp_path / suite.DATA / 'dataset_manifest.json'
    save(data, {})
    run = destination / 'runs/fixture'
    save(run / 'config.json', {})
    save(run / 'training_complete.json', {})
    checkpoint = run / 'best.pt'
    checkpoint.write_bytes(b'fixture checkpoint')
    config = dict(test_authorized=True, models=['ablation/full'], source_hashes={})
    freeze = dict(models=config['models'], source_hashes={}, dataset_manifest_sha256=release.sha(data),
        checkpoints={'ablation/full': dict(run_path=str(run.relative_to(tmp_path)),
            checkpoint_path=str(checkpoint.relative_to(tmp_path)), checkpoint_sha256=release.sha(checkpoint),
            config_sha256=release.sha(run / 'config.json'), training_complete_sha256=release.sha(run / 'training_complete.json'))})
    save(destination / 'suite_config.json', config)
    freeze['experiment_config_sha256'] = release.sha(destination / 'suite_config.json')
    save(destination / 'models_frozen.json', freeze)
    release.verify_freeze(tmp_path, destination)
    config['test_authorized'] = False
    save(destination / 'suite_config.json', config)
    with pytest.raises(PermissionError):
        release.verify_freeze(tmp_path, destination)
    config['test_authorized'] = True
    save(destination / 'suite_config.json', config)
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Fingerprint'):
        release.verify_freeze(tmp_path, destination)


def test_classical_evaluation_wrapper_consumes_diagnostic_keyword():
    class FixedEstimator:
        targets = ['fixture']
        def predict(self, **inputs):
            assert set(inputs) == set(suite.INPUT_KEYS)
            return np.ones((2, 4, 1))
    model = suite.ClassicalPredictor(FixedEstimator())
    result = model(**{k: torch.zeros(2, 1) for k in suite.INPUT_KEYS}, return_diagnostics=False)
    assert result.shape == (2, 4, 1) and result.dtype == torch.float32
    with pytest.raises(ValueError):
        model(return_diagnostics=True)


def test_real_ridge_fit_shared_cache_and_masked_validation_roundtrip(tmp_path, monkeypatch):
    import joblib
    from triweave.models.benchmark.classical import MaskedClassical
    assets = [dict(asset_id=str(i)) for i in range(311)]
    model = MaskedClassical('Ridge', assets, ['fixture'])
    y = np.arange(5 * 4, dtype=np.float32).reshape(5, 4, 1) / 1000
    label_mask = np.ones_like(y, bool)
    label_mask[0, 1] = False
    class Fixture:
        def __init__(self):
            self.base = self
            self.a = dict(train_targets=y, train_label_valid_mask=label_mask)
        def __len__(self):
            return 5
        def __getitem__(self, i):
            x = np.zeros((20, 311, 26), np.float32)
            x[:, 0, 0] = i / 10
            return dict(X=x, input_valid_mask=np.ones((20, 311), bool),
                feature_valid_mask=np.ones_like(x, bool), feature_applicable_mask=np.ones((311, 26), bool),
                Y=y[i], label_valid_mask=label_mask[i], origin_date=f'2024-01-{i+2:02d}')
    ds = Fixture()
    monkeypatch.setattr(suite, 'SLBRDataset', lambda *args: ds)
    monkeypatch.setattr(suite.phase, 'loaders', lambda *args: dict(VAL=[suite.collate([ds[0], ds[1]])]))
    save(tmp_path / suite.DATA / 'dataset_manifest.json', {})
    destination = tmp_path / 'suite'
    destination.mkdir()
    run = destination / 'run'
    run.mkdir()
    suite.train_classical_shared(tmp_path, destination, run, model, dict(model_id='baseline/Ridge'))
    marker = release.read_json(run / 'training_complete.json')
    assert marker['TRAIN_only'] and marker['estimator_groups'] == 2 and all(marker['fitted_outputs'])
    assert marker['TEST_numeric_rows_parsed'] == 0
    restored = joblib.load(run / 'estimator.joblib')['estimator']
    batch = suite.collate([ds[2]])
    inputs = {k: batch[k] for k in suite.INPUT_KEYS}
    np.testing.assert_array_equal(model.predict(**inputs), restored.predict(**inputs))
    assert 'make_estimator' not in restored.__dict__
    matrix = suite.shared_train_design(tmp_path, destination, MaskedClassical('XGBoost', assets, ['fixture']))
    assert matrix.shape == (5, 161720) and len(list((destination / 'classical_cache').glob('*.npy'))) == 1
    assert np.isfinite(pd.read_csv(run / 'validation_metrics.csv').VAL_SELECTION_SCORE).all()


def test_parallel_queue_prefers_different_families_and_classicals_stay_cpu():
    tasks = [dict(model_id=m) for m in ('ablation/full', 'baseline/GRU', 'baseline/Ridge')]
    neural = [t for t in tasks if not suite.is_classical(t['model_id'])]
    assert suite.choose_pending(neural, 'baseline')['model_id'] == 'baseline/GRU'
    assert suite.choose_pending(neural, 'ablation')['model_id'] == 'ablation/full'
    assert suite.is_classical('baseline/Ridge') and not suite.is_classical('ablation/full')


def test_scheduler_bounded_workers_and_failure_blocks_test(tmp_path, monkeypatch):
    destination = tmp_path / 'suite'
    destination.mkdir()
    (destination / 'logs').mkdir()
    ids = ['ablation/full', 'baseline/GRU', 'baseline/Ridge', 'ablation/no_graph']
    tasks = [suite.new_task(tmp_path, destination, m) for m in ids]
    launched = []
    class Process:
        def poll(self):
            return 1 if self.model == 'baseline/Ridge' else 0
    def launch(root, target, task, gpu, testing):
        launched.append((task['model_id'], gpu))
        task['status'] = 'RUNNING'
        process = Process()
        process.model = task['model_id']
        return process, task, open(tmp_path / (task['attempt'] + '.log'), 'w')
    monkeypatch.setattr(suite, 'launch', launch)
    monkeypatch.setattr(suite, 'gpu_free_mib', lambda: {0: 10000, 1: 10000})
    monkeypatch.setattr(suite, 'completed', lambda root, task: {} if task['model_id'] != 'baseline/Ridge' else None)
    monkeypatch.setattr(suite.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='Some jobs failed'):
        suite.schedule(tmp_path, destination, tasks, dict(gpus=[0, 1], min_free_gib=6))
    assert launched[:3] == [('baseline/GRU', 0), ('ablation/full', 1), ('baseline/Ridge', None)]
    assert tasks[2]['status'] == 'FAILED'
    assert all(t['status'] == 'COMPLETE' for i, t in enumerate(tasks) if i != 2)
    assert not (destination / 'test_dataset').exists()


def test_concurrent_launchers_are_rejected(tmp_path):
    with suite.suite_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another launcher'):
            with suite.suite_lock(tmp_path):
                pass


def test_no_implicit_test_or_reuse_existing_run(tmp_path):
    with pytest.raises(ValueError, match='explicit --with-test'):
        suite.run(tmp_path, SimpleNamespace(run_id='fixture', with_test=False))
    destination = suite.suite_path(tmp_path, 'fixture')
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        suite.run(tmp_path, SimpleNamespace(run_id='fixture', with_test=True, resume=False))


def test_test_evaluation_restores_frozen_checkpoint_and_summarizes_without_training(tmp_path, monkeypatch):
    destination = tmp_path / 'suite'
    run = destination / 'runs/fixture'
    run.mkdir(parents=True)
    save(destination / 'models_frozen.json', {})
    save(destination / 'test_dataset/dataset_manifest.json', {})
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.01))
            self.targets = ['fixture']
            self.config = SimpleNamespace(horizons=(1, 5, 10, 20))
        def set_epoch(self, epoch):
            assert epoch == 7
        def forward(self, X, return_diagnostics=False, **kwargs):
            return self.weight.expand(len(X), 4, 1)
    checkpoint = run / 'best.pt'
    torch.save(dict(model_state_dict=Model().state_dict(), metadata=dict(model_id='ablation/full'), best_epoch=7), checkpoint)
    record = dict(run_path=str(run.relative_to(tmp_path)), checkpoint_path=str(checkpoint.relative_to(tmp_path)),
        checkpoint_sha256=release.sha(checkpoint), best_epoch=7, val_selection_score=.02)
    config = dict(models=['ablation/full'])
    frozen = dict(checkpoints={'ablation/full': record})
    monkeypatch.setattr(release, 'verify_freeze', lambda *args, **kwargs: (config, frozen))
    monkeypatch.setattr(release, 'verify_dataset', lambda *args: {})
    monkeypatch.setattr(suite, 'manifests', lambda root: ([], ['fixture']))
    monkeypatch.setattr(suite, 'make_model', lambda *args: Model())
    batch = {k: torch.zeros(2, 1) for k in suite.INPUT_KEYS}
    batch.update(Y=torch.zeros(2, 4, 1), label_valid_mask=torch.ones(2, 4, 1, dtype=torch.bool), origin_date=['2025-01-02', '2025-01-03'])
    monkeypatch.setattr(suite, 'DataLoader', lambda *args, **kwargs: [batch])
    monkeypatch.setattr(release, 'TestDataset', lambda path: None)
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *args: pytest.fail('TEST optimizer step'))
    suite.evaluate_test(tmp_path, destination, 'ablation/full', 'cpu')
    assert release.sha(checkpoint) == record['checkpoint_sha256']
    marker = release.read_json(run / 'test_complete.json')
    assert marker['split'] == 'TEST' and not marker['TEST_used_for_selection']
    assert suite.summarize(tmp_path, destination)
    frame = pd.read_csv(destination / 'test_summary.csv')
    assert frame.best_epoch.iloc[0] == 7
    np.testing.assert_allclose(frame[[f'test_{h}d_macro_MAE' for h in (1, 5, 10, 20)]], .01)
    assert 'VAL_SELECTION_SCORE' not in pd.read_csv(tmp_path / marker['result_path'] / 'test_metrics.csv').columns
    suite.evaluate_test(tmp_path, destination, 'ablation/full', 'cpu')
    assert len(list(run.glob('test_attempt_*'))) == 1
    (tmp_path / marker['result_path'] / 'test_metrics.csv').write_text('tampered')
    with pytest.raises(ValueError, match='Fingerprint'):
        suite.summarize(tmp_path, destination)


@pytest.mark.parametrize('historical_revision', [False, True])
def test_full_test_release_uses_frozen_scaler_exact_prefix_and_boundary_masks(tmp_path, monkeypatch, historical_revision):
    """Real feature pipeline on 311 synthetic assets; no archived TEST opened."""
    from triweave.assets import YIELD_SPECS, INDEX_SPECS
    assets = []
    for ids, kind in [(list('STOCK_' + str(i) for i in range(263)), ac.KINDS[0]),
                      (list(YIELD_SPECS), ac.KINDS[1]), (list(INDEX_SPECS), ac.KINDS[2]),
                      (['FUT_' + c for c in r23.TARGETS], ac.KINDS[3])]:
        assets += [dict(index=len(assets) + j, asset_id=asset, economic_type=kind) for j, asset in enumerate(ids)]
    dates = list(pd.bdate_range('2022-12-01', periods=22).strftime('%Y-%m-%d'))
    dates += list(pd.bdate_range('2024-11-18', periods=32).strftime('%Y-%m-%d'))
    dates += list(pd.bdate_range('2025-01-02', periods=45).strftime('%Y-%m-%d'))
    calendar = dates + ['2025-03-06']
    protocol = dict(normalization=dict(minimum_train_count=1, epsilon=1e-8), publication_assumptions={
        kind: dict(local_time='19:00:00', reference_session_delay=1 if kind in ac.KINDS[1:3] else 0,
                   carry=kind in ac.KINDS[1:3], max_observation_age_calendar_days=5) for kind in ac.KINDS})
    frames = {}
    for spec in assets:
        p = np.linspace(100., 120., len(dates))
        frame = pd.DataFrame(dict(date=dates, open=p, high=p+1, low=p-1, close=p,
            value=p, volume=p*100, amount=p*1000, turnover=p/100, open_interest=p*10,
            trading_status=['1']*len(dates), __source=np.zeros(len(dates), int)))
        frames[spec['asset_id']] = frame
    old, _ = ac.align_sources(frames, assets, calendar, protocol)
    rules = [dict(asset_id='FUT_SN', date=dates[10])]
    old, accepted = r23.apply_quarantine(old, assets, rules)
    x, m, app = ac.feature_panel(old, assets)
    params = ac.fit_scaler(x, m, old['dates'], assets, protocol)
    z, fm = ac.transform(x, m, assets, params)
    data = tmp_path / suite.DATA
    data.mkdir(parents=True)
    arrays = dict(features=z, feature_valid_mask=fm, feature_applicable_mask=app, input_valid_mask=old['valid'],
        reference_dates=old['dates'], decision_time=old['decision_time'], source_observation_acceptance_mask=accepted,
        unscaled_features=x, unscaled_feature_valid_mask=m)
    for k in ('observation_day', 'available_at', 'source_index', 'age_days', 'new_observation', 'trading_status'):
        arrays[k] = old[k]
    for split in ('TRAIN', 'VAL'):
        y, lm, ends, origins = r23.labels(old, assets, split)
        prefix = split.lower() + '_'
        for k, a in dict(targets=y[origins], label_valid_mask=lm[origins], origin_indices=origins,
            anchor_indices=origins, endpoint_indices=ends[origins],
            window_indices=np.maximum(origins[:, None] + np.arange(-19, 1), -1).astype(np.int32)).items():
            arrays[prefix+k] = a
    for k, a in arrays.items():
        np.save(data / (k + '.npy'), a)
    for name, doc in [('scaler_params', params), ('asset_manifest', assets), ('target_manifest', {}),
                      ('feature_registry', ac.registry()), ('source_quarantine_manifest', rules), ('dataset_manifest', {})]:
        save(data / (name + '.json'), doc)
    source_hashes = {p.name: release.sha(p) for p in data.iterdir()}
    calfile = tmp_path / 'data/manifests/baseline15y/reference_trading_calendar.csv'
    calfile.parent.mkdir(parents=True)
    pd.DataFrame(dict(date=calendar)).to_csv(calfile, index=False)
    destination = tmp_path / 'suite'
    save(destination / 'models_frozen.json', {})
    monkeypatch.setattr(release, 'verify_freeze', lambda *args: (
        dict(test_authorization='Synthetic test fixture only'), dict(dataset_manifest_sha256=release.sha(data / 'dataset_manifest.json'))))
    monkeypatch.setattr(r23, 'definitions', lambda root: (protocol, assets))
    def sources(root, a, end, *, authorized=False):
        assert authorized and end == '2026-09-30'
        return frames, [], [], []
    monkeypatch.setattr(release, 'load_sources', sources)
    monkeypatch.setattr(ac, 'fit_scaler', lambda *args: pytest.fail('TEST must not refit scaler'))
    if historical_revision:
        frames['FUT_LU'].loc[3, ['open', 'high', 'low', 'close']] += 5
        with pytest.raises(ValueError, match='changed frozen TRAIN/VAL'):
            release.prepare(tmp_path, destination)
        assert (destination / 'test_dataset/failed_prefix_audit.json').exists()
        assert not (destination / 'test_dataset/dataset_manifest.json').exists()
    else:
        manifest = release.prepare(tmp_path, destination)
        assert manifest['origin_count'] == 44 and not manifest['scaler_refitted']
        ds = release.TestDataset(destination / 'test_dataset')
        sample = ds[0]
        assert sample['origin_date'] == '2025-01-02' and sample['X'].shape == (20, 311, 26)
        assert sample['label_valid_mask'].all() and np.isfinite(sample['X']).all()
        assert not ds.a['test_label_valid_mask'][-1, 1:].any()
        for i in ds.a['test_origin_indices']:
            assert str(ds.a['reference_dates'][i]) >= '2025-01-01'
        assert all(release.read_json(destination / 'test_dataset/audit.json')['prefix_equal'].values())
        assert release.sha(data / 'scaler_params.json') == release.sha(destination / 'test_dataset/scaler_params.json')
        assert release.prepare(tmp_path, destination) == manifest
    assert source_hashes == {p.name: release.sha(p) for p in data.iterdir()}
