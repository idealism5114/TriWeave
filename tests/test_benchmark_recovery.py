"""Numerical fixes, exact unaffected identities and frozen recovery provenance."""
from pathlib import Path
import json

import pytest
import torch
from torch.nn.utils.parametrizations import weight_norm

from triweave.models.benchmark import make_model as original
from triweave.models.benchmark.crossgnn import CrossGNNCore
from triweave.models.benchmark_recovery import DeterministicCrossGNNCore, StableWeightNorm, make_model, NORM_FLOOR
from triweave.slbr_data import manifests, INPUT_KEYS, SLBRDataset, collate
from triweave.models.slbr_triweave import masked_huber

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('seed', [0, 42])
def test_crossgnn_threshold_graph_and_gradients_equal_original_with_ties_missing(seed):
    torch.manual_seed(seed)
    old = CrossGNNCore()
    fixed = DeterministicCrossGNNCore()
    fixed.load_state_dict(old.state_dict())
    valid = torch.rand(3, 40, 27) > .2
    valid[0] = False
    a = old.variable_adjacency(valid)
    b = fixed.variable_adjacency(valid)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward(); b.square().sum().backward()
    for name in ('nodevec1', 'nodevec2'):
        torch.testing.assert_close(getattr(old, name).grad, getattr(fixed, name).grad, rtol=0, atol=0)


def test_safe_weightnorm_matches_formula_and_gradients_above_floor():
    torch.manual_seed(42)
    old = weight_norm(torch.nn.Conv1d(4, 3, 3))
    g = old.parametrizations.weight.original0
    v = old.parametrizations.weight.original1
    g2 = g.detach().clone().requires_grad_(); v2 = v.detach().clone().requires_grad_()
    a, b = old.weight, StableWeightNorm()(g2, v2)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(g.grad, g2.grad, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(v.grad, v2.grad, rtol=0, atol=2e-7)


@pytest.mark.parametrize('value', [0., 1e-45, 1e-30, 1e-13, 1e-8])
def test_safe_weightnorm_zero_and_tiny_directions_have_finite_backward(value):
    g = torch.ones(3, 1, 1, requires_grad=True)
    v = torch.full((3, 4, 3), value, requires_grad=True)
    w = StableWeightNorm()(g, v)
    w.sum().backward()
    assert torch.isfinite(w).all() and torch.isfinite(g.grad).all() and torch.isfinite(v.grad).all()
    expected = g.detach().double() * v.detach().double() / torch.linalg.vector_norm(v.detach().double(), dim=(1, 2), keepdim=True).clamp_min(NORM_FLOOR)
    torch.testing.assert_close(w, expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize('model_id', ['baseline/CrossGNN', 'baseline/TCN', 'ablation/full', 'baseline/GraphWaveNet'])
def test_factory_preserves_parameter_keys_initial_values_and_rng(model_id):
    assets, targets = manifests(ROOT)
    torch.manual_seed(42); a = original(model_id, assets, targets); rng = torch.get_rng_state()
    torch.manual_seed(42); b = make_model(model_id, assets, targets)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(a.state_dict()) == list(b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0, atol=0)


def test_real_tcn_failure_snapshot_fixed_backward_preserves_masks():
    path = ROOT / 'experiments/phase_g_benchmark_test/bench_all37_seed42_v1/runs/baseline__TCN__attempt001/nonfinite_epoch25.pt'
    dump = torch.load(path, map_location='cpu', weights_only=False)
    assets, targets = manifests(ROOT)
    model = make_model('baseline/TCN', assets, targets)
    model.load_state_dict(dump['model_state_dict'], strict=True)
    b = {k: v[:2] if isinstance(v, torch.Tensor) else v for k, v in dump['batch'].items()}
    prediction = model(**{k: b[k] for k in INPUT_KEYS})
    loss, _ = masked_huber(prediction, b['Y'], b['label_valid_mask'])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    model.eval()
    changed = {k: b[k].clone() for k in INPUT_KEYS}
    valid = b['feature_valid_mask'] & b['input_valid_mask'][..., None] & b['feature_applicable_mask'][:, None]
    changed['X'][~valid] = float('nan')
    with torch.no_grad():
        torch.testing.assert_close(model(**{k: b[k] for k in INPUT_KEYS}), model(**changed), rtol=0, atol=0)


def test_recovery_inherits_checkpoint_without_changing_parent_and_retrains_only_failures(tmp_path, monkeypatch):
    from triweave import benchmark_recovery as recovery, test_release as release
    parent = recovery.es.suite_path(tmp_path, recovery.PARENT)
    parent.mkdir(parents=True)
    (parent / 'launcher.lock').touch()
    ids = ['baseline/RNN', 'baseline/CrossGNN', 'baseline/TCN']
    cfg = dict(models=ids, source_hashes={}, test_authorization='explicit fixture', dataset_manifest_sha256='fixture')
    release.write_json(parent / 'suite_config.json', cfg)
    tasks = [recovery.es.new_task(tmp_path, parent, m) | dict(status='FAILED' if m != 'baseline/RNN' else 'COMPLETE') for m in ids]
    release.write_json(parent / 'tasks.json', tasks)
    oldrun = tmp_path / tasks[0]['run_path']
    oldrun.mkdir(parents=True)
    (oldrun / 'best.pt').write_bytes(b'checkpoint fixture')
    (oldrun / 'config.json').write_text('{"fixture": true}')
    record = dict(checkpoint_path=str((oldrun / 'best.pt').relative_to(tmp_path)),
                  checkpoint_sha256=release.sha(oldrun / 'best.pt'))
    gate = tmp_path / recovery.AUDIT / 'gpu_probe.json'
    gate.parent.mkdir(parents=True)
    release.write_json(gate, dict(PASS=True, source_hashes={}))
    original_preflight = tmp_path / recovery.bm.OUT / 'preflight.json'
    original_preflight.parent.mkdir(parents=True)
    release.write_json(original_preflight, dict(PASS=True))
    monkeypatch.setattr(recovery.es, 'validate_preflight', lambda root: dict(PASS=True))
    monkeypatch.setattr(recovery, 'model_hashes', lambda root: {})
    monkeypatch.setattr(recovery.es, 'workflow_hashes', lambda root: {})
    monkeypatch.setattr(recovery.es, 'gpu_free_mib', lambda: {0: 10000})
    monkeypatch.setattr(recovery.es, 'pid_alive', lambda task: False)
    monkeypatch.setattr(recovery.es, 'completed', lambda root, task: record if task['model_id'] == 'baseline/RNN' else None)
    before = {str(p.relative_to(parent)): release.sha(p) for p in parent.rglob('*') if p.is_file()}
    dest = recovery.initialize(tmp_path, gpus=[0])
    after = {str(p.relative_to(parent)): release.sha(p) for p in parent.rglob('*') if p.is_file()}
    assert before == after
    newtasks = release.read_json(dest / 'tasks.json')
    assert [t['model_id'] for t in newtasks if t['status'] == 'PENDING'] == ['baseline/CrossGNN', 'baseline/TCN']
    inherited = newtasks[0]
    link = tmp_path / inherited['run_path'] / 'best.pt'
    assert link.is_symlink() and release.sha(link) == record['checkpoint_sha256']
    assert (tmp_path / inherited['run_path'] / 'config.json').read_bytes() == (oldrun / 'config.json').read_bytes()
    assert inherited['run_path'] != tasks[0]['run_path']
    assert all(t['attempt'].endswith('002') for t in newtasks[1:])
    assert release.read_json(dest / 'recovery_provenance.json')['inherited_checkpoint_count'] == 1
    with pytest.raises(FileExistsError):
        recovery.initialize(tmp_path, gpus=[0])
