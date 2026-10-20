"""Shared bond weights, parameter reduction and the real mask-aware data path."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from triweave.models.shared_bond import HeterogeneousInputEncoder, SLBRConfig, TriWeaveSLBR
from triweave.models.slbr_triweave import (
    HeterogeneousInputEncoder as OldEncoder,
    TriWeaveSLBR as OldModel,
    masked_huber,
)
from triweave.slbr_data import INPUT_KEYS, SLBRDataset, collate, manifests


torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
TYPES = ('STOCK_PRICE', 'BOND_YIELD', 'BOND_WEALTH_INDEX', 'FUTURES_CONTINUOUS_PRICE')


@pytest.mark.parametrize('temporal,width', [(False, 64), (True, 32)])
def test_bond_economic_types_use_identical_parameters(temporal, width):
    encoder = HeterogeneousInputEncoder(TYPES, 26, width, temporal=temporal)
    assert encoder.type_indices.tolist() == [0, 1, 1, 2]
    assert len(encoder.projections) == len(encoder.presence) == len(encoder.norms) == 3
    x = torch.randn(2, 3, 4, 26)
    x[:, :, 2] = x[:, :, 1]
    mask = torch.ones_like(x, dtype=torch.bool)
    out, valid = encoder(x, mask.any(-1), mask, torch.ones(4, 26, dtype=torch.bool))
    torch.testing.assert_close(out[:, :, 1], out[:, :, 2], rtol=0, atol=0)
    out.sum().backward()
    assert encoder.projections[1].weight.grad is not None
    assert valid.all() and torch.isfinite(out).all()


def test_fewer_parameters_and_unchanged_nonencoder_initialization():
    assets, targets = manifests(ROOT)
    torch.manual_seed(42)
    old = OldModel(assets, targets)
    torch.manual_seed(42)
    new = TriWeaveSLBR(assets, targets)
    assert old.parameter_counts()['total_params'] == 534909
    assert new.parameter_counts()['total_params'] == 529757
    assert new.parameter_counts()['heterogeneous_encoder'] == 15456
    assert sum(new.parameter_counts()[k] for k in new.parameter_counts() if k != 'total_params') == 529757
    for name, p in new.named_parameters():
        if '.encoder.' not in name:
            torch.testing.assert_close(p, dict(old.named_parameters())[name], rtol=0, atol=0)
    for encoder in (new.spatial_branch.encoder, new.temporal_branch.market_encoder.encoder):
        assert torch.bincount(encoder.type_indices).tolist() == [263, 25, 23]
    assert new.assets == old.assets  # Metadata and economic units remain intact.
    assert new.config.name == SLBRConfig().name
    assert new.forward.__func__ is old.forward.__func__


@pytest.mark.parametrize('all_missing', [False, True])
def test_invalid_values_and_missing_bond_market_cannot_change_predictions(all_missing):
    assets, targets = manifests(ROOT)
    model = TriWeaveSLBR(assets, targets).eval()
    batch = collate([SLBRDataset(ROOT, 'TRAIN')[0]])
    inputs = {k: batch[k].clone() for k in INPUT_KEYS}
    inputs['input_valid_mask'][:, :, 263:288] = False
    if all_missing:
        inputs['input_valid_mask'][:] = False
    valid = (inputs['feature_valid_mask'] & inputs['feature_applicable_mask'][:, None]
             & inputs['input_valid_mask'][..., None])
    changed = {k: v.clone() for k, v in inputs.items()}
    changed['X'][~valid] = float('nan')
    with torch.no_grad():
        expected = model(**inputs)
        actual, diag = model(**changed, return_diagnostics=True)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    if all_missing:
        assert torch.count_nonzero(diag['spatial']) == torch.count_nonzero(diag['temporal']) == 0


def test_real_train_padding_extreme_forward_backward_and_checkpoint():
    assets, targets = manifests(ROOT)
    ds = SLBRDataset(ROOT, 'TRAIN')
    dates = ds.base.a['reference_dates'][ds.base.a['train_origin_indices']]
    extreme = int(np.where(dates == '2016-12-23')[0][0])
    batch = collate([ds[i] for i in (0, extreme)])
    model = TriWeaveSLBR(assets, targets)
    model.set_epoch(20)
    prediction = model(**{k: batch[k] for k in INPUT_KEYS})
    loss, counts = masked_huber(prediction, batch['Y'], batch['label_valid_mask'])
    (loss + .25 * model.switch_loss()).backward()
    assert prediction.shape == (2, 4, 23) and all(counts)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    saved = copy.deepcopy(model.state_dict())
    recovered = TriWeaveSLBR(assets, targets).eval()
    recovered.load_state_dict(saved, strict=True)
    with torch.no_grad():
        a = model(**{k: batch[k] for k in INPUT_KEYS})
        b = recovered(**{k: batch[k] for k in INPUT_KEYS})
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        model.load_state_dict(OldModel(assets, targets).state_dict(), strict=True)


def test_mask_and_inapplicable_fields_do_not_create_observations():
    encoder = HeterogeneousInputEncoder(TYPES, 26, 32, temporal=True)
    x = torch.full((1, 2, 4, 26), float('nan'))
    present = torch.ones(1, 2, 4, dtype=torch.bool)
    valid_fields = torch.zeros_like(x, dtype=torch.bool)
    output, valid = encoder(x, present, valid_fields, torch.ones(4, 26, dtype=torch.bool))
    assert not valid.any() and torch.count_nonzero(output) == 0
    with pytest.raises(ValueError, match='Unknown economic types'):
        HeterogeneousInputEncoder(('unknown',), 26, 64)
