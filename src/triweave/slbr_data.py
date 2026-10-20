"""Strict TRAIN/VAL-only adapter for the frozen Research23 memmap loader."""
from pathlib import Path
import json
import numpy as np
import torch
from .research23 import Research23Dataset, PROTOCOL

DATA = Path('data/derived/triweave_23_after_close_research_v1')
OUT = Path('experiments/phase_f_slbr_migration')
INPUT_KEYS = ('X', 'input_valid_mask', 'feature_valid_mask', 'feature_applicable_mask')


class SLBRDataset:
    def __init__(self, root, split):
        if split not in ('TRAIN', 'VAL'):
            raise ValueError('TEST numeric access prohibited')
        self.base = Research23Dataset(Path(root) / DATA, split)
        if self.base.manifest['protocol_id'] != PROTOCOL or not self.base.manifest['EXPLORATORY_RESEARCH_READY']:
            raise ValueError('Dataset gate failed')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        return {k: sample[k] for k in INPUT_KEYS + ('Y', 'label_valid_mask', 'origin_date')}


def collate(samples):
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in INPUT_KEYS + ('Y', 'label_valid_mask')} | {
        'origin_date': [s['origin_date'] for s in samples]}


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def manifests(root):
    path = Path(root) / DATA
    assets = json.loads((path / 'asset_manifest.json').read_text())
    target_doc = json.loads((path / 'target_manifest.json').read_text())
    targets = [x['asset_id'] for x in target_doc['targets']]
    assert len(assets) == 311 and len(targets) == 23
    assert all(a['index'] == i for i, a in enumerate(assets))
    assert targets == [a['asset_id'] for a in assets if a['economic_type'] == 'FUTURES_CONTINUOUS_PRICE']
    assert not {'FUT_AU', 'FUT_AG'} & {a['asset_id'] for a in assets}
    return assets, targets
