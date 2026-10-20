"""SLBR revision with one bond encoder per branch, shared by all25 bonds."""

from dataclasses import dataclass

import torch
from torch import nn

from ..slbr_triweave import (
    HeterogeneousInputEncoder as OriginalInputEncoder,
    SLBRConfig as OriginalConfig,
    TriWeaveSLBR as OriginalSLBR,
)


ENCODER_GROUPS = ('STOCK_PRICE', 'BOND', 'FUTURES_CONTINUOUS_PRICE')
ECONOMIC_TYPE_TO_ENCODER = {
    'STOCK_PRICE': 'STOCK_PRICE',
    'BOND_YIELD': 'BOND',
    'BOND_WEALTH_INDEX': 'BOND',
    'FUTURES_CONTINUOUS_PRICE': 'FUTURES_CONTINUOUS_PRICE',
}


@dataclass(frozen=True)
class SLBRConfig(OriginalConfig):
    name: str = 'triweave_switching_latent_balanced_readout_shared_bond_v1'


class HeterogeneousInputEncoder(OriginalInputEncoder):
    """Three registered projections; yield and wealth share the same modules.

    Economic types in the data retain their units, feature applicability and
    frozen scaler groups. Only the mapping to trainable encoders is merged.
    Reuse the original mask-aware forward without changing its arithmetic.
    """

    def __init__(self, economic_types, feature_dim, output_dim, temporal=False):
        nn.Module.__init__(self)
        unknown = sorted(set(economic_types) - ECONOMIC_TYPE_TO_ENCODER.keys())
        if unknown:
            raise ValueError(f'Unknown economic types: {unknown}')
        self.register_buffer('type_indices', torch.tensor([
            ENCODER_GROUPS.index(ECONOMIC_TYPE_TO_ENCODER[k]) for k in economic_types
        ], dtype=torch.long))
        self.projections = nn.ModuleList([
            nn.Linear(feature_dim, output_dim) for _ in ENCODER_GROUPS
        ])
        self.presence = nn.ModuleList([
            nn.Linear(feature_dim, output_dim, bias=False) for _ in ENCODER_GROUPS
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(output_dim) if temporal else nn.Identity()
            for _ in ENCODER_GROUPS
        ])
        self.temporal = temporal


class TriWeaveSLBR(OriginalSLBR):
    """Same SLBR branches/readout/gate/head, with shared bond input encoders.

    The historical four-type class and its checkpoint stay frozen. This version
    replaces both registered input encoders, leaving no fourth encoder or tied
    duplicate parameters in the resulting model. Non-encoder initialization is
    inherited unchanged for a fixed constructor seed. Training is a separate run.
    """

    def __init__(self, assets, targets, config=None):
        if not isinstance(config, SLBRConfig):
            config = SLBRConfig(**(config or {}))
        if config.name != SLBRConfig().name:
            raise ValueError('This model requires the shared-bond version identifier')
        super().__init__(assets, targets, config)
        economic_types = [a['economic_type'] for a in assets]
        self.spatial_branch.encoder = HeterogeneousInputEncoder(
            economic_types, config.feature_dim, config.hidden_dim,
        )
        self.temporal_branch.market_encoder.encoder = HeterogeneousInputEncoder(
            economic_types, config.feature_dim, config.node_dim, temporal=True,
        )
