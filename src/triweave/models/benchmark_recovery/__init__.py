"""Numerical recovery V1; immutable original benchmark implementations remain.

CrossGNN: exact third-smallest VALUE via sort, never kthvalue CUDA indices.
TCN: explicit FP64 WeightNorm reduction/backward and fixed 1e-12 norm floor.
No parameter, architecture, data, learning-rate or model-selection changes.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark import make_model as original_make_model
from ..benchmark.common import masked_softmax
from ..benchmark.crossgnn import CrossGNNCore

REPAIRED = ('baseline/CrossGNN', 'baseline/TCN')
NORM_FLOOR = 1e-12
IMPLEMENTATIONS = {
    'baseline/CrossGNN': 'CROSSGNN_SORT_THRESHOLD_V1',
    'baseline/TCN': 'TCN_EXPLICIT_FP64_WEIGHTNORM_FLOOR_1E12_V1',
}


class DeterministicCrossGNNCore(CrossGNNCore):
    def variable_adjacency(self, valid):
        scores = F.relu(self.nodevec1 @ self.nodevec2)
        positive = scores >= scores.topk(3, dim=-1).values[:, -1:]
        # Only threshold values affect the graph; tied index ordering is unused.
        negative = scores <= scores.sort(dim=-1).values[:, 2:3]
        support = valid[:, :, :, None] & valid[:, :, None, :]
        pos = masked_softmax(scores.expand_as(support.to(scores.dtype)), support & positive)
        neg = masked_softmax((1 / (scores + 1)).expand_as(pos), support & negative)
        return pos - neg


class StableWeightNorm(nn.Module):
    """Same g/v parameter identities; no fused CUDA weight_norm backward.

    The predeclared floor handles degenerate/dead filters. Below 1e-12 this
    intentionally differs from g*v/||v||; the change is recorded, never hidden.
    Invalid values are not replaced or silently skipped.
    """
    def forward(self, magnitude, direction):
        v = direction.to(torch.float64)
        norm = torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)), keepdim=True)
        weight = magnitude.to(torch.float64) * (v / norm.clamp_min(NORM_FLOOR))
        return weight.to(direction.dtype)


def make_model(model_id, assets, targets):
    model = original_make_model(model_id, assets, targets)
    if model_id == 'baseline/CrossGNN':
        # Reuse the initialized tensors and RNG consumption exactly; do not
        # construct an independently initialized second core.
        model.core.__class__ = DeterministicCrossGNNCore
    elif model_id == 'baseline/TCN':
        for block in model.tcn.network:
            for conv in (block.conv1, block.conv2):
                conv.parametrizations.weight[0] = StableWeightNorm()
    return model
