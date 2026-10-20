"""Parameter-free neutral adapter for CMGM Deep V3; no data/scaler fitting."""
import torch
from torch import nn
from ..slbr_triweave import masked_softmax, zero_invalid, market_index


def sanitize(X, input_valid_mask, feature_valid_mask, feature_applicable_mask):
    app = feature_applicable_mask
    if app.ndim == 2:app = app[None, None]
    elif app.ndim == 3:app = app[:, None]
    mask = feature_valid_mask & app & input_valid_mask[..., None]
    x = torch.where(mask, X, 0.)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite observed feature')
    return x, mask


def last_valid(states, valid):
    idx = torch.arange(valid.size(1), device=valid.device)[None].expand_as(valid)
    last = idx.masked_fill(~valid, -1).max(1).values
    out = states[torch.arange(len(states), device=states.device), last.clamp_min(0)]
    return zero_invalid(out, last >= 0)


def masked_adjacency(adjacency, valid, self_loops=False):
    """B,T,N -> B,T,N,N; normalize destination rows, excluding invalid nodes."""
    if self_loops:adjacency = adjacency + torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype)
    support = valid[..., :, None] & valid[..., None, :]
    a = torch.where(support, adjacency, 0.)
    return a / a.sum(-1, keepdim=True).clamp_min(1e-12)


class NeutralAdapter(nn.Module):
    """Stock mean/std, joint bond mean/std, then23 explicit commodity targets.

    Statistics use valid observations per feature (population variance). No
    learned encoder, price fill, standardizer or future-universe filtering.
    The311-node input is compressed to27 tokens, as in the V3 design.
    """
    def __init__(self, assets, targets, features=26):
        super().__init__()
        ids = [a['asset_id'] for a in assets]
        if len(set(ids)) != len(ids) or not set(targets) <= set(ids):raise ValueError('Asset identity mismatch')
        self.features = features;self.n_assets = len(assets)
        self.register_buffer('markets', torch.tensor([market_index(a) for a in assets]))
        self.register_buffer('target_nodes', torch.tensor([ids.index(t) for t in targets]))
        self.n_tokens = 4 + len(targets)

    def forward(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask):
        if X.shape[1:] != (20, self.n_assets, self.features):raise ValueError('Expected B,20,N,26')
        x, mask = sanitize(X, input_valid_mask, feature_valid_mask, feature_applicable_mask)
        values=[];masks=[]
        for market in (0, 1):
            idx=self.markets == market;e=x[:, :, idx];m=mask[:, :, idx]
            count=m.sum(2);valid=count>0;den=count.clamp_min(1)
            mean=e.sum(2)/den
            var=torch.where(m, (e-mean[:, :, None]).square(), 0.).sum(2)/den
            std=(var+1e-8).sqrt()-1e-4
            values += [torch.where(valid, mean, 0.), torch.where(valid, std, 0.)]
            masks += [valid, valid]
        features=torch.cat([torch.stack(values, 2), x.index_select(2, self.target_nodes)], 2)
        fm=torch.cat([torch.stack(masks, 2), mask.index_select(2, self.target_nodes)], 2)
        return features, fm.any(-1), fm
