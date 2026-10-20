"""Mask-aware port of CMGM Original SLBR, never a concat-fusion variant.

Core provenance: Commedities git 45879f5; source snapshots and module mapping
are sealed in experiments/phase_f_slbr_migration. No CMGM runtime imports.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .slbr_core import AdaptiveGraphLearner, MarkovRegimeFilter, RegimeLatentTransition

KINDS = ('STOCK_PRICE', 'BOND_YIELD', 'BOND_WEALTH_INDEX', 'FUTURES_CONTINUOUS_PRICE')
MARKETS = ('stock', 'bond', 'commodity')


def market_index(asset):
    # Frozen data manifest calls this market "future"; CMGM calls it commodity.
    return MARKETS.index('commodity' if asset['market'] in ('future', 'futures') else asset['market'])


def masked_softmax(logits, valid, dim=-1):
    """Exactly zero for an empty support, including its derivative."""
    valid = torch.broadcast_to(valid, logits.shape)
    scores = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(scores, dim=dim) * valid.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def zero_invalid(x, valid):
    return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))


@dataclass(frozen=True)
class SLBRConfig:
    name: str = 'triweave_switching_latent_balanced_readout'
    feature_dim: int = 26
    hidden_dim: int = 64
    node_dim: int = 32
    d_model: int = 128
    graph_heads: int = 8
    temporal_heads: int = 4
    temporal_layers: int = 2
    ffn_dim: int = 256
    regimes: int = 3
    z_dim: int = 64
    max_len: int = 20
    dropout: float = .3
    graph_attention_dropout: float = .1
    graph_prior_scale: float = .5
    mixhop_K: int = 2
    mixhop_beta: float = .05
    sticky_alpha: float = .5
    switch_beta_max: float = 5e-4
    switch_warmup: int = 20
    horizons: tuple = (1, 5, 10, 20)


class HeterogeneousInputEncoder(nn.Module):
    """Four economic types; missingness is encoded separately from z=0.

    Both projections consume only sanitized values. A bias-free presence
    projection distinguishes unavailable and structurally inapplicable fields
    from observed zero z-scores. Entire invalid nodes are zero after all bias/LN.
    Spatial and temporal encoders remain independent as in CMGM.
    """
    def __init__(self, economic_types, feature_dim, output_dim, temporal=False):
        super().__init__()
        self.register_buffer('type_indices', torch.tensor([KINDS.index(k) for k in economic_types]))
        self.projections = nn.ModuleList([nn.Linear(feature_dim, output_dim) for _ in KINDS])
        self.presence = nn.ModuleList([nn.Linear(feature_dim, output_dim, bias=False) for _ in KINDS])
        self.norms = nn.ModuleList([nn.LayerNorm(output_dim) if temporal else nn.Identity() for _ in KINDS])
        self.temporal = temporal

    def forward(self, x, input_valid_mask, feature_valid_mask, feature_applicable_mask):
        if feature_applicable_mask.ndim == 2:
            feature_applicable_mask = feature_applicable_mask[None, None]
        elif feature_applicable_mask.ndim == 3:
            feature_applicable_mask = feature_applicable_mask[:, None]
        fm = feature_valid_mask & feature_applicable_mask & input_valid_mask.unsqueeze(-1)
        valid = input_valid_mask & fm.any(-1)
        # torch.where BEFORE any projection: NaN/Inf in invalid slots cannot enter.
        safe = torch.where(fm, x, torch.zeros_like(x))
        result = safe.new_zeros((*safe.shape[:-1], self.projections[0].out_features))
        for k, (proj, presence, norm) in enumerate(zip(self.projections, self.presence, self.norms)):
            idx = torch.where(self.type_indices == k)[0]
            if idx.numel() == 0:
                continue
            y = proj(safe[:, :, idx]) + presence(fm[:, :, idx].to(safe.dtype))
            if self.temporal:
                y = F.relu(y)
            result[:, :, idx] = zero_invalid(norm(y), valid[:, :, idx])
        return result, valid


class MaskedEdgeAttnMixHop(nn.Module):
    """CMGM EdgeAttnMixHop equations, with invalid queries/keys removed.

    A is a *soft* log prior, not a hard adjacency: original nonedges carry
    log(1e-6). We preserve this distinction when reporting graph sparsity.
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.K, self.beta = cfg.mixhop_K, cfg.mixhop_beta
        self.n_heads, self.head_dim = cfg.graph_heads, h // cfg.graph_heads
        self.prior_scale = cfg.graph_prior_scale
        self.Ws = nn.ModuleList([nn.Linear(h, h) for _ in range(self.K + 1)])
        self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.out_proj = nn.Linear(h, h)
        self.attn_drop = nn.Dropout(cfg.graph_attention_dropout)

    def forward(self, x, adjacency, valid, return_diagnostics=False):
        b, n, h = x.shape
        x = zero_invalid(x, valid)
        hidden, out, attention_list = x, self.Ws[0](x), []
        prior = self.prior_scale * torch.log(adjacency.clamp_min(0) + 1e-6)
        support = valid[:, None, :, None] & valid[:, :, None, None]
        for hop in range(1, self.K + 1):
            q = self.q(hidden).view(b, n, self.n_heads, self.head_dim)
            k = self.k(hidden).view(b, n, self.n_heads, self.head_dim)
            v = self.v(hidden).view(b, n, self.n_heads, self.head_dim)
            logits = torch.einsum('bnhd,bmhd->bnmh', q, k) / math.sqrt(self.head_dim)
            alpha = masked_softmax(logits + prior[None, :, :, None], support, dim=2)
            if return_diagnostics:
                attention_list.append(alpha.detach())
            aggregate = torch.einsum('bnmh,bmhd->bnhd', self.attn_drop(alpha), v).reshape(b, n, h)
            aggregate = zero_invalid(self.out_proj(aggregate), valid)
            hidden = zero_invalid(self.beta * x + (1 - self.beta) * aggregate, valid)
            out = out + self.Ws[hop](hidden)
        return zero_invalid(out, valid), attention_list


class SpatialBranch(nn.Module):
    def __init__(self, cfg, assets):
        super().__init__()
        self.encoder = HeterogeneousInputEncoder([a['economic_type'] for a in assets], cfg.feature_dim, cfg.hidden_dim)
        self.register_buffer('market_indices', torch.tensor([market_index(a) for a in assets]))
        self.temporal_score = nn.Linear(cfg.hidden_dim, 1)
        self.graph_learner = AdaptiveGraphLearner(len(assets), embed_dim=10, alpha=.5, top_k=10)
        self.attn_mixhop1, self.attn_mixhop2 = MaskedEdgeAttnMixHop(cfg), MaskedEdgeAttnMixHop(cfg)
        self.gcn_norm = nn.LayerNorm(cfg.hidden_dim)
        self.type_agg = nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, x, iv, fv, fa, return_diagnostics=False):
        sequence, valid = self.encoder(x, iv, fv, fa)
        alpha = masked_softmax(self.temporal_score(sequence).squeeze(-1), valid, dim=1)
        nodes = (sequence * alpha.unsqueeze(-1)).sum(1)
        node_valid = valid.any(1)
        adjacency = self.graph_learner()  # trainable identity embeddings; no fitted correlation statistics
        h, a1 = self.attn_mixhop1(nodes, adjacency, node_valid, return_diagnostics)
        h, a2 = self.attn_mixhop2(F.relu(h), adjacency, node_valid, return_diagnostics)
        h = zero_invalid(self.gcn_norm(h), node_valid)
        pools = []
        for k in range(3):
            idx = self.market_indices == k
            m = node_valid[:, idx]
            pools.append(h[:, idx].sum(1) / m.sum(1, keepdim=True).clamp_min(1))
        result = zero_invalid(self.type_agg(torch.cat(pools, -1)), node_valid.any(-1))
        diag = dict(spatial=result, graph_nodes=h, spatial_time_attention=alpha,
                    graph_node_valid=node_valid, adjacency=adjacency,
                    graph_attention=a1 + a2, spatial_market_pools=torch.stack(pools, 1)) if return_diagnostics else {}
        return result, diag


class MarketEncoder(nn.Module):
    def __init__(self, cfg, assets):
        super().__init__()
        self.encoder = HeterogeneousInputEncoder([a['economic_type'] for a in assets], cfg.feature_dim, cfg.node_dim, temporal=True)
        self.register_buffer('market_indices', torch.tensor([market_index(a) for a in assets]))
        self.scores = nn.ModuleList([nn.Linear(cfg.node_dim, 1) for _ in MARKETS])
        self.daily_projection = nn.Sequential(nn.Linear(6 * cfg.node_dim, cfg.d_model), nn.LayerNorm(cfg.d_model))

    def forward(self, x, iv, fv, fa, return_diagnostics=False):
        encoded, valid = self.encoder(x, iv, fv, fa)
        representations, attentions, market_valid = [], [], []
        for k, score in enumerate(self.scores):
            idx = self.market_indices == k
            e, m = encoded[:, :, idx], valid[:, :, idx]
            attn = masked_softmax(score(e).squeeze(-1), m, dim=2)
            level = (e * attn.unsqueeze(-1)).sum(2)
            count = m.sum(2, keepdim=True).clamp_min(1)
            mean = e.sum(2) / count
            variance = torch.where(m.unsqueeze(-1), (e - mean.unsqueeze(2)).square(), 0.).sum(2) / count
            # Stabilized population std: zero at var=0 with finite backward.
            dispersion = (variance + 1e-8).sqrt() - 1e-4
            mv = m.any(2)
            representations.append(zero_invalid(torch.cat([level, dispersion], -1), mv))
            market_valid.append(mv)
            if return_diagnostics:
                attentions.append(attn)
        valid_steps = torch.stack(market_valid, -1).any(-1)
        tokens = zero_invalid(self.daily_projection(torch.cat(representations, -1)), valid_steps)
        diag = dict(market_attention=attentions, market_representations=torch.stack(representations, 2),
                    market_valid=torch.stack(market_valid, -1), daily_tokens=tokens) if return_diagnostics else {}
        return tokens, valid_steps, diag


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.head_dim = cfg.temporal_heads, cfg.d_model // cfg.temporal_heads
        self.q, self.k, self.v, self.out = [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(4)]

    def forward(self, x, bias, valid, return_diagnostics=False):
        b, t, d = x.shape
        q, k, v = [layer(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for layer in (self.q, self.k, self.v)]
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        support = causal[None, None] & valid[:, None, None, :] & valid[:, None, :, None]
        alpha = masked_softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + bias, support)
        y = (alpha @ v).transpose(1, 2).reshape(b, t, d)
        return zero_invalid(self.out(y), valid), alpha.detach() if return_diagnostics else None


class LongMemoryBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = CausalAttention(cfg)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(nn.Linear(cfg.d_model, cfg.ffn_dim), nn.ReLU(), nn.Linear(cfg.ffn_dim, cfg.d_model))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, bias, valid, return_diagnostics=False):
        attended, alpha = self.attention(x, bias, valid, return_diagnostics)
        x = zero_invalid(self.norm1(x + self.dropout(attended)), valid)
        x = zero_invalid(self.norm2(x + self.dropout(self.ffn(x))), valid)
        return x, alpha


class TemporalBranch(nn.Module):
    def __init__(self, cfg, assets):
        super().__init__()
        self.max_len = cfg.max_len
        self.market_encoder = MarketEncoder(cfg, assets)
        self.base_rpe = nn.Parameter(torch.randn(cfg.temporal_heads, 2 * cfg.max_len - 1) * .02)
        self.layers = nn.ModuleList([LongMemoryBlock(cfg) for _ in range(cfg.temporal_layers)])

    def forward(self, x, iv, fv, fa, return_diagnostics=False):
        hidden, valid, diag = self.market_encoder(x, iv, fv, fa, return_diagnostics)
        t = hidden.size(1)
        if t > self.max_len:
            raise ValueError('History exceeds frozen max_len')
        positions = torch.arange(t, device=x.device)
        delta = positions[:, None] - positions[None, :]
        bias = self.base_rpe[:, delta + self.max_len - 1][None]
        attentions = []
        for layer in self.layers:
            hidden, alpha = layer(hidden, bias, valid, return_diagnostics)
            if return_diagnostics:
                attentions.append(alpha)
        if return_diagnostics:
            diag.update(long_memory=hidden, temporal_attention=attentions, temporal_valid=valid)
        return hidden, valid, diag


class SwitchingRegime(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.regime_filter = MarkovRegimeFilter(cfg.d_model, cfg.regimes, sticky_alpha=cfg.sticky_alpha,
             beta_max=cfg.switch_beta_max, warmup_epochs=cfg.switch_warmup)
        self.latent_transition = RegimeLatentTransition(cfg.d_model, cfg.z_dim, hidden_dim=128, K=cfg.regimes)
        self.K, self.z_dim = cfg.regimes, cfg.z_dim

    def forward(self, memory, valid):
        b, t, _ = memory.shape
        p = memory.new_full((b, self.K), 1. / self.K)
        z = memory.new_zeros(b, self.z_dim)
        transition = self.regime_filter.transition_matrix()
        probabilities, priors, evidence, states, kls = [], [], [], [], []
        for i in range(t):
            prior, e, proposed_p, kl = self.regime_filter.step(memory[:, i], p, transition)
            proposed_z, _ = self.latent_transition(memory[:, i], z, proposed_p)
            p = torch.where(valid[:, i, None], proposed_p, p)
            z = torch.where(valid[:, i, None], proposed_z, z)
            probabilities.append(p); states.append(z)
            priors.append(torch.where(valid[:, i, None], prior, p))
            evidence.append(zero_invalid(e, valid[:, i]))
            kls.append(torch.where(valid[:, i], kl, 0.))
        self.regime_filter._last_switch_loss = torch.stack(kls, 1).sum() / valid.sum().clamp_min(1)
        return torch.stack(states, 1), dict(regime_probabilities=torch.stack(probabilities, 1),
                    regime_priors=torch.stack(priors, 1), regime_evidence=torch.stack(evidence, 1))


class BalancedReadout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.long_memory_readout = nn.Linear(cfg.d_model, cfg.hidden_dim)
        self.micro_state_readout = nn.Linear(cfg.z_dim, cfg.hidden_dim)
        self.long_memory_norm = nn.LayerNorm(cfg.hidden_dim)
        self.micro_state_norm = nn.LayerNorm(cfg.hidden_dim)
        self.state_readout = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, memory, states, valid):
        positions = torch.arange(valid.size(1), device=valid.device)[None].expand_as(valid)
        last = positions.masked_fill(~valid, -1).max(1).values
        row = torch.arange(valid.size(0), device=valid.device)
        h = memory[row, last.clamp_min(0)]; z = states[row, last.clamp_min(0)]
        has = last >= 0
        long = zero_invalid(self.long_memory_norm(self.long_memory_readout(h)), has)
        micro = zero_invalid(self.micro_state_norm(self.micro_state_readout(z)), has)
        return zero_invalid(self.state_readout(torch.cat([long, micro], -1)), has), long, micro


class ChannelFusionGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.gate_fc, self.gcn_proj, self.lstm_proj = nn.Linear(2 * h, h), nn.Linear(h, h), nn.Linear(h, h)

    def forward(self, spatial, temporal, valid):
        # Original direction: gate weights TEMPORAL, (1-gate) SPATIAL.
        gate = torch.sigmoid(self.gate_fc(torch.cat([spatial, temporal], -1)))
        spatial_p = zero_invalid(self.gcn_proj(spatial), valid)
        temporal_p = zero_invalid(self.lstm_proj(temporal), valid)
        return gate * temporal_p + (1 - gate) * spatial_p, gate


class TriWeaveSLBR(nn.Module):
    def __init__(self, assets, targets, config=None):
        super().__init__()
        self.config = config if isinstance(config, SLBRConfig) else SLBRConfig(**(config or {}))
        self.assets = assets; self.targets = targets
        ids = [a['asset_id'] for a in assets]
        if len(set(ids)) != len(ids) or len(set(targets)) != len(targets) or not set(targets) <= set(ids):
            raise ValueError('Ambiguous asset/target identity mapping')
        self.target_to_index = {target: i for i, target in enumerate(targets)}
        self.target_to_node = {target: ids.index(target) for target in targets}
        c = self.config
        self.spatial_branch, self.temporal_branch = SpatialBranch(c, assets), TemporalBranch(c, assets)
        self.switching_regime, self.balanced_readout = SwitchingRegime(c), BalancedReadout(c)
        self.fusion_gate = ChannelFusionGate(c)
        self.prediction_head = nn.Sequential(nn.Linear(c.hidden_dim, c.hidden_dim), nn.ReLU(), nn.Dropout(c.dropout),
                                   nn.Linear(c.hidden_dim, len(c.horizons) * len(targets)))

    def forward(self, X, input_valid_mask, feature_valid_mask, feature_applicable_mask, return_diagnostics=False):
        if X.ndim != 4 or X.shape[2:] != (len(self.assets), self.config.feature_dim):
            raise ValueError('Input identity/feature shape does not match frozen manifests')
        args = X, input_valid_mask, feature_valid_mask, feature_applicable_mask, return_diagnostics
        spatial, sd = self.spatial_branch(*args)
        memory, valid, td = self.temporal_branch(*args)
        states, rd = self.switching_regime(memory, valid)
        temporal, long, micro = self.balanced_readout(memory, states, valid)
        fused, gate = self.fusion_gate(spatial, temporal, valid.any(-1))
        prediction = self.prediction_head(fused).view(X.size(0), len(self.config.horizons), len(self.targets))
        if not return_diagnostics:
            return prediction
        diagnostics = dict(**sd, **td, **rd, temporal=temporal, balanced_long=long, balanced_micro=micro,
                           latent_states=states, fusion_gate=gate, fused=fused)
        # Diagnostics are read-only tensors from this SAME forward, with no RNG draws.
        return prediction, _detach(diagnostics)

    def set_epoch(self, epoch):
        return self.switching_regime.regime_filter.set_epoch(epoch)

    def switch_loss(self):
        return self.switching_regime.regime_filter.switch_loss()

    def parameter_counts(self):
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        enc = sum(p.numel() for branch in (self.spatial_branch.encoder, self.temporal_branch.market_encoder.encoder) for p in branch.parameters())
        counts['heterogeneous_encoder'] = enc
        counts['spatial_branch'] -= sum(p.numel() for p in self.spatial_branch.encoder.parameters())
        counts['temporal_branch'] -= sum(p.numel() for p in self.temporal_branch.market_encoder.encoder.parameters())
        counts['total_params'] = sum(p.numel() for p in self.parameters())
        return counts


def _detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach(v) for v in value]
    return value


def masked_huber(prediction, target, mask, delta=.02):
    """Equal means of nonempty horizons; select before arithmetic with labels."""
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError('Masked Huber requires matching [B,H,C] and a boolean mask')
    losses, denominators = [], []
    for h in range(prediction.size(1)):
        mh = mask[:, h]
        n = int(mh.sum().item()); denominators.append(n)
        if n:
            losses.append(F.huber_loss(prediction[:, h][mh], target[:, h][mh], delta=delta))
    if not losses:
        return None, denominators
    return torch.stack(losses).mean(), denominators
