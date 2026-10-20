"""Unmodified CMGM Phase6 fusion operators; see migration source_manifest.json."""
import torch
from torch import nn
from torch.nn import functional as F

class ResidualFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.main = nn.Linear(dim * 2, dim)
        self.residual = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_s, x_t], dim=-1)
        nonlinear = F.gelu(self.main(x))
        residual = self.residual(x)
        return self.norm(nonlinear + residual)


class BranchPreservingFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.spatial_proj = nn.Linear(dim, dim)
        self.temporal_proj = nn.Linear(dim, dim)
        self.interaction = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        s = self.spatial_proj(x_s)
        t = self.temporal_proj(x_t)
        c = F.gelu(self.interaction(torch.cat([x_s, x_t], dim=-1)))
        return self.norm(self.out(torch.cat([s, t, c], dim=-1)))


