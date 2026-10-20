"""Unmodified CMGM core classes from git 45879f5 (see Phase F manifest)."""
import torch
from torch import nn
from torch.nn import functional as F

class AdaptiveGraphLearner(nn.Module):
    """
    Learnable directed graph adjacency matrix with improved gradient flow.

    Args:
        num_nodes: Number of nodes in the graph (e.g. 284)
        embed_dim: Node embedding dimension (default: 10)
        alpha:     Initial saturation rate (default: 0.5, learnable)
        top_k:     Number of neighbors to keep per node (default: 10)
    """

    def __init__(self, num_nodes: int, embed_dim: int = 10,
                 alpha: float = 0.5, top_k: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.top_k = top_k

        # Learnable α (initialized small to avoid tanh saturation)
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Learnable node embeddings
        self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))

        # Learnable transformations
        self.Θ1 = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.Θ2 = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier uniform initialization for all parameters."""
        for p in [self.E1, self.E2]:
            nn.init.xavier_uniform_(p)
        for p in [self.Θ1, self.Θ2]:
            nn.init.xavier_uniform_(p)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            A: (num_nodes, num_nodes) — non-negative, soft top-k adjacency
        """
        # Projected embeddings (no tanh — keeps gradient flowing)
        M1 = self.E1 @ self.Θ1                              # (N, d)
        M2 = self.E2 @ self.Θ2                              # (N, d)

        # Anti-symmetric similarity → adjacency
        # Single tanh after anti-symmetric difference (not before)
        S = M1 @ M2.T - M2 @ M1.T                           # (N, N)
        A = F.relu(torch.tanh(self.alpha * S))               # (N, N)

        # Soft top-k: differentiable sigmoid gating
        # Instead of hard zeroing non-top-k positions, use a smooth gate
        if self.top_k is not None and self.top_k < self.num_nodes:
            # threshold = k-th largest value per row, shape (N, 1)
            threshold = torch.topk(A, self.top_k, dim=1).values[:, -1:]
            # Sigmoid gate: ~1 for values above threshold, ~0 below
            # gradient flows through both A and threshold
            gate = torch.sigmoid((A - threshold) * 10.0)
            A = A * gate

        return A


class MarkovRegimeFilter(nn.Module):
    """One-head discriminative Markov filter used only for latent dynamics."""

    def __init__(self, d_model: int = 128, K: int = 3,
                 sticky_alpha: float = 0.5, tau: float = 1.0,
                 beta_max: float = 5e-4, warmup_epochs: int = 20,
                 eps: float = 1e-8):
        super().__init__()
        self.K = int(K)
        self.sticky_alpha = float(sticky_alpha)
        self.tau = float(tau)
        self.beta_max = float(beta_max)
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)
        self.current_epoch = 1
        self.current_beta = 0.0
        self.transition_logits = nn.Parameter(torch.zeros(self.K, self.K))
        self.regime_evidence = nn.Linear(d_model, self.K)
        self._last_switch_loss = None

    def transition_matrix(self) -> torch.Tensor:
        learned = F.softmax(self.transition_logits, dim=-1)
        identity = torch.eye(
            self.K, dtype=learned.dtype, device=learned.device
        )
        return self.sticky_alpha * identity + (1.0 - self.sticky_alpha) * learned

    def set_epoch(self, epoch: int) -> float:
        self.current_epoch = int(epoch)
        if self.warmup_epochs <= 1:
            progress = 1.0
        else:
            progress = min(
                max((self.current_epoch - 1) / (self.warmup_epochs - 1), 0.0),
                1.0,
            )
        self.current_beta = self.beta_max * progress
        return self.current_beta

    def step(self, h_t: torch.Tensor, p_prev: torch.Tensor,
             transition: torch.Tensor):
        prior_t = p_prev @ transition
        evidence_t = self.regime_evidence(h_t)
        posterior_logits = (
            torch.log(prior_t + self.eps) + evidence_t / self.tau
        )
        p_t = F.softmax(posterior_logits, dim=-1)
        kl_t = (
            p_t
            * (
                torch.log(p_t + self.eps)
                - torch.log(prior_t + self.eps)
            )
        ).sum(dim=-1)
        return prior_t, evidence_t, p_t, kl_t

    def switch_loss(self) -> torch.Tensor:
        if self._last_switch_loss is None:
            return self.transition_logits.sum() * 0.0
        return self.current_beta * self._last_switch_loss


class RegimeLatentTransition(nn.Module):
    """K independent deterministic generators G_k([H_t,Z_(t-1)])."""

    def __init__(self, h_dim: int = 128, z_dim: int = 64,
                 hidden_dim: int = 128, K: int = 3):
        super().__init__()
        self.K = int(K)
        self.z_dim = int(z_dim)
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim + z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, z_dim),
            )
            for _ in range(self.K)
        ])

    def forward(self, h_t: torch.Tensor, z_prev: torch.Tensor,
                probabilities: torch.Tensor):
        transition_input = torch.cat([h_t, z_prev], dim=-1)
        candidates = torch.stack(
            [generator(transition_input) for generator in self.generators],
            dim=1,
        )
        z_t = torch.einsum("bk,bkd->bd", probabilities, candidates)
        return z_t, candidates


