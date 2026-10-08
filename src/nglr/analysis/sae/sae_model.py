"""Sparse autoencoder variants (TopK and ReLU+L1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """x -> encoder -> ReLU -> TopK(k) -> decoder -> reconstruction."""

    def __init__(self, input_dim: int, feature_dim: int, k: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.k = k
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.decoder = nn.Linear(feature_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x))
        kk = min(self.k, z.shape[-1])
        vals, idx = torch.topk(z, kk, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, idx, vals)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z


class L1SAE(nn.Module):
    """Generic SAE: x -> encoder -> ReLU -> decoder (L1 on codes in the loss)."""

    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.k = 0  # unused; variable sparsity
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.decoder = nn.Linear(feature_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Unit-norm decoder columns (common SAE constraint)."""
        w = self.decoder.weight
        w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))
