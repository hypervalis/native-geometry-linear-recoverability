"""Block-Sparse Featurizers (Fel et al. 2026, arXiv:2606.25234).

Vanilla:       z = Π_k (x W + b),   x̂ = z D          (untied encoder/decoder)
Grassmannian:  z = Π_k (γ x Dᵀ),   x̂ = z D          (tied; each block on Stiefel)

Π_k keeps the k blocks of largest ‖z_g‖₂. Codes are signed (no ReLU).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def block_topk(z: torch.Tensor, n_blocks: int, block_dim: int, k_blocks: int) -> torch.Tensor:
    z_blocks = z.view(*z.shape[:-1], n_blocks, block_dim)
    norms = z_blocks.norm(dim=-1)
    kk = min(k_blocks, n_blocks)
    _, idx = torch.topk(norms, kk, dim=-1)
    mask = torch.zeros_like(norms)
    mask.scatter_(-1, idx, 1.0)
    return (z_blocks * mask.unsqueeze(-1)).reshape(*z.shape)


def compute_block_norms(
    z: torch.Tensor, n_blocks: int, block_dim: int
) -> torch.Tensor:
    return z.view(*z.shape[:-1], n_blocks, block_dim).norm(dim=-1)


class VanillaBSF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_blocks: int,
        block_dim: int,
        k_blocks: int,
    ) -> None:
        super().__init__()
        if k_blocks > n_blocks:
            raise ValueError(f"k_blocks={k_blocks} > n_blocks={n_blocks}")
        self.input_dim = input_dim
        self.n_blocks = n_blocks
        self.block_dim = block_dim
        self.k_blocks = k_blocks
        self.code_dim = n_blocks * block_dim
        self.encoder = nn.Linear(input_dim, self.code_dim)
        self.decoder = nn.Linear(self.code_dim, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return block_topk(
            self.encoder(x), self.n_blocks, self.block_dim, self.k_blocks
        )

    def block_norms(self, z: torch.Tensor) -> torch.Tensor:
        return compute_block_norms(z, self.n_blocks, self.block_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


class GrassmannianBSF(nn.Module):
    """Tied dictionary; each block's columns form an orthonormal frame in R^d.

    z = Π_k(γ x Dᵀ),  x̂ = z D + bias,
    with D_g ∈ St(block_dim, input_dim) enforced by QR after each step.
    """

    def __init__(
        self,
        input_dim: int,
        n_blocks: int,
        block_dim: int,
        k_blocks: int,
    ) -> None:
        super().__init__()
        if k_blocks > n_blocks:
            raise ValueError(f"k_blocks={k_blocks} > n_blocks={n_blocks}")
        self.input_dim = input_dim
        self.n_blocks = n_blocks
        self.block_dim = block_dim
        self.k_blocks = k_blocks
        self.code_dim = n_blocks * block_dim
        # D stored as (d, code): columns within each block are orthonormal
        self.weight = nn.Parameter(torch.empty(input_dim, self.code_dim))
        nn.init.orthogonal_(self.weight)
        self._project_blocks_()
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def _project_blocks_(self) -> None:
        """Batched QR-orthonormalize each block's columns (Stiefel projection)."""
        d, gb = self.weight.shape
        b = self.block_dim
        g = self.n_blocks
        # (G, d, b)
        blocks = self.weight.view(d, g, b).permute(1, 0, 2).contiguous()
        Q, R = torch.linalg.qr(blocks, mode="reduced")
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        signs = torch.sign(diag)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(-2)
        self.weight.copy_(Q.permute(1, 0, 2).contiguous().view(d, gb))

    def project_blocks_(self) -> None:
        self._project_blocks_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # z_pre = γ x Dᵀ  with D = weight.T  ⇒  γ x @ weight
        z = self.gamma * (x @ self.weight)
        return block_topk(z, self.n_blocks, self.block_dim, self.k_blocks)

    def block_norms(self, z: torch.Tensor) -> torch.Tensor:
        return compute_block_norms(z, self.n_blocks, self.block_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        # x̂ = z D + bias = z @ weight.T + bias
        return z @ self.weight.T + self.bias, z
