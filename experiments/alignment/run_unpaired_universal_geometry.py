#!/usr/bin/env python3
"""Unpaired universal geometry / vec2vec-style overnight smoke.

Physics vit_base ↔ dinov3: recover cross-model relational geometry without
paired training, and measure how latent bottleneck Z affects identity vs
geometry saturation.

Outputs land under:
  outputs/unpaired_universal_geometry/smoke/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent
_REPO = _EXPERIMENTS.parent

for _p in (
    _EXPERIMENTS / "bipartite-matching",
    _EXPERIMENTS / "universetbd_shared_basis_mknn",
    _EXPERIMENTS / "SAE-shared-basis",
):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from _shared import (  # noqa: E402
    MODELS,
    idf_weights,
    l2n as l2n_np,
    load_col,
    load_sae as _shared_load_sae,
    platonic_root,
    resolve_path,
    sae_dir as _shared_sae_dir,
)

# Soft JS helpers (optional)
_HAS_SOFT = False
try:
    import run_decoder_metric_geometry as dmg  # noqa: E402

    calibrate_tau_for_keff_cosine = dmg.calibrate_tau_for_keff_cosine
    js_similarity_rows = dmg.js_similarity_rows
    hellinger_rows = getattr(dmg, "hellinger_rows", None)
    tv_rows = getattr(dmg, "tv_rows", None)
    cosine_logits = dmg.cosine_logits
    _HAS_SOFT = True
except Exception:  # noqa: BLE001
    dmg = None  # type: ignore
    calibrate_tau_for_keff_cosine = None  # type: ignore
    js_similarity_rows = None  # type: ignore
    hellinger_rows = None  # type: ignore
    tv_rows = None  # type: ignore
    cosine_logits = None  # type: ignore

# Prefer GPU knn/mknn from sae_affine_basis_mknn_gpu when available
try:
    from sae_affine_basis_mknn_gpu import knn_cos as _knn_cos_affine  # noqa: E402
    from sae_affine_basis_mknn_gpu import mknn as _mknn_affine  # noqa: E402

    _HAS_AFFINE_KNN = True
except Exception:  # noqa: BLE001
    _HAS_AFFINE_KNN = False
    _knn_cos_affine = None  # type: ignore
    _mknn_affine = None  # type: ignore


SAE_TAG_PREFER = (
    "F2048_k20_seed0",
    "F2048_k22_seed0",
    "F2048_k19_seed0",
    "F2048_k64_seed0",
    "F2048_k32_seed0",
    "F2048_k128_seed0",
)

ORACLE_Z_FULL = (8, 16, 32, 64, 128, 256, 512)
ORACLE_Z_SMOKE = (16, 64, 256)
BOTTLENECK_Z_SMOKE = (16, 64, 256)
MKNN_KS = (5, 10, 20, 50, 100)
SOFT_KEFFS = (10, 20, 50, 100, 250, 500, 1000)
N_GEOMETRY_DEFAULT = 512
N_SOFT_QUERY = 256
N_IDENTITY_SHUFFLES = 100
DECODER_ALPHA = 0.35

EPS = 1e-12
LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument(
        "--out-dir",
        default="outputs/unpaired_universal_geometry/smoke",
    )
    p.add_argument("--pair", default="vit_base,dinov3")
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--zs", default="16,64,256")
    p.add_argument("--oracle-zs", default="16,64,256")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reps", default="dense,sae_shared")
    p.add_argument("--skip-soft", action="store_true")
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument("--include-decoder-metric", action="store_true")
    p.add_argument("--n-oracle-train", type=int, default=2500)
    p.add_argument("--n-a-train", type=int, default=5500)
    p.add_argument("--n-b-train", type=int, default=5500)
    p.add_argument("--n-geometry", type=int, default=N_GEOMETRY_DEFAULT)
    p.add_argument("--n-soft-query", type=int, default=N_SOFT_QUERY)
    p.add_argument("--n-identity-shuffles", type=int, default=N_IDENTITY_SHUFFLES)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-mmd", type=float, default=1.0)
    p.add_argument("--w-cycle", type=float, default=1.0)
    p.add_argument("--w-geom", type=float, default=1.0)
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--run-orthogonal-null", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def l2n_t(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def empty_parquet(path: Path, columns: list[str], note: str) -> None:
    df = pd.DataFrame(columns=columns)
    df.attrs["note"] = note
    df.to_parquet(path, index=False)
    path.with_suffix(".skip_note.txt").write_text(note + "\n")


def df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    if df is None or len(df) == 0:
        cols = list(df.columns) if df is not None else ["note"]
        empty_parquet(path, cols, "empty result table")
        return
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Graph / ranking helpers  (@torch.no_grad — avoid InferenceMode mutation bugs)
# ---------------------------------------------------------------------------


@torch.no_grad()
def knn_cos(Z: torch.Tensor, k: int, row_batch: int = 256) -> torch.Tensor:
    if _HAS_AFFINE_KNN and _knn_cos_affine is not None:
        return _knn_cos_affine(Z, k=k, row_batch=row_batch)
    Z = l2n_t(Z)
    n = Z.shape[0]
    k = min(k, max(1, n - 1))
    out = torch.empty(n, k, device=Z.device, dtype=torch.long)
    for s in range(0, n, row_batch):
        e = min(n, s + row_batch)
        sim = Z[s:e] @ Z.T
        b = e - s
        sim[torch.arange(b, device=Z.device), torch.arange(s, e, device=Z.device)] = (
            -torch.inf
        )
        out[s:e] = torch.topk(sim, k=k, dim=1).indices
    return out


@torch.no_grad()
def mknn_score(nn1: torch.Tensor, nn2: torch.Tensor, k: int) -> float:
    """Hard mKNN: mean |top-k(nn1) ∩ top-k(nn2)| / k (must slice to k)."""
    k = min(int(k), int(nn1.shape[1]), int(nn2.shape[1]))
    nn1k, nn2k = nn1[:, :k], nn2[:, :k]
    if _HAS_AFFINE_KNN and _mknn_affine is not None:
        return float(_mknn_affine(nn1k, nn2k, k=k))
    a = nn1k.cpu().numpy()
    b = nn2k.cpu().numpy()
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


@torch.no_grad()
def identity_ranks(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    *,
    row_batch: int = 256,
) -> torch.Tensor:
    """Return 0-based rank of diagonal identity in cosine similarity.

    Assumes queries[i] should match gallery[i] (paired eval gallery).
    """
    Q = l2n_t(queries)
    G = l2n_t(gallery)
    n = Q.shape[0]
    ranks = torch.empty(n, device=Q.device, dtype=torch.long)
    arange_g = torch.arange(n, device=Q.device)
    for s in range(0, n, row_batch):
        e = min(n, s + row_batch)
        sim = Q[s:e] @ G.T  # [b, n]
        # rank: number of gallery items with strictly higher similarity
        true = sim[torch.arange(e - s, device=Q.device), arange_g[s:e]]
        ranks[s:e] = (sim > true[:, None]).sum(dim=1)
    return ranks


def rank_metrics(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    n = max(len(ranks), 1)
    out = {
        "top1": float(np.mean(ranks < 1)),
        "top5": float(np.mean(ranks < 5)),
        "top10": float(np.mean(ranks < 10)),
        "top100": float(np.mean(ranks < 100)),
        "mrr": float(np.mean(1.0 / (ranks + 1.0))),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "n": float(n),
    }
    return out


@torch.no_grad()
def pairwise_cosine_gram(X: torch.Tensor) -> torch.Tensor:
    Xn = l2n_t(X)
    return Xn @ Xn.T


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between feature matrices (rows = samples)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    num = float(np.linalg.norm(xty, "fro") ** 2)
    den = float(np.linalg.norm(xtx, "fro") * np.linalg.norm(yty, "fro"))
    if den < EPS:
        return 0.0
    return num / den


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if den < EPS:
        return float("nan")
    return float((ra * rb).sum() / den)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a**2).sum() * (b**2).sum()))
    if den < EPS:
        return float("nan")
    return float((a * b).sum() / den)


def latent_effective_ranks(Z: np.ndarray) -> dict[str, float]:
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # covariance spectrum via SVD of centered features
    s = np.linalg.svd(Zc, compute_uv=False)
    eigs = (s**2) / max(Zc.shape[0] - 1, 1)
    eigs = np.maximum(eigs, 0.0)
    tot = float(eigs.sum())
    if tot < EPS:
        return {
            "stable_rank": 0.0,
            "entropy_rank": 0.0,
            "participation_ratio": 0.0,
            "pca90": 0.0,
            "pca95": 0.0,
            "pca99": 0.0,
        }
    p = eigs / tot
    stable = float((eigs.sum() ** 2) / max((eigs**2).sum(), EPS))
    ent = float(np.exp(-(p * np.log(p + EPS)).sum()))
    pr = float((eigs.sum() ** 2) / max((eigs**2).sum(), EPS))
    csum = np.cumsum(p)

    def zfrac(f: float) -> float:
        return float(np.searchsorted(csum, f) + 1)

    return {
        "stable_rank": stable,
        "entropy_rank": ent,
        "participation_ratio": pr,
        "pca90": zfrac(0.90),
        "pca95": zfrac(0.95),
        "pca99": zfrac(0.99),
    }


# ---------------------------------------------------------------------------
# MMD (multi-bandwidth RBF)
# ---------------------------------------------------------------------------


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: [n,d], y: [m,d]
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    d2 = x2 + y2.T - 2.0 * (x @ y.T)
    return torch.exp(-d2 / (2.0 * sigma * sigma + EPS))


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Iterable[float] | None = None,
) -> torch.Tensor:
    if sigmas is None:
        # heuristic bandwidths from pooled pairwise median-ish scales
        with torch.no_grad():
            z = torch.cat([x.detach(), y.detach()], dim=0)
            if z.shape[0] > 512:
                idx = torch.randperm(z.shape[0], device=z.device)[:512]
                z = z[idx]
            d2 = torch.cdist(z, z, p=2).pow(2)
            med = float(d2[d2 > 0].median().item()) if (d2 > 0).any() else 1.0
            base = math.sqrt(max(med, EPS))
        sigma_list = [base / 4, base / 2, base, base * 2, base * 4]
    else:
        sigma_list = [float(s) for s in sigmas]
    total = x.new_zeros(())
    for s in sigma_list:
        kxx = _rbf_kernel(x, x, float(s))
        kyy = _rbf_kernel(y, y, float(s))
        kxy = _rbf_kernel(x, y, float(s))
        # unbiased-ish with diag kept for stability in small batches
        total = total + (kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    return total / max(len(sigma_list), 1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dims: list[int], act: Callable = nn.GELU):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualEncoder(nn.Module):
    """LayerNorm → MLP → bottleneck Z → residual MLP → decoder.

    Optional shared transform T (identity or small MLP) in latent space.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        z_dim: int,
        hidden: int = 512,
        shared_t: str = "identity",
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden = int(hidden)
        self.ln_a = nn.LayerNorm(dim_a)
        self.ln_b = nn.LayerNorm(dim_b)
        self.enc_a = MLP([dim_a, hidden, hidden, z_dim])
        self.enc_b = MLP([dim_b, hidden, hidden, z_dim])
        self.res_a = MLP([z_dim, hidden, z_dim])
        self.res_b = MLP([z_dim, hidden, z_dim])
        self.dec_a = MLP([z_dim, hidden, hidden, dim_a])
        self.dec_b = MLP([z_dim, hidden, hidden, dim_b])
        if shared_t == "mlp":
            self.T = MLP([z_dim, hidden, z_dim])
        else:
            self.T = nn.Identity()

    def encode_a(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_a(self.ln_a(x))
        return h + self.res_a(h)

    def encode_b(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_b(self.ln_b(x))
        return h + self.res_b(h)

    def decode_a(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_a(self.T(z))

    def decode_b(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_b(self.T(z))

    def forward_a(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_a(x)
        return z, self.decode_a(z)

    def forward_b(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_b(x)
        return z, self.decode_b(z)

    def translate_a_to_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_b(self.encode_a(x))

    def translate_b_to_a(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_a(self.encode_b(x))


class BottleneckMLP(nn.Module):
    """Paired oracle: xA → hidden → Z → hidden → xB."""

    def __init__(self, dim_a: int, dim_b: int, z_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_a),
            nn.Linear(dim_a, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_b),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def recon_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    xn = l2n_t(x)
    xhn = l2n_t(xhat)
    cos = 1.0 - (xn * xhn).sum(dim=-1).mean()
    mse = F.mse_loss(xhn, xn)
    return cos + mse


def geom_gram_loss(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    kx = pairwise_cosine_gram(x)
    kz = pairwise_cosine_gram(z)
    return F.mse_loss(kz, kx)


def cycle_loss(model: DualEncoder, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    # A→B→A
    xb_hat = model.translate_a_to_b(xa)
    xa_cyc = model.translate_b_to_a(xb_hat)
    # B→A→B
    xa_hat = model.translate_b_to_a(xb)
    xb_cyc = model.translate_a_to_b(xa_hat)
    return recon_loss(xa, xa_cyc) + recon_loss(xb, xb_cyc)


@dataclass
class LossWeights:
    recon: float = 1.0
    mmd: float = 1.0
    cycle: float = 1.0
    geom: float = 1.0


def unpaired_batch_loss(
    model: DualEncoder,
    xa: torch.Tensor,
    xb: torch.Tensor,
    w: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    za, xa_hat = model.forward_a(xa)
    zb, xb_hat = model.forward_b(xb)
    l_recon = recon_loss(xa, xa_hat) + recon_loss(xb, xb_hat)
    l_mmd = mmd_rbf(za, zb)
    l_cycle = cycle_loss(model, xa, xb)
    l_geom = geom_gram_loss(xa, za) + geom_gram_loss(xb, zb)
    total = (
        w.recon * l_recon
        + w.mmd * l_mmd
        + w.cycle * l_cycle
        + w.geom * l_geom
    )
    stats = {
        "loss": float(total.detach().item()),
        "recon": float(l_recon.detach().item()),
        "mmd": float(l_mmd.detach().item()),
        "cycle": float(l_cycle.detach().item()),
        "geom": float(l_geom.detach().item()),
    }
    return total, stats


# ---------------------------------------------------------------------------
# SAE resolution / representations
# ---------------------------------------------------------------------------


def resolve_sae_dir(root: Path, model: str) -> Path:
    parquet, col = MODELS[model]
    stem = Path(parquet).stem
    base = root / "outputs" / "sae" / stem / col
    if not base.is_dir():
        # fall back to _shared helper path layout
        for tag in SAE_TAG_PREFER:
            cand = _shared_sae_dir(root, model, tag)
            if (cand / "model.pt").is_file():
                return cand
        raise FileNotFoundError(f"No SAE directory for {model} under {base}")
    tags = {
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / "model.pt").is_file()
    }
    for tag in SAE_TAG_PREFER:
        if tag in tags:
            return base / tag
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "model.pt").is_file():
            return p
    raise FileNotFoundError(f"No SAE checkpoint for {model} in {base}")


@torch.no_grad()
def encode_sae(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    """Encode with no_grad (not inference_mode) for safety with later mutations."""
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    model = bundle["model"]
    model.eval()
    for i in range(0, len(xs), bs):
        xb = torch.as_tensor(xs[i : i + bs], device=device, dtype=torch.float32)
        _, z = model(xb)
        outs.append(z.detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def fit_ridge_map_b_to_a(
    C_a: np.ndarray,
    C_b: np.ndarray,
    train_idx: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit C_A ≈ C_B W + b on train_idx only (standardized Ridge)."""
    x_tr = C_b[train_idx]
    y_tr = C_a[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    return {
        "x_scaler": x_sc,
        "y_scaler": y_sc,
        "ridge": ridge,
        "coef": ridge.coef_.astype(np.float64),
        "intercept": ridge.intercept_.astype(np.float64),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "alpha": float(alpha),
    }


def map_b_to_a(bundle: dict[str, Any], C_b: np.ndarray) -> np.ndarray:
    xs = bundle["x_scaler"].transform(C_b)
    ys = bundle["ridge"].predict(xs)
    y = bundle["y_scaler"].inverse_transform(ys)
    return np.maximum(y, 0.0).astype(np.float32)


def apply_decoder_shrinkage(
    codes: np.ndarray,
    sae_bundle: dict,
    alpha: float = DECODER_ALPHA,
) -> np.ndarray:
    """Cheap fixed-alpha decoder-metric style features (optional).

    Uses decoder weight columns as a linear metric: features ~ (1-α)·codes + α·(W_dec @ codes)
    projected / normalized in ambient-ish space when possible; falls back to scaled codes.
    """
    model = sae_bundle["model"]
    W = None
    for name, p in model.named_parameters():
        if "decoder" in name.lower() and p.ndim == 2:
            W = p.detach().float().cpu().numpy()
            break
    if W is None:
        return (codes * (1.0 - alpha)).astype(np.float32)
    # W: [d_in, F] or [F, d_in]; try both
    if W.shape[1] == codes.shape[1]:
        ambient = codes @ W.T
    elif W.shape[0] == codes.shape[1]:
        ambient = codes @ W
    else:
        return (codes * (1.0 - alpha)).astype(np.float32)
    # blend code energy with decoder ambient embedding (concat then L2)
    feat = np.concatenate(
        [(1.0 - alpha) * codes, alpha * ambient.astype(np.float32)], axis=1
    )
    return l2n_np(feat.astype(np.float32))


def write_representation_audit(
    path: Path,
    *,
    model_a: str,
    model_b: str,
    sae_a: Path,
    sae_b: Path,
    n_oracle: int,
    n_a_train: int,
    n_b_train: int,
    n_eval: int,
) -> None:
    text = f"""# Representation audit — unpaired universal geometry

This file is generated at runtime by `run_unpaired_universal_geometry.py`.
It documents the exact representation conventions used for the smoke.

## Models

- Domain A: `{model_a}`
- Domain B: `{model_b}`
- SAE A: `{sae_a}`
- SAE B: `{sae_b}`

Preferred SAE tag order: `{list(SAE_TAG_PREFER)}`.

## Dense

- Ambient embeddings loaded from Physics parquet columns (`MODELS`).
- Features = L2-normalized embeddings (`l2n`).
- No paired fitting.

## sae_shared (shared-basis SAE)

Convention matches the improved cross-model mKNN protocol:

1. Encode TopK SAE codes `C_A`, `C_B` for all subsampled rows (existing checkpoints only; no SAE training).
2. Fit Ridge map **only on `paired_oracle_train`** (n={n_oracle}):
   - Target: `C_A`
   - Predictor: `C_B`
   - Form: standardized Ridge `C_A ≈ map(C_B)` i.e. `y_sc(C_A) ≈ x_sc(C_B) @ W.T + b` (sklearn Ridge).
3. IDF weights computed **only from A_train codes on basis A** (n={n_a_train}):
   - `idf_A = log((n+1)/(df+1)) + 1` with `df = (# nonzero over A_train)`.
4. Features used by unpaired / oracle models:
   - A: `C_A * idf_A`
   - B: `mapped_to_A(C_B) * idf_A` then (optionally) L2-normalized for training stability

### Leakage rules (enforced)

- **Never** fit `W` / scalers on `paired_eval`, `A_train`, or `B_train`.
- `paired_oracle_train` is the sole supervised correspondence set for the Ridge shared-basis map and for paired-oracle translators.
- Unpaired DualEncoder training uses only `A_train` / `B_train` (disjoint IDs); checkpoint selection uses unsupervised val slices (10% of A_train/B_train), never paired metrics.
- `paired_eval` (n={n_eval}) is evaluation-only.

## Optional sae_decoder_shrinkage

- CLI `--include-decoder-metric` (default **off** for smoke).
- Fixed `alpha={DECODER_ALPHA}` from prior decoder-metric experiments (not tuned here).
- Applied after shared-basis mapping / IDF when enabled.

## Split roles

| role | n (smoke defaults) | used for |
|------|-------------------:|----------|
| paired_oracle_train | {n_oracle} | Ridge W; paired oracle translators |
| A_train | {n_a_train} | unpaired encoder A; IDF_A |
| B_train | {n_b_train} | unpaired encoder B |
| paired_eval | {n_eval} | all held-out paired metrics |

All roles are pairwise disjoint (asserted at runtime). Galaxy index = row id after the deterministic `max_n` subsample (seed=0).
"""
    path.write_text(text)


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------


def build_split(
    n: int,
    n_oracle: int,
    n_a: int,
    n_b: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    need = n_oracle + n_a + n_b
    if need >= n:
        raise ValueError(
            f"split sizes too large: oracle+A+B={need} >= n={n}; need room for eval"
        )
    oracle = perm[:n_oracle]
    a_tr = perm[n_oracle : n_oracle + n_a]
    b_tr = perm[n_oracle + n_a : n_oracle + n_a + n_b]
    eval_ids = perm[n_oracle + n_a + n_b :]
    sets = {
        "paired_oracle_train": np.sort(oracle),
        "A_train": np.sort(a_tr),
        "B_train": np.sort(b_tr),
        "paired_eval": np.sort(eval_ids),
    }
    # pairwise disjoint assertions
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = np.intersect1d(sets[names[i]], sets[names[j]])
            assert inter.size == 0, f"{names[i]} ∩ {names[j]} nonempty ({inter.size})"
    return sets


def write_split_manifest(path: Path, split: dict[str, np.ndarray]) -> None:
    rows = []
    for role, ids in split.items():
        for gid in ids:
            rows.append({"galaxy_idx": int(gid), "role": role})
    pd.DataFrame(rows).sort_values(["role", "galaxy_idx"]).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Soft retrieval (optional)
# ---------------------------------------------------------------------------


@torch.no_grad()
def soft_js_metrics(
    Q_pred: torch.Tensor,
    Q_true: torch.Tensor,
    G_true: torch.Tensor,
    keffs: Iterable[int],
    batch: int = 128,
) -> list[dict[str, float]]:
    """Soft neighbourhood agreement over a *shared* true-target gallery.

    For each query index i:
      P_pred = softmax(cos(Q_pred_i, G_true) / τ)
      P_true = softmax(cos(Q_true_i, G_true) / τ)
    so both distributions are over the same gallery columns.
    """
    if not _HAS_SOFT or calibrate_tau_for_keff_cosine is None:
        return []
    if Q_pred.shape[0] != Q_true.shape[0]:
        raise ValueError("Q_pred and Q_true must have same number of queries")
    Qp = l2n_t(Q_pred)
    Qt = l2n_t(Q_true)
    Gt = l2n_t(G_true)
    rows = []
    for keff in keffs:
        tau_p, med_p = calibrate_tau_for_keff_cosine(Qp, Gt, float(keff), batch)
        tau_t, med_t = calibrate_tau_for_keff_cosine(Qt, Gt, float(keff), batch)
        # shared temperature for fair JS comparison
        tau = 0.5 * (tau_p + tau_t)
        logp = cosine_logits(Qp, Gt, tau, batch)
        logt = cosine_logits(Qt, Gt, tau, batch)
        js = js_similarity_rows(logp, logt)
        row = {
            "keff_target": float(keff),
            "tau": float(tau),
            "keff_pred_med": float(med_p),
            "keff_true_med": float(med_t),
            "js_sim_mean": float(js.mean().item()),
            "js_sim_median": float(js.median().item()),
        }
        if hellinger_rows is not None:
            row["hellinger_mean"] = float(hellinger_rows(logp, logt).mean().item())
        if tv_rows is not None:
            row["tv_mean"] = float(tv_rows(logp, logt).mean().item())
        rows.append(row)
    return rows


@torch.no_grad()
def distance_geometry_metrics(
    Za: torch.Tensor,
    Zb: torch.Tensor,
    Ytrue: torch.Tensor,
) -> dict[str, float]:
    """Coordinate-free geometry on a subset: cosine-distance Spearman/Pearson + CKA."""
    An = l2n_t(Za).cpu().numpy()
    Bn = l2n_t(Zb).cpu().numpy()
    Tn = l2n_t(Ytrue).cpu().numpy()
    # pairwise cosine distances (1 - cos), upper triangle
    def dist_vec(X: np.ndarray) -> np.ndarray:
        G = X @ X.T
        D = 1.0 - G
        iu = np.triu_indices(len(X), k=1)
        return D[iu]

    da = dist_vec(An)
    db = dist_vec(Bn)
    dt = dist_vec(Tn)
    return {
        "spearman_latentA_vs_trueB": spearman_corr(da, dt),
        "spearman_latentB_vs_trueB": spearman_corr(db, dt),
        "spearman_transA_vs_trueB": spearman_corr(db, dt),  # placeholder overwrite below
        "pearson_latentA_vs_trueB": pearson_corr(da, dt),
        "cka_latentA_trueB": linear_cka(An, Tn),
        "cka_latentB_trueB": linear_cka(Bn, Tn),
    }


# ---------------------------------------------------------------------------
# Paired oracle
# ---------------------------------------------------------------------------


def fit_ridge_full(
    Xa: np.ndarray, Xb: np.ndarray, train_idx: np.ndarray, alpha: float
) -> dict[str, Any]:
    x_sc = StandardScaler().fit(Xa[train_idx])
    y_sc = StandardScaler().fit(Xb[train_idx])
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(Xa[train_idx]), y_sc.transform(Xb[train_idx]))
    return {"x_scaler": x_sc, "y_scaler": y_sc, "ridge": ridge}


def predict_ridge(bundle: dict[str, Any], Xa: np.ndarray) -> np.ndarray:
    return bundle["y_scaler"].inverse_transform(
        bundle["ridge"].predict(bundle["x_scaler"].transform(Xa))
    ).astype(np.float32)


def reduced_rank_predict(
    bundle: dict[str, Any], Xa: np.ndarray, rank: int
) -> np.ndarray:
    ridge = bundle["ridge"]
    # sklearn: y = X @ coef_.T + intercept_
    W = ridge.coef_.T  # [d_in, d_out]
    b = ridge.intercept_
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(rank, U.shape[1])
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]
    Xs = bundle["x_scaler"].transform(Xa)
    Ys = Xs @ Wr + b
    return bundle["y_scaler"].inverse_transform(Ys).astype(np.float32)


def train_bottleneck_oracle(
    Xa_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xa_va: np.ndarray,
    Xb_va: np.ndarray,
    z_dim: int,
    hidden: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> BottleneckMLP:
    model = BottleneckMLP(Xa_tr.shape[1], Xb_tr.shape[1], z_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xa_tr)
    best_state = None
    best_val = float("inf")
    xa_va = to_torch(Xa_va, device)
    xb_va = to_torch(Xb_va, device)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xa = to_torch(Xa_tr[idx], device)
            xb = to_torch(Xb_tr[idx], device)
            pred = model(xa)
            loss = recon_loss(xb, pred)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(recon_loss(xb_va, model(xa_va)).item())
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def eval_translation(
    pred_b: torch.Tensor,
    true_b: torch.Tensor,
    *,
    row_batch: int,
    mknn_ks: Iterable[int] = MKNN_KS,
) -> dict[str, Any]:
    ranks = identity_ranks(pred_b, true_b, row_batch=row_batch).cpu().numpy()
    metrics = rank_metrics(ranks)
    pn = l2n_t(pred_b)
    tn = l2n_t(true_b)
    metrics["cosine"] = float((pn * tn).sum(dim=1).mean().item())
    metrics["mse"] = float(F.mse_loss(pn, tn).item())
    # mKNN translated vs true B
    max_k = max(mknn_ks)
    nn_pred = knn_cos(pred_b, k=max_k, row_batch=row_batch)
    nn_true = knn_cos(true_b, k=max_k, row_batch=row_batch)
    for k in mknn_ks:
        metrics[f"mknn_k{k}"] = mknn_score(nn_pred, nn_true, k)
    return metrics


def run_paired_oracle(
    FA: np.ndarray,
    FB: np.ndarray,
    split: dict[str, np.ndarray],
    *,
    oracle_zs: list[int],
    hidden: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    ridge_alpha: float,
    row_batch: int,
    rep_name: str,
) -> pd.DataFrame:
    tr = split["paired_oracle_train"]
    ev = split["paired_eval"]
    # small val slice from oracle train for MLP selection
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    n_va = max(1, int(0.1 * len(tr)))
    va_local = tr[perm[:n_va]]
    tr_fit = tr[perm[n_va:]]

    rows: list[dict[str, Any]] = []
    log(f"[oracle/{rep_name}] Ridge full …")
    ridge = fit_ridge_full(FA, FB, tr, alpha=ridge_alpha)
    pred = to_torch(predict_ridge(ridge, FA[ev]), device)
    true = to_torch(FB[ev], device)
    m = eval_translation(pred, true, row_batch=row_batch)
    m.update(
        {
            "rep": rep_name,
            "method": "ridge_full",
            "Z": int(min(FA.shape[1], FB.shape[1])),
            "n_train": int(len(tr)),
            "n_eval": int(len(ev)),
        }
    )
    rows.append(m)

    for z in oracle_zs:
        log(f"[oracle/{rep_name}] reduced-rank Z={z} …")
        pred = to_torch(reduced_rank_predict(ridge, FA[ev], z), device)
        m = eval_translation(pred, true, row_batch=row_batch)
        m.update(
            {
                "rep": rep_name,
                "method": "ridge_reduced_rank",
                "Z": int(z),
                "n_train": int(len(tr)),
                "n_eval": int(len(ev)),
            }
        )
        rows.append(m)

    for z in oracle_zs:
        log(f"[oracle/{rep_name}] bottleneck MLP Z={z} …")
        model = train_bottleneck_oracle(
            FA[tr_fit],
            FB[tr_fit],
            FA[va_local],
            FB[va_local],
            z_dim=z,
            hidden=hidden,
            device=device,
            epochs=min(epochs, 80),
            batch_size=batch_size,
            lr=lr,
        )
        with torch.no_grad():
            pred = model(to_torch(FA[ev], device))
        m = eval_translation(pred, true, row_batch=row_batch)
        m.update(
            {
                "rep": rep_name,
                "method": "bottleneck_mlp",
                "Z": int(z),
                "n_params": count_params(model),
                "hidden": hidden,
                "n_train": int(len(tr_fit)),
                "n_eval": int(len(ev)),
            }
        )
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unpaired training
# ---------------------------------------------------------------------------


def train_dual_encoder(
    FA: np.ndarray,
    FB: np.ndarray,
    a_train: np.ndarray,
    b_train: np.ndarray,
    *,
    z_dim: int,
    hidden: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    weights: LossWeights,
    seed: int,
) -> tuple[DualEncoder, pd.DataFrame, dict[str, Any]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    a_perm = rng.permutation(a_train)
    b_perm = rng.permutation(b_train)
    n_va_a = max(1, int(val_frac * len(a_perm)))
    n_va_b = max(1, int(val_frac * len(b_perm)))
    a_va, a_tr = a_perm[:n_va_a], a_perm[n_va_a:]
    b_va, b_tr = b_perm[:n_va_b], b_perm[n_va_b:]

    model = DualEncoder(FA.shape[1], FB.shape[1], z_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    curve_rows: list[dict[str, Any]] = []
    best_state = None
    best_val = float("inf")
    best_ep = -1

    xa_va = to_torch(FA[a_va], device)
    xb_va = to_torch(FB[b_va], device)

    n_tr = min(len(a_tr), len(b_tr))
    for ep in range(epochs):
        model.train()
        a_order = rng.permutation(a_tr)[:n_tr]
        b_order = rng.permutation(b_tr)[:n_tr]
        ep_stats = []
        for s in range(0, n_tr, batch_size):
            ia = a_order[s : s + batch_size]
            ib = b_order[s : s + batch_size]
            # align batch lengths
            m = min(len(ia), len(ib))
            if m < 4:
                continue
            xa = to_torch(FA[ia[:m]], device)
            xb = to_torch(FB[ib[:m]], device)
            loss, stats = unpaired_batch_loss(model, xa, xb, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_stats.append(stats)
        sched.step()

        model.eval()
        with torch.no_grad():
            # unsupervised val: same weighted losses on held-out A/B slices
            # subsample if large
            va_n = min(len(a_va), len(b_va), 1024)
            _, va_stats = unpaired_batch_loss(
                model, xa_va[:va_n], xb_va[:va_n], weights
            )
            # latent diagnostics
            za = model.encode_a(xa_va[:va_n])
            zb = model.encode_b(xb_va[:va_n])
            zcat = torch.cat([za, zb], dim=0).cpu().numpy()
            er = latent_effective_ranks(zcat)
            lat_std = float(zcat.std())

        mean_train = {
            k: float(np.mean([d[k] for d in ep_stats])) if ep_stats else float("nan")
            for k in ("loss", "recon", "mmd", "cycle", "geom")
        }
        row = {
            "epoch": ep,
            "lr": float(sched.get_last_lr()[0]),
            "train_loss": mean_train["loss"],
            "train_recon": mean_train["recon"],
            "train_mmd": mean_train["mmd"],
            "train_cycle": mean_train["cycle"],
            "train_geom": mean_train["geom"],
            "val_loss": va_stats["loss"],
            "val_recon": va_stats["recon"],
            "val_mmd": va_stats["mmd"],
            "val_cycle": va_stats["cycle"],
            "val_geom": va_stats["geom"],
            "latent_std": lat_std,
            **{f"lat_{k}": v for k, v in er.items()},
        }
        curve_rows.append(row)
        if va_stats["loss"] < best_val:
            best_val = va_stats["loss"]
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            log(
                f"  ep {ep+1:03d}/{epochs} train={mean_train['loss']:.4f} "
                f"val={va_stats['loss']:.4f} best={best_val:.4f}@{best_ep+1}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {
        "best_epoch": int(best_ep),
        "best_val_loss": float(best_val),
        "n_params": count_params(model),
        "hidden": hidden,
        "Z": z_dim,
        "n_a_train": int(len(a_tr)),
        "n_b_train": int(len(b_tr)),
        "n_a_val": int(len(a_va)),
        "n_b_val": int(len(b_va)),
    }
    return model, pd.DataFrame(curve_rows), meta


@torch.no_grad()
def evaluate_unpaired_model(
    model: DualEncoder,
    FA: np.ndarray,
    FB: np.ndarray,
    eval_ids: np.ndarray,
    *,
    device: torch.device,
    row_batch: int,
    n_geometry: int,
    n_soft_query: int,
    n_shuffles: int,
    skip_soft: bool,
    seed: int,
) -> dict[str, Any]:
    ev = eval_ids
    xa = to_torch(FA[ev], device)
    xb = to_torch(FB[ev], device)
    pred_b = model.translate_a_to_b(xa)
    pred_a = model.translate_b_to_a(xb)
    za = model.encode_a(xa)
    zb = model.encode_b(xb)

    # Identity ranking vs paired_eval gallery
    id_ab = eval_translation(pred_b, xb, row_batch=row_batch)
    id_ba = eval_translation(pred_a, xa, row_batch=row_batch)

    # Shared-latent cross-model mKNN
    max_k = max(MKNN_KS)
    nn_za = knn_cos(za, k=max_k, row_batch=row_batch)
    nn_zb = knn_cos(zb, k=max_k, row_batch=row_batch)
    latent_mknn = {f"latent_mknn_k{k}": mknn_score(nn_za, nn_zb, k) for k in MKNN_KS}
    zn = l2n_t(za)
    zbn = l2n_t(zb)
    paired_lat_cos = float((zn * zbn).sum(dim=1).mean().item())
    paired_lat_l2 = float((za - zb).norm(dim=1).mean().item())
    # shuffle latent pairing
    rng = np.random.default_rng(seed + 17)
    shuf = rng.permutation(len(ev))
    shuf_cos = float((zn * zbn[shuf]).sum(dim=1).mean().item())
    shuf_l2 = float((za - zb[shuf]).norm(dim=1).mean().item())

    # Soft JS (subsample queries; shared true-B gallery)
    soft_rows: list[dict[str, Any]] = []
    if not skip_soft and _HAS_SOFT:
        nq = min(n_soft_query, len(ev))
        qidx = rng.choice(len(ev), size=nq, replace=False)
        soft_rows = soft_js_metrics(
            pred_b[qidx], xb[qidx], xb, keffs=SOFT_KEFFS, batch=64
        )

    # Distance geometry on subset
    ng = min(n_geometry, len(ev))
    gidx = rng.choice(len(ev), size=ng, replace=False)
    # translated features vs true B geometry
    pred_sub = pred_b[gidx]
    true_sub = xb[gidx]
    za_sub = za[gidx]
    zb_sub = zb[gidx]
    geom = distance_geometry_metrics(za_sub, zb_sub, true_sub)
    # overwrite translated spearman using pred vs true distances
    def _dvec(X: torch.Tensor) -> np.ndarray:
        Xn = l2n_t(X).cpu().numpy()
        G = Xn @ Xn.T
        D = 1.0 - G
        iu = np.triu_indices(len(Xn), k=1)
        return D[iu]

    d_pred = _dvec(pred_sub)
    d_true = _dvec(true_sub)
    geom["spearman_transA_vs_trueB"] = spearman_corr(d_pred, d_true)
    geom["pearson_transA_vs_trueB"] = pearson_corr(d_pred, d_true)
    geom["cka_transA_trueB"] = linear_cka(
        l2n_t(pred_sub).cpu().numpy(), l2n_t(true_sub).cpu().numpy()
    )

    # Latent ranks
    lat_ranks = latent_effective_ranks(
        torch.cat([za, zb], dim=0).cpu().numpy()
    )

    # Identity shuffle null
    null_top1 = []
    null_mrr = []
    ranks_true = identity_ranks(pred_b, xb, row_batch=row_batch).cpu().numpy()
    for _ in range(n_shuffles):
        perm = rng.permutation(len(ev))
        # rank of true identity under shuffled gallery labels ≈ compare pred to shuffled targets
        # Equivalent: permute true targets and recompute diagonal ranks
        ranks_s = identity_ranks(pred_b, xb[perm], row_batch=row_batch).cpu().numpy()
        # Under shuffled correspondence, the "true" index i matches gallery[perm^{-1}?];
        # identity_ranks assumes query i ↔ gallery i. So shuffle gallery rows:
        rm = rank_metrics(ranks_s)
        null_top1.append(rm["top1"])
        null_mrr.append(rm["mrr"])

    out = {
        "identity_A_to_B": id_ab,
        "identity_B_to_A": id_ba,
        "latent_mknn": latent_mknn,
        "paired_latent_cosine": paired_lat_cos,
        "paired_latent_l2": paired_lat_l2,
        "shuffled_latent_cosine": shuf_cos,
        "shuffled_latent_l2": shuf_l2,
        "geometry": geom,
        "latent_ranks": lat_ranks,
        "soft_rows": soft_rows,
        "null_top1_mean": float(np.mean(null_top1)) if null_top1 else float("nan"),
        "null_top1_std": float(np.std(null_top1)) if null_top1 else float("nan"),
        "null_mrr_mean": float(np.mean(null_mrr)) if null_mrr else float("nan"),
        "null_mrr_std": float(np.std(null_mrr)) if null_mrr else float("nan"),
        "true_ranks": ranks_true,
    }
    return out


# ---------------------------------------------------------------------------
# Saturation / aggregates / figures / report
# ---------------------------------------------------------------------------


def saturation_table(df: pd.DataFrame, metric_cols: list[str], z_col: str = "Z") -> pd.DataFrame:
    """Z90/Z95/Z99 per metric; higher-is-better assumed except *rank* metrics."""
    rows = []
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=["rep", "method", "metric", "best", "Z90", "Z95", "Z99"]
        )
    group_cols = [c for c in ("rep", "method", "direction", "ablation") if c in df.columns]
    if not group_cols:
        groups = [("all", df)]
    else:
        groups = list(df.groupby(group_cols, dropna=False))

    for gkey, g in groups:
        if not isinstance(gkey, tuple):
            gkey = (gkey,)
        meta = dict(zip(group_cols, gkey))
        for metric in metric_cols:
            if metric not in g.columns:
                continue
            # aggregate across seeds
            gg = g.groupby(z_col, as_index=False)[metric].mean().sort_values(z_col)
            vals = gg[metric].to_numpy(dtype=np.float64)
            zs = gg[z_col].to_numpy(dtype=np.int64)
            if len(vals) == 0 or np.all(~np.isfinite(vals)):
                continue
            lower_better = "rank" in metric.lower() and "mrr" not in metric.lower()
            if lower_better:
                best = float(np.nanmin(vals))
                # convert to score = -val for thresholding
                score = -vals
                best_score = float(np.nanmax(score))
            else:
                best = float(np.nanmax(vals))
                score = vals
                best_score = best
            if abs(best_score) < EPS:
                z90 = z95 = z99 = int(zs[0])
            else:
                def first_hit(frac: float) -> int:
                    thr = frac * best_score
                    for z, sc in zip(zs, score):
                        if np.isfinite(sc) and sc >= thr:
                            return int(z)
                    return int(zs[-1])

                z90, z95, z99 = first_hit(0.90), first_hit(0.95), first_hit(0.99)
            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "best": best,
                    "Z90": z90,
                    "Z95": z95,
                    "Z99": z99,
                }
            )
    return pd.DataFrame(rows)


def _safe_plot(path: Path, plot_fn: Callable[[], None]) -> None:
    try:
        plot_fn()
        plt.tight_layout()
        plt.savefig(path, dpi=140)
    except Exception as exc:  # noqa: BLE001
        path.with_suffix(".error.txt").write_text(str(exc) + "\n")
    finally:
        plt.close("all")


def make_figures(
    fig_dir: Path,
    oracle_df: pd.DataFrame,
    unpaired_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    curves_df: pd.DataFrame,
) -> None:
    ensure_dir(fig_dir)

    def line_by(df: pd.DataFrame, y: str, title: str, fname: str, hue: str = "rep"):
        if df is None or len(df) == 0 or y not in df.columns:
            return

        def _p():
            for key, g in df.groupby(hue):
                gg = g.groupby("Z", as_index=False)[y].mean()
                plt.plot(gg["Z"], gg[y], marker="o", label=str(key))
            plt.xlabel("Z")
            plt.ylabel(y)
            plt.title(title)
            plt.legend()
            plt.xscale("log", base=2)

        _safe_plot(fig_dir / fname, _p)

    if oracle_df is not None and len(oracle_df):
        o = oracle_df[oracle_df["method"].isin(["ridge_reduced_rank", "bottleneck_mlp"])]
        line_by(o, "top1", "Paired oracle top-1 vs Z", "paired_oracle_top1_vs_Z.png", hue="method")
        line_by(o, "mrr", "Paired oracle MRR vs Z", "paired_oracle_mrr_vs_Z.png", hue="method")
        line_by(o, "mknn_k10", "Paired oracle mKNN@10 vs Z", "paired_oracle_mknn10_vs_Z.png", hue="method")

    if unpaired_df is not None and len(unpaired_df):
        u = unpaired_df[unpaired_df.get("ablation", "full").fillna("full") == "full"] if "ablation" in unpaired_df.columns else unpaired_df
        line_by(u, "top1", "Unpaired top-1 vs Z", "unpaired_top1_vs_Z.png")
        line_by(u, "mrr", "Unpaired MRR vs Z", "unpaired_mrr_vs_Z.png")
        line_by(u, "median_rank", "Unpaired median rank vs Z", "unpaired_median_rank_vs_Z.png")
        for k in (5, 10, 20, 50, 100):
            col = f"mknn_k{k}"
            if col in u.columns:
                line_by(u, col, f"Unpaired mKNN@{k} vs Z", f"unpaired_mknn_k{k}_vs_Z.png")
        if "cka_transA_trueB" in u.columns:
            line_by(u, "cka_transA_trueB", "Kernel CKA vs Z", "cka_vs_Z.png")
        if "spearman_transA_vs_trueB" in u.columns:
            line_by(
                u,
                "spearman_transA_vs_trueB",
                "Distance Spearman vs Z",
                "distance_spearman_vs_Z.png",
            )
        if "js_keff10" in u.columns:
            line_by(u, "js_keff10", "JS sim keff=10 vs Z", "js_keff10_vs_Z.png")
        if "js_keff500" in u.columns:
            line_by(u, "js_keff500", "JS sim keff=500 vs Z", "js_keff500_vs_Z.png")
        if {"top1", "mknn_k100"}.issubset(u.columns):
            def _p():
                for rep, g in u.groupby("rep"):
                    gg = g.groupby("Z", as_index=False)[["top1", "mknn_k100"]].mean()
                    plt.plot(gg["Z"], gg["top1"], marker="o", label=f"{rep} top1")
                    plt.plot(
                        gg["Z"],
                        gg["mknn_k100"],
                        marker="s",
                        linestyle="--",
                        label=f"{rep} mknn100",
                    )
                plt.xlabel("Z")
                plt.title("Point identity vs broad geometry")
                plt.legend()
                plt.xscale("log", base=2)

            _safe_plot(fig_dir / "identity_vs_geometry_saturation.png", _p)
        if "lat_stable_rank" in u.columns:
            def _p():
                for rep, g in u.groupby("rep"):
                    gg = g.groupby("Z", as_index=False)["lat_stable_rank"].mean()
                    plt.plot(gg["Z"], gg["lat_stable_rank"], marker="o", label=str(rep))
                plt.plot([16, 256], [16, 256], "k--", alpha=0.4, label="nominal")
                plt.xlabel("nominal Z")
                plt.ylabel("latent stable rank")
                plt.title("Nominal Z vs effective rank")
                plt.legend()

            _safe_plot(fig_dir / "nominal_vs_effective_rank.png", _p)
        if {"paired_latent_cosine", "shuffled_latent_cosine"}.issubset(u.columns):
            def _p():
                for rep, g in u.groupby("rep"):
                    gg = g.groupby("Z", as_index=False)[
                        ["paired_latent_cosine", "shuffled_latent_cosine"]
                    ].mean()
                    plt.plot(gg["Z"], gg["paired_latent_cosine"], marker="o", label=f"{rep} paired")
                    plt.plot(
                        gg["Z"],
                        gg["shuffled_latent_cosine"],
                        marker="x",
                        linestyle="--",
                        label=f"{rep} shuffled",
                    )
                plt.xlabel("Z")
                plt.ylabel("latent cosine")
                plt.title("Paired vs shuffled latent distance (cosine)")
                plt.legend()
                plt.xscale("log", base=2)

            _safe_plot(fig_dir / "paired_vs_shuffled_latent.png", _p)

        # dense vs SAE
        if "rep" in u.columns and u["rep"].nunique() > 1:
            line_by(u, "top1", "Dense vs SAE top-1", "dense_vs_sae_top1.png")
            line_by(u, "mknn_k10", "Dense vs SAE mKNN@10", "dense_vs_sae_mknn10.png")

        # supervised vs unpaired capacity
        if oracle_df is not None and len(oracle_df) and "top1" in u.columns:
            def _p():
                oo = oracle_df[oracle_df["method"] == "bottleneck_mlp"]
                for rep in sorted(set(u["rep"]).union(set(oo["rep"]) if "rep" in oo else [])):
                    uu = u[u["rep"] == rep].groupby("Z", as_index=False)["top1"].mean()
                    plt.plot(uu["Z"], uu["top1"], marker="o", label=f"unpaired/{rep}")
                    if len(oo):
                        orep = oo[oo["rep"] == rep] if "rep" in oo.columns else oo
                        if len(orep):
                            gg = orep.groupby("Z", as_index=False)["top1"].mean()
                            plt.plot(
                                gg["Z"],
                                gg["top1"],
                                marker="s",
                                linestyle="--",
                                label=f"oracle/{rep}",
                            )
                plt.xlabel("Z")
                plt.ylabel("top-1")
                plt.title("Supervised vs unpaired capacity")
                plt.legend()
                plt.xscale("log", base=2)

            _safe_plot(fig_dir / "supervised_vs_unpaired_capacity.png", _p)

    if sat_df is not None and len(sat_df):
        def _p():
            sub = sat_df.copy()
            metrics = list(sub["metric"].unique())[:12]
            sub = sub[sub["metric"].isin(metrics)]
            x = np.arange(len(sub))
            plt.bar(x - 0.2, sub["Z90"], width=0.2, label="Z90")
            plt.bar(x, sub["Z95"], width=0.2, label="Z95")
            plt.bar(x + 0.2, sub["Z99"], width=0.2, label="Z99")
            plt.xticks(x, [f"{r}/{m}" for r, m in zip(sub.get("rep", ["?"]*len(sub)), sub["metric"])], rotation=90, fontsize=7)
            plt.ylabel("Z")
            plt.title("Z90/Z95/Z99 by metric")
            plt.legend()

        _safe_plot(fig_dir / "Z90_Z95_Z99_by_metric.png", _p)

    if curves_df is not None and len(curves_df):
        def _p():
            g = curves_df
            # plot a few representative curves
            keys = g[["rep", "Z", "seed"]].drop_duplicates().head(6)
            for _, r in keys.iterrows():
                mask = (
                    (g["rep"] == r["rep"])
                    & (g["Z"] == r["Z"])
                    & (g["seed"] == r["seed"])
                )
                gg = g[mask]
                plt.plot(gg["epoch"], gg["val_loss"], label=f"{r['rep']} Z{r['Z']} s{r['seed']}")
            plt.xlabel("epoch")
            plt.ylabel("val unsupervised loss")
            plt.title("Training curves (val)")
            plt.legend(fontsize=7)

        _safe_plot(fig_dir / "training_loss_curves.png", _p)


def compute_gates(
    oracle_df: pd.DataFrame,
    unpaired_df: pd.DataFrame,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    # Gate A
    if oracle_df is not None and len(oracle_df):
        best_top1 = float(oracle_df["top1"].max())
        best_mrr = float(oracle_df["mrr"].max())
        gates["A_paired_oracle_works"] = bool(best_top1 > 0.05 or best_mrr > 0.05)
        gates["A_detail"] = {"best_top1": best_top1, "best_mrr": best_mrr}
    else:
        gates["A_paired_oracle_works"] = False
        gates["A_detail"] = {"skipped": True}

    if unpaired_df is None or len(unpaired_df) == 0:
        for k in ("B", "C", "D", "E"):
            gates[f"{k}_pass"] = False
        return gates

    u = unpaired_df.copy()
    if "ablation" in u.columns:
        u = u[u["ablation"].fillna("full") == "full"]

    # Gate B: beat nulls
    beat = False
    if "null_top1_mean" in u.columns:
        beat = bool(((u["top1"] - u["null_top1_mean"]) > 2.0 * u["null_top1_std"].clip(lower=1e-6)).any())
    if "null_mrr_mean" in u.columns:
        beat = beat or bool(
            ((u["mrr"] - u["null_mrr_mean"]) > 2.0 * u["null_mrr_std"].clip(lower=1e-6)).any()
        )
    if "mknn_k10" in u.columns:
        beat = beat or bool((u["mknn_k10"] > 0.05).any())
    gates["B_unpaired_learns_something"] = beat

    # Gate C: relational geometry
    geom_ok = False
    if "mknn_k10" in u.columns:
        geom_ok = bool((u["mknn_k10"] > 0.05).any())
    if "cka_transA_trueB" in u.columns:
        geom_ok = geom_ok or bool((u["cka_transA_trueB"] > 0.1).any())
    if "js_keff50" in u.columns:
        geom_ok = geom_ok or bool((u["js_keff50"] > 0.05).any())
    gates["C_relational_geometry_recoverable"] = geom_ok

    # Gate D: dimension matters
    dim_matters = False
    for rep, g in u.groupby("rep"):
        gg = g.groupby("Z", as_index=False)["mrr"].mean()
        if len(gg) >= 2 and float(gg["mrr"].max() - gg["mrr"].min()) > 0.01:
            dim_matters = True
    gates["D_dimension_matters"] = dim_matters

    # Gate E: SAE helps
    sae_helps = False
    if set(u["rep"]) >= {"dense", "sae_shared"}:
        for metric in ("top1", "mrr", "mknn_k10", "cka_transA_trueB"):
            if metric not in u.columns:
                continue
            d = u[u["rep"] == "dense"].groupby("Z")[metric].mean()
            s = u[u["rep"] == "sae_shared"].groupby("Z")[metric].mean()
            common = sorted(set(d.index) & set(s.index))
            if common and any(s.loc[z] > d.loc[z] + 1e-3 for z in common):
                sae_helps = True
                break
    gates["E_sae_helps"] = sae_helps
    return gates


def write_report(
    path: Path,
    *,
    config: dict[str, Any],
    gates: dict[str, Any],
    oracle_df: pd.DataFrame,
    unpaired_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    dense_vs_sae: pd.DataFrame,
    anti_cheat: list[str],
) -> None:
    def best_row(df: pd.DataFrame, metric: str) -> str:
        if df is None or len(df) == 0 or metric not in df.columns:
            return "n/a"
        i = df[metric].idxmax() if "rank" not in metric else df[metric].idxmin()
        r = df.loc[i]
        return (
            f"{metric}={r[metric]:.4f} (rep={r.get('rep','?')}, Z={r.get('Z','?')}, "
            f"seed={r.get('seed','?')})"
        )

    u = unpaired_df
    if u is not None and len(u) and "ablation" in u.columns:
        u_full = u[u["ablation"].fillna("full") == "full"]
    else:
        u_full = u

    answers: dict[str, str] = {}
    answers["1"] = (
        "Yes, partially — Gate B "
        f"{'PASS' if gates.get('B_unpaired_learns_something') else 'FAIL'}: "
        "unpaired metrics vs identity-shuffle nulls / mKNN threshold."
        if u_full is not None and len(u_full)
        else "Inconclusive (no unpaired results)."
    )
    answers["2"] = best_row(u_full, "top1") + "; " + best_row(u_full, "mrr")
    answers["3"] = best_row(u_full, "mknn_k10") + "; " + best_row(u_full, "cka_transA_trueB")
    # 4 hierarchy
    if sat_df is not None and len(sat_df):
        def z95(metric: str) -> float:
            sub = sat_df[sat_df["metric"] == metric]
            return float(sub["Z95"].median()) if len(sub) else float("nan")

        z_broad = z95("mknn_k100")
        z_fine = z95("mknn_k10")
        z_id = z95("top1")
        answers["4"] = (
            f"Z95(broad mknn100)={z_broad}, Z95(fine mknn10)={z_fine}, Z95(top1)={z_id}. "
            + (
                "Broad geometry saturates earlier than identity."
                if np.isfinite(z_broad) and np.isfinite(z_id) and z_broad < z_id
                else "No clear broad-before-identity hierarchy in this smoke."
            )
        )
        answers["6"] = f"broad geometry Z90/95/99 from sat table (mknn_k100 / js_keff500)."
        answers["7"] = f"fine geometry Z95(mknn_k10)={z_fine}"
        answers["8"] = f"identity Z95(top1)={z_id}"
    else:
        answers["4"] = "Saturation table empty."
        answers["6"] = answers["7"] = answers["8"] = "n/a"

    answers["5"] = (
        "See unpaired_*_vs_Z figures; Gate D="
        f"{'PASS' if gates.get('D_dimension_matters') else 'FAIL'}."
    )
    if oracle_df is not None and len(oracle_df) and sat_df is not None and len(sat_df):
        answers["9"] = (
            "Compare oracle bottleneck_mlp saturation vs unpaired in "
            "dimension_saturation.csv / supervised_vs_unpaired_capacity.png."
        )
    else:
        answers["9"] = "Oracle skipped or unavailable."

    answers["10"] = (
        f"Gate E={'PASS' if gates.get('E_sae_helps') else 'FAIL'}. See dense_vs_sae.csv."
    )
    answers["11"] = (
        "Compare Z95 columns for dense vs sae_shared in dimension_saturation.csv."
    )
    answers["12"] = (
        best_row(u_full, "lat_stable_rank")
        if u_full is not None and "lat_stable_rank" in getattr(u_full, "columns", [])
        else "See latent_rank_results.parquet."
    )

    # ablations
    if u is not None and "ablation" in u.columns and u["ablation"].nunique() > 1:
        answers["13"] = "See ablation rows (drop_geom) vs full in unpaired_results."
        answers["14"] = "See ablation rows (drop_cycle) vs full."
        answers["15"] = "See ablation rows (drop_mmd) vs full."
    else:
        answers["13"] = answers["14"] = answers["15"] = (
            "Ablations not run (pass --run-ablations)."
        )

    if u_full is not None and len(u_full) and "paired_latent_cosine" in u_full.columns:
        d = float(
            (u_full["paired_latent_cosine"] - u_full["shuffled_latent_cosine"]).mean()
        )
        answers["16"] = f"Mean paired−shuffled latent cosine = {d:.4f}."
    else:
        answers["16"] = "n/a"

    answers["17"] = (
        "If Z95 identity ≫ Z95 broad geometry → high-dim relational geometry with "
        "lower-dim coarse backbone; if all Z95 small → low-dim shared manifold; "
        "if unpaired fails while oracle works → structure exists but unpaired objectives "
        "insufficient. See gates + saturation."
    )
    answers["18"] = answers["4"]

    lines = [
        "# Unpaired Universal Geometry — Smoke Report",
        "",
        "## Config",
        "```json",
        json.dumps(config, indent=2, default=str),
        "```",
        "",
        "## Anti-cheating checks",
    ]
    lines += [f"- {c}" for c in anti_cheat]
    lines += ["", "## Gates A–E", "```json", json.dumps(gates, indent=2, default=str), "```", ""]
    lines += ["## Answers (questions 1–18)", ""]
    qs = [
        "Can correspondence be recovered without paired training galaxies?",
        "How strong is unpaired identity recovery?",
        "How strong is unpaired relational-geometry recovery?",
        "Does broad relational geometry emerge before exact point identity?",
        "How does performance depend on latent dimension Z?",
        "What Z is required for 90/95/99% of broad geometry performance?",
        "What Z is required for fine geometry?",
        "What Z is required for exact identity?",
        "Does the supervised oracle saturate at a different Z?",
        "Does SAE improve unpaired alignment relative to dense?",
        "Does SAE reduce the latent dimension required?",
        "Is nominal Z actually used, or does the latent collapse?",
        "Does relational-geometry preservation matter critically?",
        "Does cycle consistency matter?",
        "Does distribution matching matter?",
        "Does the learned universal space bring paired galaxies closer than shuffled?",
        "Low-dimensional universal manifold or high-dimensional relational geometry?",
        "Evidence for hierarchy broad < fine < identity in required dimensionality?",
    ]
    for i, q in enumerate(qs, start=1):
        lines.append(f"### Q{i}. {q}")
        lines.append(answers.get(str(i), "n/a"))
        lines.append("")

    lines += [
        "## Gallery note",
        "",
        "Identity ranking uses the **paired_eval** gallery (not full n) for smoke speed.",
        "Null baselines: 100 identity shuffles of eval correspondence for top-1/MRR.",
        "",
        "## Interpretation matrix pointers",
        "",
        "- Broad@low Z, identity@high Z → low-dim relational backbone + high-dim identity.",
        "- All metrics saturate low → genuinely low-dimensional shared structure.",
        "- Both require high Z → high-dimensional universal relational geometry.",
        "- SAE lower Z than dense → SAE compresses alignment-friendly coordinates.",
        "- Unpaired fails / oracle succeeds → shared structure not identifiable unpaired.",
        "- Geometry-loss ablation destroys alignment → relational kernel is essential.",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_pair_embeddings(
    root: Path, model_a: str, model_b: str, max_n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pq_a, col_a = MODELS[model_a]
    pq_b, col_b = MODELS[model_b]
    Xa = load_col(resolve_path(root, pq_a), col_a, l2=False).astype(np.float32)
    Xb = load_col(resolve_path(root, pq_b), col_b, l2=False).astype(np.float32)
    n = min(len(Xa), len(Xb), max_n)
    rng = np.random.default_rng(seed)
    if len(Xa) != len(Xb):
        log(f"WARN: length mismatch {len(Xa)} vs {len(Xb)}; using min={n}")
    # common row ids 0..min-1 then subsample
    n_common = min(len(Xa), len(Xb))
    if n_common > max_n:
        sel = np.sort(rng.choice(n_common, size=max_n, replace=False))
    else:
        sel = np.arange(n_common, dtype=np.int64)
        n = n_common
    return Xa[sel], Xb[sel], sel


def build_representations(
    root: Path,
    model_a: str,
    model_b: str,
    Xa: np.ndarray,
    Xb: np.ndarray,
    split: dict[str, np.ndarray],
    device: torch.device,
    ridge_alpha: float,
    reps: list[str],
    include_decoder: bool,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    meta: dict[str, Any] = {}

    if "dense" in reps:
        out["dense"] = (l2n_np(Xa), l2n_np(Xb))
        meta["dense"] = {"dim": int(Xa.shape[1])}

    need_sae = any(r.startswith("sae") for r in reps) or include_decoder
    if need_sae or "sae_shared" in reps or "sae_decoder_shrinkage" in reps:
        sae_a_path = resolve_sae_dir(root, model_a)
        sae_b_path = resolve_sae_dir(root, model_b)
        log(f"SAE A: {sae_a_path}")
        log(f"SAE B: {sae_b_path}")
        bundle_a = _shared_load_sae(sae_a_path, device)
        bundle_b = _shared_load_sae(sae_b_path, device)
        # re-wrap encode with no_grad path
        Ca = encode_sae(bundle_a, Xa, device)
        Cb = encode_sae(bundle_b, Xb, device)
        oracle = split["paired_oracle_train"]
        a_tr = split["A_train"]
        ridge_map = fit_ridge_map_b_to_a(Ca, Cb, oracle, alpha=ridge_alpha)
        Cb_mapped = map_b_to_a(ridge_map, Cb)
        idf_a = idf_weights(Ca[a_tr])
        FA = (Ca * idf_a[None, :]).astype(np.float32)
        FB = (Cb_mapped * idf_a[None, :]).astype(np.float32)
        # L2 for training stability
        FA_n = l2n_np(FA)
        FB_n = l2n_np(FB)
        meta["sae"] = {
            "sae_a": str(sae_a_path),
            "sae_b": str(sae_b_path),
            "ridge_alpha": ridge_alpha,
            "idf_from": "A_train",
            "ridge_fit_on": "paired_oracle_train",
            "feature_dim": int(FA.shape[1]),
        }
        if "sae_shared" in reps:
            out["sae_shared"] = (FA_n, FB_n)
        if include_decoder or "sae_decoder_shrinkage" in reps:
            FA_d = apply_decoder_shrinkage(FA, bundle_a, alpha=DECODER_ALPHA)
            FB_d = apply_decoder_shrinkage(FB, bundle_a, alpha=DECODER_ALPHA)
            out["sae_decoder_shrinkage"] = (l2n_np(FA_d), l2n_np(FB_d))
            meta["sae_decoder_shrinkage"] = {"alpha": DECODER_ALPHA}
        meta["sae_paths"] = (sae_a_path, sae_b_path)
    return out, meta


def flatten_eval_row(
    ev: dict[str, Any],
    *,
    rep: str,
    Z: int,
    seed: int,
    meta: dict[str, Any],
    ablation: str = "full",
    direction: str = "A_to_B",
) -> dict[str, Any]:
    idm = ev["identity_A_to_B"] if direction == "A_to_B" else ev["identity_B_to_A"]
    row: dict[str, Any] = {
        "rep": rep,
        "Z": int(Z),
        "seed": int(seed),
        "ablation": ablation,
        "direction": direction,
        "n_params": meta.get("n_params"),
        "hidden": meta.get("hidden"),
        "best_epoch": meta.get("best_epoch"),
        "best_val_loss": meta.get("best_val_loss"),
        **{k: idm[k] for k in idm if k != "true_ranks"},
        **ev["latent_mknn"],
        "paired_latent_cosine": ev["paired_latent_cosine"],
        "paired_latent_l2": ev["paired_latent_l2"],
        "shuffled_latent_cosine": ev["shuffled_latent_cosine"],
        "shuffled_latent_l2": ev["shuffled_latent_l2"],
        "null_top1_mean": ev["null_top1_mean"],
        "null_top1_std": ev["null_top1_std"],
        "null_mrr_mean": ev["null_mrr_mean"],
        "null_mrr_std": ev["null_mrr_std"],
        **{f"geom_{k}": v for k, v in ev["geometry"].items()},
        **{f"lat_{k}": v for k, v in ev["latent_ranks"].items()},
    }
    # promote common geometry keys
    for k in (
        "spearman_transA_vs_trueB",
        "cka_transA_trueB",
        "pearson_transA_vs_trueB",
    ):
        if k in ev["geometry"]:
            row[k] = ev["geometry"][k]
    for soft in ev.get("soft_rows") or []:
        ke = int(soft["keff_target"])
        row[f"js_keff{ke}"] = soft.get("js_sim_mean")
        if "hellinger_mean" in soft:
            row[f"hellinger_keff{ke}"] = soft["hellinger_mean"]
        if "tv_mean" in soft:
            row[f"tv_keff{ke}"] = soft["tv_mean"]
    return row


def main() -> None:
    args = parse_args()
    t0 = time.time()
    root = platonic_root(args.platonic_root)
    out_dir = resolve_path(root, args.out_dir)
    # also allow out_dir relative to repo when platonic outputs live elsewhere
    if not out_dir.parent.exists():
        out_dir = (_REPO / args.out_dir).resolve()
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    device_s = args.device
    if device_s.startswith("cuda") and not torch.cuda.is_available():
        if args.allow_cpu:
            log("CUDA unavailable; falling back to CPU")
            device_s = "cpu"
        else:
            raise RuntimeError("CUDA requested but unavailable (pass --allow-cpu)")
    device = torch.device(device_s)

    pair = parse_str_list(args.pair)
    if len(pair) != 2:
        raise ValueError("--pair must be modelA,modelB")
    model_a, model_b = pair[0], pair[1]
    zs = parse_int_list(args.zs)
    oracle_zs = parse_int_list(args.oracle_zs)
    reps = parse_str_list(args.reps)
    if args.include_decoder_metric and "sae_decoder_shrinkage" not in reps:
        reps.append("sae_decoder_shrinkage")

    weights = LossWeights(
        recon=args.w_recon, mmd=args.w_mmd, cycle=args.w_cycle, geom=args.w_geom
    )

    config = {
        "pair": [model_a, model_b],
        "max_n": args.max_n,
        "seeds": args.seeds,
        "zs": zs,
        "oracle_zs": oracle_zs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "device": str(device),
        "reps": reps,
        "skip_soft": bool(args.skip_soft),
        "skip_oracle": bool(args.skip_oracle),
        "run_ablations": bool(args.run_ablations),
        "include_decoder_metric": bool(args.include_decoder_metric),
        "n_oracle_train": args.n_oracle_train,
        "n_a_train": args.n_a_train,
        "n_b_train": args.n_b_train,
        "n_geometry": args.n_geometry,
        "n_soft_query": args.n_soft_query,
        "n_identity_shuffles": args.n_identity_shuffles,
        "loss_weights": weights.__dict__,
        "sae_tag_prefer": list(SAE_TAG_PREFER),
        "seed_base": args.seed_base,
        "platonic_root": str(root),
        "out_dir": str(out_dir),
        "has_soft_helpers": _HAS_SOFT,
        "has_affine_knn": _HAS_AFFINE_KNN,
        "identity_gallery": "paired_eval",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"Out dir: {out_dir}")
    log(f"Device: {device}")

    # ---- data ----
    log("Loading embeddings …")
    Xa_raw, Xb_raw, sel = load_pair_embeddings(
        root, model_a, model_b, args.max_n, seed=0
    )
    n = len(Xa_raw)
    log(f"Subsample n={n} (galaxy_idx = row id in this subsample)")

    split = build_split(
        n,
        args.n_oracle_train,
        args.n_a_train,
        args.n_b_train,
        seed=0,
    )
    write_split_manifest(out_dir / "split_manifest.csv", split)
    anti_cheat = [
        f"A_train ∩ B_train = ∅ (nA={len(split['A_train'])}, nB={len(split['B_train'])})",
        f"A_train ∩ paired_eval = ∅ (n_eval={len(split['paired_eval'])})",
        f"B_train ∩ paired_eval = ∅",
        f"paired_oracle_train ∩ (A_train∪B_train∪paired_eval) = ∅ "
        f"(n_oracle={len(split['paired_oracle_train'])})",
        "Unpaired checkpoint selection uses unsupervised val loss only",
        "Paired metrics computed only after checkpoint selection",
        "Ridge shared-basis W fit only on paired_oracle_train",
        "IDF_A fit only on A_train codes",
    ]
    for a, b in (
        ("A_train", "B_train"),
        ("A_train", "paired_eval"),
        ("B_train", "paired_eval"),
        ("paired_oracle_train", "A_train"),
        ("paired_oracle_train", "B_train"),
        ("paired_oracle_train", "paired_eval"),
    ):
        assert np.intersect1d(split[a], split[b]).size == 0

    # ---- representations ----
    log("Building representations …")
    rep_map, rep_meta = build_representations(
        root,
        model_a,
        model_b,
        Xa_raw,
        Xb_raw,
        split,
        device,
        args.ridge_alpha,
        reps,
        args.include_decoder_metric,
    )
    sae_paths = rep_meta.get("sae_paths")
    if sae_paths is None:
        # still try to resolve for audit
        try:
            sae_paths = (resolve_sae_dir(root, model_a), resolve_sae_dir(root, model_b))
        except Exception:  # noqa: BLE001
            sae_paths = (Path("MISSING"), Path("MISSING"))

    write_representation_audit(
        out_dir / "representation_audit.md",
        model_a=model_a,
        model_b=model_b,
        sae_a=sae_paths[0],
        sae_b=sae_paths[1],
        n_oracle=len(split["paired_oracle_train"]),
        n_a_train=len(split["A_train"]),
        n_b_train=len(split["B_train"]),
        n_eval=len(split["paired_eval"]),
    )

    # ---- storage ----
    oracle_rows: list[pd.DataFrame] = []
    unpaired_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    mknn_rows: list[dict[str, Any]] = []
    soft_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    lat_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    null_rows: list[dict[str, Any]] = []

    # ---- oracle ----
    if not args.skip_oracle:
        for rep_name, (FA, FB) in rep_map.items():
            log(f"===== Paired oracle / {rep_name} =====")
            odf = run_paired_oracle(
                FA,
                FB,
                split,
                oracle_zs=oracle_zs,
                hidden=args.hidden,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                ridge_alpha=args.ridge_alpha,
                row_batch=args.row_batch,
                rep_name=rep_name,
            )
            oracle_rows.append(odf)
    else:
        log("Skipping paired oracle (--skip-oracle)")

    oracle_df = (
        pd.concat(oracle_rows, ignore_index=True) if oracle_rows else pd.DataFrame()
    )
    df_to_parquet(oracle_df, out_dir / "paired_oracle_results.parquet")

    # ---- unpaired ----
    ablation_sched = [("full", weights)]
    if args.run_ablations:
        ablation_sched += [
            ("drop_geom", LossWeights(args.w_recon, args.w_mmd, args.w_cycle, 0.0)),
            ("drop_cycle", LossWeights(args.w_recon, args.w_mmd, 0.0, args.w_geom)),
            ("drop_mmd", LossWeights(args.w_recon, 0.0, args.w_cycle, args.w_geom)),
        ]

    for rep_name, (FA, FB) in rep_map.items():
        for z in zs:
            for seed_i in range(args.seeds):
                seed = args.seed_base + 1009 * seed_i + 17 * z
                for abl_name, w in ablation_sched:
                    # ablations only for Z=64 dense one seed
                    if abl_name != "full":
                        if not (
                            rep_name == "dense"
                            and z == 64
                            and seed_i == 0
                        ):
                            continue
                    log(
                        f"===== Unpaired {rep_name} Z={z} seed={seed_i} "
                        f"ablation={abl_name} ====="
                    )
                    FA_use, FB_use = FA, FB
                    if args.run_orthogonal_null and abl_name == "full" and seed_i == 0 and z == zs[0]:
                        # cheap null: random orthogonal preprocess of B
                        rng = np.random.default_rng(123)
                        d = FB.shape[1]
                        Q, _ = np.linalg.qr(rng.normal(size=(d, d)).astype(np.float32))
                        FB_null = l2n_np(FB @ Q)
                        log("  (also running orthogonal-B null config)")
                        model_n, curves_n, meta_n = train_dual_encoder(
                            FA,
                            FB_null,
                            split["A_train"],
                            split["B_train"],
                            z_dim=z,
                            hidden=args.hidden,
                            device=device,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            val_frac=args.val_frac,
                            weights=w,
                            seed=seed + 999,
                        )
                        ev_n = evaluate_unpaired_model(
                            model_n,
                            FA,
                            FB_null,
                            split["paired_eval"],
                            device=device,
                            row_batch=args.row_batch,
                            n_geometry=args.n_geometry,
                            n_soft_query=args.n_soft_query,
                            n_shuffles=min(20, args.n_identity_shuffles),
                            skip_soft=True,
                            seed=seed + 999,
                        )
                        null_rows.append(
                            {
                                "rep": rep_name,
                                "Z": z,
                                "seed": seed_i,
                                "null_type": "random_orthogonal_B",
                                "top1": ev_n["identity_A_to_B"]["top1"],
                                "mrr": ev_n["identity_A_to_B"]["mrr"],
                                "mknn_k10": ev_n["identity_A_to_B"].get("mknn_k10"),
                            }
                        )

                    model, curves, meta = train_dual_encoder(
                        FA_use,
                        FB_use,
                        split["A_train"],
                        split["B_train"],
                        z_dim=z,
                        hidden=args.hidden,
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        val_frac=args.val_frac,
                        weights=w,
                        seed=seed,
                    )
                    curves = curves.copy()
                    curves["rep"] = rep_name
                    curves["Z"] = z
                    curves["seed"] = seed_i
                    curves["ablation"] = abl_name
                    curve_rows.append(curves)

                    ev = evaluate_unpaired_model(
                        model,
                        FA_use,
                        FB_use,
                        split["paired_eval"],
                        device=device,
                        row_batch=args.row_batch,
                        n_geometry=args.n_geometry,
                        n_soft_query=args.n_soft_query,
                        n_shuffles=args.n_identity_shuffles,
                        skip_soft=args.skip_soft or abl_name != "full",
                        seed=seed,
                    )
                    row = flatten_eval_row(
                        ev,
                        rep=rep_name,
                        Z=z,
                        seed=seed_i,
                        meta=meta,
                        ablation=abl_name,
                        direction="A_to_B",
                    )
                    unpaired_rows.append(row)
                    identity_rows.append(
                        {
                            "rep": rep_name,
                            "Z": z,
                            "seed": seed_i,
                            "ablation": abl_name,
                            "direction": "A_to_B",
                            **{
                                k: ev["identity_A_to_B"][k]
                                for k in (
                                    "top1",
                                    "top5",
                                    "top10",
                                    "top100",
                                    "mrr",
                                    "median_rank",
                                    "mean_rank",
                                    "cosine",
                                    "mse",
                                )
                            },
                            "null_top1_mean": ev["null_top1_mean"],
                            "null_mrr_mean": ev["null_mrr_mean"],
                        }
                    )
                    for k in MKNN_KS:
                        mknn_rows.append(
                            {
                                "rep": rep_name,
                                "Z": z,
                                "seed": seed_i,
                                "ablation": abl_name,
                                "k": k,
                                "translation_mknn": ev["identity_A_to_B"].get(
                                    f"mknn_k{k}"
                                ),
                                "latent_mknn": ev["latent_mknn"].get(
                                    f"latent_mknn_k{k}"
                                ),
                            }
                        )
                    for soft in ev.get("soft_rows") or []:
                        soft_rows.append(
                            {
                                "rep": rep_name,
                                "Z": z,
                                "seed": seed_i,
                                "ablation": abl_name,
                                **soft,
                            }
                        )
                    dist_rows.append(
                        {
                            "rep": rep_name,
                            "Z": z,
                            "seed": seed_i,
                            "ablation": abl_name,
                            **ev["geometry"],
                        }
                    )
                    lat_rows.append(
                        {
                            "rep": rep_name,
                            "Z": z,
                            "seed": seed_i,
                            "ablation": abl_name,
                            **ev["latent_ranks"],
                            "paired_latent_cosine": ev["paired_latent_cosine"],
                            "shuffled_latent_cosine": ev["shuffled_latent_cosine"],
                        }
                    )
                    null_rows.append(
                        {
                            "rep": rep_name,
                            "Z": z,
                            "seed": seed_i,
                            "ablation": abl_name,
                            "null_type": "identity_shuffle",
                            "null_top1_mean": ev["null_top1_mean"],
                            "null_top1_std": ev["null_top1_std"],
                            "null_mrr_mean": ev["null_mrr_mean"],
                            "null_mrr_std": ev["null_mrr_std"],
                            "obs_top1": ev["identity_A_to_B"]["top1"],
                            "obs_mrr": ev["identity_A_to_B"]["mrr"],
                        }
                    )

    unpaired_df = pd.DataFrame(unpaired_rows)
    identity_df = pd.DataFrame(identity_rows)
    mknn_df = pd.DataFrame(mknn_rows)
    soft_df = pd.DataFrame(soft_rows)
    dist_df = pd.DataFrame(dist_rows)
    lat_df = pd.DataFrame(lat_rows)
    null_df = pd.DataFrame(null_rows)
    curves_df = (
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    )

    df_to_parquet(unpaired_df, out_dir / "unpaired_results.parquet")
    df_to_parquet(identity_df, out_dir / "identity_retrieval_results.parquet")
    df_to_parquet(mknn_df, out_dir / "mknn_results.parquet")
    if len(soft_df) == 0:
        empty_parquet(
            out_dir / "soft_retrieval_results.parquet",
            ["rep", "Z", "seed", "keff_target", "js_sim_mean", "note"],
            "soft skipped (--skip-soft or helpers unavailable)",
        )
    else:
        df_to_parquet(soft_df, out_dir / "soft_retrieval_results.parquet")
    df_to_parquet(dist_df, out_dir / "distance_geometry_results.parquet")
    df_to_parquet(lat_df, out_dir / "latent_rank_results.parquet")
    df_to_parquet(curves_df, out_dir / "training_curves.parquet")
    df_to_parquet(null_df, out_dir / "null_results.parquet")

    # seed summary
    if len(unpaired_df):
        metric_cols = [
            c
            for c in (
                "top1",
                "mrr",
                "median_rank",
                "mknn_k10",
                "mknn_k100",
                "cka_transA_trueB",
                "spearman_transA_vs_trueB",
                "js_keff10",
                "js_keff500",
            )
            if c in unpaired_df.columns
        ]
        gcols = ["rep", "Z", "ablation"] if "ablation" in unpaired_df.columns else ["rep", "Z"]
        seed_summary = (
            unpaired_df.groupby(gcols, dropna=False)[metric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        seed_summary.columns = [
            "_".join([c for c in col if c]).rstrip("_")
            if isinstance(col, tuple)
            else col
            for col in seed_summary.columns.to_flat_index()
        ]
        seed_summary.to_csv(out_dir / "seed_summary.csv", index=False)
    else:
        pd.DataFrame({"note": ["no unpaired results"]}).to_csv(
            out_dir / "seed_summary.csv", index=False
        )

    # saturation
    sat_metrics = [
        c
        for c in (
            "top1",
            "mrr",
            "median_rank",
            "mknn_k5",
            "mknn_k10",
            "mknn_k20",
            "mknn_k100",
            "cka_transA_trueB",
            "spearman_transA_vs_trueB",
            "js_keff10",
            "js_keff20",
            "js_keff500",
            "js_keff1000",
            "cosine",
        )
        if len(unpaired_df) and c in unpaired_df.columns
    ]
    u_full = (
        unpaired_df[unpaired_df["ablation"].fillna("full") == "full"]
        if len(unpaired_df) and "ablation" in unpaired_df.columns
        else unpaired_df
    )
    sat_u = saturation_table(u_full, sat_metrics)
    sat_u["regime"] = "unpaired"
    sat_o = pd.DataFrame()
    if len(oracle_df):
        o_metrics = [c for c in sat_metrics if c in oracle_df.columns]
        sat_o = saturation_table(oracle_df, o_metrics)
        sat_o["regime"] = "oracle"
    sat_df = (
        pd.concat([sat_u, sat_o], ignore_index=True)
        if len(sat_u) or len(sat_o)
        else pd.DataFrame()
    )
    if len(sat_df):
        sat_df.to_csv(out_dir / "dimension_saturation.csv", index=False)
    else:
        pd.DataFrame(
            columns=["rep", "method", "metric", "best", "Z90", "Z95", "Z99", "regime", "note"]
        ).to_csv(out_dir / "dimension_saturation.csv", index=False)

    # dense vs sae
    if len(u_full) and "rep" in u_full.columns:
        rows = []
        metrics = [
            c
            for c in ("top1", "mrr", "mknn_k10", "mknn_k100", "cka_transA_trueB")
            if c in u_full.columns
        ]
        for z in sorted(u_full["Z"].unique()):
            row: dict[str, Any] = {"Z": int(z)}
            for rep in sorted(u_full["rep"].unique()):
                sub = u_full[(u_full["Z"] == z) & (u_full["rep"] == rep)]
                for m in metrics:
                    row[f"{rep}_{m}_mean"] = float(sub[m].mean()) if len(sub) else float("nan")
            rows.append(row)
        dense_vs_sae = pd.DataFrame(rows)
        dense_vs_sae.to_csv(out_dir / "dense_vs_sae.csv", index=False)
    else:
        dense_vs_sae = pd.DataFrame({"note": ["n/a"]})
        dense_vs_sae.to_csv(out_dir / "dense_vs_sae.csv", index=False)

    # aggregate summary
    gates = compute_gates(oracle_df, unpaired_df)
    agg = {
        "n": n,
        "n_eval": len(split["paired_eval"]),
        "elapsed_sec": time.time() - t0,
        **{f"gate_{k}": v for k, v in gates.items() if not isinstance(v, dict)},
    }
    if len(u_full):
        for m in ("top1", "mrr", "mknn_k10", "mknn_k100"):
            if m in u_full.columns:
                agg[f"unpaired_best_{m}"] = float(u_full[m].max() if "rank" not in m else u_full[m].min())
    pd.DataFrame([agg]).to_csv(out_dir / "aggregate_summary.csv", index=False)
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2, default=str) + "\n")

    log("Making figures …")
    make_figures(fig_dir, oracle_df, unpaired_df, sat_df, curves_df)

    write_report(
        out_dir / "unpaired_universal_geometry_report.md",
        config=config,
        gates=gates,
        oracle_df=oracle_df,
        unpaired_df=unpaired_df,
        sat_df=sat_df,
        dense_vs_sae=dense_vs_sae,
        anti_cheat=anti_cheat,
    )

    # ensure all required files exist
    required = [
        "config.json",
        "split_manifest.csv",
        "representation_audit.md",
        "paired_oracle_results.parquet",
        "unpaired_results.parquet",
        "identity_retrieval_results.parquet",
        "mknn_results.parquet",
        "soft_retrieval_results.parquet",
        "distance_geometry_results.parquet",
        "latent_rank_results.parquet",
        "training_curves.parquet",
        "seed_summary.csv",
        "dimension_saturation.csv",
        "dense_vs_sae.csv",
        "null_results.parquet",
        "aggregate_summary.csv",
        "unpaired_universal_geometry_report.md",
    ]
    for name in required:
        p = out_dir / name
        if not p.exists():
            if name.endswith(".parquet"):
                empty_parquet(p, ["note"], f"missing placeholder for {name}")
            else:
                p.write_text(f"missing placeholder for {name}\n")

    log(f"Done in {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
