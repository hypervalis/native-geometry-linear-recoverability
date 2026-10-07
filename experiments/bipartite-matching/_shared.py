"""Shared helpers for bipartite-matching experiments.

Self-contained: data loading, SAE encoding, IDF weights, kNN graphs, mKNN,
and the standardized Ridge map W between two SAE code spaces.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

MODELS = {
    "vit_base": ("data_hf/physics/vit_base_test.parquet", "vit_base_galaxies"),
    "dinov3": ("data_hf/physics/dinov3_vitb16_test.parquet", "dinov3_vitb16_galaxies"),
    "clip_base": ("data_hf/physics/clip_base_test.parquet", "clip_base_galaxies"),
    "convnext_base": (
        "data_hf/physics/convnext_base_test.parquet",
        "convnext_base_galaxies",
    ),
    "vit_large": ("data_hf/physics/vit_large_test.parquet", "vit_large_galaxies"),
}

SAE_TAG = "F2048_k64_seed0"
LABELS_NPZ = "data_hf/physics/vit_base_test_labels.npz"
DEFAULT_PROPERTIES = ["mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass"]


def platonic_root(cli_value: str | None = None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "platonic-universe").resolve()


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def ensure_sae_import() -> Path:
    """Put the vendored / sibling sae_model on sys.path."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "sae",
        here.parent / "SAE-shared-basis" / "sae",
        Path.home() / "platonic-universe" / "experiments" / "sae",
    ]
    for p in candidates:
        if (p / "sae_model.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    raise FileNotFoundError(
        "sae_model.py not found; expected sibling SAE-shared-basis/sae or "
        "PLATONIC_ROOT/experiments/sae"
    )


def sae_dir(root: Path, model: str, tag: str = SAE_TAG) -> Path:
    parquet, col = MODELS[model]
    return root / "outputs" / "sae" / Path(parquet).stem / col / tag


# ---------------------------------------------------------------------------
# Data / SAE
# ---------------------------------------------------------------------------


def load_col(path: Path, column: str, l2: bool = False) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    X = np.vstack(table.column(0).to_pylist()).astype(np.float32)
    if l2:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(n, 1e-12)
    return X


def l2n(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, eps)).astype(np.float32)


def load_sae(sae_path: Path, device: torch.device) -> dict:
    ensure_sae_import()
    from sae_model import TopKSAE  # noqa: PLC0415

    cfg = json.loads((sae_path / "config.json").read_text())
    sc = np.load(sae_path / "scaler_stats.npz")
    model = TopKSAE(cfg["dim"], cfg["feature_dim"], cfg["k"]).to(device)
    model.load_state_dict(
        torch.load(sae_path / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return {
        "model": model,
        "mean": sc["mean"].astype(np.float32),
        "scale": sc["scale"].astype(np.float32),
    }


@torch.inference_mode()
def encode(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    for i in range(0, len(xs), bs):
        _, z = bundle["model"](torch.as_tensor(xs[i : i + bs], device=device))
        outs.append(z.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def load_model_codes(
    root: Path, model: str, sel: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense embeddings, SAE codes) for a named model at rows sel."""
    parquet, col = MODELS[model]
    X = load_col(root / parquet, col, l2=False).astype(np.float32)[sel]
    C = encode(load_sae(sae_dir(root, model), device), X, device)
    return X, C


# ---------------------------------------------------------------------------
# Graphs / mKNN
# ---------------------------------------------------------------------------


def idf_weights(C_train: np.ndarray) -> np.ndarray:
    df = (C_train > 0).sum(axis=0).astype(np.float64)
    n = len(C_train)
    return (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def knn_graph(Z: np.ndarray, k: int) -> np.ndarray:
    """Cosine kNN graph, self excluded. Returns [n, k] indices."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(Z)
    out = nn.kneighbors(Z, return_distance=False)
    res = np.empty((len(Z), k), dtype=np.int64)
    for i in range(len(Z)):
        row = out[i]
        res[i] = row[row != i][:k]
    return res


def mknn(nn1: np.ndarray, nn2: np.ndarray) -> float:
    k = nn1.shape[1]
    inter = sum(
        len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(nn1, nn2)
    )
    return inter / (len(nn1) * k)


# ---------------------------------------------------------------------------
# Ridge shared-basis map
# ---------------------------------------------------------------------------


class RidgeMap:
    """Standardized Ridge map src codes -> dst codes, mKNN evaluation."""

    def __init__(
        self,
        C_src: np.ndarray,
        C_dst: np.ndarray,
        train_idx: np.ndarray,
        alpha: float = 1.0,
    ):
        self.x_sc = StandardScaler().fit(C_src[train_idx])
        self.y_sc = StandardScaler().fit(C_dst[train_idx])
        self.Xs = self.x_sc.transform(C_src).astype(np.float64)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(self.Xs[train_idx], self.y_sc.transform(C_dst[train_idx]))
        self.W = ridge.coef_.T  # [d_src, d_dst]; pred = Xs @ W + b
        self.b = ridge.intercept_

    def stable_rank(self) -> float:
        S = np.linalg.svd(self.W, compute_uv=False)
        return float((S.sum() ** 2) / (S**2).sum())

    def mapped_codes(self, rows: np.ndarray, Wv: np.ndarray | None = None) -> np.ndarray:
        Wv = self.W if Wv is None else Wv
        mapped = self.y_sc.inverse_transform(self.Xs[rows] @ Wv + self.b)
        return np.maximum(mapped, 0.0).astype(np.float32)

    def eval_mknn(
        self,
        rows: np.ndarray,
        g_true: np.ndarray,
        w_idf: np.ndarray,
        k: int,
        Wv: np.ndarray | None = None,
    ) -> float:
        mapped = self.mapped_codes(rows, Wv)
        return mknn(g_true, knn_graph(mapped * w_idf[None, :], k))


def topk_rows(W: np.ndarray, k_row: int) -> np.ndarray:
    """Keep only each row's k_row largest-|.| entries."""
    Wk = np.zeros_like(W)
    top = np.argpartition(-np.abs(W), k_row - 1, axis=1)[:, :k_row]
    rows = np.repeat(np.arange(W.shape[0]), k_row)
    Wk[rows, top.ravel()] = W[rows, top.ravel()]
    return Wk


def rank_truncate(W: np.ndarray, r: int) -> np.ndarray:
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r]
