"""Shared helpers for SAE shared-basis experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def platonic_root(cli_value: str | None = None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        Path.home() / "platonic-universe",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return candidates[0]


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def ensure_sae_import() -> Path:
    """Put vendored / sibling sae_model on sys.path; return the chosen dir."""
    candidates = [
        Path(__file__).resolve().parent / "sae",
        Path(__file__).resolve().parents[1] / "sae",
        Path.home() / "platonic-universe" / "experiments" / "sae",
    ]
    for p in candidates:
        if (p / "sae_model.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    raise FileNotFoundError(
        "sae_model.py not found. Expected vendored copy at "
        f"{candidates[0]} (shipped with this package) or PLATONIC_ROOT/experiments/sae/"
    )


def load_col(path: Path, column: str, l2: bool = False) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    X = np.vstack(table.column(0).to_pylist()).astype(np.float32)
    if l2:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(n, 1e-12)
    return X


def load_aligned_pair(
    path1: Path,
    col1: str,
    path2: Path,
    col2: str,
    *,
    allow_truncate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load two embedding columns; require equal length unless allow_truncate."""
    X1 = load_col(path1, col1)
    X2 = load_col(path2, col2)
    if len(X1) != len(X2):
        msg = (
            f"Cross-matched length mismatch: {path1.name}:{col1} n={len(X1)} vs "
            f"{path2.name}:{col2} n={len(X2)}. Rows must be positionally aligned."
        )
        if not allow_truncate:
            raise ValueError(msg + " Pass --allow-truncate to use min length (discouraged).")
        n = min(len(X1), len(X2))
        print(f"WARNING: {msg} Truncating to n={n}.", flush=True)
        X1, X2 = X1[:n], X2[:n]
    return X1, X2


def binary_metrics_topk(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    """Active-set overlap using *positive* TopK of predictions (SAE-like), not |y|."""
    true_a = y_true > 0
    kk = min(k, y_pred.shape[1])
    # Prefer largest positive entries; negatives never count as activations.
    score = np.where(y_pred > 0, y_pred, -np.inf)
    # If a row has fewer than kk positives, pad with remaining largest (still may be -inf)
    top = np.argpartition(-score, kk - 1, axis=1)[:, :kk]
    pred_a = np.zeros_like(true_a)
    for i in range(len(y_pred)):
        pred_a[i, top[i]] = score[i, top[i]] > 0
    # If no positive predictions, fall back to empty set (all False)
    tp = (true_a & pred_a).sum(axis=1).astype(np.float64)
    union = (true_a | pred_a).sum(axis=1).astype(np.float64)
    return {
        "precision_at_k": float((tp / np.maximum(pred_a.sum(axis=1), 1)).mean()),
        "recall_at_k": float((tp / np.maximum(true_a.sum(axis=1), 1)).mean()),
        "jaccard_at_k": float((tp / np.maximum(union, 1)).mean()),
    }


def standardize_with_fit(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((X - mean) / np.maximum(scale, 1e-12)).astype(np.float32)


def singular_chart_coords(
    C_basis: np.ndarray,
    C_other: np.ndarray,
    fit: dict,
    U: np.ndarray,
    Vt: np.ndarray,
    r: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project *standardized* codes into paired singular charts of W_std.

    W_std maps other_std → basis_std. SVD(W)=U Σ Vt with U on other, Vt on basis.
    """
    r = min(r, U.shape[1], Vt.shape[0])
    other_s = standardize_with_fit(C_other, fit["x_mean"], fit["x_scale"])
    basis_s = standardize_with_fit(C_basis, fit["y_mean"], fit["y_scale"])
    zb = basis_s @ Vt[:r].T
    zo = other_s @ U[:, :r]
    return zb, zo


def ridge_ref_compatible(ref: dict, *, n: int, col1: str, col2: str, seed: int) -> bool:
    meta = ref.get("meta") or {}
    if not meta:
        # older ridge dumps nest differently
        meta = {
            "n": ref.get("n") or ref.get("meta", {}).get("n"),
            "seed": ref.get("seed"),
            "col1": ref.get("col1"),
            "col2": ref.get("col2"),
        }
    checks = []
    if meta.get("n") is not None:
        checks.append(int(meta["n"]) == int(n))
    if meta.get("seed") is not None:
        checks.append(int(meta["seed"]) == int(seed))
    if meta.get("col1") is not None:
        checks.append(meta["col1"] == col1)
    if meta.get("col2") is not None:
        checks.append(meta["col2"] == col2)
    return bool(checks) and all(checks)
