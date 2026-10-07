"""Unit checks for Ridge affine reconstruction used in translation ablation.

Self-contained (no torch/pyarrow) so the reconstruction identity can be checked
without the full experiment stack.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def recover_affine_map(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> dict:
    """Mirror of run_translation_ablation.recover_affine_map (math only)."""
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))

    W = np.asarray(ridge.coef_, dtype=np.float64)
    b_std = np.asarray(ridge.intercept_, dtype=np.float64)
    sx = np.asarray(x_sc.scale_, dtype=np.float64)
    sy = np.asarray(y_sc.scale_, dtype=np.float64)
    mx = np.asarray(x_sc.mean_, dtype=np.float64)
    my = np.asarray(y_sc.mean_, dtype=np.float64)
    sx = np.where(sx > 0, sx, 1.0)
    sy = np.where(sy > 0, sy, 1.0)

    A_eff = (sy[:, None] * W) / sx[None, :]
    b_eff = my + sy * b_std - A_eff @ mx

    x64 = x.astype(np.float64)
    pipeline = y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float64)
    reconstructed = x64 @ A_eff.T + b_eff
    max_abs_err = float(np.max(np.abs(pipeline - reconstructed)))

    return {
        "A_eff": A_eff,
        "b_eff": b_eff,
        "max_abs_err": max_abs_err,
        "mu_x": mx,
        "mu_y": my,
    }


def test_affine_reconstruction_matches_sklearn_pipeline():
    rng = np.random.default_rng(0)
    n, d_x, d_y = 200, 8, 6
    X = rng.normal(size=(n, d_x)).astype(np.float64)
    W_true = rng.normal(size=(d_y, d_x))
    b_true = rng.normal(size=(d_y))
    Y = (X @ W_true.T + b_true + 0.05 * rng.normal(size=(n, d_y))).astype(np.float64)
    train_idx = np.arange(160)

    fit = recover_affine_map(X.astype(np.float32), Y.astype(np.float32), train_idx, alpha=1.0)
    # Float32 reconstruction of the sklearn pipeline. The paper's
    # verification tolerance is 1e-4; this bound stays well inside it.
    assert fit["max_abs_err"] < 1e-5

    x_te = X[160:]
    x_sc = StandardScaler().fit(X[train_idx])
    y_sc = StandardScaler().fit(Y[train_idx])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(x_sc.transform(X[train_idx]), y_sc.transform(Y[train_idx]))
    pipe = y_sc.inverse_transform(ridge.predict(x_sc.transform(x_te)))
    recon = x_te @ fit["A_eff"].T + fit["b_eff"]
    assert np.allclose(pipe, recon, atol=1e-5, rtol=0)


def test_linear_only_has_no_translation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5)).astype(np.float32)
    Y = rng.normal(size=(100, 4)).astype(np.float32)
    train_idx = np.arange(80)
    fit = recover_affine_map(X, Y, train_idx, alpha=1.0)
    A = fit["A_eff"]
    assert np.allclose(A @ np.zeros(A.shape[1]), 0.0)
    X0 = X.astype(np.float64) - X.astype(np.float64).mean(axis=0)
    assert np.allclose((X0 @ A.T).mean(axis=0), 0.0, atol=1e-10)


def test_centered_uses_train_means_only():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 3)).astype(np.float64)
    Y = rng.normal(size=(120, 3)).astype(np.float64)
    train_idx = np.arange(90)
    test_idx = np.arange(90, 120)
    fit = recover_affine_map(X.astype(np.float32), Y.astype(np.float32), train_idx, alpha=1.0)
    mu_x, mu_y = fit["mu_x"], fit["mu_y"]
    assert np.allclose(mu_x, X[train_idx].mean(axis=0))
    assert np.allclose(mu_y, Y[train_idx].mean(axis=0))
    assert not np.allclose(mu_x, X[test_idx].mean(axis=0))
