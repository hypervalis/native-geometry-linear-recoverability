#!/usr/bin/env python3
"""Patchwise residual diagnostic for frozen Dense→Dense Ridge maps.

Partitions Legacy source space with k-means (train only), estimates patch
residual means from train residuals only, evaluates held-out MSE reduction
vs global Ridge. Optional local-linear residual correction with source PCA.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
FAMILY_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
K_GRID = (4, 8, 16, 32)
LOCAL_PCA = 32
B_NULL = 200
MIN_PATCH_TRAIN = 20


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def load_col(path: Path, col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[col])
    return np.vstack(table.column(0).to_pylist()).astype(np.float32)


def load_pair_arrays(
    root: Path, cfg: dict, max_n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    X1 = load_col(resolve_path(root, cfg["parquet1"]), cfg["col1"])
    X2 = load_col(resolve_path(root, cfg["parquet2"]), cfg["col2"])
    n_full = min(len(X1), len(X2))
    X1, X2 = X1[:n_full], X2[:n_full]
    n_cap = int(cfg.get("default_max_n", 0) or 0)
    n_use = max_n
    if n_cap > 0:
        n_use = min(n_use, n_cap) if n_use > 0 else n_cap
    rng = np.random.default_rng(seed)
    if n_use and n_full > n_use:
        sel = np.sort(rng.choice(n_full, size=n_use, replace=False))
        X1, X2 = X1[sel], X2[sel]
    return X1.astype(np.float32), X2.astype(np.float32)


def family_ols_slope(logp: np.ndarray, y: np.ndarray) -> float:
    if len(logp) < 2:
        return float("nan")
    lr = LinearRegression()
    lr.fit(logp.reshape(-1, 1), y)
    return float(lr.coef_[0])


def fit_ridge_predict(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, alpha: float
) -> np.ndarray:
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    return y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Row-wise cosine between a[i] and b[i] (or broadcast b if 1D vector rows)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(an * bn, axis=1)))


def median_cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.median(np.sum(an * bn, axis=1)))


def patch_means_from_labels(
    resid_tr: np.ndarray, labels_tr: np.ndarray, K: int
) -> np.ndarray:
    d = resid_tr.shape[1]
    C = np.zeros((K, d), dtype=np.float64)
    for p in range(K):
        m = labels_tr == p
        if np.any(m):
            C[p] = resid_tr[m].mean(axis=0)
    return C


def eval_patch_correction(
    resid_te: np.ndarray,
    y_te: np.ndarray,
    yhat_te: np.ndarray,
    labels_te: np.ndarray,
    C: np.ndarray,
) -> dict[str, float]:
    c_pred = C[labels_te]
    yhat_patch = yhat_te + c_pred.astype(np.float32)
    mse_g = mse(yhat_te, y_te)
    mse_p = mse(yhat_patch, y_te)
    return {
        "mse_global": mse_g,
        "mse_patch": mse_p,
        "G": mse_g - mse_p,
        "rel_mse_reduction": (mse_g - mse_p) / max(mse_g, 1e-300),
        "mean_cos": mean_cosine(resid_te, c_pred),
        "median_cos": median_cosine(resid_te, c_pred),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--b-null", type=int, default=B_NULL)
    ap.add_argument("--skip-local", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "patch_residual_diagnostic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "tmp_ridge_geom/official_legacy_pairs.yaml",
            ROOT / "configs/official_legacy_pairs.yaml",
        ]
        pairs_path = next(p for p in cands if p.is_file())

    pairs = yaml.safe_load(pairs_path.read_text())
    names = [
        n
        for n, cfg in pairs.items()
        if isinstance(cfg, dict) and "legacysurvey" in str(cfg.get("parquet2", ""))
    ]

    meta = {
        "seed": args.seed,
        "test_size": args.test_size,
        "ridge_alpha": args.alpha,
        "K_grid": list(K_GRID),
        "kmeans": "MiniBatchKMeans on Legacy train only",
        "patch_correction": "mean train residual per patch; apply to test by nearest centroid",
        "local_linear": (
            None
            if args.skip_local
            else f"source PCA->{LOCAL_PCA} then Ridge(alpha={args.alpha}) on train residuals"
        ),
        "b_null": args.b_null,
        "leakage": "test residuals never used to estimate c_p or B_p",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    score_rows = []
    null_rows = []
    patch_diag_rows = []

    for name in names:
        cfg = pairs[name]
        X_hsc, X_leg = load_pair_arrays(root, cfg, args.max_n, args.seed)
        # X_hsc=col1 target, X_leg=col2 source
        n = len(X_leg)
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed, shuffle=True
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

        params_m = float(cfg.get("approx_params_m", 1))
        logp = math.log10(max(params_m * 1e6, 1.0))
        fam = str(cfg["family"])

        yhat = fit_ridge_predict(X_leg, X_hsc, train_idx, args.alpha)
        resid = (X_hsc - yhat).astype(np.float32)
        resid_tr, resid_te = resid[train_idx], resid[test_idx]
        y_te, yhat_te = X_hsc[test_idx], yhat[test_idx]
        X_tr, X_te = X_leg[train_idx], X_leg[test_idx]
        mse_global = mse(yhat_te, y_te)

        print(f"{name}: global test MSE={mse_global:.6g}", flush=True)

        for K in K_GRID:
            km = MiniBatchKMeans(
                n_clusters=K,
                random_state=args.seed,
                batch_size=2048,
                n_init=3,
                max_iter=200,
            )
            labels_tr = km.fit_predict(X_tr)
            labels_te = km.predict(X_te)
            sizes = np.bincount(labels_tr, minlength=K)
            n_small = int(np.sum(sizes < MIN_PATCH_TRAIN))

            C = patch_means_from_labels(resid_tr.astype(np.float64), labels_tr, K)
            ev = eval_patch_correction(resid_te, y_te, yhat_te, labels_te, C)

            # Local linear residual correction (secondary)
            mse_local = float("nan")
            rel_local = float("nan")
            delta_local = float("nan")
            if not args.skip_local:
                # PCA on source train; predict residual
                n_comp = min(LOCAL_PCA, X_tr.shape[1], len(train_idx) - 1)
                pca = PCA(n_components=n_comp, random_state=args.seed).fit(X_tr)
                Z_tr = pca.transform(X_tr)
                Z_te = pca.transform(X_te)
                yhat_local = yhat_te.copy()
                for p in range(K):
                    tr_m = labels_tr == p
                    te_m = labels_te == p
                    if tr_m.sum() < max(MIN_PATCH_TRAIN, n_comp + 2) or not np.any(te_m):
                        # fall back to patch mean only
                        if np.any(te_m):
                            yhat_local[te_m] = yhat_te[te_m] + C[p].astype(np.float32)
                        continue
                    mu = Z_tr[tr_m].mean(axis=0)
                    dZ_tr = Z_tr[tr_m] - mu
                    dZ_te = Z_te[te_m] - mu
                    # residual Ridge: r ≈ B δz + c  (Ridge with intercept)
                    rr = Ridge(alpha=args.alpha, fit_intercept=True)
                    rr.fit(dZ_tr, resid_tr[tr_m])
                    r_hat = rr.predict(dZ_te).astype(np.float32)
                    yhat_local[te_m] = yhat_te[te_m] + r_hat
                mse_local = mse(yhat_local, y_te)
                rel_local = (mse_global - mse_local) / max(mse_global, 1e-300)
                delta_local = ev["mse_patch"] - mse_local

            # Permutation null: shuffle train labels preserving sizes
            rng = np.random.default_rng(args.seed + 17_000 + K + hash(name) % 10_000)
            G_null = np.empty(args.b_null, dtype=np.float64)
            cos_null = np.empty(args.b_null, dtype=np.float64)
            for b in range(args.b_null):
                shuf = labels_tr.copy()
                rng.shuffle(shuf)
                C_b = patch_means_from_labels(resid_tr.astype(np.float64), shuf, K)
                # Test still assigned by real k-means centroids; shuffled train labels
                # break spatial coherence of residual means.
                ev_b = eval_patch_correction(resid_te, y_te, yhat_te, labels_te, C_b)
                G_null[b] = ev_b["G"]
                cos_null[b] = ev_b["mean_cos"]
                null_rows.append(
                    {
                        "model": name,
                        "family": fam,
                        "K": K,
                        "perm_id": b,
                        "G": ev_b["G"],
                        "mean_cos": ev_b["mean_cos"],
                    }
                )

            p_G = (1 + np.sum(G_null >= ev["G"])) / (args.b_null + 1)
            p_cos = (1 + np.sum(cos_null >= ev["mean_cos"])) / (args.b_null + 1)

            # Patch residual norms vs size
            for p in range(K):
                patch_diag_rows.append(
                    {
                        "model": name,
                        "family": fam,
                        "K": K,
                        "patch": p,
                        "n_train": int(sizes[p]),
                        "n_test": int(np.sum(labels_te == p)),
                        "c_norm": float(np.linalg.norm(C[p])),
                    }
                )

            score_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "size_name": cfg.get("size_name"),
                    "parameter_count": int(round(params_m * 1e6)),
                    "log10_params": logp,
                    "K": K,
                    "n_small_patches": n_small,
                    "mse_global": mse_global,
                    "mse_patch": ev["mse_patch"],
                    "mse_local": mse_local,
                    "G": ev["G"],
                    "rel_mse_reduction": ev["rel_mse_reduction"],
                    "rel_mse_reduction_local": rel_local,
                    "delta_local_vs_patch": delta_local,
                    "mean_cos": ev["mean_cos"],
                    "median_cos": ev["median_cos"],
                    "p_perm_G": p_G,
                    "p_perm_cos": p_cos,
                    "null_G_mean": float(G_null.mean()),
                    "null_G_sd": float(G_null.std()),
                }
            )
            print(
                f"  K={K}: rel_red={ev['rel_mse_reduction']*100:.3f}% "
                f"cos={ev['mean_cos']:.4f} p_G={p_G:.4f} "
                f"local_rel={rel_local*100 if np.isfinite(rel_local) else float('nan'):.3f}%",
                flush=True,
            )

    scores = pd.DataFrame(score_rows)
    scores.to_csv(out_dir / "patch_residual_scores.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out_dir / "patch_residual_null.csv", index=False)
    pd.DataFrame(patch_diag_rows).to_csv(
        out_dir / "patch_centroid_norms.csv", index=False
    )

    # Summary by K
    sum_rows = []
    for K in K_GRID:
        g = scores[scores["K"] == K]
        sum_rows.append(
            {
                "K": K,
                "n_rungs": len(g),
                "mean_rel_mse_reduction": float(g["rel_mse_reduction"].mean()),
                "median_rel_mse_reduction": float(g["rel_mse_reduction"].median()),
                "mean_mean_cos": float(g["mean_cos"].mean()),
                "median_mean_cos": float(g["mean_cos"].median()),
                "frac_p_G_le_005": float((g["p_perm_G"] <= 0.05).mean()),
                "mean_p_G": float(g["p_perm_G"].mean()),
                "mean_rel_local": float(g["rel_mse_reduction_local"].mean()),
                "mean_delta_local": float(g["delta_local_vs_patch"].mean()),
                "median_delta_local": float(g["delta_local_vs_patch"].median()),
            }
        )
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(out_dir / "patch_residual_summary.csv", index=False)

    # Family slopes of rel reduction vs logP
    trend_rows = []
    for K in K_GRID:
        for fam in FAMILY_ORDER:
            g = scores[(scores["K"] == K) & (scores["family"] == fam)].sort_values(
                "log10_params"
            )
            if len(g) < 2:
                continue
            slope = family_ols_slope(
                g["log10_params"].to_numpy(float),
                g["rel_mse_reduction"].to_numpy(float),
            )
            trend_rows.append(
                {
                    "K": K,
                    "family": fam,
                    "slope_rel_red_vs_log10P": slope,
                    "mean_rel_red": float(g["rel_mse_reduction"].mean()),
                }
            )
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(out_dir / "patch_residual_vs_size_slopes.csv", index=False)

    # Figure
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for fam in FAMILY_ORDER:
        ys = []
        for K in K_GRID:
            g = scores[(scores["family"] == fam) & (scores["K"] == K)]
            ys.append(float(g["rel_mse_reduction"].mean()) * 100)
        ax.plot(K_GRID, ys, "o-", label=FAMILY_LABEL.get(fam, fam), markersize=5)
    # overall mean ± sd across rungs
    means = [
        float(scores[scores.K == K]["rel_mse_reduction"].mean()) * 100 for K in K_GRID
    ]
    sds = [
        float(scores[scores.K == K]["rel_mse_reduction"].std()) * 100 for K in K_GRID
    ]
    ax.errorbar(
        K_GRID,
        means,
        yerr=sds,
        fmt="k--",
        capsize=3,
        label="all-rung mean±sd",
        linewidth=1.5,
    )
    if not args.skip_local:
        means_l = [
            float(scores[scores.K == K]["rel_mse_reduction_local"].mean()) * 100
            for K in K_GRID
        ]
        ax.plot(K_GRID, means_l, "s:", color="0.3", label="local-linear (mean)")
    ax.set_xticks(K_GRID)
    ax.set_xlabel("number of patches K")
    ax.set_ylabel("relative held-out MSE reduction (%)")
    ax.set_title("Patch residual correction of global Dense+Ridge")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "patch_residual_diagnostic.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_patch_residual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("summary:\n", summary.to_string(index=False), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
