#!/usr/bin/env python3
"""Final compact PCA rank sweep for Dense→Dense Ridge alignment.

Reproduces the paper_alignment_controls PCA256 convention exactly:
  - PCA fitted separately on Legacy (X2) and HSC (X1)
  - PCA fit on inner train split (80% of train_idx; same seed as controls)
  - transform all rows; Ridge on full train_idx
  - StandardScaler + Ridge(alpha=1) applied *after* PCA (inside fit_ridge_map)
  - sklearn PCA defaults: centered, whiten=False
  - test-only gallery mKNN@10

Does not overwrite paper_alignment_controls/. Writes outputs/final_pca_rank_sweep/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
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
RANK_LADDER = (8, 16, 32, 64, 128, 256, 512)


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def load_col(path: Path, col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[col])
    return np.vstack(table.column(0).to_pylist()).astype(np.float32)


def load_pair_arrays(
    root: Path, cfg: dict, max_n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
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
    return X1, X2, n_full


def family_ols_slope(logp: np.ndarray, y: np.ndarray) -> float:
    if len(logp) < 2:
        return float("nan")
    lr = LinearRegression()
    lr.fit(logp.reshape(-1, 1), y)
    return float(lr.coef_[0])


@torch.inference_mode()
def knn_cos(Z: torch.Tensor, k: int, row_batch: int) -> torch.Tensor:
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = Z.shape[0]
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


def mknn(nn1: torch.Tensor, nn2: torch.Tensor, k: int) -> float:
    a, b = nn1[:, :k].cpu().numpy(), nn2[:, :k].cpu().numpy()
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


@torch.inference_mode()
def mknn_pair(
    A: np.ndarray, B: np.ndarray, k: int, device: torch.device, row_batch: int
) -> float:
    ta = torch.as_tensor(np.ascontiguousarray(A), device=device)
    tb = torch.as_tensor(np.ascontiguousarray(B), device=device)
    return mknn(knn_cos(ta, k, row_batch), knn_cos(tb, k, row_batch), k)


def fit_ridge_map(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, float, float]:
    """Return mapped embeddings + train/test MSE in original Y space."""
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    xs = x_sc.transform(x_tr)
    ys = y_sc.transform(y_tr)
    ridge.fit(xs, ys)
    mapped = y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)
    # MSE diagnostic (not used for selection)
    train_mse = float(np.mean((mapped[train_idx] - y[train_idx]) ** 2))
    # caller may pass test_idx separately; compute full residual and leave test to caller
    return mapped, train_mse, float(np.mean((mapped - y) ** 2))


def clustered_mean_ci(
    vals: np.ndarray, families: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    fams = np.unique(families)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(fams, size=len(fams), replace=True)
        parts = [vals[families == f] for f in chosen]
        boots.append(float(np.mean(np.concatenate(parts))))
    boots = np.asarray(boots)
    return float(np.mean(vals)), float(np.quantile(boots, 0.025)), float(
        np.quantile(boots, 0.975)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    ap.add_argument(
        "--k90-csv",
        type=Path,
        default=None,
        help="optional transfer_complexity_k90.csv for comparison",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "final_pca_rank_sweep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "configs/official_legacy_pairs.yaml",
            ROOT / "configs/official_legacy_pairs.yaml",
            root / "tmp_ridge_geom/official_legacy_pairs.yaml",
        ]
        pairs_path = next(p for p in cands if p.is_file())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = yaml.safe_load(pairs_path.read_text())
    names = [
        n
        for n, cfg in pairs.items()
        if isinstance(cfg, dict) and "legacysurvey" in str(cfg.get("parquet2", ""))
    ]

    meta = {
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "ridge_alpha": args.alpha,
        "rank_ladder": list(RANK_LADDER) + ["full"],
        "pca_fitting_set": "inner_tr = 80% of train_idx (train_test_split seed=0)",
        "pca_centering": "sklearn PCA default (mean-center)",
        "pca_whitening": False,
        "legacy_hsc_pca": "independent (separate PCA on X_leg and X_hsc)",
        "standard_scaler": "after PCA, inside Ridge fit (StandardScaler on Z_x and Z_y)",
        "direction": "Legacy→HSC (col2→col1)",
        "gallery": "test-only",
        "reproduces": "paper_alignment_controls pca_ridge_scores.csv rank=256",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    score_rows = []
    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        # X1=HSC target, X2=Legacy source
        n = len(X1)
        d_in, d_out = int(X2.shape[1]), int(X1.shape[1])
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed, shuffle=True
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
        inner_tr, _inner_va = train_test_split(
            train_idx, test_size=0.2, random_state=args.seed, shuffle=True
        )
        params_m = float(cfg.get("approx_params_m", 1))
        logp = math.log10(max(params_m * 1e6, 1.0))
        fam = str(cfg["family"])

        # Dense + Dense+Ridge baselines (same split)
        m_dense = mknn_pair(X2[test_idx], X1[test_idx], args.k, device, args.row_batch)
        mapped_full, train_mse_full, _ = fit_ridge_map(
            X2, X1, train_idx, alpha=args.alpha
        )
        m_dense_ridge = mknn_pair(
            mapped_full[test_idx], X1[test_idx], args.k, device, args.row_batch
        )
        test_mse_full = float(
            np.mean((mapped_full[test_idx] - X1[test_idx]) ** 2)
        )

        # Rank ladder
        valid_ranks = [r for r in RANK_LADDER if r < min(d_in, d_out, len(inner_tr) - 1)]
        for r in valid_ranks:
            n_comp = min(r, len(inner_tr) - 1, d_in, d_out)
            pca_x = PCA(n_components=n_comp).fit(X2[inner_tr])
            pca_y = PCA(n_components=n_comp).fit(X1[inner_tr])
            Z2 = pca_x.transform(X2).astype(np.float32)
            Z1 = pca_y.transform(X1).astype(np.float32)
            mapped, train_mse, _ = fit_ridge_map(Z2, Z1, train_idx, alpha=args.alpha)
            mknn = mknn_pair(
                mapped[test_idx], Z1[test_idx], args.k, device, args.row_batch
            )
            test_mse = float(np.mean((mapped[test_idx] - Z1[test_idx]) ** 2))
            score_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "size_name": cfg.get("size_name"),
                    "parameter_count": int(round(params_m * 1e6)),
                    "log10_params": logp,
                    "input_dim": d_in,
                    "output_dim": d_out,
                    "rank": r,
                    "rank_label": str(r),
                    "mknn": mknn,
                    "dense_raw": m_dense,
                    "dense_ridge": m_dense_ridge,
                    "pca_minus_dense_ridge": mknn - m_dense_ridge,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "train_mse_full_ridge": train_mse_full,
                    "test_mse_full_ridge": test_mse_full,
                }
            )
            print(
                f"{name} rank={r}: mknn={mknn:.4f} "
                f"ΔvsDR={mknn-m_dense_ridge:+.4f}",
                flush=True,
            )

        # full = Dense+Ridge (no PCA)
        score_rows.append(
            {
                "family": fam,
                "model": name,
                "size_name": cfg.get("size_name"),
                "parameter_count": int(round(params_m * 1e6)),
                "log10_params": logp,
                "input_dim": d_in,
                "output_dim": d_out,
                "rank": max(d_in, d_out),
                "rank_label": "full",
                "mknn": m_dense_ridge,
                "dense_raw": m_dense,
                "dense_ridge": m_dense_ridge,
                "pca_minus_dense_ridge": 0.0,
                "train_mse": train_mse_full,
                "test_mse": test_mse_full,
                "train_mse_full_ridge": train_mse_full,
                "test_mse_full_ridge": test_mse_full,
            }
        )
        print(
            f"{name} full: dense={m_dense:.4f} denseR={m_dense_ridge:.4f}",
            flush=True,
        )

    scores = pd.DataFrame(score_rows)
    scores.to_csv(out_dir / "pca_rank_scores.csv", index=False)

    # Aggregate by rank_label
    agg_rows = []
    for lab, g in scores.groupby("rank_label", sort=False):
        # numeric order
        S = g["pca_minus_dense_ridge"].to_numpy()
        mu, lo, hi = clustered_mean_ci(
            S, g["family"].to_numpy(), n_boot=2000, seed=args.seed
        )
        agg_rows.append(
            {
                "rank_label": lab,
                "rank_numeric": int(g["rank"].iloc[0]) if lab != "full" else -1,
                "n_rungs": len(g),
                "mean_mknn": float(g["mknn"].mean()),
                "median_mknn": float(g["mknn"].median()),
                "mean_dense_raw": float(g["dense_raw"].mean()),
                "mean_dense_ridge": float(g["dense_ridge"].mean()),
                "mean_S_pca": float(S.mean()),
                "median_S_pca": float(np.median(S)),
                "n_positive_S": int(np.sum(S > 0)),
                "S_family_boot_mean": mu,
                "S_family_boot_ci_lo": lo,
                "S_family_boot_ci_hi": hi,
                "mean_train_mse": float(g["train_mse"].mean()),
                "mean_test_mse": float(g["test_mse"].mean()),
            }
        )
    # sort: numeric ranks then full
    agg = pd.DataFrame(agg_rows)
    order = {str(r): i for i, r in enumerate(RANK_LADDER)}
    order["full"] = 999
    agg["_ord"] = agg["rank_label"].map(order)
    agg = agg.sort_values("_ord").drop(columns="_ord")
    agg.to_csv(out_dir / "pca_rank_aggregate.csv", index=False)

    # Best ranks
    best_rows = []
    for model, g in scores[scores["rank_label"] != "full"].groupby("model"):
        j = g["mknn"].idxmax()
        best_rows.append(g.loc[j].to_dict())
    best = pd.DataFrame(best_rows)
    best.to_csv(out_dir / "pca_best_rank_per_rung.csv", index=False)
    fam_best = []
    for fam in FAMILY_ORDER:
        g = scores[(scores["family"] == fam) & (scores["rank_label"] != "full")]
        if g.empty:
            continue
        means = g.groupby("rank_label")["mknn"].mean()
        fam_best.append(
            {
                "family": fam,
                "best_rank": means.idxmax(),
                "best_mean_mknn": float(means.max()),
                "means": means.to_dict(),
            }
        )
    # aggregate best (among PCA ranks only)
    pca_only = scores[scores["rank_label"] != "full"]
    agg_means = pca_only.groupby("rank_label")["mknn"].mean()
    (out_dir / "best_rank_summary.json").write_text(
        json.dumps(
            {
                "aggregate_best_rank": str(agg_means.idxmax()),
                "aggregate_best_mean_mknn": float(agg_means.max()),
                "per_family_best": fam_best,
                "per_rung_best_rank_counts": best["rank_label"]
                .value_counts()
                .to_dict(),
            },
            indent=2,
        )
        + "\n"
    )

    # Scaling: β per family per rank vs Dense+Ridge
    # Dense+Ridge β from full rows
    beta_dr = {}
    full = scores[scores["rank_label"] == "full"]
    for fam in FAMILY_ORDER:
        sub = full[full["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        beta_dr[fam] = family_ols_slope(
            sub["log10_params"].to_numpy(float), sub["mknn"].to_numpy(float)
        )

    scaling_rows = []
    t_rows = []
    for lab in list(map(str, RANK_LADDER)) + ["full"]:
        g = scores[scores["rank_label"] == lab]
        if g.empty:
            continue
        dbetas = {}
        betas = {}
        for fam in FAMILY_ORDER:
            sub = g[g["family"] == fam].sort_values("log10_params")
            if len(sub) < 2 or fam not in beta_dr:
                continue
            b = family_ols_slope(
                sub["log10_params"].to_numpy(float), sub["mknn"].to_numpy(float)
            )
            betas[fam] = b
            dbetas[fam] = b - beta_dr[fam]
            scaling_rows.append(
                {
                    "rank_label": lab,
                    "family": fam,
                    "beta": b,
                    "beta_dense_ridge": beta_dr[fam],
                    "delta_beta": dbetas[fam],
                    "n_rungs": len(sub),
                    "two_point_only": len(sub) == 2,
                }
            )
        if dbetas:
            T = float(np.mean(list(dbetas.values())))
            t_rows.append(
                {
                    "rank_label": lab,
                    "T_pca": T,
                    "mean_delta_beta": T,
                    "median_delta_beta": float(np.median(list(dbetas.values()))),
                    "n_pos_families": int(sum(v > 0 for v in dbetas.values())),
                    "n_families": len(dbetas),
                    **{f"delta_beta_{f}": dbetas[f] for f in dbetas},
                    **{f"beta_{f}": betas[f] for f in betas},
                }
            )
    pd.DataFrame(scaling_rows).to_csv(out_dir / "pca_family_slopes.csv", index=False)
    tdf = pd.DataFrame(t_rows)
    tdf.to_csv(out_dir / "pca_scaling_T_by_rank.csv", index=False)

    # Compare to k90 if available
    k90_path = args.k90_csv
    if k90_path is None:
        cands = [
            root / "outputs/ridge_scaling_geometry/transfer_complexity_k90.csv",
            ROOT / "outputs/ridge_scaling_geometry/transfer_complexity_k90.csv",
        ]
        k90_path = next((p for p in cands if p.is_file()), None)
    if k90_path is not None and k90_path.is_file():
        k90 = pd.read_csv(k90_path)
        cmp_rows = []
        for _, br in best.iterrows():
            row = {
                "family": br["family"],
                "model": br["model"],
                "embedding_dim": br["input_dim"],
                "best_pca_rank": int(br["rank"]),
                "best_pca_rank_frac": float(br["rank"] / br["input_dim"]),
                "best_pca_mknn": float(br["mknn"]),
            }
            m = k90[k90["model"] == br["model"]]
            if len(m):
                row["k90_transfer"] = float(m.iloc[0]["k90_transfer"])
                row["k90_transfer_frac"] = float(m.iloc[0]["k90_transfer_frac"])
            cmp_rows.append(row)
        pd.DataFrame(cmp_rows).to_csv(
            out_dir / "pca_best_vs_ridge_k90.csv", index=False
        )

    # Figures
    # A: mean mKNN vs rank
    plot_labs = [lab for lab in list(map(str, RANK_LADDER)) + ["full"] if lab in set(agg["rank_label"])]
    xs = np.arange(len(plot_labs))
    means = [float(agg.loc[agg.rank_label == lab, "mean_mknn"].iloc[0]) for lab in plot_labs]
    mean_dense = float(scores.groupby("model")["dense_raw"].first().mean())
    mean_dr = float(scores.groupby("model")["dense_ridge"].first().mean())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, means, "o-", color="C2", label="PCA+Ridge")
    ax.axhline(mean_dense, color="C0", ls="--", label=f"Raw Dense ({mean_dense:.3f})")
    ax.axhline(mean_dr, color="C1", ls="--", label=f"Dense+Ridge ({mean_dr:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_labs)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("mean held-out mKNN@10")
    ax.set_title("PCA+Ridge rank sweep (test-only gallery)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_pca_mean_mknn_vs_rank.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_A_mean_mknn_vs_rank.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # B: family curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fam in FAMILY_ORDER:
        ys = []
        labs_f = []
        for lab in plot_labs:
            g = scores[(scores["family"] == fam) & (scores["rank_label"] == lab)]
            if g.empty:
                continue
            labs_f.append(lab)
            ys.append(float(g["mknn"].mean()))
        if not ys:
            continue
        ax.plot(
            [plot_labs.index(l) for l in labs_f],
            ys,
            "o-",
            label=FAMILY_LABEL.get(fam, fam),
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_labs)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("family-mean mKNN@10")
    ax.set_title("PCA+Ridge by architecture family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_pca_family_curves.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_B_family_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # C: T_PCA vs rank
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t_plot = tdf[tdf["rank_label"].isin(plot_labs)].copy()
    t_plot["_ord"] = t_plot["rank_label"].map(order)
    t_plot = t_plot.sort_values("_ord")
    ax.plot(
        range(len(t_plot)),
        t_plot["T_pca"],
        "o-",
        color="C3",
        label=r"$T_{\mathrm{PCA}}=\mathrm{mean}_F(\beta_d-\beta_{\mathrm{Dense+Ridge}})$",
    )
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xticks(range(len(t_plot)))
    ax.set_xticklabels(t_plot["rank_label"])
    ax.set_xlabel("PCA rank")
    ax.set_ylabel(r"$T_{\mathrm{PCA}}(d)$")
    ax.set_title("PCA scaling interaction vs Dense+Ridge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_pca_scaling_T_vs_rank.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_C_scaling_T_vs_rank.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_dir}", flush=True)
    print("aggregate means:\n", agg[["rank_label", "mean_mknn", "mean_S_pca", "n_positive_S"]].to_string(index=False), flush=True)
    print("T_pca:\n", tdf[["rank_label", "T_pca", "n_pos_families"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
