#!/usr/bin/env python3
"""Fixed-rank scaling control: independent PCA256/128 native mKNN vs PCA+Ridge.

Compares supervised slope amplification at equal representation dimension:
  M_{PCA_r}           — raw cosine mKNN on independent survey PCAs (no Ridge)
  M_{PCA_r+Ridge}     — same PCAs + StandardScaler + Ridge(alpha=1)

Uses the exact independent-PCA convention from run_final_pca_rank_sweep.py.
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
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FAMILY_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
PRIMARY_RANKS = (256, 128)


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


def sign_test_p(n_pos: int, n_fam: int = 5) -> float:
    """One-sided P(# positive >= n_pos) under p=0.5 per family."""
    from math import comb

    return sum(comb(n_fam, k) for k in range(n_pos, n_fam + 1)) / (2**n_fam)


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
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> np.ndarray:
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    return y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)


def fit_independent_pca(
    X1: np.ndarray, X2: np.ndarray, inner_tr: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    d_in, d_out = X2.shape[1], X1.shape[1]
    n_comp = min(rank, len(inner_tr) - 1, d_in, d_out)
    pca_x = PCA(n_components=n_comp).fit(X2[inner_tr])
    pca_y = PCA(n_components=n_comp).fit(X1[inner_tr])
    Z2 = pca_x.transform(X2).astype(np.float32)
    Z1 = pca_y.transform(X1).astype(np.float32)
    meta = {
        "rank_requested": rank,
        "rank_actual": int(n_comp),
        "pca_fit_subset": "inner_tr (80% of train_idx)",
        "centering": "sklearn PCA default (per survey)",
        "whitening": False,
        "legacy_basis": "PCA_L on X2[inner_tr]",
        "hsc_basis": "PCA_H on X1[inner_tr]",
        "post_pca_ridge_scaler": "StandardScaler on Z_x, Z_y (train_idx)",
    }
    return Z2, Z1, meta


def plot_fixed_rank(df: pd.DataFrame, rank: int, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True)
    for ax, fam in zip(axes, FAMILY_ORDER):
        sub = df[(df["family"] == fam) & (df["rank"] == rank)].sort_values("log10_params")
        if sub.empty:
            ax.set_title(FAMILY_LABEL[fam])
            continue
        ax.plot(sub["log10_params"], sub["raw_pca_mknn"], "o-", label="Raw PCA", ms=4)
        ax.plot(
            sub["log10_params"], sub["pca_ridge_mknn"], "s-", label="PCA+Ridge", ms=4
        )
        ax.set_title(FAMILY_LABEL[fam], fontsize=10)
        ax.set_xlabel(r"$\log_{10}P$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mKNN@10")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Fixed rank {rank}: native PCA vs PCA+Ridge (test-only gallery)", y=1.15)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_rank(scores: pd.DataFrame, rank: int) -> dict:
    sub = scores[scores["rank"] == rank].copy()
    fam_rows = []
    for fam in FAMILY_ORDER:
        fs = sub[sub["family"] == fam]
        if fs.empty:
            continue
        logp = fs["log10_params"].to_numpy()
        beta_raw = family_ols_slope(logp, fs["raw_pca_mknn"].to_numpy())
        beta_ridge = family_ols_slope(logp, fs["pca_ridge_mknn"].to_numpy())
        fam_rows.append(
            {
                "rank": rank,
                "family": fam,
                "beta_raw_pca": beta_raw,
                "beta_pca_ridge": beta_ridge,
                "delta_beta": beta_ridge - beta_raw,
                "n_rungs": len(fs),
                "two_point_only": len(fs) == 2,
            }
        )
    fam_df = pd.DataFrame(fam_rows)
    n_pos = int((fam_df["delta_beta"] > 0).sum())
    T = float(fam_df["delta_beta"].mean())
    ex_ij = fam_df[fam_df["family"] != "ijepa"]
    return {
        "rank": rank,
        "family_slopes": fam_df,
        "T": T,
        "n_pos_families": n_pos,
        "n_families": len(fam_df),
        "sign_test_p_one_sided": sign_test_p(n_pos, len(fam_df)),
        "T_excluding_ijepa": float(ex_ij["delta_beta"].mean()) if len(ex_ij) else float("nan"),
        "n_pos_excluding_ijepa": int((ex_ij["delta_beta"] > 0).sum()) if len(ex_ij) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ranks", type=int, nargs="+", default=list(PRIMARY_RANKS))
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "fixed_rank_scaling"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "paper_working" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "experiments/universetbd_shared_basis_mknn/official_legacy_pairs.yaml",
            ROOT / "experiments/universetbd_shared_basis_mknn/official_legacy_pairs.yaml",
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

    idx = np.arange(args.max_n if args.max_n else 16384)
    # establish n from first pair
    X1_0, X2_0, _ = load_pair_arrays(root, pairs[names[0]], args.max_n, args.seed)
    n = len(X1_0)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    inner_tr, _ = train_test_split(
        train_idx, test_size=0.2, random_state=args.seed, shuffle=True
    )

    protocol = {
        "pca_convention": "independent per survey (Legacy PCA_L, HSC PCA_H)",
        "pca_fit_subset": "inner_tr = 80% of train_idx",
        "centering": "sklearn PCA default",
        "whitening": False,
        "ridge": f"StandardScaler + Ridge(alpha={args.alpha}, fit_intercept=True) on full train_idx",
        "raw_pca_mknn": "cosine mKNN@10 on Z_L vs Z_H, test-only gallery, no Ridge",
        "direction": "Legacy→HSC",
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "ranks": args.ranks,
        "pairs_yaml": str(pairs_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    rows = []
    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        assert len(X1) == n
        params_m = float(cfg.get("approx_params_m", 1))
        logp = math.log10(max(params_m * 1e6, 1.0))
        fam = str(cfg["family"])
        d_in, d_out = X2.shape[1], X1.shape[1]

        for rank in args.ranks:
            if rank >= min(d_in, d_out, len(inner_tr) - 1):
                print(f"skip {name} rank={rank} (dim={min(d_in,d_out)})", flush=True)
                continue
            Z2, Z1, pca_meta = fit_independent_pca(X1, X2, inner_tr, rank)
            m_raw = mknn_pair(Z2[test_idx], Z1[test_idx], args.k, device, args.row_batch)
            mapped = fit_ridge_map(Z2, Z1, train_idx, alpha=args.alpha)
            m_ridge = mknn_pair(
                mapped[test_idx], Z1[test_idx], args.k, device, args.row_batch
            )
            rows.append(
                {
                    "family": fam,
                    "model": name,
                    "size_name": cfg.get("size_name"),
                    "parameter_count": int(round(params_m * 1e6)),
                    "log10_params": logp,
                    "rank": rank,
                    "rank_actual": pca_meta["rank_actual"],
                    "raw_pca_mknn": m_raw,
                    "pca_ridge_mknn": m_ridge,
                    "delta_alignment": m_ridge - m_raw,
                }
            )
            print(
                f"{name} r={rank}: raw={m_raw:.4f} ridge={m_ridge:.4f} "
                f"Δ={m_ridge-m_raw:+.4f}",
                flush=True,
            )

    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "fixed_rank_scores.csv", index=False)

    summaries = []
    slope_frames = []
    for rank in args.ranks:
        if rank not in scores["rank"].values:
            continue
        sm = summarize_rank(scores, rank)
        summaries.append({k: v for k, v in sm.items() if k != "family_slopes"})
        slope_frames.append(sm["family_slopes"])
        if rank == 256:
            plot_fixed_rank(scores, rank, fig_dir / "fixed_rank_256_scaling.png")

    if slope_frames:
        pd.concat(slope_frames, ignore_index=True).to_csv(
            out_dir / "fixed_rank_family_slopes.csv", index=False
        )
    pd.DataFrame(summaries).to_csv(out_dir / "fixed_rank_scaling_summary.csv", index=False)
    (out_dir / "fixed_rank_scaling_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
