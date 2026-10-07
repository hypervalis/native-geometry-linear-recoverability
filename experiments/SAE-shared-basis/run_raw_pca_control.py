#!/usr/bin/env python3
"""Raw PCA control: pooled shared PCA basis, no Ridge.

Independent per-survey PCA is NOT used here (bases would be incomparable).
PCA is fit on stacked train embeddings from both surveys (no pairing required),
then both surveys are projected with the same V_d and scored with raw cosine
mKNN (same knn_cos l2-normalization as raw Dense).

Compares against frozen Dense / Dense+Ridge / independent-PCA+Ridge from
outputs/final_pca_rank_sweep/pca_rank_scores.csv.
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--pca-ridge-csv",
        type=Path,
        default=None,
        help="pca_rank_scores.csv from final_pca_rank_sweep",
    )
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "raw_pca_control"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "paper_working" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "tmp_ridge_geom/official_legacy_pairs.yaml",
            ROOT / "experiments/universetbd_shared_basis_mknn/official_legacy_pairs.yaml",
        ]
        pairs_path = next(p for p in cands if p.is_file())

    if args.pca_ridge_csv is not None:
        pr_path = args.pca_ridge_csv.expanduser().resolve()
    else:
        cands = [
            root / "outputs/final_pca_rank_sweep/pca_rank_scores.csv",
            ROOT / "outputs/final_pca_rank_sweep/pca_rank_scores.csv",
        ]
        pr_path = next(p for p in cands if p.is_file())
    pr = pd.read_csv(pr_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = yaml.safe_load(pairs_path.read_text())
    names = [
        n
        for n, cfg in pairs.items()
        if isinstance(cfg, dict) and "legacysurvey" in str(cfg.get("parquet2", ""))
    ]

    meta = {
        "pca_basis": "single pooled PCA on stacked [X_Legacy_train; X_HSC_train]",
        "pairing_used_for_pca": False,
        "pca_centering": "sklearn PCA default (mean-center)",
        "pca_whitening": False,
        "standard_scaler_before_pca": False,
        "post_projection": "l2-normalize inside knn_cos (same as raw Dense)",
        "dense_raw_note": "raw Dense also uses knn_cos row-wise l2-normalization; no StandardScaler",
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "gallery": "test-only",
        "direction": "Legacy→HSC (col2 source / col1 target for naming; raw PCA is symmetric cosine)",
        "pca_ridge_source": str(pr_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    rows = []
    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        # X1=HSC, X2=Legacy; dims equal on official ladder
        assert X1.shape[1] == X2.shape[1], f"dim mismatch {name}"
        d = int(X1.shape[1])
        n = len(X1)
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed, shuffle=True
        )
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

        params_m = float(cfg.get("approx_params_m", 1))
        logp = math.log10(max(params_m * 1e6, 1.0))
        fam = str(cfg["family"])

        # Raw Dense reference (recomputed; should match frozen)
        m_dense = mknn_pair(X2[test_idx], X1[test_idx], args.k, device, args.row_batch)

        # Pooled train stack (no pairing)
        X_pool = np.vstack([X2[train_idx], X1[train_idx]]).astype(np.float64)

        valid_ranks = [r for r in RANK_LADDER if r < d and r < len(X_pool)]
        for r in valid_ranks:
            pca = PCA(n_components=r, whiten=False)
            pca.fit(X_pool)
            Z2 = pca.transform(X2).astype(np.float32)
            Z1 = pca.transform(X1).astype(np.float32)
            m_pca = mknn_pair(
                Z2[test_idx], Z1[test_idx], args.k, device, args.row_batch
            )
            rows.append(
                {
                    "family": fam,
                    "model": name,
                    "size_name": cfg.get("size_name"),
                    "parameter_count": int(round(params_m * 1e6)),
                    "log10_params": logp,
                    "embedding_dim": d,
                    "pca_rank": r,
                    "rank_label": str(r),
                    "raw_dense_mknn": m_dense,
                    "raw_pca_mknn": m_pca,
                    "raw_pca_minus_dense": m_pca - m_dense,
                }
            )
            print(
                f"{name} rawPCA d={r}: {m_pca:.4f} "
                f"(dense={m_dense:.4f}, Δ={m_pca-m_dense:+.4f})",
                flush=True,
            )

        # full reference = raw Dense
        rows.append(
            {
                "family": fam,
                "model": name,
                "size_name": cfg.get("size_name"),
                "parameter_count": int(round(params_m * 1e6)),
                "log10_params": logp,
                "embedding_dim": d,
                "pca_rank": d,
                "rank_label": "full",
                "raw_dense_mknn": m_dense,
                "raw_pca_mknn": m_dense,
                "raw_pca_minus_dense": 0.0,
            }
        )
        print(f"{name} full=rawDense={m_dense:.4f}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_pca_scores.csv", index=False)

    # Merge with independent-PCA+Ridge (different PCA construction — documented)
    pr_map = pr.rename(
        columns={
            "mknn": "pca_ridge_mknn",
            "dense_raw": "dense_raw_from_pca_sweep",
            "dense_ridge": "dense_ridge_mknn",
        }
    )[
        [
            "model",
            "rank_label",
            "pca_ridge_mknn",
            "dense_raw_from_pca_sweep",
            "dense_ridge_mknn",
        ]
    ]
    merged = raw.merge(pr_map, on=["model", "rank_label"], how="left")
    # Prefer recomputed dense; use dense_ridge from sweep
    merged["ridge_gain_after_pca"] = (
        merged["pca_ridge_mknn"] - merged["raw_pca_mknn"]
    )
    merged["ridge_gain_full"] = (
        merged["dense_ridge_mknn"] - merged["raw_dense_mknn"]
    )
    merged["I_pca_x_ridge"] = (
        merged["ridge_gain_after_pca"] - merged["ridge_gain_full"]
    )
    merged.to_csv(out_dir / "raw_pca_vs_pca_ridge.csv", index=False)

    # Aggregates by rank
    agg_rows = []
    for lab in list(map(str, RANK_LADDER)) + ["full"]:
        g = merged[merged["rank_label"] == lab]
        if g.empty:
            continue
        lift = g["raw_pca_minus_dense"].to_numpy()
        row = {
            "rank_label": lab,
            "n_rungs": len(g),
            "mean_raw_pca": float(g["raw_pca_mknn"].mean()),
            "mean_raw_dense": float(g["raw_dense_mknn"].mean()),
            "mean_lift_vs_dense": float(lift.mean()),
            "median_lift_vs_dense": float(np.median(lift)),
            "n_pos_lift": int(np.sum(lift > 0)),
            "mean_pca_ridge": float(g["pca_ridge_mknn"].mean())
            if g["pca_ridge_mknn"].notna().any()
            else float("nan"),
            "mean_dense_ridge": float(g["dense_ridge_mknn"].mean())
            if g["dense_ridge_mknn"].notna().any()
            else float("nan"),
            "mean_ridge_gain_after_pca": float(g["ridge_gain_after_pca"].mean())
            if g["ridge_gain_after_pca"].notna().any()
            else float("nan"),
            "mean_I": float(g["I_pca_x_ridge"].mean())
            if g["I_pca_x_ridge"].notna().any()
            else float("nan"),
            "median_I": float(g["I_pca_x_ridge"].median())
            if g["I_pca_x_ridge"].notna().any()
            else float("nan"),
            "n_pos_I": int((g["I_pca_x_ridge"] > 0).sum())
            if g["I_pca_x_ridge"].notna().any()
            else 0,
        }
        # family means of lift
        for fam in FAMILY_ORDER:
            gf = g[g["family"] == fam]
            if len(gf):
                row[f"lift_{fam}"] = float(gf["raw_pca_minus_dense"].mean())
                if gf["I_pca_x_ridge"].notna().any():
                    row[f"I_{fam}"] = float(gf["I_pca_x_ridge"].mean())
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "raw_pca_aggregate.csv", index=False)

    # Focused 256 / 512 on common support
    focus = []
    for lab in ["256", "512"]:
        g = merged[merged["rank_label"] == lab].dropna(subset=["pca_ridge_mknn"])
        focus.append(
            {
                "rank_label": lab,
                "n": len(g),
                "raw_dense": float(g["raw_dense_mknn"].mean()),
                "raw_pca": float(g["raw_pca_mknn"].mean()),
                "dense_ridge": float(g["dense_ridge_mknn"].mean()),
                "pca_ridge": float(g["pca_ridge_mknn"].mean()),
                "mean_lift_raw_pca": float(g["raw_pca_minus_dense"].mean()),
                "mean_I": float(g["I_pca_x_ridge"].mean()),
                "n_pos_lift": int((g["raw_pca_minus_dense"] > 0).sum()),
                "n_pos_I": int((g["I_pca_x_ridge"] > 0).sum()),
            }
        )
    pd.DataFrame(focus).to_csv(out_dir / "raw_pca_focus_256_512.csv", index=False)

    # Scaling: raw PCA vs raw Dense
    beta_dense = {}
    full = raw[raw["rank_label"] == "full"]
    for fam in FAMILY_ORDER:
        sub = full[full["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        beta_dense[fam] = family_ols_slope(
            sub["log10_params"].to_numpy(float),
            sub["raw_dense_mknn"].to_numpy(float),
        )

    scale_rows = []
    t_rows = []
    for lab in list(map(str, RANK_LADDER)) + ["full"]:
        g = raw[raw["rank_label"] == lab]
        if g.empty:
            continue
        db = {}
        for fam in FAMILY_ORDER:
            sub = g[g["family"] == fam].sort_values("log10_params")
            if len(sub) < 2 or fam not in beta_dense:
                continue
            b = family_ols_slope(
                sub["log10_params"].to_numpy(float),
                sub["raw_pca_mknn"].to_numpy(float),
            )
            db[fam] = b - beta_dense[fam]
            scale_rows.append(
                {
                    "rank_label": lab,
                    "family": fam,
                    "beta_raw_pca": b,
                    "beta_raw_dense": beta_dense[fam],
                    "delta_beta": db[fam],
                    "n_rungs": len(sub),
                }
            )
        if db:
            t_rows.append(
                {
                    "rank_label": lab,
                    "mean_delta_beta": float(np.mean(list(db.values()))),
                    "n_pos_families": int(sum(v > 0 for v in db.values())),
                    "n_families": len(db),
                    **{f"delta_beta_{f}": db[f] for f in db},
                }
            )
    pd.DataFrame(scale_rows).to_csv(out_dir / "raw_pca_family_slopes.csv", index=False)
    pd.DataFrame(t_rows).to_csv(out_dir / "raw_pca_scaling_vs_dense.csv", index=False)

    # Also slopes for Dense+Ridge and PCA+Ridge at same ranks (from merged)
    slope_cmp = []
    for lab in list(map(str, RANK_LADDER)) + ["full"]:
        g = merged[merged["rank_label"] == lab]
        if g.empty or g["pca_ridge_mknn"].isna().all():
            continue
        for method, col in [
            ("raw_dense", "raw_dense_mknn"),
            ("raw_pca", "raw_pca_mknn"),
            ("dense_ridge", "dense_ridge_mknn"),
            ("pca_ridge", "pca_ridge_mknn"),
        ]:
            betas = []
            for fam in FAMILY_ORDER:
                sub = g[g["family"] == fam].sort_values("log10_params")
                if len(sub) < 2 or sub[col].isna().any():
                    continue
                betas.append(
                    family_ols_slope(
                        sub["log10_params"].to_numpy(float),
                        sub[col].to_numpy(float),
                    )
                )
            if betas:
                slope_cmp.append(
                    {
                        "rank_label": lab,
                        "method": method,
                        "mean_family_beta": float(np.mean(betas)),
                        "n_families": len(betas),
                    }
                )
    pd.DataFrame(slope_cmp).to_csv(out_dir / "slope_decomposition.csv", index=False)

    # Figures
    order = {str(r): i for i, r in enumerate(RANK_LADDER)}
    order["full"] = 999
    plot_labs = [
        lab
        for lab in list(map(str, RANK_LADDER)) + ["full"]
        if lab in set(agg["rank_label"])
    ]
    xs = np.arange(len(plot_labs))
    mean_dense = float(raw.groupby("model")["raw_dense_mknn"].first().mean())
    mean_dr = float(
        merged.groupby("model")["dense_ridge_mknn"].first().mean()
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        xs,
        [float(agg.loc[agg.rank_label == lab, "mean_raw_pca"].iloc[0]) for lab in plot_labs],
        "o-",
        color="C0",
        label="Raw PCA (pooled shared basis)",
    )
    pr_means = []
    for lab in plot_labs:
        v = agg.loc[agg.rank_label == lab, "mean_pca_ridge"]
        pr_means.append(float(v.iloc[0]) if len(v) and np.isfinite(v.iloc[0]) else np.nan)
    ax.plot(xs, pr_means, "s-", color="C2", label="PCA+Ridge (indep. bases)*")
    ax.axhline(mean_dense, color="C0", ls="--", alpha=0.7, label=f"Raw Dense ({mean_dense:.3f})")
    ax.axhline(mean_dr, color="C1", ls="--", alpha=0.7, label=f"Dense+Ridge ({mean_dr:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_labs)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("mean mKNN@10")
    ax.set_title("2×2: Raw PCA vs PCA+Ridge (*indep. PCA+Ridge from prior sweep)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "raw_pca_rank_curve.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_A_rank_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    I_vals = [
        float(agg.loc[agg.rank_label == lab, "mean_I"].iloc[0]) for lab in plot_labs
    ]
    ax.plot(xs, I_vals, "o-", color="C3", label=r"$I_{\mathrm{PCA}\times\mathrm{Ridge}}$")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_labs)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("mean interaction")
    ax.set_title("PCA×Ridge interaction vs rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_ridge_interaction.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_B_interaction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("aggregate:\n", agg[["rank_label", "mean_raw_pca", "mean_lift_vs_dense", "n_pos_lift", "mean_I", "n_pos_I"]].to_string(index=False), flush=True)
    print("focus 256/512:\n", pd.DataFrame(focus).to_string(index=False), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
