#!/usr/bin/env python3
"""SAE TopK sensitivity: inference-time K sweep with frozen paper protocol.

Uses the preferred F=2048 seed=0 SAE checkpoint per rung (heterogeneous training
TopK in 18--23) and overrides ``model.k`` at encode time. This is inference-time
sparsity sensitivity, not retraining.

Frozen:
  Legacy→HSC, n=16384, test_size=0.2, seed=0, Ridge α=1, train-only IDF,
  mKNN@10, test-only gallery, Dense+Ridge baseline from the same split.
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sae_affine_basis_mknn_gpu import encode as encode_sae  # noqa: E402
from sae_affine_basis_mknn_gpu import idf_np, load_sae  # noqa: E402

SAE_TAG_PREFER = (
    "F2048_k18_seed0",
    "F2048_k19_seed0",
    "F2048_k20_seed0",
    "F2048_k21_seed0",
    "F2048_k22_seed0",
    "F2048_k23_seed0",
    "F2048_k32_seed0",
    "F2048_k64_seed0",
)

DEFAULT_K_GRID = (10, 20, 40)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument(
        "--pairs-yaml",
        default="configs/official_legacy_pairs.yaml",
    )
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--k-mknn", type=int, default=10)
    p.add_argument("--topk-grid", default="10,20,40")
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument(
        "--dense-scores",
        default="outputs/paper_alignment_controls/test_only_gallery_scores.csv",
        help="Reuse frozen Dense / Dense+Ridge / native scores (k=10).",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/paper_robustness/sae_topk_sensitivity",
    )
    p.add_argument("--pairs", default="")
    return p.parse_args()


def platonic_root(cli: str | None) -> Path:
    if cli:
        return Path(cli).expanduser().resolve()
    env = __import__("os").environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "platonic-universe").resolve()


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def load_col(path: Path, col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[col])
    return np.vstack(table.column(0).to_pylist()).astype(np.float32)


def resolve_sae_dir(root: Path, parquet_rel: str, col: str) -> Path | None:
    base = root / "outputs" / "sae" / Path(parquet_rel).stem / col
    if not base.is_dir():
        return None
    tags = {p.name for p in base.iterdir() if p.is_dir() and (p / "model.pt").is_file()}
    for tag in SAE_TAG_PREFER:
        if tag in tags:
            return base / tag
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "model.pt").is_file():
            return p
    return None


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
    return X1, X2


def fit_ridge_map(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> np.ndarray:
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    mapped = y_sc.inverse_transform(ridge.predict(x_sc.transform(x)))
    return mapped.astype(np.float32)


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


def apply_idf(C: np.ndarray, idf: np.ndarray) -> np.ndarray:
    return (C * idf[None, :]).astype(np.float32)


def family_ols_slope(logp: np.ndarray, y: np.ndarray) -> float:
    if len(logp) < 2:
        return float("nan")
    lr = LinearRegression()
    lr.fit(logp.reshape(-1, 1), y)
    return float(lr.coef_[0])


def clustered_mean_ci(
    vals: np.ndarray, families: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    fams = np.unique(families)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(fams, size=len(fams), replace=True)
        parts = [vals[families == f] for f in draw]
        boots.append(float(np.mean(np.concatenate(parts))))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean(vals)), float(lo), float(hi)


def set_inference_topk(bundle: dict, k: int) -> int:
    """Override encode-time TopK; returns training-time k from checkpoint."""
    train_k = int(bundle["k"])
    bundle["model"].k = int(k)
    return train_k


def encode_with_k(
    bundle: dict, X: np.ndarray, device: torch.device, k: int
) -> tuple[np.ndarray, int]:
    train_k = set_inference_topk(bundle, k)
    codes = encode_sae(bundle, X, device)
    return codes, train_k


def main() -> None:
    args = parse_args()
    root = platonic_root(args.platonic_root)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    k_grid = [int(x) for x in args.topk_grid.split(",") if x.strip()]
    pairs_path = resolve_path(root, args.pairs_yaml)
    if not pairs_path.is_file():
        alt = Path(__file__).resolve().parents[3] / args.pairs_yaml
        if alt.is_file():
            pairs_path = alt
    pairs = yaml.safe_load(pairs_path.read_text())
    if args.pairs.strip():
        want = {x.strip() for x in args.pairs.split(",") if x.strip()}
        pairs = {k: v for k, v in pairs.items() if k in want}

    dense_path = resolve_path(root, args.dense_scores)
    dense_df = pd.read_csv(dense_path)
    dense_df = dense_df[dense_df["k"] == args.k_mknn].set_index("pair")

    t0 = time.time()
    rows: list[dict] = []
    for pair, cfg in pairs.items():
        if pair not in dense_df.index:
            raise KeyError(f"missing Dense baseline for {pair} in {dense_path}")
        base = dense_df.loc[pair]
        X1, X2 = load_pair_arrays(root, cfg, args.max_n, args.seed)
        idx = np.arange(len(X1))
        train_idx, test_idx = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed
        )
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        sae1 = resolve_sae_dir(root, cfg["parquet1"], cfg["col1"])
        sae2 = resolve_sae_dir(root, cfg["parquet2"], cfg["col2"])
        if sae1 is None or sae2 is None:
            raise FileNotFoundError(f"missing SAE for {pair}: {sae1} {sae2}")
        b1 = load_sae(sae1, device)
        b2 = load_sae(sae2, device)

        for K in k_grid:
            C1, train_k1 = encode_with_k(b1, X1, device, K)
            C2, train_k2 = encode_with_k(b2, X2, device, K)
            mapped = fit_ridge_map(C2, C1, train_idx, alpha=args.alpha)
            idf = idf_np(C1[train_idx])
            te = test_idx
            m_sae = mknn_pair(
                apply_idf(C1[te], idf),
                apply_idf(mapped[te], idf),
                args.k_mknn,
                device,
                args.row_batch,
            )
            m_dense = float(base["dense"])
            m_dense_ridge = float(base["dense_ridge"])
            rows.append(
                {
                    "topk": K,
                    "pair": pair,
                    "family": cfg["family"],
                    "size_name": cfg["size_name"],
                    "size_rank": cfg["size_rank"],
                    "approx_params_m": cfg["approx_params_m"],
                    "log10P": float(np.log10(cfg["approx_params_m"] * 1e6)),
                    "train_topk_hsc": train_k1,
                    "train_topk_legacy": train_k2,
                    "sae_tag_hsc": sae1.name,
                    "sae_tag_legacy": sae2.name,
                    "M_native": m_dense,
                    "M_Dense_Ridge": m_dense_ridge,
                    "M_SAE_Ridge": m_sae,
                    "S_SAE": m_sae - m_dense_ridge,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
            print(
                f"{pair} K={K}: SAE={m_sae:.5f} DenseR={m_dense_ridge:.5f} "
                f"S={m_sae - m_dense_ridge:+.5f} (train k={train_k1}/{train_k2})",
                flush=True,
            )

    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_rung.csv", index=False)

    slope_rows = []
    summary_rows = []
    for K, g in per.groupby("topk"):
        g = g.sort_values(["family", "approx_params_m"])
        fam_deltas = {}
        for fam, fg in g.groupby("family"):
            logp = fg["log10P"].to_numpy()
            beta_sae = family_ols_slope(logp, fg["M_SAE_Ridge"].to_numpy())
            beta_dense = family_ols_slope(logp, fg["M_Dense_Ridge"].to_numpy())
            dbeta = beta_sae - beta_dense
            fam_deltas[fam] = dbeta
            slope_rows.append(
                {
                    "topk": int(K),
                    "family": fam,
                    "n_rungs": int(len(fg)),
                    "beta_SAE_Ridge": beta_sae,
                    "beta_Dense_Ridge": beta_dense,
                    "delta_beta_SAE": dbeta,
                    "mean_S_SAE": float(fg["S_SAE"].mean()),
                    "n_pos_S_SAE": int((fg["S_SAE"] > 0).sum()),
                }
            )
        T = float(np.mean(list(fam_deltas.values())))
        mu, lo, hi = clustered_mean_ci(
            g["S_SAE"].to_numpy(), g["family"].to_numpy(), seed=0
        )
        summary_rows.append(
            {
                "topk": int(K),
                "n_rungs": int(len(g)),
                "mean_M_SAE_Ridge": float(g["M_SAE_Ridge"].mean()),
                "mean_M_Dense_Ridge": float(g["M_Dense_Ridge"].mean()),
                "mean_S_SAE": float(g["S_SAE"].mean()),
                "S_SAE_family_boot_lo": lo,
                "S_SAE_family_boot_hi": hi,
                "n_pos_S_SAE_rungs": int((g["S_SAE"] > 0).sum()),
                "T_SAE": T,
                "n_pos_delta_beta_families": int(sum(v > 0 for v in fam_deltas.values())),
                "delta_beta_astropt": fam_deltas.get("astropt", float("nan")),
                "delta_beta_convnext": fam_deltas.get("convnext", float("nan")),
                "delta_beta_dinov2": fam_deltas.get("dinov2", float("nan")),
                "delta_beta_vit": fam_deltas.get("vit", float("nan")),
                "delta_beta_ijepa": fam_deltas.get("ijepa", float("nan")),
            }
        )

    slopes = pd.DataFrame(slope_rows)
    summary = pd.DataFrame(summary_rows).sort_values("topk")
    slopes.to_csv(out_dir / "family_slopes_by_topk.csv", index=False)
    summary.to_csv(out_dir / "summary_by_topk.csv", index=False)

    readme = f"""# SAE TopK sensitivity

## What K means here

**Inference-time TopK override**, not retraining.

Each of the 16 official Legacy↔HSC rungs has a single trained F=2048 seed=0
checkpoint with training TopK in {{18,...,23}} (heterogeneous across rungs; no
common training-time K across the full ladder). We load that checkpoint and set
`TopKSAE.k` before `encode`, so sparsity at evaluation varies while encoder /
decoder weights stay fixed.

This is **not** training-time TopK sensitivity.

## Available trained TopK (paper protocol preference)

Per-rung preferred tags (Legacy and HSC match):
{json.dumps({r['pair']: {'train_k': int(r['train_topk_hsc']), 'tag': r['sae_tag_hsc']} for r in per[per['topk']==k_grid[0]].to_dict(orient='records')}, indent=2)}

Independently trained common-K grids (e.g. F2048_K8/16/24/32 on all 16 rungs)
are **not** available.

## Chosen K grid

`{list(k_grid)}`

Reason: spans below / near / above the operating training range 18--23, and is
supported for all 16 rungs via inference-time override on the preferred
checkpoint. Same K is used for Legacy and HSC within each rung.

## Protocol (frozen)

- SAE width F=2048, seed=0 checkpoints
- Mapping direction Legacy → HSC (col2 → col1)
- n=16384, test_size={args.test_size}, random_state={args.seed}
- train-only Ridge (α={args.alpha}), StandardScaler on X and Y
- train-only IDF on HSC SAE codes
- mKNN k={args.k_mknn}, self-exclusion
- **test-only gallery** (same as paper alignment controls)
- Dense / Dense+Ridge / native taken from `{args.dense_scores}` at k={args.k_mknn}
  (same frozen split; not retuned per K)
- Ridge not retuned per K; K not chosen using held-out mKNN

## Missing rungs

None for K∈{list(k_grid)} — all 16 rungs evaluated at each K.

## Primary quantities

- `S_SAE = M_SAE+Ridge - M_Dense+Ridge`
- `Δβ_SAE,F = β_SAE+Ridge,F - β_Dense+Ridge,F` within family
- `T_SAE,K = mean_F Δβ_SAE,F,K`

Elapsed wall time: {time.time() - t0:.1f}s on {device}.
"""
    (out_dir / "README.md").write_text(readme)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    xs = summary["topk"].to_numpy()
    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].plot(xs, summary["mean_S_SAE"], "o-", color="C0", label="mean $S_{\\mathrm{SAE}}$")
    axes[0].fill_between(
        xs,
        summary["S_SAE_family_boot_lo"],
        summary["S_SAE_family_boot_hi"],
        color="C0",
        alpha=0.2,
        label="family-boot 95% CI",
    )
    axes[0].set_xlabel("SAE TopK (inference)")
    axes[0].set_ylabel(r"$M_{\mathrm{SAE+Ridge}}-M_{\mathrm{Dense+Ridge}}$")
    axes[0].set_title("A. residual vs Dense+Ridge")
    axes[0].legend(fontsize=7, loc="best")

    axes[1].axhline(0.0, color="0.7", lw=0.8)
    fam_cols = [
        ("delta_beta_astropt", "AstroPT"),
        ("delta_beta_convnext", "ConvNeXt"),
        ("delta_beta_dinov2", "DINOv2"),
        ("delta_beta_vit", "ViT"),
        ("delta_beta_ijepa", "I-JEPA"),
    ]
    for col, lab in fam_cols:
        axes[1].plot(xs, summary[col], "o-", alpha=0.35, lw=1, markersize=4, label=lab)
    axes[1].plot(xs, summary["T_SAE"], "o-", color="k", lw=2, markersize=6, label=r"$T_{\mathrm{SAE}}$")
    axes[1].set_xlabel("SAE TopK (inference)")
    axes[1].set_ylabel(r"$\Delta\beta^{\mathrm{SAE}}$ / $T_{\mathrm{SAE}}$")
    axes[1].set_title("B. scaling interaction vs Dense+Ridge")
    axes[1].legend(fontsize=6, loc="best", ncol=2)

    fig_path = out_dir / "sae_topk_sensitivity.png"
    fig.savefig(fig_path, dpi=200)
    repo_fig = Path(__file__).resolve().parents[3] / "figures" / "sae_topk_sensitivity.png"
    repo_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(repo_fig, dpi=200)
    plt.close(fig)

    meta = {
        "topk_grid": k_grid,
        "topk_meaning": "inference_time_override",
        "device": str(device),
        "elapsed_s": time.time() - t0,
        "n_rungs": int(per["pair"].nunique()),
        "summary": summary.to_dict(orient="records"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
