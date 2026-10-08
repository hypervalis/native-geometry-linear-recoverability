#!/usr/bin/env python3
"""Alignment controls: Dense+Ridge vs SAE/BSF on a disjoint test-only gallery.

Does not overwrite existing size-scaling outputs. Writes only to
``outputs/paper_alignment_controls/``.

Frozen paper split:
  n=16384, test_size=0.2, random_state=0, Legacy→HSC (col2→col1).
Ridge recipe (same as SAE/BSF shared-basis maps):
  StandardScaler on X and Y, sklearn Ridge(alpha=1.0, fit_intercept=True).
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_BIP = _HERE.parent / "bsf"
for p in (_HERE, _BIP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sae_affine_basis_mknn_gpu import encode as encode_sae  # noqa: E402
from sae_affine_basis_mknn_gpu import idf_np, load_sae  # noqa: E402
from pipeline_isomap_bsf_shared_mknn import encode_full as encode_bsf  # noqa: E402
from pipeline_isomap_bsf_shared_mknn import load_bsf_generic  # noqa: E402

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
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--k-extra", default="50")
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--n-shuffle", type=int, default=20)
    p.add_argument("--pca-ranks", default="32,64,128,256")
    p.add_argument("--phases", default="scores,shuffle,pca,analyze")
    p.add_argument("--out-dir", default="outputs/paper_alignment_controls")
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


def resolve_bsf_dir(root: Path, parquet_rel: str, col: str) -> Path | None:
    base = root / "outputs" / "bsf" / Path(parquet_rel).stem / col
    if not base.is_dir():
        return None
    cands = [
        p
        for p in base.iterdir()
        if p.is_dir() and ((p / "model.pt").is_file() or (p / "model.pth").is_file())
    ]
    if not cands:
        return None

    def score(p: Path) -> tuple:
        name = p.name
        k = 999
        if "_k" in name:
            try:
                k = int(name.split("_k")[1].split("_")[0])
            except Exception:
                k = 999
        return (abs(k - 21), k, name)

    return sorted(cands, key=score)[0]


def fit_ridge_map(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    alpha: float,
    y_perm: np.ndarray | None = None,
) -> np.ndarray:
    x_tr = x[train_idx]
    y_tr = y[train_idx] if y_perm is None else y_perm
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


def encode_pair_codes(
    root: Path,
    cfg: dict,
    X1: np.ndarray,
    X2: np.ndarray,
    device: torch.device,
    cache_dir: Path,
    pair: str,
) -> dict:
    cache = cache_dir / f"{pair}_codes.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    sae1 = resolve_sae_dir(root, cfg["parquet1"], cfg["col1"])
    sae2 = resolve_sae_dir(root, cfg["parquet2"], cfg["col2"])
    bsf1 = resolve_bsf_dir(root, cfg["parquet1"], cfg["col1"])
    bsf2 = resolve_bsf_dir(root, cfg["parquet2"], cfg["col2"])
    if sae1 is None or sae2 is None:
        raise FileNotFoundError(f"missing SAE for {pair}: {sae1} {sae2}")
    if bsf1 is None or bsf2 is None:
        raise FileNotFoundError(f"missing BSF for {pair}: {bsf1} {bsf2}")

    s1 = load_sae(sae1, device)
    s2 = load_sae(sae2, device)
    C1 = encode_sae(s1, X1, device)
    C2 = encode_sae(s2, X2, device)
    b1 = load_bsf_generic(bsf1, device)
    b2 = load_bsf_generic(bsf2, device)
    B1 = encode_bsf(b1, X1, device)
    B2 = encode_bsf(b2, X2, device)
    out = {
        "C1": C1,
        "C2": C2,
        "B1": B1,
        "B2": B2,
        "sae1_tag": np.asarray(sae1.name),
        "sae2_tag": np.asarray(sae2.name),
        "bsf1_tag": np.asarray(bsf1.name),
        "bsf2_tag": np.asarray(bsf2.name),
    }
    np.savez_compressed(cache, **out)
    return out


def scores_for_maps(
    X1: np.ndarray,
    X2: np.ndarray,
    codes: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    alpha: float,
    ks: list[int],
    device: torch.device,
    row_batch: int,
    y_perm_hsc_dense: np.ndarray | None = None,
    y_perm_hsc_sae: np.ndarray | None = None,
    y_perm_hsc_bsf: np.ndarray | None = None,
) -> dict[str, dict[int, float]]:
    mapped_dense = fit_ridge_map(X2, X1, train_idx, alpha=alpha, y_perm=y_perm_hsc_dense)
    mapped_sae = fit_ridge_map(
        codes["C2"], codes["C1"], train_idx, alpha=alpha, y_perm=y_perm_hsc_sae
    )
    mapped_bsf = fit_ridge_map(
        codes["B2"], codes["B1"], train_idx, alpha=alpha, y_perm=y_perm_hsc_bsf
    )
    idf = idf_np(codes["C1"][train_idx])
    te = test_idx
    reps = {
        "dense": (X1[te], X2[te]),
        "dense_ridge": (X1[te], mapped_dense[te]),
        "sae_ridge": (
            apply_idf(codes["C1"][te], idf),
            apply_idf(mapped_sae[te], idf),
        ),
        "bsf_ridge": (codes["B1"][te], mapped_bsf[te]),
    }
    out: dict[str, dict[int, float]] = {m: {} for m in reps}
    for method, (A, B) in reps.items():
        for k in ks:
            out[method][k] = mknn_pair(A, B, k, device, row_batch)
    return out


def freeze_baseline(root: Path, out_dir: Path) -> None:
    src = (
        root
        / "outputs/universetbd_shared_basis_mknn_ks/size_scaling/mknn_by_size.parquet"
    )
    dst_dir = out_dir / "frozen_baseline"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        pd.read_parquet(src).to_parquet(dst_dir / "mknn_by_size.parquet", index=False)
        (dst_dir / "source.txt").write_text(str(src) + "\n")


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


def analyze(out_dir: Path, k: int = 10) -> None:
    scores = pd.read_csv(out_dir / "test_only_gallery_scores.csv")
    kdf = scores[scores["k"] == k].copy()
    kdf["log10P"] = np.log10(kdf["approx_params_m"].to_numpy() * 1e6)
    kdf["A_ridge"] = kdf["dense_ridge"] - kdf["dense"]
    kdf["S_sae"] = kdf["sae_ridge"] - kdf["dense_ridge"]
    kdf["S_bsf"] = kdf["bsf_ridge"] - kdf["dense_ridge"]
    kdf["raw_sae"] = kdf["sae_ridge"] - kdf["dense"]
    kdf["raw_bsf"] = kdf["bsf_ridge"] - kdf["dense"]

    summary: dict = {
        "k": k,
        "n_rungs": int(len(kdf)),
        "chance": 10 / 3276,
        "mean_dense": float(kdf["dense"].mean()),
        "mean_dense_ridge": float(kdf["dense_ridge"].mean()),
        "mean_sae": float(kdf["sae_ridge"].mean()),
        "mean_bsf": float(kdf["bsf_ridge"].mean()),
        "mean_raw_sae_vs_dense": float(kdf["raw_sae"].mean()),
        "mean_raw_bsf_vs_dense": float(kdf["raw_bsf"].mean()),
        "mean_A_ridge": float(kdf["A_ridge"].mean()),
        "mean_S_sae": float(kdf["S_sae"].mean()),
        "mean_S_bsf": float(kdf["S_bsf"].mean()),
        "n_pos_S_sae": int((kdf["S_sae"] > 0).sum()),
        "n_pos_S_bsf": int((kdf["S_bsf"] > 0).sum()),
    }
    mu, lo, hi = clustered_mean_ci(kdf["S_sae"].to_numpy(), kdf["family"].to_numpy())
    summary["S_sae_family_boot"] = {"mean": mu, "ci95": [lo, hi]}
    mu, lo, hi = clustered_mean_ci(kdf["S_bsf"].to_numpy(), kdf["family"].to_numpy())
    summary["S_bsf_family_boot"] = {"mean": mu, "ci95": [lo, hi]}

    slope_rows = []
    for fam, g in kdf.groupby("family"):
        g = g.sort_values("approx_params_m")
        logp = np.log10(g["approx_params_m"].to_numpy() * 1e6)
        row = {"family": fam, "n_rungs": int(len(g)), "two_point_only": bool(len(g) == 2)}
        for col, key in (
            ("dense", "beta_dense"),
            ("dense_ridge", "beta_dense_ridge"),
            ("sae_ridge", "beta_sae"),
            ("bsf_ridge", "beta_bsf"),
        ):
            row[key] = family_ols_slope(logp, g[col].to_numpy())
        row["delta_beta_sae"] = row["beta_sae"] - row["beta_dense_ridge"]
        row["delta_beta_bsf"] = row["beta_bsf"] - row["beta_dense_ridge"]
        slope_rows.append(row)
    pd.DataFrame(slope_rows).to_csv(out_dir / "family_slopes.csv", index=False)

    d_rows = []
    for fam, g in kdf.groupby("family"):
        g = g.sort_values("size_rank")
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            d_rows.append(
                {
                    "family": fam,
                    "from": a["size_name"],
                    "to": b["size_name"],
                    "D_sae_vs_dense": (b["sae_ridge"] - a["sae_ridge"])
                    - (b["dense"] - a["dense"]),
                    "D_bsf_vs_dense": (b["bsf_ridge"] - a["bsf_ridge"])
                    - (b["dense"] - a["dense"]),
                    "D_sae_vs_dense_ridge": (b["sae_ridge"] - a["sae_ridge"])
                    - (b["dense_ridge"] - a["dense_ridge"]),
                    "D_bsf_vs_dense_ridge": (b["bsf_ridge"] - a["bsf_ridge"])
                    - (b["dense_ridge"] - a["dense_ridge"]),
                }
            )
    ddf = pd.DataFrame(d_rows)
    ddf.to_csv(out_dir / "controlled_interactions.csv", index=False)
    summary["mean_D_sae_vs_dense_ridge"] = float(ddf["D_sae_vs_dense_ridge"].mean())
    summary["mean_D_bsf_vs_dense_ridge"] = float(ddf["D_bsf_vs_dense_ridge"].mean())
    summary["n_pos_D_sae_vs_dense_ridge"] = int((ddf["D_sae_vs_dense_ridge"] > 0).sum())
    summary["n_pos_D_bsf_vs_dense_ridge"] = int((ddf["D_bsf_vs_dense_ridge"] > 0).sum())

    shuffle_path = out_dir / "refit_shuffle_summary.csv"
    if shuffle_path.is_file():
        sh = pd.read_csv(shuffle_path)
        shk = sh[sh["k"] == k]
        summary["shuffle"] = (
            shk.groupby("method")[["real", "null_mean", "p_emp"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

    pca_path = out_dir / "pca_ridge_scores.csv"
    if pca_path.is_file():
        pca = pd.read_csv(pca_path)
        pk = pca[pca["k"] == k]
        summary["pca"] = (
            pk.groupby("rank")["mknn"].mean().reset_index().to_dict(orient="records")
        )

    mean_s = 0.5 * (summary["mean_S_sae"] + summary["mean_S_bsf"])
    mean_a = summary["mean_A_ridge"]
    mean_raw = 0.5 * (
        summary["mean_raw_sae_vs_dense"] + summary["mean_raw_bsf_vs_dense"]
    )
    if abs(mean_s) < 0.25 * max(mean_raw, 1e-8) and mean_a > 0.5 * mean_raw:
        outcome = 1
        outcome_text = (
            "Dense+Ridge explains the lift; sparse representation attribution does not survive."
        )
    elif mean_s > 0 and mean_a > 0.15 * mean_raw:
        outcome = 2
        outcome_text = (
            "Dense+Ridge explains part of the lift, but SAE/BSF retain additional correspondence."
        )
    else:
        outcome = 3
        outcome_text = (
            "Dense+Ridge has little effect; the SAE/BSF lift is largely representation-specific."
        )
    dmean = 0.5 * (
        summary["mean_D_sae_vs_dense_ridge"] + summary["mean_D_bsf_vs_dense_ridge"]
    )
    if dmean > 5e-4:
        scaling = "positive"
    elif dmean < -5e-4:
        scaling = "negative"
    else:
        scaling = "heterogeneous / near zero"
    summary["outcome"] = outcome
    summary["outcome_text"] = outcome_text
    summary["scaling_after_dense_ridge"] = scaling
    (out_dir / "alignment_control_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    methods = ["dense", "dense_ridge", "sae_ridge", "bsf_ridge"]
    colors = {
        "dense": "#444444",
        "dense_ridge": "#1f77b4",
        "sae_ridge": "#d62728",
        "bsf_ridge": "#2ca02c",
    }
    labels = {
        "dense": "Dense",
        "dense_ridge": "Dense+Ridge",
        "sae_ridge": "SAE+Ridge",
        "bsf_ridge": "BSF+Ridge",
    }
    fams = sorted(kdf["family"].unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(2.4 * len(fams), 3.2), sharey=True)
    if len(fams) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams):
        g = kdf[kdf["family"] == fam].sort_values("log10P")
        for m in methods:
            ax.plot(
                g["log10P"], g[m], marker="o", color=colors[m], label=labels[m], lw=1.4
            )
        ax.set_title(fam)
        ax.set_xlabel(r"$\log_{10} P$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mKNN@10 (test-only gallery)")
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "absolute_alignment.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="0.5", lw=1)
    for fam, g in kdf.groupby("family"):
        g = g.sort_values("log10P")
        ax.plot(g["log10P"], g["S_sae"], marker="o", label=f"{fam} SAE")
        ax.plot(g["log10P"], g["S_bsf"], marker="s", ls="--", label=f"{fam} BSF")
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(r"$S = M_R - M_{\mathrm{dense+Ridge}}$")
    ax.set_title("Residual structured lift over Dense+Ridge")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_representation_lift.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Alignment-control results (test-only gallery)",
        "",
        f"k={k}, n_rungs={len(kdf)}, chance ≈ {summary['chance']:.4f}",
        "",
        "| Family | Size | Dense | Dense+Ridge | SAE | BSF | SAE−DenseRidge | BSF−DenseRidge |",
        "| ------ | ---: | ----: | ----------: | --: | --: | -------------: | -------------: |",
    ]
    for r in kdf.sort_values(["family", "size_rank"]).itertuples():
        lines.append(
            f"| {r.family} | {r.size_name} | {r.dense:.4f} | {r.dense_ridge:.4f} | "
            f"{r.sae_ridge:.4f} | {r.bsf_ridge:.4f} | {r.S_sae:+.4f} | {r.S_bsf:+.4f} |"
        )
    lines += [
        "",
        f"Mean raw SAE lift over Dense: {summary['mean_raw_sae_vs_dense']:+.4f}",
        f"Mean raw BSF lift over Dense: {summary['mean_raw_bsf_vs_dense']:+.4f}",
        f"Mean Dense+Ridge lift over Dense: {summary['mean_A_ridge']:+.4f}",
        f"Mean SAE residual over Dense+Ridge: {summary['mean_S_sae']:+.4f} "
        f"(family bootstrap 95% CI {summary['S_sae_family_boot']['ci95']})",
        f"Mean BSF residual over Dense+Ridge: {summary['mean_S_bsf']:+.4f} "
        f"(family bootstrap 95% CI {summary['S_bsf_family_boot']['ci95']})",
        f"Positive SAE residual rungs: {summary['n_pos_S_sae']}/{len(kdf)}",
        f"Positive BSF residual rungs: {summary['n_pos_S_bsf']}/{len(kdf)}",
        "",
        f"**Outcome {outcome}.** {outcome_text}",
        f"**Scaling after Dense+Ridge:** {scaling}.",
        "",
    ]
    (out_dir / "alignment_control_results.md").write_text("\n".join(lines))
    (out_dir / "alignment_control_audit.md").write_text(
        "\n".join(
            [
                "# Alignment-control audit",
                "",
                "Gallery: test IDs only. Mapping train IDs excluded from neighbours.",
                "Direction: Legacy→HSC. Ridge α=1, intercept=True, StandardScaler on X and Y.",
                "",
                "## Q1. Does supervised Dense+Ridge account for the SAE/BSF lift?",
                f"Dense+Ridge mean lift over dense is {summary['mean_A_ridge']:+.4f}, "
                f"vs raw SAE/BSF lifts {summary['mean_raw_sae_vs_dense']:+.4f} / "
                f"{summary['mean_raw_bsf_vs_dense']:+.4f}.",
                "",
                "## Q2. Do SAE/BSF retain positive lift over an equally supervised dense alignment?",
                f"Residual S_SAE={summary['mean_S_sae']:+.4f} "
                f"({summary['n_pos_S_sae']}/{len(kdf)} rungs); "
                f"S_BSF={summary['mean_S_bsf']:+.4f} "
                f"({summary['n_pos_S_bsf']}/{len(kdf)} rungs).",
                "",
                "## Q3. Does the result survive a gallery containing only unseen mapping-test objects?",
                "These numbers *are* that gallery. Chance is 10/3276. Do not mix with full-gallery scores.",
                "",
                "## Q4. Does correct Legacy↔HSC correspondence matter to the fitted map?",
                json.dumps(summary.get("shuffle", []), indent=2),
                "",
                "## Q5. Is there a consistent positive representation×scale interaction after Dense+Ridge?",
                f"Mean D_SAE vs Dense+Ridge = {summary['mean_D_sae_vs_dense_ridge']:+.4g} "
                f"({summary['n_pos_D_sae_vs_dense_ridge']}/{len(ddf)} positive). "
                f"Mean D_BSF vs Dense+Ridge = {summary['mean_D_bsf_vs_dense_ridge']:+.4g} "
                f"({summary['n_pos_D_bsf_vs_dense_ridge']}/{len(ddf)} positive). "
                "Family-slope Δβ in family_slopes.csv. I-JEPA is two-point only.",
                "",
                "## Q6. Does a generic PCA/Ridge bottleneck explain the residual lift?",
                json.dumps(summary.get("pca", []), indent=2),
                "",
                "## Decision",
                f"Outcome {outcome}: {outcome_text}",
                f"Scaling: {scaling}",
                "",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    root = platonic_root(args.platonic_root)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "code_cache"
    cache_dir.mkdir(exist_ok=True)
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ks = [args.k] + [
        int(x) for x in args.k_extra.split(",") if x.strip() and int(x) != args.k
    ]
    pca_ranks = [int(x) for x in args.pca_ranks.split(",") if x.strip()]

    pairs_path = resolve_path(root, args.pairs_yaml)
    if not pairs_path.is_file():
        alt = Path(__file__).resolve().parents[3] / args.pairs_yaml
        if alt.is_file():
            pairs_path = alt
    pairs = yaml.safe_load(pairs_path.read_text())
    names = list(pairs)
    if args.pairs:
        names = [n.strip() for n in args.pairs.split(",") if n.strip()]

    freeze_baseline(root, out_dir)
    (out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "n": args.max_n,
                "test_size": args.test_size,
                "seed": args.seed,
                "alpha": args.alpha,
                "direction": "Legacy→HSC (col2→col1)",
                "ridge": "StandardScaler(X,Y) + Ridge(alpha=1, fit_intercept=True)",
                "gallery": "test_ids_only",
                "sae": "shared_side1_basis_idf",
                "bsf": "shared_side1_basis_cosine",
            },
            indent=2,
        )
    )

    if "scores" in phases:
        score_rows = []
        for name in names:
            cfg = pairs[name]
            t0 = time.time()
            X1, X2, n_full = load_pair_arrays(root, cfg, args.max_n, args.seed)
            idx = np.arange(len(X1))
            train_idx, test_idx = train_test_split(
                idx, test_size=args.test_size, random_state=args.seed, shuffle=True
            )
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)
            codes = encode_pair_codes(root, cfg, X1, X2, device, cache_dir, name)
            scores = scores_for_maps(
                X1,
                X2,
                codes,
                train_idx,
                test_idx,
                alpha=args.alpha,
                ks=ks,
                device=device,
                row_batch=args.row_batch,
            )
            for k in ks:
                score_rows.append(
                    {
                        "pair": name,
                        "family": cfg.get("family"),
                        "size_name": cfg.get("size_name"),
                        "size_rank": cfg.get("size_rank"),
                        "approx_params_m": cfg.get("approx_params_m"),
                        "n": len(X1),
                        "n_full": n_full,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "k": int(k),
                        "dense": scores["dense"][k],
                        "dense_ridge": scores["dense_ridge"][k],
                        "sae_ridge": scores["sae_ridge"][k],
                        "bsf_ridge": scores["bsf_ridge"][k],
                        "dense_ridge_minus_dense": scores["dense_ridge"][k]
                        - scores["dense"][k],
                        "sae_minus_dense_ridge": scores["sae_ridge"][k]
                        - scores["dense_ridge"][k],
                        "bsf_minus_dense_ridge": scores["bsf_ridge"][k]
                        - scores["dense_ridge"][k],
                        "sae1_tag": str(codes["sae1_tag"]),
                        "bsf1_tag": str(codes["bsf1_tag"]),
                        "sec": time.time() - t0,
                    }
                )
            print(
                f"[{name}] k={args.k} dense={scores['dense'][args.k]:.4f} "
                f"denseR={scores['dense_ridge'][args.k]:.4f} "
                f"sae={scores['sae_ridge'][args.k]:.4f} "
                f"bsf={scores['bsf_ridge'][args.k]:.4f} "
                f"n_train={len(train_idx)} n_test={len(test_idx)}",
                flush=True,
            )
        pd.DataFrame(score_rows).to_csv(
            out_dir / "test_only_gallery_scores.csv", index=False
        )

    if "shuffle" in phases:
        raw_rows = []
        sum_rows = []
        scores_df = pd.read_csv(out_dir / "test_only_gallery_scores.csv")
        for name in names:
            cfg = pairs[name]
            X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
            idx = np.arange(len(X1))
            train_idx, test_idx = train_test_split(
                idx, test_size=args.test_size, random_state=args.seed, shuffle=True
            )
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)
            codes = encode_pair_codes(root, cfg, X1, X2, device, cache_dir, name)
            real = scores_df[(scores_df.pair == name) & (scores_df.k == args.k)].iloc[0]
            real_map = {
                "dense_ridge": float(real.dense_ridge),
                "sae_ridge": float(real.sae_ridge),
                "bsf_ridge": float(real.bsf_ridge),
            }
            nulls = {m: [] for m in real_map}
            for b in range(args.n_shuffle):
                rng = np.random.default_rng(b)
                perm = rng.permutation(len(train_idx))
                sc = scores_for_maps(
                    X1,
                    X2,
                    codes,
                    train_idx,
                    test_idx,
                    alpha=args.alpha,
                    ks=[args.k],
                    device=device,
                    row_batch=args.row_batch,
                    y_perm_hsc_dense=X1[train_idx][perm],
                    y_perm_hsc_sae=codes["C1"][train_idx][perm],
                    y_perm_hsc_bsf=codes["B1"][train_idx][perm],
                )
                for method in real_map:
                    val = sc[method][args.k]
                    nulls[method].append(val)
                    raw_rows.append(
                        {
                            "pair": name,
                            "family": cfg.get("family"),
                            "method": method,
                            "k": args.k,
                            "b": b,
                            "mknn": val,
                            "dense": float(real.dense),
                            "lift_vs_dense": val - float(real.dense),
                        }
                    )
                print(f"[{name}] shuffle {b + 1}/{args.n_shuffle}", flush=True)
            B = args.n_shuffle
            for method, arr in nulls.items():
                a = np.asarray(arr, float)
                real_v = real_map[method]
                p = (1 + int(np.sum(a >= real_v))) / (B + 1)
                sum_rows.append(
                    {
                        "pair": name,
                        "family": cfg.get("family"),
                        "method": method,
                        "k": args.k,
                        "real": real_v,
                        "null_mean": float(a.mean()),
                        "null_sd": float(a.std(ddof=1) if len(a) > 1 else 0.0),
                        "null_p025": float(np.quantile(a, 0.025)),
                        "null_p50": float(np.quantile(a, 0.50)),
                        "null_p975": float(np.quantile(a, 0.975)),
                        "p_emp": p,
                        "B": B,
                    }
                )
        pd.DataFrame(raw_rows).to_csv(out_dir / "refit_shuffle_raw.csv", index=False)
        pd.DataFrame(sum_rows).to_csv(out_dir / "refit_shuffle_summary.csv", index=False)

    if "pca" in phases:
        pca_rows = []
        for name in names:
            cfg = pairs[name]
            X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
            idx = np.arange(len(X1))
            train_idx, test_idx = train_test_split(
                idx, test_size=args.test_size, random_state=args.seed, shuffle=True
            )
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)
            inner_tr, inner_va = train_test_split(
                train_idx, test_size=0.2, random_state=args.seed, shuffle=True
            )
            best_rank, best_va = None, -1.0
            for r in pca_ranks:
                pca_x = PCA(n_components=min(r, inner_tr.size - 1)).fit(X2[inner_tr])
                pca_y = PCA(n_components=min(r, inner_tr.size - 1)).fit(X1[inner_tr])
                Z2 = pca_x.transform(X2)
                Z1 = pca_y.transform(X1)
                mapped_va = fit_ridge_map(Z2, Z1, inner_tr, alpha=args.alpha)
                va = mknn_pair(
                    Z1[inner_va], mapped_va[inner_va], args.k, device, args.row_batch
                )
                if va > best_va:
                    best_va, best_rank = va, r
                mapped_all = fit_ridge_map(Z2, Z1, train_idx, alpha=args.alpha)
                te = mknn_pair(
                    Z1[test_idx], mapped_all[test_idx], args.k, device, args.row_batch
                )
                pca_rows.append(
                    {
                        "pair": name,
                        "family": cfg.get("family"),
                        "size_name": cfg.get("size_name"),
                        "rank": r,
                        "k": args.k,
                        "mknn": te,
                        "val_mknn": va,
                        "selected": False,
                    }
                )
            for row in pca_rows:
                if row["pair"] == name:
                    row["selected"] = row["rank"] == best_rank
            print(f"[{name}] PCA selected rank={best_rank} val={best_va:.4f}", flush=True)
        pd.DataFrame(pca_rows).to_csv(out_dir / "pca_ridge_scores.csv", index=False)

    if "analyze" in phases:
        analyze(out_dir, k=args.k)
        print("wrote", out_dir / "alignment_control_audit.md", flush=True)


if __name__ == "__main__":
    main()
