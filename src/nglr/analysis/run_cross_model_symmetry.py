#!/usr/bin/env python3
"""Cross-model same-dataset (Legacy) symmetry experiment.

Hold the observation channel fixed; test mutual linear recoverability and
directional asymmetry across independently trained model representations.

Protocol: PCA256 (train-only per model), Ridge(alpha=1), mKNN@10, test-only
gallery, cross-family unordered pairs only.
"""
from __future__ import annotations

import argparse
import itertools
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WS_ROOT = Path(__file__).resolve().parents[3]
FAMILY_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
PCA_RANK = 256
K_EVAL = 10
EPS_REL = 1e-8

SCALE_BANDS = {
    "small": (15, 30),
    "medium": (80, 100),
    "large": (200, 350),
    "xlarge": (600, 1100),
}


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def load_col(path: Path, col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[col])
    return np.vstack(table.column(0).to_pylist()).astype(np.float32)


def load_legacy_col(
    root: Path, cfg: dict, max_n: int, seed: int
) -> np.ndarray:
    path = resolve_path(root, cfg["parquet2"])
    col = cfg["col2"]
    X = load_col(path, col)
    n_full = len(X)
    n_cap = int(cfg.get("default_max_n", 0) or 0)
    n_use = max_n
    if n_cap > 0:
        n_use = min(n_use, n_cap) if n_use > 0 else n_cap
    rng = np.random.default_rng(seed)
    if n_use and n_full > n_use:
        sel = np.sort(rng.choice(n_full, size=n_use, replace=False))
        X = X[sel]
    return X


def scale_band(params_m: float) -> str | None:
    for name, (lo, hi) in SCALE_BANDS.items():
        if lo <= params_m <= hi:
            return name
    return None


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


def mknn_from_nn(nn1: np.ndarray, nn2: np.ndarray, k: int) -> float:
    a, b = nn1[:, :k], nn2[:, :k]
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


@torch.inference_mode()
def mknn_pair(
    A: np.ndarray, B: np.ndarray, k: int, device: torch.device, row_batch: int
) -> float:
    ta = torch.as_tensor(np.ascontiguousarray(A), device=device)
    tb = torch.as_tensor(np.ascontiguousarray(B), device=device)
    nn1 = knn_cos(ta, k, row_batch).cpu().numpy()
    nn2 = knn_cos(tb, k, row_batch).cpu().numpy()
    return mknn_from_nn(nn1, nn2, k)


def fit_ridge_map(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> np.ndarray:
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    return y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)


def fit_pca256(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray | None:
    d = X.shape[1]
    if d < PCA_RANK:
        return None
    n_comp = min(PCA_RANK, len(train_idx) - 1, d)
    if n_comp < PCA_RANK:
        return None
    pca = PCA(n_components=PCA_RANK).fit(X[train_idx])
    return pca.transform(X).astype(np.float32)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [
        float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ols_multi(
    y: np.ndarray, s_min: np.ndarray, s_gap: np.ndarray
) -> dict[str, float]:
    X = np.column_stack([np.ones(len(y)), s_min, s_gap])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "alpha": float(beta[0]),
        "beta_min": float(beta[1]),
        "beta_gap": float(beta[2]),
        "se_beta_min": float(se[1]),
        "se_beta_gap": float(se[2]),
        "n": int(n),
        "r2": float(1 - resid @ resid / max(((y - y.mean()) ** 2).sum(), 1e-12)),
    }


def model_sort_key(m: dict) -> tuple:
    fam = m["family"]
    fam_i = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99
    return (fam_i, m["log10_params"], m["short_name"])


def plot_heatmap(
    models: list[dict],
    mat: np.ndarray,
    title: str,
    path: Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    labels = [m["short_name"] for m in models]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary_scatter(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        ("native_mknn", "Native mKNN@10"),
        ("gain_symmetric", r"Symmetric gain $G_{\rm sym}$"),
        ("asymmetry_abs", r"Abs. asymmetry $A_{\rm dir}$"),
    ]
    for ax, (col, ylab) in zip(axes, panels):
        sc = ax.scatter(
            df["min_log_params"],
            df[col],
            c=df["log_size_gap"],
            s=18 + 40 * (df["log_size_gap"] / df["log_size_gap"].max().clip(0.1)),
            alpha=0.75,
            cmap="coolwarm",
            edgecolors="none",
        )
        ax.set_xlabel(r"$s_{\min}=\min(\log_{10}P_A,\log_{10}P_B)$")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes, label=r"$s_{\rm gap}$", fraction=0.02, pad=0.02)
    fig.suptitle("Cross-model Legacy symmetry (PCA256, k=10)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    band_df: pd.DataFrame,
    reg: dict,
    dir_stats: dict,
    path: Path,
    meta: dict,
) -> None:
    n = len(df)
    mean_native = df["native_mknn"].mean()
    mean_g_ab = df["gain_A_to_B"].mean()
    mean_g_ba = df["gain_B_to_A"].mean()
    mean_gsym = df["gain_symmetric"].mean()
    mean_adir = df["asymmetry_abs"].mean()
    mean_arel = df["asymmetry_rel"].mean()

    # Verdict logic
    bmin_native = reg["native"]["beta_min"]
    bmin_gsym = reg["gain_symmetric"]["beta_min"]
    bmin_adir = reg["asymmetry_abs"]["beta_min"]
    bgap_adir = reg["asymmetry_abs"]["beta_gap"]

    if bmin_native > 0 and bmin_gsym > 0 and bmin_adir < 0:
        verdict = "A"
        verdict_text = (
            "Larger same-dataset cross-family models become increasingly "
            "mutually and symmetrically linearly recoverable."
        )
    elif bmin_gsym > 0 and bmin_adir >= -0.001:
        verdict = "B"
        verdict_text = (
            "Mutual recoverability increases with scale, but directional "
            "asymmetry persists."
        )
    elif bmin_native > 0 and bmin_gsym <= 0:
        verdict = "C"
        verdict_text = (
            "Native correspondence increases with scale while the additional "
            "value of Ridge decreases."
        )
    elif abs(bmin_native) < 0.002 and abs(bmin_gsym) < 0.002:
        verdict = "D"
        verdict_text = (
            "Cross-model recoverability shows no clean scale dependence."
        )
    else:
        verdict = "E"
        verdict_text = (
            f"Mixed pattern: native β_min={bmin_native:.4f}, "
            f"G_sym β_min={bmin_gsym:.4f}, A_dir β_min={bmin_adir:.4f}."
        )

    lines = [
        "# Cross-model same-dataset symmetry (Legacy only)",
        "",
        "Exploratory first pass: PCA256 (train-only per model), Ridge(α=1), "
        "mKNN@10 test-only gallery, cross-family pairs only.",
        "",
        f"- **Pairs scored:** {n}",
        f"- **Models:** {meta['n_models']}",
        f"- **Runtime:** {meta['elapsed_sec']:.1f}s on {meta['device']}",
        "",
        "## Q1 — Native cross-model mKNN",
        "",
        f"Mean native mKNN@10 over {n} cross-family pairs: **{mean_native:.4f}** "
        f"(median {df['native_mknn'].median():.4f}; "
        f"range [{df['native_mknn'].min():.4f}, {df['native_mknn'].max():.4f}]).",
        "",
        "Chance level (test gallery, k=10): "
        f"**{meta['chance_mknn']:.5f}**.",
        "",
        "## Q2 — Does Ridge increase correspondence in both directions?",
        "",
        f"Mean ridge mKNN A→B: **{df['ridge_A_to_B_mknn'].mean():.4f}**; "
        f"B→A: **{df['ridge_B_to_A_mknn'].mean():.4f}**.",
        "",
        f"Pairs with ridge gain > 0 in A→B: "
        f"**{(df['gain_A_to_B'] > 0).sum()}/{n}**; "
        f"B→A: **{(df['gain_B_to_A'] > 0).sum()}/{n}**.",
        "",
        "## Q3 — Directional gain symmetry",
        "",
        f"| Statistic | Mean | Median |",
        f"|-----------|------|--------|",
        f"| Gain A→B | {mean_g_ab:.4f} | {df['gain_A_to_B'].median():.4f} |",
        f"| Gain B→A | {mean_g_ba:.4f} | {df['gain_B_to_A'].median():.4f} |",
        f"| $G_{{\\rm sym}}$ | {mean_gsym:.4f} | {df['gain_symmetric'].median():.4f} |",
        f"| $A_{{\\rm dir}}$ | {mean_adir:.4f} | {df['asymmetry_abs'].median():.4f} |",
        f"| $A_{{\\rm rel}}$ | {mean_arel:.4f} | {df['asymmetry_rel'].median():.4f} |",
        "",
        "## Q4 — Does $G_{\\rm sym}$ increase with scale?",
        "",
        "### Approximate matched-scale bands (both models in band)",
        "",
        band_df.to_markdown(index=False),
        "",
        "### All-pair OLS ($s_{\\min}$, $s_{\\rm gap}$)",
        "",
        f"- Native: β_min={reg['native']['beta_min']:.5f} "
        f"(SE {reg['native']['se_beta_min']:.5f}), "
        f"β_gap={reg['native']['beta_gap']:.5f}",
        f"- $G_{{\\rm sym}}$: β_min={reg['gain_symmetric']['beta_min']:.5f} "
        f"(SE {reg['gain_symmetric']['se_beta_min']:.5f}), "
        f"β_gap={reg['gain_symmetric']['beta_gap']:.5f}",
        "",
        "## Q5 — Does directional asymmetry decrease with scale?",
        "",
        f"- $A_{{\\rm dir}}$: β_min={bmin_adir:.5f} "
        f"(SE {reg['asymmetry_abs']['se_beta_min']:.5f}), "
        f"β_gap={bgap_adir:.5f} "
        f"(SE {reg['asymmetry_abs']['se_beta_gap']:.5f})",
        "",
        "## Q6 — Larger→smaller vs smaller→larger",
        "",
        f"- Mean gain **large→small**: {dir_stats['mean_large_to_small']:.4f} "
        f"({dir_stats['n_large_to_small']} ordered directions)",
        f"- Mean gain **small→large**: {dir_stats['mean_small_to_large']:.4f} "
        f"({dir_stats['n_small_to_large']} ordered directions)",
        f"- Difference (large→small minus small→large): "
        f"**{dir_stats['diff']:.4f}**",
        "",
        "Within matched-scale bands (same band, ordered by relative size):",
        "",
        dir_stats["band_table_md"],
        "",
        "## Q7 — Comparison with cross-survey asymmetry",
        "",
        "Cross-survey Legacy↔HSC showed **directional asymmetry** "
        "(forward $T_{L\\to H}$ stronger than reverse $T_{H\\to L}$, "
        "family-heterogeneous). Same-dataset cross-model removes survey "
        "information differences; any remaining asymmetry reflects "
        "representation / training differences only.",
        "",
        f"Here mean $|G_{{A\\to B}}-G_{{B\\to A}}|$ = {mean_adir:.4f} vs mean "
        f"$G_{{\\rm sym}}$ = {mean_gsym:.4f} "
        f"(ratio {mean_adir / max(mean_gsym, 1e-6):.2f}). "
        "Do not compare raw mKNN magnitudes to cross-survey numbers "
        "(different task: cross-model vs cross-survey).",
        "",
        "## Q8 — Verdict",
        "",
        f"### {verdict}",
        "",
        f"> {verdict_text}",
        "",
        "## Outputs",
        "",
        "- `outputs/cross_model_symmetry/cross_model_pair_scores.csv`",
        "- `figures/cross_model_*_heatmap.png`",
        "- `figures/cross_model_symmetry_summary.png`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "cross_model_symmetry"
    )
    fig_dir = (
        args.fig_dir.expanduser().resolve()
        if args.fig_dir
        else WS_ROOT / "figures"
    )
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else WS_ROOT / "outputs/cross_model_symmetry/cross_model_symmetry_report.md"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "configs/official_legacy_pairs.yaml",
            WS_ROOT
            / "configs/official_legacy_pairs.yaml",
        ]
        pairs_path = next(p for p in cands if p.is_file())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = yaml.safe_load(pairs_path.read_text())
    names = sorted(
        n
        for n, cfg in pairs.items()
        if isinstance(cfg, dict) and "legacysurvey" in str(cfg.get("col2", ""))
    )
    print(f"models={len(names)} device={device}", flush=True)

    models: list[dict] = []
    embeddings: dict[str, np.ndarray] = {}
    n_ref = None
    for name in names:
        cfg = pairs[name]
        X = load_legacy_col(root, cfg, args.max_n, args.seed)
        if n_ref is None:
            n_ref = len(X)
        assert len(X) == n_ref, f"row mismatch {name}: {len(X)} vs {n_ref}"
        params_m = float(cfg.get("approx_params_m", 1))
        fam = str(cfg["family"]).lower()
        short = name.replace("official_legacy_cross_", "")
        models.append(
            {
                "pair_key": name,
                "family": fam,
                "short_name": short,
                "paper_model": cfg.get("paper_model", short),
                "size_name": cfg.get("size_name", ""),
                "params_m": params_m,
                "parameter_count": int(round(params_m * 1e6)),
                "log10_params": math.log10(max(params_m * 1e6, 1.0)),
                "native_dim": int(X.shape[1]),
                "scale_band": scale_band(params_m),
            }
        )
        embeddings[name] = X
        print(f"  {short}: d={X.shape[1]} P={params_m}M band={scale_band(params_m)}", flush=True)

    models.sort(key=model_sort_key)
    n = int(n_ref)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"n={n} train={len(train_idx)} test={len(test_idx)}", flush=True)

    t0 = time.time()
    pca_emb: dict[str, np.ndarray] = {}
    invalid_pca: set[str] = set()
    for m in models:
        key = m["pair_key"]
        Z = fit_pca256(embeddings[key], train_idx)
        if Z is None:
            invalid_pca.add(key)
            print(f"  INVALID PCA256: {m['short_name']} (d={m['native_dim']})", flush=True)
        else:
            pca_emb[key] = Z

    pair_rows = []
    n_pairs = 0
    for ma, mb in itertools.combinations(models, 2):
        if ma["family"] == mb["family"]:
            continue
        if ma["pair_key"] in invalid_pca or mb["pair_key"] in invalid_pca:
            continue
        Za = pca_emb[ma["pair_key"]]
        Zb = pca_emb[mb["pair_key"]]
        Zate, Zbte = Za[test_idx], Zb[test_idx]

        native = mknn_pair(Zate, Zbte, K_EVAL, device, args.row_batch)
        mapped_ab = fit_ridge_map(Za, Zb, train_idx, alpha=args.alpha)
        mapped_ba = fit_ridge_map(Zb, Za, train_idx, alpha=args.alpha)
        ridge_ab = mknn_pair(mapped_ab[test_idx], Zbte, K_EVAL, device, args.row_batch)
        ridge_ba = mknn_pair(mapped_ba[test_idx], Zate, K_EVAL, device, args.row_batch)

        g_ab = ridge_ab - native
        g_ba = ridge_ba - native
        g_sym = 0.5 * (g_ab + g_ba)
        a_signed = g_ab - g_ba
        a_abs = abs(a_signed)
        denom = abs(g_ab) + abs(g_ba) + EPS_REL
        a_rel = a_abs / denom

        log_a, log_b = ma["log10_params"], mb["log10_params"]
        pair_rows.append(
            {
                "family_A": ma["family"],
                "model_A": ma["short_name"],
                "params_A": ma["parameter_count"],
                "family_B": mb["family"],
                "model_B": mb["short_name"],
                "params_B": mb["parameter_count"],
                "native_mknn": native,
                "ridge_A_to_B_mknn": ridge_ab,
                "ridge_B_to_A_mknn": ridge_ba,
                "gain_A_to_B": g_ab,
                "gain_B_to_A": g_ba,
                "gain_symmetric": g_sym,
                "asymmetry_signed": a_signed,
                "asymmetry_abs": a_abs,
                "asymmetry_rel": a_rel,
                "log_params_A": log_a,
                "log_params_B": log_b,
                "min_log_params": min(log_a, log_b),
                "max_log_params": max(log_a, log_b),
                "mean_log_params": 0.5 * (log_a + log_b),
                "log_size_gap": abs(log_a - log_b),
            }
        )
        n_pairs += 1
        if n_pairs % 20 == 0:
            print(f"  scored {n_pairs} pairs...", flush=True)

    df = pd.DataFrame(pair_rows)
    df.to_csv(out_dir / "cross_model_pair_scores.csv", index=False)
    print(f"scored {len(df)} cross-family pairs", flush=True)

    # Heatmaps
    n_m = len([m for m in models if m["pair_key"] not in invalid_pca])
    valid_models = [m for m in models if m["pair_key"] not in invalid_pca]
    key_to_i = {m["pair_key"]: i for i, m in enumerate(valid_models)}
    short_to_key = {m["short_name"]: m["pair_key"] for m in valid_models}

    def fill_matrix(col: str) -> np.ndarray:
        mat = np.full((n_m, n_m), np.nan)
        np.fill_diagonal(mat, np.nan)
        for _, r in df.iterrows():
            ka = short_to_key[r["model_A"]]
            kb = short_to_key[r["model_B"]]
            i, j = key_to_i[ka], key_to_i[kb]
            mat[i, j] = mat[j, i] = r[col]
        return mat

    plot_heatmap(
        valid_models,
        fill_matrix("native_mknn"),
        "Native cross-model mKNN@10 (Legacy, PCA256)",
        fig_dir / "cross_model_native_heatmap.png",
    )
    plot_heatmap(
        valid_models,
        fill_matrix("gain_symmetric"),
        r"Symmetric alignment gain $G_{\rm sym}$",
        fig_dir / "cross_model_symmetric_gain_heatmap.png",
        cmap="RdYlGn",
    )
    plot_heatmap(
        valid_models,
        fill_matrix("asymmetry_abs"),
        r"Absolute directional asymmetry $A_{\rm dir}$",
        fig_dir / "cross_model_asymmetry_heatmap.png",
        cmap="magma",
    )
    plot_summary_scatter(df, fig_dir / "cross_model_symmetry_summary.png")

    # Matched-scale bands
    band_rows = []
    for band_name in SCALE_BANDS:
        sub = df[
            df.apply(
                lambda r: (
                    scale_band(r["params_A"] / 1e6) == band_name
                    and scale_band(r["params_B"] / 1e6) == band_name
                ),
                axis=1,
            )
        ]
        if len(sub) == 0:
            continue
        for col, label in [
            ("native_mknn", "native"),
            ("gain_symmetric", "G_sym"),
            ("asymmetry_abs", "A_dir"),
        ]:
            vals = sub[col].to_numpy(float)
            lo, hi = bootstrap_ci(vals)
            band_rows.append(
                {
                    "band": band_name,
                    "metric": label,
                    "n_pairs": len(sub),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(out_dir / "matched_scale_summary.csv", index=False)

    # All-pair regression
    s_min = df["min_log_params"].to_numpy(float)
    s_gap = df["log_size_gap"].to_numpy(float)
    reg = {
        "native": ols_multi(df["native_mknn"].to_numpy(float), s_min, s_gap),
        "gain_symmetric": ols_multi(df["gain_symmetric"].to_numpy(float), s_min, s_gap),
        "asymmetry_abs": ols_multi(df["asymmetry_abs"].to_numpy(float), s_min, s_gap),
    }
    (out_dir / "all_pair_regression.json").write_text(json.dumps(reg, indent=2) + "\n")

    # Direction by relative size
    l2s_gains, s2l_gains = [], []
    for _, r in df.iterrows():
        if r["params_A"] > r["params_B"]:
            l2s_gains.append(r["gain_A_to_B"])
            s2l_gains.append(r["gain_B_to_A"])
        elif r["params_B"] > r["params_A"]:
            l2s_gains.append(r["gain_B_to_A"])
            s2l_gains.append(r["gain_A_to_B"])
    l2s = np.array(l2s_gains, float)
    s2l = np.array(s2l_gains, float)

    band_dir_rows = []
    for band_name in SCALE_BANDS:
        sub_l2s, sub_s2l = [], []
        for _, r in df.iterrows():
            pa, pb = r["params_A"] / 1e6, r["params_B"] / 1e6
            if scale_band(pa) != band_name or scale_band(pb) != band_name:
                continue
            if r["params_A"] > r["params_B"]:
                sub_l2s.append(r["gain_A_to_B"])
                sub_s2l.append(r["gain_B_to_A"])
            elif r["params_B"] > r["params_A"]:
                sub_l2s.append(r["gain_B_to_A"])
                sub_s2l.append(r["gain_A_to_B"])
        if sub_l2s:
            band_dir_rows.append(
                {
                    "band": band_name,
                    "mean_large_to_small": float(np.mean(sub_l2s)),
                    "mean_small_to_large": float(np.mean(sub_s2l)),
                    "diff": float(np.mean(sub_l2s) - np.mean(sub_s2l)),
                    "n_directions": len(sub_l2s),
                }
            )
    band_dir_df = pd.DataFrame(band_dir_rows)
    band_dir_df.to_csv(out_dir / "direction_by_size_band.csv", index=False)

    dir_stats = {
        "mean_large_to_small": float(l2s.mean()),
        "mean_small_to_large": float(s2l.mean()),
        "diff": float(l2s.mean() - s2l.mean()),
        "n_large_to_small": len(l2s),
        "n_small_to_large": len(s2l),
        "band_table_md": band_dir_df.to_markdown(index=False)
        if len(band_dir_df)
        else "_No matched-scale bands with ordered directions._",
    }

    meta = {
        "n_models": len(valid_models),
        "n_pairs": len(df),
        "invalid_pca": list(invalid_pca),
        "seed": args.seed,
        "test_size": args.test_size,
        "n": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "ridge_alpha": args.alpha,
        "pca_rank": PCA_RANK,
        "k": K_EVAL,
        "chance_mknn": K_EVAL / (len(test_idx) - 1),
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "pairs_yaml": str(pairs_path),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    write_report(df, band_df, reg, dir_stats, report_path, meta)

    print("\n=== SUMMARY ===", flush=True)
    print(f"pairs={len(df)} native_mean={df['native_mknn'].mean():.4f}", flush=True)
    print(
        f"G_sym mean={df['gain_symmetric'].mean():.4f} "
        f"A_dir mean={df['asymmetry_abs'].mean():.4f}",
        flush=True,
    )
    print(
        f"large->small={dir_stats['mean_large_to_small']:.4f} "
        f"small->large={dir_stats['mean_small_to_large']:.4f}",
        flush=True,
    )
    print(f"report={report_path}", flush=True)
    print(f"done in {meta['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
