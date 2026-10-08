#!/usr/bin/env python3
"""Ablate Ridge translation: affine vs linear-only vs centered.

Frozen protocol matching run_ridge_scaling_geometry.py:
  n=16384, seed=0, test_size=0.2, Legacy→HSC, test-only gallery,
  StandardScaler + Ridge(alpha=1, fit_intercept=True), k=10.

Does not refit Ridge for the linear-only condition — ablates b_eff from the
already-fitted affine map.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from math import comb
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

WS_ROOT = Path(__file__).resolve().parents[3]
FAMILY_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
FAM_ALIAS = {
    "astropt": "astropt",
    "astroptv2": "astropt",
    "convnext": "convnext",
    "convnextv2": "convnext",
    "dinov2": "dinov2",
    "dino": "dinov2",
    "vit": "vit",
    "ijepa": "ijepa",
}
K_EVAL = 10
EPS_ETA = 1e-12


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def load_col(path: Path, col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[col])
    return np.vstack(table.column(0).to_pylist()).astype(np.float32)


def load_pair_arrays(
    root: Path, cfg: dict, max_n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    p1 = cfg.get("parquet1", cfg.get("parquet_1"))
    c1 = cfg.get("col1", cfg.get("col_1"))
    p2 = cfg.get("parquet2", cfg.get("parquet_2"))
    c2 = cfg.get("col2", cfg.get("col_2"))
    X1 = load_col(resolve_path(root, p1), c1)
    X2 = load_col(resolve_path(root, p2), c2)
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


def recover_affine_map(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> dict:
    """Fit StandardScaler+Ridge; recover original-coordinate affine map.

    Returns A_eff, b_eff such that y_hat = A_eff @ x + b_eff matches
    y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).
    """
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))

    W = np.asarray(ridge.coef_, dtype=np.float64)  # (d_y, d_x)
    b_std = np.asarray(ridge.intercept_, dtype=np.float64)

    sx = np.asarray(x_sc.scale_, dtype=np.float64)
    sy = np.asarray(y_sc.scale_, dtype=np.float64)
    mx = np.asarray(x_sc.mean_, dtype=np.float64)
    my = np.asarray(y_sc.mean_, dtype=np.float64)
    sx = np.where(sx > 0, sx, 1.0)
    sy = np.where(sy > 0, sy, 1.0)

    # x_std = (x - mx) / sx
    # y_std_hat = W x_std + b_std
    # y_hat = sy * y_std_hat + my
    #       = diag(sy) W diag(1/sx) x - diag(sy) W diag(1/sx) mx + sy b_std + my
    A_eff = (sy[:, None] * W) / sx[None, :]
    b_eff = my + sy * b_std - A_eff @ mx

    x64 = x.astype(np.float64)
    pipeline = y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float64)
    reconstructed = x64 @ A_eff.T + b_eff
    linear_only = x64 @ A_eff.T

    max_abs_err = float(np.max(np.abs(pipeline - reconstructed)))
    mean_abs_err = float(np.mean(np.abs(pipeline - reconstructed)))

    # Linear-only has no translation: A @ 0 = 0
    assert np.allclose(A_eff @ np.zeros(A_eff.shape[1]), 0.0)

    return {
        "A_eff": A_eff,
        "b_eff": b_eff,
        "pipeline_mapped": pipeline.astype(np.float32),
        "affine_mapped": reconstructed.astype(np.float32),
        "linear_only_mapped": linear_only.astype(np.float32),
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "mu_x": mx,
        "mu_y": my,
        "d_x": int(x.shape[1]),
        "d_y": int(y.shape[1]),
        "b_eff_l2": float(np.linalg.norm(b_eff)),
        "A_eff_frob": float(np.linalg.norm(A_eff, ord="fro")),
    }


def plot_ablation(df: pd.DataFrame, delta: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.scatter(df["gain_affine"], df["gain_linear_only"], s=36, c="#1f4e79", zorder=3)
    lim_lo = min(df["gain_affine"].min(), df["gain_linear_only"].min(), 0.0) - 0.002
    lim_hi = max(df["gain_affine"].max(), df["gain_linear_only"].max()) + 0.002
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.6)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"Affine gain $G_{\mathrm{affine}}$")
    ax.set_ylabel(r"Linear-only gain $G_A$")
    ax.set_title("Per-rung gain: affine vs linear-only")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    fams = [f for f in FAMILY_ORDER if f in delta["delta_beta_affine"]]
    x = np.arange(len(fams))
    w = 0.35
    ax.bar(
        x - w / 2,
        [delta["delta_beta_affine"][f] for f in fams],
        w,
        label=r"$\Delta\beta^{\mathrm{affine}}$",
        color="#1f4e79",
    )
    ax.bar(
        x + w / 2,
        [delta["delta_beta_linear"][f] for f in fams],
        w,
        label=r"$\Delta\beta^{A}$",
        color="#c44e52",
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL.get(f, f) for f in fams], rotation=15)
    ax.set_ylabel(r"Family slope amplification $\Delta\beta_F$")
    ax.set_title(r"Family $\Delta\beta$: affine vs linear-only")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    ap.add_argument("--verify-tol", type=float, default=1e-4)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "translation_ablation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "configs/official_legacy_pairs.yaml",
            root / "experiments/SAE-shared-basis/official_legacy_pairs.yaml",
            root / "tmp_ridge_geom/official_legacy_pairs.yaml",
            WS_ROOT
            / "configs/official_legacy_pairs.yaml",
        ]
        pairs_path = next(p for p in cands if p.is_file())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = yaml.safe_load(pairs_path.read_text())
    names = [
        n
        for n, cfg in pairs.items()
        if isinstance(cfg, dict)
        and (
            "legacysurvey" in str(cfg.get("col2", cfg.get("col_2", "")))
            or "legacysurvey" in str(cfg.get("parquet2", cfg.get("parquet_2", "")))
        )
    ]
    print(f"pairs={len(names)} device={device} yaml={pairs_path}", flush=True)

    first_cfg = pairs[names[0]]
    X1_ref, X2_ref = load_pair_arrays(root, first_cfg, args.max_n, args.seed)
    n = len(X1_ref)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"n={n} train={len(train_idx)} test={len(test_idx)}", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    verify_errs: list[float] = []

    for name in names:
        cfg = pairs[name]
        X_hsc, X_leg = load_pair_arrays(root, cfg, args.max_n, args.seed)
        assert len(X_hsc) == n, f"row mismatch {name}"
        X, Y = X_leg.astype(np.float32), X_hsc.astype(np.float32)

        params_m = float(
            cfg.get("approx_params_m", cfg.get("approx_params_millions", 1))
        )
        fam = FAM_ALIAS.get(str(cfg["family"]).lower(), str(cfg["family"]).lower())
        logp = math.log10(max(params_m * 1e6, 1.0))

        fit = recover_affine_map(X, Y, train_idx, alpha=args.alpha)
        verify_errs.append(fit["max_abs_err"])
        if fit["max_abs_err"] > args.verify_tol:
            raise RuntimeError(
                f"Affine reconstruction failed for {name}: "
                f"max_abs_err={fit['max_abs_err']:.3e} > {args.verify_tol}"
            )

        pipe_te = fit["pipeline_mapped"][test_idx]
        aff_te = fit["affine_mapped"][test_idx]
        lin_te = fit["linear_only_mapped"][test_idx]
        assert np.allclose(pipe_te, aff_te, atol=args.verify_tol, rtol=0)

        # Centered diagnostic: train means only
        mu_x, mu_y = fit["mu_x"], fit["mu_y"]
        Xc = (X.astype(np.float64) - mu_x).astype(np.float32)
        Yc = (Y.astype(np.float64) - mu_y).astype(np.float32)
        lin_c = (Xc.astype(np.float64) @ fit["A_eff"].T).astype(np.float32)

        Xte, Yte = X[test_idx], Y[test_idx]
        m_native = mknn_pair(Xte, Yte, K_EVAL, device, args.row_batch)
        m_affine = mknn_pair(aff_te, Yte, K_EVAL, device, args.row_batch)
        m_linear = mknn_pair(lin_te, Yte, K_EVAL, device, args.row_batch)
        m_cn = mknn_pair(Xc[test_idx], Yc[test_idx], K_EVAL, device, args.row_batch)
        m_cl = mknn_pair(lin_c[test_idx], Yc[test_idx], K_EVAL, device, args.row_batch)

        g_aff = m_affine - m_native
        g_lin = m_linear - m_native
        eta = g_lin / g_aff if abs(g_aff) > EPS_ETA else float("nan")

        rows.append(
            {
                "family": fam,
                "model": name,
                "size_name": cfg.get("size_name"),
                "paper_model": cfg.get("paper_model"),
                "parameter_count": int(round(params_m * 1e6)),
                "log10_params": logp,
                "d_x": fit["d_x"],
                "d_y": fit["d_y"],
                "mknn_native": m_native,
                "mknn_affine": m_affine,
                "mknn_linear_only": m_linear,
                "gain_affine": g_aff,
                "gain_linear_only": g_lin,
                "eta_linear": eta,
                "mknn_centered_native": m_cn,
                "mknn_centered_linear": m_cl,
                "gain_centered": m_cl - m_cn,
                "b_eff_l2": fit["b_eff_l2"],
                "A_eff_frob": fit["A_eff_frob"],
                "affine_recon_max_abs_err": fit["max_abs_err"],
            }
        )
        print(
            f"{name}: native={m_native:.4f} aff={m_affine:.4f} "
            f"lin={m_linear:.4f} eta={eta:.3f} "
            f"err={fit['max_abs_err']:.2e}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["family", "log10_params"])
    df.to_csv(out_dir / "translation_ablation_rung_scores.csv", index=False)

    def slopes(col: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for fam in FAMILY_ORDER:
            sub = df[df["family"] == fam]
            if len(sub) < 2:
                continue
            out[fam] = family_ols_slope(
                sub["log10_params"].to_numpy(float), sub[col].to_numpy(float)
            )
        return out

    beta_native = slopes("mknn_native")
    beta_affine = slopes("mknn_affine")
    beta_linear = slopes("mknn_linear_only")
    beta_cn = slopes("mknn_centered_native")
    beta_cl = slopes("mknn_centered_linear")

    d_aff = {f: beta_affine[f] - beta_native[f] for f in beta_native}
    d_lin = {f: beta_linear[f] - beta_native[f] for f in beta_native}
    d_cen = {f: beta_cl[f] - beta_cn[f] for f in beta_cn}

    T_aff = float(np.mean(list(d_aff.values())))
    T_lin = float(np.mean(list(d_lin.values())))
    T_cen = float(np.mean(list(d_cen.values())))
    n_pos_lin = int(sum(v > 0 for v in d_lin.values()))
    n_pos_aff = int(sum(v > 0 for v in d_aff.values()))

    d_lin_ex = [d_lin[f] for f in d_lin if f != "ijepa"]
    d_aff_ex = [d_aff[f] for f in d_aff if f != "ijepa"]
    T_lin_ex = float(np.mean(d_lin_ex))
    T_aff_ex = float(np.mean(d_aff_ex))

    etas = df["eta_linear"].to_numpy(float)
    etas = etas[np.isfinite(etas)]

    n_fam = len(d_lin)
    sign_p_lin = (
        sum(comb(n_fam, k) for k in range(n_pos_lin, n_fam + 1)) / (2**n_fam)
        if n_fam
        else float("nan")
    )

    summary = {
        "T_affine": T_aff,
        "T_linear_only": T_lin,
        "T_ratio_linear_over_affine": (
            T_lin / T_aff if abs(T_aff) > EPS_ETA else float("nan")
        ),
        "T_affine_excl_ijepa": T_aff_ex,
        "T_linear_only_excl_ijepa": T_lin_ex,
        "T_centered": T_cen,
        "n_positive_affine": n_pos_aff,
        "n_positive_linear_only": n_pos_lin,
        "n_families": n_fam,
        "sign_p_linear_only": float(sign_p_lin),
        "delta_beta_affine": d_aff,
        "delta_beta_linear": d_lin,
        "delta_beta_centered": d_cen,
        "eta_mean": float(np.mean(etas)),
        "eta_median": float(np.median(etas)),
        "eta_min": float(np.min(etas)),
        "eta_max": float(np.max(etas)),
        "mean_gain_affine": float(df["gain_affine"].mean()),
        "mean_gain_linear_only": float(df["gain_linear_only"].mean()),
        "mean_mknn_native": float(df["mknn_native"].mean()),
        "mean_mknn_affine": float(df["mknn_affine"].mean()),
        "mean_mknn_linear_only": float(df["mknn_linear_only"].mean()),
        "mean_mknn_centered_native": float(df["mknn_centered_native"].mean()),
        "mean_mknn_centered_linear": float(df["mknn_centered_linear"].mean()),
        "affine_recon_max_err_worst": float(max(verify_errs)),
        "affine_recon_max_err_mean": float(np.mean(verify_errs)),
        "verify_tol": args.verify_tol,
        "n_rungs": int(len(df)),
        "seed": args.seed,
        "test_size": args.test_size,
        "n": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "ridge_alpha": args.alpha,
        "k": K_EVAL,
        "direction": "Legacy→HSC",
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "pairs_yaml": str(pairs_path),
    }
    (out_dir / "translation_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    pd.DataFrame(
        [
            {
                "statistic": "T_affine",
                "value": T_aff,
                "n_positive_families": n_pos_aff,
            },
            {
                "statistic": "T_linear_only",
                "value": T_lin,
                "n_positive_families": n_pos_lin,
            },
            {
                "statistic": "T_ratio_linear_over_affine",
                "value": summary["T_ratio_linear_over_affine"],
                "n_positive_families": n_pos_lin,
            },
            {
                "statistic": "T_linear_only_excl_ijepa",
                "value": T_lin_ex,
                "n_positive_families": int(sum(v > 0 for v in d_lin_ex)),
            },
            {
                "statistic": "T_centered",
                "value": T_cen,
                "n_positive_families": int(sum(v > 0 for v in d_cen.values())),
            },
            {"statistic": "eta_mean", "value": summary["eta_mean"], "n_positive_families": ""},
            {
                "statistic": "eta_median",
                "value": summary["eta_median"],
                "n_positive_families": "",
            },
        ]
    ).to_csv(out_dir / "translation_ablation_aggregate.csv", index=False)

    pd.DataFrame(
        [
            {
                "family": f,
                "delta_beta_affine": d_aff[f],
                "delta_beta_linear_only": d_lin[f],
                "delta_beta_centered": d_cen.get(f, float("nan")),
            }
            for f in FAMILY_ORDER
            if f in d_aff
        ]
    ).to_csv(out_dir / "family_delta_beta_translation.csv", index=False)

    plot_ablation(
        df,
        {"delta_beta_affine": d_aff, "delta_beta_linear": d_lin},
        fig_dir / "translation_ablation_summary.png",
    )
    paper_fig = WS_ROOT / "figures" / "translation_ablation_summary.png"
    if paper_fig.parent.is_dir():
        shutil.copy2(fig_dir / "translation_ablation_summary.png", paper_fig)

    print("\n=== TRANSLATION ABLATION ===", flush=True)
    print(f"T_affine       = {T_aff:.5f}  ({n_pos_aff}/{n_fam} positive)", flush=True)
    print(f"T_linear_only  = {T_lin:.5f}  ({n_pos_lin}/{n_fam} positive)", flush=True)
    print(
        f"T_A / T_aff    = {summary['T_ratio_linear_over_affine']:.3f}",
        flush=True,
    )
    print(f"T_A excl I-JEPA= {T_lin_ex:.5f}", flush=True)
    print(f"T_centered     = {T_cen:.5f}", flush=True)
    print(
        f"eta mean/med/range = {summary['eta_mean']:.3f} / "
        f"{summary['eta_median']:.3f} / "
        f"[{summary['eta_min']:.3f}, {summary['eta_max']:.3f}]",
        flush=True,
    )
    print(
        f"affine recon worst max|err| = {summary['affine_recon_max_err_worst']:.2e}",
        flush=True,
    )
    print(f"done in {summary['elapsed_sec']:.1f}s → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
