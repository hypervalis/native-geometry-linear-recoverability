#!/usr/bin/env python3
"""Singular-spectrum analysis of Dense→Dense Ridge maps.

Uses the same protocol as run_ridge_scaling_geometry.py / run_alignment_controls.py.
Does not overwrite prior CSVs except by writing new spectrum-specific artifacts
under outputs/ridge_scaling_geometry/.
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
EPS_PRIMARY = 1e-8
EPS_ROBUST = (1e-6, 1e-8, 1e-10)
RANK_GRID = (8, 16, 32, 64, 128, 256)
ENERGY_GRID = (0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


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


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


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


def fit_ridge_components(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, alpha: float
) -> dict:
    """Fit Ridge in scaled space; return W, b, scalers, full SVD, mapped embeddings."""
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    xs = x_sc.transform(x_tr)
    ys = y_sc.transform(y_tr)
    ridge.fit(xs, ys)
    W = np.asarray(ridge.coef_, dtype=np.float64)  # (d_out, d_in)
    b = np.asarray(ridge.intercept_, dtype=np.float64)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    mapped = y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)
    return {
        "W": W,
        "b": b,
        "x_sc": x_sc,
        "y_sc": y_sc,
        "U": U,
        "s": s,
        "Vt": Vt,
        "mapped": mapped,
        "input_dim": int(W.shape[1]),
        "output_dim": int(W.shape[0]),
    }


def apply_truncated_map(
    x: np.ndarray,
    fit: dict,
    *,
    k: int | None = None,
    energy: float | None = None,
) -> np.ndarray:
    """Apply rank-k or energy-truncated W without refitting."""
    U, s, Vt = fit["U"], fit["s"], fit["Vt"]
    if energy is not None:
        e2 = s**2
        cum = np.cumsum(e2) / max(e2.sum(), 1e-300)
        k = int(np.searchsorted(cum, energy) + 1)
        k = min(max(k, 1), len(s))
    assert k is not None
    k = min(int(k), len(s))
    Wk = (U[:, :k] * s[:k]) @ Vt[:k, :]
    xs = fit["x_sc"].transform(x)
    ys_hat = xs @ Wk.T + fit["b"]
    return fit["y_sc"].inverse_transform(ys_hat).astype(np.float32)


def active_mask(s: np.ndarray, eps: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return np.zeros(0, dtype=bool)
    return s >= eps * float(s.max())


def spectrum_shape_metrics(s: np.ndarray, eps: float) -> dict[str, float]:
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s) & (s >= 0)]
    n_total = int(s.size)
    if n_total == 0:
        return {"n_total": 0, "n_active": 0, "active_fraction": float("nan")}
    mask = active_mask(s, eps)
    active = s[mask]
    n_active = int(active.size)
    if n_active == 0:
        active = s[:1]
        n_active = 1
    log_s = np.log(np.clip(active, 1e-300, None))
    g = float(np.exp(np.mean(log_s)))
    tilde = active / g
    A_log = float(np.std(np.log(tilde), ddof=0))
    e2 = active**2
    p = e2 / max(e2.sum(), 1e-300)
    H = float(-np.sum(p * np.log(np.clip(p, 1e-300, None))))
    H_norm = H / math.log(n_active) if n_active > 1 else float("nan")
    r_eff = float(np.exp(H))
    r_eff_norm = r_eff / n_active
    c = float(np.mean(active))
    D_sim = float(np.linalg.norm(active - c) / (np.linalg.norm(active) + 1e-300))
    q95, q05 = np.quantile(active, [0.95, 0.05])
    S_95_5 = float(np.log(q95 / max(q05, 1e-300)))
    kappa_95_5 = float(q95 / max(q05, 1e-300))
    # energy ranks on active spectrum (ordered descending — already sorted)
    cum = np.cumsum(e2) / max(e2.sum(), 1e-300)
    energy_ranks = {}
    for thr, name in [
        (0.50, "r50"),
        (0.75, "r75"),
        (0.90, "r90"),
        (0.95, "r95"),
        (0.99, "r99"),
    ]:
        r = int(np.searchsorted(cum, thr) + 1)
        energy_ranks[name] = r
        energy_ranks[f"{name}_frac"] = r / n_active
    return {
        "eps": eps,
        "n_total": n_total,
        "n_active": n_active,
        "active_fraction": n_active / n_total,
        "geom_mean": g,
        "H": H,
        "H_norm": H_norm,
        "r_eff": r_eff,
        "r_eff_norm": r_eff_norm,
        "A_log": A_log,
        "D_sim": D_sim,
        "S_95_5": S_95_5,
        "kappa_95_5": kappa_95_5,
        "sigma_max": float(active.max()),
        "sigma_min_active": float(active.min()),
        **energy_ranks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    ap.add_argument("--skip-trunc", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "ridge_scaling_geometry"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (
        args.fig_dir.expanduser().resolve()
        if args.fig_dir
        else ROOT / "paper_working" / "figures"
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "experiments/universetbd_shared_basis_mknn/official_legacy_pairs.yaml",
            ROOT / "experiments/universetbd_shared_basis_mknn/official_legacy_pairs.yaml",
        ]
        pairs_path = next(p for p in cands if p.is_file())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = yaml.safe_load(pairs_path.read_text())
    names = [
        n
        for n, cfg in pairs.items()
        if isinstance(cfg, dict) and "legacysurvey" in str(cfg.get("parquet2", ""))
    ]

    X1_0, _, _ = load_pair_arrays(root, pairs[names[0]], args.max_n, args.seed)
    n = len(X1_0)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(
        f"n={n} train={len(train_idx)} test={len(test_idx)} device={device} "
        f"pairs={len(names)}",
        flush=True,
    )

    meta = {
        "analysis": "singular_spectrum_of_dense_ridge_maps",
        "seed": args.seed,
        "test_size": args.test_size,
        "ridge_alpha": args.alpha,
        "fit_intercept": True,
        "normalization": "StandardScaler on X and Y (train-only)",
        "map_analyzed": "sklearn Ridge.coef_ W in scaled space (y_sc ≈ x_sc @ W.T + b)",
        "eps_primary": EPS_PRIMARY,
        "eps_robustness": list(EPS_ROBUST),
        "primary_spectral_flatness": "H_norm",
        "direction": "Legacy→HSC (col2→col1)",
        "gallery": "test-only",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "singular_spectrum_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    full_rows = []
    summary_rows = []
    robust_rows = []
    trunc_rows = []
    map_meta_rows = []
    spectra: dict[str, dict] = {}

    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        assert len(X1) == n
        X, Y = X2.astype(np.float32), X1.astype(np.float32)  # Legacy→HSC
        fam = str(cfg["family"])
        params_m = float(cfg.get("approx_params_m", 1))
        log10p = math.log10(max(params_m * 1e6, 1.0))
        pcount = int(round(params_m * 1e6))

        fit = fit_ridge_components(X, Y, train_idx, args.alpha)
        s = fit["s"]
        spectra[name] = {
            "family": fam,
            "log10_params": log10p,
            "parameter_count": pcount,
            "s": s,
            "size_name": cfg.get("size_name"),
        }

        map_meta_rows.append(
            {
                "family": fam,
                "model": name,
                "parameter_count": pcount,
                "log10_params": log10p,
                "input_dim": fit["input_dim"],
                "output_dim": fit["output_dim"],
                "ridge_alpha": args.alpha,
                "fit_intercept": True,
                "normalization": "StandardScaler(X), StandardScaler(Y), train-only",
                "intercept_l2_scaled": float(np.linalg.norm(fit["b"])),
            }
        )

        for i, sv in enumerate(s, start=1):
            full_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "parameter_count": pcount,
                    "rank_index": i,
                    "singular_value": float(sv),
                    "singular_value_squared": float(sv**2),
                }
            )

        m_primary = spectrum_shape_metrics(s, EPS_PRIMARY)

        for eps in EPS_ROBUST:
            m = spectrum_shape_metrics(s, eps)
            robust_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "parameter_count": pcount,
                    "log10_params": log10p,
                    **m,
                }
            )

        # Dense + full Ridge scores
        Xte, Yte = X[test_idx], Y[test_idx]
        m_dense = mknn_pair(Xte, Yte, args.k, device, args.row_batch)
        m_full = mknn_pair(fit["mapped"][test_idx], Yte, args.k, device, args.row_batch)
        lift_full = m_full - m_dense

        summary_rows.append(
            {
                "family": fam,
                "model": name,
                "size_name": cfg.get("size_name"),
                "parameter_count": pcount,
                "log10_params": log10p,
                "mknn_dense": m_dense,
                "mknn_dense_ridge": m_full,
                "lift_ridge": lift_full,
                **{k: m_primary[k] for k in m_primary if k != "eps"},
                "eps": EPS_PRIMARY,
            }
        )
        print(
            f"{name}: H_norm={m_primary['H_norm']:.4f} A_log={m_primary['A_log']:.4f} "
            f"r90_frac={m_primary['r90_frac']:.3f} ridge={m_full:.4f}",
            flush=True,
        )

        if args.skip_trunc:
            continue

        nn_y = knn_cos(
            torch.as_tensor(np.ascontiguousarray(Yte), device=device),
            args.k,
            args.row_batch,
        )
        r_full = min(fit["input_dim"], fit["output_dim"], len(s))

        # Rank grid + full
        ks = sorted({min(r, r_full) for r in RANK_GRID if r <= r_full} | {r_full})
        for kk in ks:
            mapped_k = apply_truncated_map(X, fit, k=kk)
            m_k = mknn(
                knn_cos(
                    torch.as_tensor(mapped_k[test_idx], device=device),
                    args.k,
                    args.row_batch,
                ),
                nn_y,
                args.k,
            )
            trunc_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "parameter_count": pcount,
                    "log10_params": log10p,
                    "mode": "rank",
                    "k": kk,
                    "k_frac": kk / r_full,
                    "energy_retained": float(np.sum(s[:kk] ** 2) / np.sum(s**2)),
                    "mknn": m_k,
                    "mknn_dense": m_dense,
                    "mknn_full_ridge": m_full,
                    "lift": m_k - m_dense,
                    "lift_frac_of_full": (m_k - m_dense) / max(lift_full, 1e-12),
                }
            )

        # Energy grid
        for e in ENERGY_GRID:
            mapped_e = apply_truncated_map(X, fit, energy=e)
            # recover k used
            e2 = s**2
            cum = np.cumsum(e2) / max(e2.sum(), 1e-300)
            kk = int(np.searchsorted(cum, e) + 1) if e < 1.0 else len(s)
            kk = min(max(kk, 1), len(s))
            m_e = mknn(
                knn_cos(
                    torch.as_tensor(mapped_e[test_idx], device=device),
                    args.k,
                    args.row_batch,
                ),
                nn_y,
                args.k,
            )
            trunc_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "parameter_count": pcount,
                    "log10_params": log10p,
                    "mode": "energy",
                    "k": kk,
                    "k_frac": kk / r_full,
                    "energy_retained": e,
                    "mknn": m_e,
                    "mknn_dense": m_dense,
                    "mknn_full_ridge": m_full,
                    "lift": m_e - m_dense,
                    "lift_frac_of_full": (m_e - m_dense) / max(lift_full, 1e-12),
                }
            )

    # Normalized spectra CSV
    norm_rows = []
    for name, sp in spectra.items():
        s = sp["s"]
        m = spectrum_shape_metrics(s, EPS_PRIMARY)
        mask = active_mask(s, EPS_PRIMARY)
        active_idx = np.flatnonzero(mask) if mask.any() else np.array([0])
        active = s[active_idx]
        g = m["geom_mean"]
        r = len(active)
        for j, (idx0, sv) in enumerate(zip(active_idx, active), start=1):
            norm_rows.append(
                {
                    "family": sp["family"],
                    "model": name,
                    "parameter_count": sp["parameter_count"],
                    "rank_index": int(idx0 + 1),
                    "norm_rank": j / r,
                    "singular_value": float(sv),
                    "singular_value_norm": float(sv / g),
                    "log10_singular_value_norm": float(np.log10(sv / g)),
                }
            )

    pd.DataFrame(full_rows).to_csv(out_dir / "singular_spectra_full.csv", index=False)
    pd.DataFrame(norm_rows).to_csv(
        out_dir / "singular_spectra_normalized.csv", index=False
    )
    summary = pd.DataFrame(summary_rows).sort_values(["family", "log10_params"])
    summary.to_csv(out_dir / "singular_spectrum_summary.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(
        out_dir / "singular_spectrum_eps_robustness.csv", index=False
    )
    pd.DataFrame(map_meta_rows).to_csv(out_dir / "ridge_map_metadata.csv", index=False)

    # Transfer complexity k90
    transfer_rows = []
    if trunc_rows:
        tdf = pd.DataFrame(trunc_rows)
        tdf.to_csv(out_dir / "truncated_svd_mknn.csv", index=False)
        for name, g in tdf[tdf["mode"] == "rank"].groupby("model"):
            g = g.sort_values("k")
            row0 = summary[summary["model"] == name].iloc[0]
            hit = g[g["lift_frac_of_full"] >= 0.9]
            if len(hit):
                k90 = int(hit.iloc[0]["k"])
                k90_frac = float(hit.iloc[0]["k_frac"])
            else:
                k90 = int(g.iloc[-1]["k"])
                k90_frac = float(g.iloc[-1]["k_frac"])
            transfer_rows.append(
                {
                    "family": row0["family"],
                    "model": name,
                    "parameter_count": int(row0["parameter_count"]),
                    "log10_params": float(row0["log10_params"]),
                    "k90_transfer": k90,
                    "k90_transfer_frac": k90_frac,
                    "H_norm": float(row0["H_norm"]),
                    "mknn_dense_ridge": float(row0["mknn_dense_ridge"]),
                    "lift_ridge": float(row0["lift_ridge"]),
                }
            )
        pd.DataFrame(transfer_rows).to_csv(
            out_dir / "transfer_complexity_k90.csv", index=False
        )

    # Family slopes vs log10P
    slope_rows = []
    metrics = ["H_norm", "r_eff_norm", "A_log", "D_sim", "S_95_5", "r90_frac", "r95_frac"]
    for fam in FAMILY_ORDER:
        sub = summary[summary["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        for met in metrics:
            slope_rows.append(
                {
                    "family": fam,
                    "metric": met,
                    "slope_vs_log10P": family_ols_slope(
                        sub["log10_params"].to_numpy(float), sub[met].to_numpy(float)
                    ),
                    "n_rungs": int(len(sub)),
                    "two_point_only": bool(len(sub) == 2),
                    "delta_small_to_large": float(sub[met].iloc[-1] - sub[met].iloc[0]),
                }
            )
    if transfer_rows:
        tsum = pd.DataFrame(transfer_rows)
        for fam in FAMILY_ORDER:
            sub = tsum[tsum["family"] == fam].sort_values("log10_params")
            if len(sub) < 2:
                continue
            slope_rows.append(
                {
                    "family": fam,
                    "metric": "k90_transfer_frac",
                    "slope_vs_log10P": family_ols_slope(
                        sub["log10_params"].to_numpy(float),
                        sub["k90_transfer_frac"].to_numpy(float),
                    ),
                    "n_rungs": int(len(sub)),
                    "two_point_only": bool(len(sub) == 2),
                    "delta_small_to_large": float(
                        sub["k90_transfer_frac"].iloc[-1]
                        - sub["k90_transfer_frac"].iloc[0]
                    ),
                }
            )
    pd.DataFrame(slope_rows).to_csv(
        out_dir / "singular_spectrum_vs_size_slopes.csv", index=False
    )

    # Spearman recoverability vs spectrum
    corr_rows = []
    for xcol, ycol in [
        ("mknn_dense_ridge", "H_norm"),
        ("lift_ridge", "H_norm"),
        ("mknn_dense_ridge", "A_log"),
        ("lift_ridge", "A_log"),
        ("mknn_dense_ridge", "D_sim"),
        ("lift_ridge", "r_eff_norm"),
        ("lift_ridge", "r90_frac"),
    ]:
        rho = spearman(summary[xcol].to_numpy(float), summary[ycol].to_numpy(float))
        corr_rows.append(
            {"scope": "all_rungs", "x": xcol, "y": ycol, "spearman": rho, "n": len(summary)}
        )
        for fam in FAMILY_ORDER:
            sub = summary[summary["family"] == fam]
            if len(sub) < 3:
                continue
            corr_rows.append(
                {
                    "scope": fam,
                    "x": xcol,
                    "y": ycol,
                    "spearman": spearman(
                        sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)
                    ),
                    "n": len(sub),
                }
            )
    pd.DataFrame(corr_rows).to_csv(
        out_dir / "recoverability_vs_spectrum_spearman.csv", index=False
    )

    # Adjacent transitions: Δspectral vs D_align
    adj_rows = []
    for fam in FAMILY_ORDER:
        sub = summary[summary["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        for i in range(len(sub) - 1):
            a, b = sub.iloc[i], sub.iloc[i + 1]
            d_dense = b["mknn_dense"] - a["mknn_dense"]
            d_ridge = b["mknn_dense_ridge"] - a["mknn_dense_ridge"]
            adj_rows.append(
                {
                    "family": fam,
                    "from_model": a["model"],
                    "to_model": b["model"],
                    "D_align": d_ridge - d_dense,
                    "delta_H_norm": b["H_norm"] - a["H_norm"],
                    "delta_A_log": b["A_log"] - a["A_log"],
                    "delta_D_sim": b["D_sim"] - a["D_sim"],
                    "delta_r90_frac": b["r90_frac"] - a["r90_frac"],
                }
            )
    adj = pd.DataFrame(adj_rows)
    adj.to_csv(out_dir / "adjacent_Dalign_vs_spectrum_delta.csv", index=False)
    if len(adj):
        adj_corr = []
        for y in ["delta_H_norm", "delta_A_log", "delta_D_sim", "delta_r90_frac"]:
            adj_corr.append(
                {
                    "x": "D_align",
                    "y": y,
                    "spearman": spearman(
                        adj["D_align"].to_numpy(float), adj[y].to_numpy(float)
                    ),
                    "n": len(adj),
                }
            )
        pd.DataFrame(adj_corr).to_csv(
            out_dir / "adjacent_Dalign_vs_spectrum_spearman.csv", index=False
        )

    # Sign consistency for H_norm vs size
    h_slopes = {
        r["family"]: r["slope_vs_log10P"]
        for r in slope_rows
        if r["metric"] == "H_norm"
    }
    n_pos = sum(1 for v in h_slopes.values() if v > 0)
    n_neg = sum(1 for v in h_slopes.values() if v < 0)
    sign_summary = {
        "metric": "H_norm_vs_log10P",
        "n_families": len(h_slopes),
        "n_positive_slope": n_pos,
        "n_negative_slope": n_neg,
        "slopes": h_slopes,
    }
    (out_dir / "H_norm_size_sign_summary.json").write_text(
        json.dumps(sign_summary, indent=2) + "\n"
    )

    # ---- Figures: per-family spectra ----
    for fam in FAMILY_ORDER:
        sub_names = [n for n, sp in spectra.items() if sp["family"] == fam]
        if not sub_names:
            continue
        sub_names = sorted(sub_names, key=lambda n: spectra[n]["log10_params"])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sub_names)))
        for color, name in zip(cmap, sub_names):
            sp = spectra[name]
            s = sp["s"]
            m = spectrum_shape_metrics(s, EPS_PRIMARY)
            mask = active_mask(s, EPS_PRIMARY)
            active = s[mask] if mask.any() else s[:1]
            g = m["geom_mean"]
            tilde = active / g
            x = np.arange(1, len(tilde) + 1) / len(tilde)
            label = f"{sp.get('size_name', name)} (P={sp['parameter_count']:.2g})"
            ax.plot(x, np.log10(tilde), color=color, lw=1.6, label=label)
        ax.set_xlabel(r"normalized rank $i/r$")
        ax.set_ylabel(r"$\log_{10}(\tilde\sigma_i)$")
        ax.set_title(f"{FAMILY_LABEL[fam]}: geom-mean–normalized singular spectrum")
        ax.legend(fontsize=7, loc="best")
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        fig.tight_layout()
        fname = f"singular_spectrum_{FAMILY_LABEL[fam].replace('-', '')}.png"
        # I-JEPA label → IJEPA
        if fam == "ijepa":
            fname = "singular_spectrum_IJEPA.png"
        elif fam == "astropt":
            fname = "singular_spectrum_AstroPT.png"
        elif fam == "convnext":
            fname = "singular_spectrum_ConvNeXt.png"
        elif fam == "dinov2":
            fname = "singular_spectrum_DINOv2.png"
        elif fam == "vit":
            fname = "singular_spectrum_ViT.png"
        fig.savefig(fig_dir / fname, dpi=160, bbox_inches="tight")
        fig.savefig(out_dir / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Recoverability vs spectral entropy
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam in FAMILY_ORDER:
        sub = summary[summary["family"] == fam].sort_values("log10_params")
        if sub.empty:
            continue
        ax.plot(
            sub["mknn_dense_ridge"],
            sub["H_norm"],
            "o-",
            label=FAMILY_LABEL[fam],
            markersize=6,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(r.get("size_name", "")),
                (r["mknn_dense_ridge"], r["H_norm"]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel(r"recoverability $R = M_{\mathrm{Dense+Ridge}}$")
    ax.set_ylabel(r"spectral flatness $G = H_{\mathrm{norm}}$")
    ax.set_title("Recoverability vs alignment-map spectral entropy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "recoverability_vs_spectral_entropy.png", dpi=160, bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "recoverability_vs_spectral_entropy.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)

    # Transfer efficiency curves (optional aggregate figure)
    if trunc_rows:
        tdf = pd.DataFrame(trunc_rows)
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
        for ax, fam in zip(axes, FAMILY_ORDER):
            sub = tdf[(tdf["family"] == fam) & (tdf["mode"] == "rank")]
            for name, g in sub.groupby("model"):
                g = g.sort_values("k_frac")
                lp = float(g["log10_params"].iloc[0])
                ax.plot(g["k_frac"], g["lift_frac_of_full"], "o-", markersize=3, label=f"{lp:.1f}")
            ax.set_title(FAMILY_LABEL[fam])
            ax.set_xlabel(r"$k/r$")
            ax.axhline(0.9, color="0.5", ls="--", lw=0.8)
        axes[0].set_ylabel("fraction of full Ridge lift")
        fig.suptitle("Truncated-SVD transfer efficiency", y=1.02)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "truncated_svd_transfer_efficiency.png", dpi=160, bbox_inches="tight"
        )
        fig.savefig(
            out_dir / "truncated_svd_transfer_efficiency.png", dpi=160, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"Wrote {out_dir}", flush=True)
    print(f"H_norm size signs: {sign_summary}", flush=True)


if __name__ == "__main__":
    main()
