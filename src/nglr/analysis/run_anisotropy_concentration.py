#!/usr/bin/env python3
"""Anisotropy concentration of Dense+Ridge effective maps A_eff.

Uses singular values of A_eff = diag(σy) W diag(1/σx) from
outputs/ridge_scaling_geometry/effective_map_spectrum/ (same pipeline as
run_singular_spectrum_analysis.py with composed effective map).

Does not refit Ridge.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "outputs/ridge_scaling_geometry/effective_map_spectrum"
OUT = ROOT / "outputs/ridge_scaling_geometry/anisotropy_concentration"
FIG = ROOT / "figures"
PAPER_FIG = ROOT / "figures"

EPS = 1e-8
LOG_FLOOR = 1e-300
FAM_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAM_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}


def rankdata(a: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(a, float)).rank(method="average").to_numpy()


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if den == 0:
        return float("nan")
    return float((x * y).sum() / den)


def betainc_reg(a: float, b: float, x: float, n: int = 8000) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    ts = np.linspace(0.0, x, n + 1)
    if a < 1:
        ts = ts[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.exp(
            (a - 1) * np.log(np.clip(ts, 1e-300, None))
            + (b - 1) * np.log1p(-ts)
            - logB
        )
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapezoid(vals, ts if a >= 1 else ts))


def spearmanr(x, y) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rho = pearsonr(rankdata(x), rankdata(y))
    if n < 3 or not np.isfinite(rho):
        return float(rho), float("nan")
    if abs(rho) >= 1 - 1e-14:
        return float(np.clip(rho, -1, 1)), 0.0
    df = n - 2
    t = rho * math.sqrt(df / (1 - rho * rho))
    xbeta = df / (df + t * t)
    p = betainc_reg(df / 2.0, 0.5, xbeta)
    return float(rho), float(min(max(p, 0.0), 1.0))


def family_perm_spearman(df, xcol, ycol, n_perm=9999, seed=0) -> float:
    """Permute y within families; p_MC = (b+1)/(B+1)."""
    rng = np.random.default_rng(seed)
    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)
    fam = df["family"].to_numpy()
    rho_obs, _ = spearmanr(x, y)
    b = 0
    y_work = y.copy()
    for _ in range(n_perm):
        for f in np.unique(fam):
            m = fam == f
            y_work[m] = rng.permutation(y[m])
        rho_p, _ = spearmanr(x, y_work)
        if abs(rho_p) >= abs(rho_obs) - 1e-15:
            b += 1
    return (b + 1) / (n_perm + 1)


def active_mask(s: np.ndarray, eps: float = EPS) -> np.ndarray:
    s = np.asarray(s, float)
    if s.size == 0:
        return np.zeros(0, dtype=bool)
    return s >= eps * float(s.max())


def centered_log_deviations(s: np.ndarray) -> tuple[np.ndarray, int]:
    """a_i = log σ_i - mean(log σ); floor tiny values before log.

    Returns (a, n_floored) where floored means σ was raised to LOG_FLOOR.
    """
    s = np.asarray(s, float)
    floored = int((s < LOG_FLOOR).sum())
    log_s = np.log(np.clip(s, LOG_FLOOR, None))
    a = log_s - log_s.mean()
    return a, floored


def concentration_stats(a: np.ndarray) -> dict:
    m = a * a
    M = float(m.sum())
    d = len(a)
    if M <= 0 or d == 0:
        return {
            "M_anis": M,
            "A_log_recomputed": float("nan"),
            "r50_anis": float("nan"),
            "r80_anis": float("nan"),
            "r90_anis": float("nan"),
            "r95_anis": float("nan"),
            "f50_anis": float("nan"),
            "f80_anis": float("nan"),
            "f90_anis": float("nan"),
            "f95_anis": float("nan"),
            "r_eff_anis": float("nan"),
            "f_eff_anis": float("nan"),
            "m_sorted": np.array([]),
            "C": np.array([]),
        }
    A_log = math.sqrt(M / d)  # = std(a, ddof=0)
    order = np.argsort(-m)
    m_sorted = m[order]
    C = np.cumsum(m_sorted) / M
    thr = {}
    for name, t in [("50", 0.50), ("80", 0.80), ("90", 0.90), ("95", 0.95)]:
        r = int(np.searchsorted(C, t) + 1)
        r = min(max(r, 1), d)
        thr[f"r{name}_anis"] = r
        thr[f"f{name}_anis"] = r / d
    q = m / M
    H = float(-np.sum(q * np.log(np.clip(q, 1e-300, None))))
    r_eff = float(np.exp(H))
    return {
        "M_anis": M,
        "A_log_recomputed": A_log,
        **thr,
        "r_eff_anis": r_eff,
        "f_eff_anis": r_eff / d,
        "m_sorted": m_sorted,
        "C": C,
    }


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    # slope of y ~ a + b x
    b, _a = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(b)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SPEC_DIR / "singular_spectrum_summary.csv")
    full = pd.read_csv(SPEC_DIR / "singular_spectra_full.csv")
    k90 = pd.read_csv(SPEC_DIR / "transfer_complexity_k90.csv")

    # Manifest
    manifest = summary[
        [
            "family",
            "model",
            "size_name",
            "parameter_count",
            "log10_params",
            "n_total",
            "n_active",
            "A_log",
            "H_norm",
            "D_sim",
        ]
    ].copy()
    manifest["d"] = manifest["n_total"]
    manifest["P_millions"] = manifest["parameter_count"] / 1e6
    manifest.to_csv(OUT / "manifest.csv", index=False)

    parity_rows = []
    per_rows = []
    curve_store = {}

    for _, row in summary.iterrows():
        model = row["model"]
        fam = row["family"]
        s_all = (
            full.loc[full["model"] == model, "singular_value"]
            .to_numpy(float)
        )
        assert len(s_all) == int(row["n_total"]), (model, len(s_all), row["n_total"])

        mask = active_mask(s_all, EPS)
        s_active = s_all[mask]
        n_inactive = int((~mask).sum())
        # Values that needed floor among active (should be rare)
        a, n_floor_active = centered_log_deviations(s_active)
        stats = concentration_stats(a)
        existing = float(row["A_log"])
        recomputed = float(stats["A_log_recomputed"])
        parity_rows.append(
            {
                "model": model,
                "family": fam,
                "existing_A_log": existing,
                "recomputed_A_log": recomputed,
                "absolute_difference": abs(existing - recomputed),
                "n_total": int(row["n_total"]),
                "n_active": int(mask.sum()),
                "n_inactive": n_inactive,
                "n_singular_values_floored": n_floor_active,
            }
        )

        # Curve on fractional axis
        d = len(a)
        f = (np.arange(1, d + 1)) / d
        C = stats["C"]
        curve_df = pd.DataFrame({"f": f, "C": C, "m_sorted": stats["m_sorted"]})
        curve_df.to_csv(OUT / "curves" / f"{model}.npz", index=False)
        curve_store[model] = {
            "family": fam,
            "size_name": row["size_name"],
            "log10_params": float(row["log10_params"]),
            "f": f,
            "C": C,
        }

        # Lift comparison
        k90_row = k90.loc[k90["model"] == model]
        lift_f90 = (
            float(k90_row["k90_transfer_frac"].iloc[0]) if len(k90_row) else float("nan")
        )

        per_rows.append(
            {
                "family": fam,
                "model": model,
                "size_name": row["size_name"],
                "P_millions": float(row["parameter_count"]) / 1e6,
                "log10_P": float(row["log10_params"]),
                "d": int(row["n_total"]),
                "d_active": int(mask.sum()),
                "A_log": existing,
                "A_log_recomputed": recomputed,
                "H_norm": float(row["H_norm"]),
                "D_sim": float(row["D_sim"]),
                "r50_anis": stats["r50_anis"],
                "r80_anis": stats["r80_anis"],
                "r90_anis": stats["r90_anis"],
                "r95_anis": stats["r95_anis"],
                "f50_anis": stats["f50_anis"],
                "f80_anis": stats["f80_anis"],
                "f90_anis": stats["f90_anis"],
                "f95_anis": stats["f95_anis"],
                "r_eff_anis": stats["r_eff_anis"],
                "f_eff_anis": stats["f_eff_anis"],
                "M_anis": stats["M_anis"],
                "n_singular_values_floored": n_floor_active,
                "n_inactive": n_inactive,
                "fraction_directions_for_90pct_lift": lift_f90,
            }
        )

    parity = pd.DataFrame(parity_rows)
    per = pd.DataFrame(per_rows)
    parity.to_csv(OUT / "alog_parity_check.csv", index=False)
    per.to_csv(OUT / "per_model_summary.csv", index=False)

    max_diff = float(parity["absolute_difference"].max())
    if max_diff > 1e-6:
        print(f"WARNING: A_log parity max abs diff = {max_diff}")
    else:
        print(f"A_log parity OK (max abs diff = {max_diff:.3e})")

    # Family scale trends (descriptive)
    trend_rows = []
    for fam in FAM_ORDER:
        g = per[per["family"] == fam].sort_values("log10_P")
        x = g["log10_P"].to_numpy()
        y90 = g["f90_anis"].to_numpy()
        yeff = g["f_eff_anis"].to_numpy()
        trend_rows.append(
            {
                "family": fam,
                "n_rungs": int(len(g)),
                "beta_f90": ols_slope(x, y90),
                "beta_feff": ols_slope(x, yeff),
                "first_to_last_delta_f90": float(y90[-1] - y90[0]),
                "first_to_last_delta_feff": float(yeff[-1] - yeff[0]),
                "two_point_only": bool(len(g) == 2),
            }
        )
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(OUT / "family_scale_trends.csv", index=False)

    # Pooled correlations
    corr_rows = []
    for ycol in ["f90_anis", "f_eff_anis"]:
        rho, p = spearmanr(per["log10_P"], per[ycol])
        p_perm = family_perm_spearman(per, "log10_P", ycol)
        corr_rows.append(
            {
                "x": "log10_P",
                "y": ycol,
                "n": 16,
                "rho": rho,
                "p_conventional": p,
                "p_family_perm": p_perm,
                "note": "descriptive pooled; rungs not IID across families",
            }
        )
    for ycol in ["f90_anis", "f_eff_anis"]:
        rho, p = spearmanr(per["A_log"], per[ycol])
        corr_rows.append(
            {
                "x": "A_log",
                "y": ycol,
                "n": 16,
                "rho": rho,
                "p_conventional": p,
                "p_family_perm": "",
                "note": "descriptive; magnitude vs concentration",
            }
        )
    rho_l, p_l = spearmanr(
        per["f90_anis"], per["fraction_directions_for_90pct_lift"]
    )
    corr_rows.append(
        {
            "x": "f90_anis",
            "y": "fraction_directions_for_90pct_lift",
            "n": 16,
            "rho": rho_l,
            "p_conventional": p_l,
            "p_family_perm": "",
            "note": "anisotropy concentration vs functional lift concentration (distinct)",
        }
    )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "pooled_correlations.csv", index=False)

    # -------- Figures --------
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Figure 1: concentration curves by family
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.6), sharey=True, constrained_layout=True)
    cmap = plt.cm.viridis
    for ax, fam in zip(axes, FAM_ORDER):
        g = per[per["family"] == fam].sort_values("log10_P")
        logs = g["log10_P"].to_numpy()
        lo, hi = logs.min(), logs.max()
        for _, r in g.iterrows():
            cstore = curve_store[r["model"]]
            if hi > lo:
                t = (r["log10_P"] - lo) / (hi - lo)
            else:
                t = 0.5
            color = cmap(0.15 + 0.7 * t)
            ax.plot(
                cstore["f"],
                cstore["C"],
                color=color,
                lw=1.4,
                label=str(r["size_name"]),
            )
        ax.axhline(0.90, color="0.75", lw=0.7, ls="--")
        for xv in (0.05, 0.10, 0.25, 0.50):
            ax.axvline(xv, color="0.9", lw=0.5, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(FAM_LABEL[fam])
        ax.set_xlabel(r"$r/d$")
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel(r"$C(r)$ anisotropy mass")
    fig.suptitle("Cumulative anisotropy concentration of $A_{\\mathrm{eff}}$", fontsize=10)
    p1 = FIG / "anisotropy_concentration_curves.png"
    fig.savefig(p1, dpi=200)
    fig.savefig(PAPER_FIG / "anisotropy_concentration_curves.png", dpi=200)
    fig.savefig(OUT / "anisotropy_concentration_curves.png", dpi=200)
    plt.close(fig)

    # Figure 2: vs scale
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    colors = {
        "astropt": "#4C72B0",
        "convnext": "#DD8452",
        "dinov2": "#55A868",
        "vit": "#C44E52",
        "ijepa": "#8172B3",
    }
    for ax, ycol, ylab in [
        (axes[0], "f90_anis", r"$f_{90}^{\mathrm{anis}}=r_{90}/d$"),
        (axes[1], "f_eff_anis", r"$f_{\mathrm{eff}}^{\mathrm{anis}}=r_{\mathrm{eff}}/d$"),
    ]:
        for fam in FAM_ORDER:
            g = per[per["family"] == fam].sort_values("log10_P")
            ax.plot(
                g["log10_P"],
                g[ycol],
                "o-",
                color=colors[fam],
                lw=1.3,
                markersize=5,
                label=FAM_LABEL[fam],
            )
        ax.set_xlabel(r"$\log_{10}P$")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_title("(a) 90% anisotropy mass")
    axes[1].set_title("(b) entropy effective fraction")
    p2 = FIG / "anisotropy_concentration_vs_scale.png"
    fig.savefig(p2, dpi=200)
    fig.savefig(PAPER_FIG / "anisotropy_concentration_vs_scale.png", dpi=200)
    fig.savefig(OUT / "anisotropy_concentration_vs_scale.png", dpi=200)
    plt.close(fig)

    # -------- Report --------
    f90 = per["f90_anis"]
    feff = per["f_eff_anis"]
    n_neg90 = int((trends["beta_f90"] < 0).sum())
    n_negeff = int((trends["beta_feff"] < 0).sum())
    rho90 = corr.loc[(corr.x == "log10_P") & (corr.y == "f90_anis")].iloc[0]
    rhoeff = corr.loc[(corr.x == "log10_P") & (corr.y == "f_eff_anis")].iloc[0]
    rhoA90 = corr.loc[(corr.x == "A_log") & (corr.y == "f90_anis")].iloc[0]
    rhoAlift = corr.loc[corr.y == "fraction_directions_for_90pct_lift"].iloc[0]

    report = f"""# Anisotropy concentration of Dense+Ridge effective maps

## Protocol

- Source: `{SPEC_DIR.relative_to(ROOT)}`
- Map: $A_{{\\mathrm{{eff}}}}=\\mathrm{{diag}}(\\sigma_y)W\\,\\mathrm{{diag}}(1/\\sigma_x)$ (Legacy→HSC, α=1, frozen split)
- Singular values: active set with relative floor `eps={EPS}` × σ_max (same as existing $A_{{\\log}}$)
- Log floor for numerical zeros: `{LOG_FLOOR}`
- **No Ridge refit**

## A. How concentrated is anisotropy?

Across the 16 maps, 90% of centered log-singular-value anisotropy mass is carried by
**{100*f90.min():.1f}–{100*f90.max():.1f}%** of singular directions
(median **{100*f90.median():.1f}%**; mean {100*f90.mean():.1f}%).

Entropy-effective fraction $f_{{\\mathrm{{eff}}}}^{{\\mathrm{{anis}}}}$:
range {100*feff.min():.1f}–{100*feff.max():.1f}% (median {100*feff.median():.1f}%).

## B. Does concentration change with model size?

Descriptive within-family OLS slopes of $f_{{90}}$ vs $\\log_{{10}}P$
(negative ⇒ more concentrated with size):

| family | n | β_f90 | Δ first→last f90 | β_feff | Δ first→last feff |
|---|---:|---:|---:|---:|---:|
"""
    for _, t in trends.iterrows():
        report += (
            f"| {t['family']} | {t['n_rungs']} | {t['beta_f90']:+.4f} | "
            f"{t['first_to_last_delta_f90']:+.4f} | {t['beta_feff']:+.4f} | "
            f"{t['first_to_last_delta_feff']:+.4f} |\n"
        )
    report += f"""
- Families with $β_{{f90}}<0$: **{n_neg90}/5**
- Families with $β_{{feff}}<0$: **{n_negeff}/5**

Pooled descriptive Spearman (n=16; not IID):
- $\\rho(\\log_{{10}}P, f_{{90}})$ = {rho90['rho']:.3f}, $p$={rho90['p_conventional']:.3f},
  $p_{{\\mathrm{{perm}}}}$={rho90['p_family_perm']:.3f} (family-preserving)
- $\\rho(\\log_{{10}}P, f_{{\\mathrm{{eff}}}})$ = {rhoeff['rho']:.3f}, $p$={rhoeff['p_conventional']:.3f},
  $p_{{\\mathrm{{perm}}}}$={rhoeff['p_family_perm']:.3f}

## C. Magnitude vs concentration

- $\\rho(A_{{\\log}}, f_{{90}})$ = {rhoA90['rho']:.3f}, $p$={rhoA90['p_conventional']:.3f}
- $A_{{\\log}}$ range: {per['A_log'].min():.3f}–{per['A_log'].max():.3f}

Strong overall anisotropy does **not** imply dispersed anisotropy (they are separate axes).

## D. Anisotropy concentration vs functional lift concentration

Existing truncated-SVD / transfer complexity: fraction of directions recovering ≥90% of
Dense+Ridge mKNN lift (`k90_transfer_frac` from effective_map_spectrum).

- Lift $f_{{90}}$ range: {100*per['fraction_directions_for_90pct_lift'].min():.1f}–{100*per['fraction_directions_for_90pct_lift'].max():.1f}%
  (median {100*per['fraction_directions_for_90pct_lift'].median():.1f}%)
- Anisotropy $f_{{90}}$ median {100*f90.median():.1f}% vs lift median
  {100*per['fraction_directions_for_90pct_lift'].median():.1f}%
- Spearman $\\rho(f_{{90}}^{{\\mathrm{{anis}}}}, f_{{90}}^{{\\mathrm{{lift}}}})$ =
  {rhoAlift['rho']:.3f}, $p$={rhoAlift['p_conventional']:.3f}

These measure different things; do not equate spectral anisotropy mass with mKNN-useful subspace.

## Parity and numerics

- Max $|A_{{\\log}}^{{\\mathrm{{existing}}}}-A_{{\\log}}^{{\\mathrm{{recomputed}}}}|$ = {max_diff:.3e}
- Inactive singular values (relative eps): {int(parity['n_inactive'].sum())} total across models
  (per-model max {int(parity['n_inactive'].max())})
- Active values floored at {LOG_FLOOR}: {int(parity['n_singular_values_floored'].sum())}

## Figures

- `{p1.relative_to(ROOT)}`
- `{p2.relative_to(ROOT)}`
"""
    (OUT / "REPORT.md").write_text(report)

    meta = {
        "source": str(SPEC_DIR.relative_to(ROOT)),
        "eps_active": EPS,
        "log_floor": LOG_FLOOR,
        "max_A_log_parity_abs_diff": max_diff,
        "n_models": 16,
        "f90_median": float(f90.median()),
        "f90_range": [float(f90.min()), float(f90.max())],
        "n_families_beta_f90_neg": n_neg90,
        "n_families_beta_feff_neg": n_negeff,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(report)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
