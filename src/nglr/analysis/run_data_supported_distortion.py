#!/usr/bin/env python3
"""Data-supported local stretch distortion of the effective Dense+Ridge map.

Measures anisotropic deformation on held-out Legacy local edges:
  s_ij = log( |A_eff δ_ij| / |δ_ij| )
where A_eff = diag(σ_y) W diag(1/σ_x) in original coordinates.
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

ROOT = Path(__file__).resolve().parents[3]
FAMILY_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def load_col(path: Path, col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[col])
    return np.vstack(table.column(0).to_pylist()).astype(np.float64)


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


def effective_linear_map(
    W: np.ndarray, x_sc: StandardScaler, y_sc: StandardScaler
) -> np.ndarray:
    sx = np.asarray(x_sc.scale_, dtype=np.float64)
    sy = np.asarray(y_sc.scale_, dtype=np.float64)
    sx = np.where(sx > 0, sx, 1.0)
    sy = np.where(sy > 0, sy, 1.0)
    return (sy[:, None] * W) / sx[None, :]


def fit_ridge_A(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, alpha: float
) -> np.ndarray:
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    W = np.asarray(ridge.coef_, dtype=np.float64)
    return effective_linear_map(W, x_sc, y_sc)


def spectrum_metrics(s: np.ndarray) -> dict[str, float]:
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return {"A_log": float("nan"), "D_sim": float("nan"), "H_norm": float("nan")}
    tilde = s / np.exp(np.mean(np.log(s)))
    A_log = float(np.std(np.log(tilde), ddof=0))
    c = float(s.mean())
    D_sim = float(np.linalg.norm(s - c) / max(np.linalg.norm(s), 1e-300))
    p = (s**2) / max((s**2).sum(), 1e-300)
    H = -float(np.sum(p * np.log(p + 1e-300)))
    H_norm = H / math.log(len(s)) if len(s) > 1 else float("nan")
    return {"A_log": A_log, "D_sim": D_sim, "H_norm": H_norm}


def family_ols_slope(logp: np.ndarray, y: np.ndarray) -> float:
    if len(logp) < 2:
        return float("nan")
    lr = LinearRegression()
    lr.fit(logp.reshape(-1, 1), y)
    return float(lr.coef_[0])


@torch.inference_mode()
def knn_indices_cos(X: np.ndarray, k: int, device: torch.device, row_batch: int) -> np.ndarray:
    Z = torch.as_tensor(np.ascontiguousarray(X), device=device, dtype=torch.float32)
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
    return out.cpu().numpy()


def local_stretch_stats(
    X: np.ndarray,
    A: np.ndarray,
    nn_idx: np.ndarray,
    *,
    compute_angle: bool = False,
) -> tuple[dict[str, float], np.ndarray]:
    """Edge stretch s_ij and optional angular distortion on Legacy test points."""
    n, k = nn_idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.reshape(-1)
    D = X[cols] - X[rows]
    norms = np.linalg.norm(D, axis=1)
    good = norms > 1e-12
    D = D[good]
    norms = norms[good]
    AD = D @ A.T
    s_arr = np.log(np.linalg.norm(AD, axis=1) / norms)
    angle_errs: list[float] = []
    if compute_angle and k >= 2:
        for i in range(n):
            xi = X[i]
            for a in range(min(3, k - 1)):
                for b in range(a + 1, min(a + 4, k)):
                    j, k2 = nn_idx[i, a], nn_idx[i, b]
                    u = X[j] - xi
                    v = X[k2] - xi
                    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                    if nu < 1e-12 or nv < 1e-12:
                        continue
                    cu = float(np.dot(u, v) / (nu * nv))
                    Au, Av = A @ u, A @ v
                    nAu, nAv = np.linalg.norm(Au), np.linalg.norm(Av)
                    if nAu < 1e-12 or nAv < 1e-12:
                        continue
                    angle_errs.append(abs(float(np.dot(Au, Av) / (nAu * nAv)) - cu))

    out = {
        "n_edges": int(len(s_arr)),
        "sigma_local": float(np.std(s_arr, ddof=0)) if len(s_arr) else float("nan"),
        "mean_s": float(np.mean(s_arr)) if len(s_arr) else float("nan"),
        "R_local_q90_q10": float(np.quantile(s_arr, 0.90) - np.quantile(s_arr, 0.10))
        if len(s_arr)
        else float("nan"),
        "E_angle": float(np.mean(angle_errs)) if angle_errs else float("nan"),
    }
    return out, s_arr


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def plot_distortion(summary: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam in FAMILY_ORDER:
        sub = summary[summary["family"] == fam].sort_values("log10_params")
        if sub.empty:
            continue
        ax.plot(
            sub["log10_params"],
            sub["sigma_local"],
            "o-",
            label=FAMILY_LABEL[fam],
            ms=5,
        )
    ax.set_xlabel(r"$\log_{10}P$")
    ax.set_ylabel(r"$\sigma_{\mathrm{local}}$ (log stretch std)")
    ax.set_title("Data-supported local distortion vs model size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--k-edge", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    ap.add_argument("--compute-angle", action="store_true")
    ap.add_argument("--save-edges", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "data_supported_distortion"
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

    X1_0, X2_0, _ = load_pair_arrays(root, pairs[names[0]], args.max_n, args.seed)
    n = len(X1_0)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    summary_rows = []
    edge_rows = []
    data: dict[str, dict] = {}
    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        assert len(X1) == n
        params_m = float(cfg.get("approx_params_m", 1))
        data[name] = {
            "X_hsc": X1,
            "X_leg": X2,
            "family": str(cfg["family"]),
            "log10_params": math.log10(max(params_m * 1e6, 1.0)),
            "parameter_count": int(round(params_m * 1e6)),
        }
        print(f"loaded {name} d={X2.shape[1]}", flush=True)

    for name, d in data.items():
        X1, X2 = d["X_hsc"], d["X_leg"]
        logp = d["log10_params"]
        fam = d["family"]
        A = fit_ridge_A(X2, X1, train_idx, args.alpha)
        spec = spectrum_metrics(np.linalg.svd(A, compute_uv=False))
        Xte = X2[test_idx]

        for k_edge in args.k_edge:
            nn = knn_indices_cos(Xte, k_edge, device, args.row_batch)
            stats, s_arr = local_stretch_stats(
                Xte, A, nn, compute_angle=args.compute_angle
            )
            summary_rows.append(
                {
                    "family": fam,
                    "model": name,
                    "parameter_count": d["parameter_count"],
                    "log10_params": logp,
                    "k_edge": k_edge,
                    **stats,
                    **spec,
                }
            )
            if args.save_edges and len(s_arr) <= 500_000:
                for val in s_arr[:: max(1, len(s_arr) // 20000)]:
                    edge_rows.append(
                        {
                            "model": name,
                            "k_edge": k_edge,
                            "s_ij": float(val),
                        }
                    )
            print(
                f"{name} k={k_edge}: sigma={stats['sigma_local']:.4f} "
                f"R={stats['R_local_q90_q10']:.4f} A_log={spec['A_log']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "local_edge_summary.csv", index=False)
    if edge_rows:
        pd.DataFrame(edge_rows).to_csv(out_dir / "local_edge_distortion.csv", index=False)

    # family slopes for primary k=10
    prim = summary[summary["k_edge"] == args.k_edge[0]]
    slope_rows = []
    for fam in FAMILY_ORDER:
        fs = prim[prim["family"] == fam]
        if len(fs) < 2:
            continue
        slope_rows.append(
            {
                "family": fam,
                "slope_sigma_local": family_ols_slope(
                    fs["log10_params"].to_numpy(), fs["sigma_local"].to_numpy()
                ),
                "n_rungs": len(fs),
            }
        )
    pd.DataFrame(slope_rows).to_csv(out_dir / "sigma_local_size_slopes.csv", index=False)

    corr_rows = []
    for col in ["A_log", "D_sim", "H_norm"]:
        corr_rows.append(
            {
                "x": col,
                "y": "sigma_local",
                "spearman": spearman(prim[col].to_numpy(), prim["sigma_local"].to_numpy()),
                "n": len(prim),
                "k_edge": args.k_edge[0],
            }
        )
    pd.DataFrame(corr_rows).to_csv(out_dir / "sigma_local_vs_spectrum.csv", index=False)

    plot_distortion(prim, fig_dir / "local_edge_distortion_vs_size.png")

    meta = {
        "map": "A_eff = diag(sy) W diag(1/sx)",
        "edges": "Legacy test-only kNN within test gallery",
        "k_edge": args.k_edge,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
