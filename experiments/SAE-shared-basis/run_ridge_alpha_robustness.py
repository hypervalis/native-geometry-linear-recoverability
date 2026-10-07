#!/usr/bin/env python3
"""Ridge alpha robustness for Dense+Ridge scaling statistic T(alpha)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

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
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


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
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> np.ndarray:
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    return y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
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
        else root / "outputs" / "ridge_alpha_robustness"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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

    X1_0, X2_0, _ = load_pair_arrays(root, pairs[names[0]], args.max_n, args.seed)
    n = len(X1_0)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    # dense slopes (alpha-independent)
    dense_scores = []
    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        params_m = float(cfg.get("approx_params_m", 1))
        logp = math.log10(max(params_m * 1e6, 1.0))
        m = mknn_pair(X2[test_idx], X1[test_idx], args.k, device, args.row_batch)
        dense_scores.append(
            {
                "model": name,
                "family": str(cfg["family"]),
                "log10_params": logp,
                "mknn_dense": m,
            }
        )
    dense_df = pd.DataFrame(dense_scores)
    beta_dense = {
        fam: family_ols_slope(
            dense_df.loc[dense_df["family"] == fam, "log10_params"].to_numpy(),
            dense_df.loc[dense_df["family"] == fam, "mknn_dense"].to_numpy(),
        )
        for fam in FAMILY_ORDER
    }

    rows = []
    summaries = []
    for alpha in args.alphas:
        alpha_rows = []
        for name in names:
            cfg = pairs[name]
            X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
            params_m = float(cfg.get("approx_params_m", 1))
            logp = math.log10(max(params_m * 1e6, 1.0))
            fam = str(cfg["family"])
            mapped = fit_ridge_map(X2, X1, train_idx, alpha=alpha)
            m = mknn_pair(mapped[test_idx], X1[test_idx], args.k, device, args.row_batch)
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "family": fam,
                    "model": name,
                    "log10_params": logp,
                    "mknn_dense": dense_df.loc[dense_df["model"] == name, "mknn_dense"].iloc[0],
                    "mknn_ridge": m,
                }
            )
        adf = pd.DataFrame(alpha_rows)
        fam_deltas = []
        for fam in FAMILY_ORDER:
            fs = adf[adf["family"] == fam]
            br = family_ols_slope(fs["log10_params"].to_numpy(), fs["mknn_ridge"].to_numpy())
            fam_deltas.append(br - beta_dense[fam])
        T = float(np.mean(fam_deltas))
        summaries.append(
            {
                "alpha": alpha,
                "T": T,
                "n_pos_families": int(sum(d > 0 for d in fam_deltas)),
                **{f"delta_beta_{f}": d for f, d in zip(FAMILY_ORDER, fam_deltas)},
            }
        )
        rows.extend(alpha_rows)
        print(f"alpha={alpha}: T={T:.5f} pos={sum(d>0 for d in fam_deltas)}/5", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "ridge_alpha_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "ridge_alpha_scaling.csv", index=False)
    (out_dir / "ridge_alpha_scaling.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
