#!/usr/bin/env python3
"""Final robustness: reverse direction + k-sensitivity + pooled theta.

Frozen protocol matching run_ridge_scaling_geometry.py:
  n=16384, seed=0, test_size=0.2, test-only gallery,
  StandardScaler + Ridge(alpha=1, fit_intercept=True).
"""
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

WS_ROOT = Path(__file__).resolve().parents[3]
FAMILY_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
K_GRID = (5, 10, 25, 50)
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


@torch.inference_mode()
def knn_pair(
    A: np.ndarray, B: np.ndarray, k: int, device: torch.device, row_batch: int
) -> tuple[np.ndarray, np.ndarray]:
    ta = torch.as_tensor(np.ascontiguousarray(A), device=device)
    tb = torch.as_tensor(np.ascontiguousarray(B), device=device)
    nn1 = knn_cos(ta, k, row_batch).cpu().numpy()
    nn2 = knn_cos(tb, k, row_batch).cpu().numpy()
    return nn1, nn2


def mknn_from_nn(nn1: np.ndarray, nn2: np.ndarray, k: int) -> float:
    a, b = nn1[:, :k], nn2[:, :k]
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


def fit_ridge_map(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, alpha: float
) -> np.ndarray:
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    return y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)


def slopes_and_T(df: pd.DataFrame, dense_col: str, ridge_col: str) -> dict:
    delta: dict[str, float] = {}
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        logp = sub["log10_params"].to_numpy(float)
        bd = family_ols_slope(logp, sub[dense_col].to_numpy(float))
        br = family_ols_slope(logp, sub[ridge_col].to_numpy(float))
        delta[fam] = br - bd
    vals = list(delta.values())
    excl = [delta[f] for f in delta if f != "ijepa"]
    n_pos = int(sum(v > 0 for v in vals))
    return {
        "delta_beta": delta,
        "T": float(np.mean(vals)),
        "T_excl_ijepa": float(np.mean(excl)),
        "n_positive": n_pos,
        "n_families": len(vals),
        "sign_p_one_sided": float(0.5 ** len(vals)),
    }


def pooled_theta(df_long: pd.DataFrame) -> dict:
    """M = a_F + g_F log10P + d_F R + theta (R*log10P) + e."""
    d = df_long.copy()
    d["R"] = (d["method"] == "Ridge").astype(float)
    d["RxlogP"] = d["R"] * d["log10_params"]
    fams = list(FAMILY_ORDER)
    cols = [np.ones((len(d), 1))]
    names = ["intercept"]
    for fam in fams[1:]:
        cols.append((d["family"] == fam).to_numpy(float)[:, None])
        names.append(f"fe_{fam}")
    for fam in fams:
        cols.append(
            ((d["family"] == fam).to_numpy(float) * d["log10_params"].to_numpy(float))[
                :, None
            ]
        )
        names.append(f"gamma_{fam}")
    for fam in fams:
        cols.append(
            ((d["family"] == fam).to_numpy(float) * d["R"].to_numpy(float))[:, None]
        )
        names.append(f"delta_{fam}")
    cols.append(d["RxlogP"].to_numpy(float)[:, None])
    names.append("theta")
    X = np.hstack(cols)
    y = d["mknn"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    ti = names.index("theta")
    theta = float(beta[ti])
    se_t = float(se[ti])
    z = theta / se_t if se_t > 0 else float("inf")
    p_two = float(math.erfc(abs(z) / math.sqrt(2)))
    return {
        "theta": theta,
        "se": se_t,
        "ci95_lo": theta - 1.96 * se_t,
        "ci95_hi": theta + 1.96 * se_t,
        "z": float(z),
        "p_two_sided": p_two,
        "n": int(n),
        "p_params": int(p),
        "sigma2": sigma2,
    }


def run_direction(
    data: dict[str, dict],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    direction: str,
    k_eval: tuple[int, ...],
    alpha: float,
    device: torch.device,
    row_batch: int,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    k_max = max(k_eval)
    rows = []
    for name, d in data.items():
        if direction == "L2H":
            X, Y = d["X_leg"], d["X_hsc"]
        elif direction == "H2L":
            X, Y = d["X_hsc"], d["X_leg"]
        else:
            raise ValueError(direction)
        mapped = fit_ridge_map(X, Y, train_idx, alpha=alpha)
        Xte, Yte, Mte = X[test_idx], Y[test_idx], mapped[test_idx]
        nn_xd, nn_yd = knn_pair(Xte, Yte, k_max, device, row_batch)
        nn_md, nn_ym = knn_pair(Mte, Yte, k_max, device, row_batch)
        row = {
            "family": d["family"],
            "model": name,
            "size_name": d["size_name"],
            "parameter_count": d["parameter_count"],
            "log10_params": d["log10_params"],
            "direction": direction,
        }
        for k in k_eval:
            row[f"mknn_dense_k{k}"] = mknn_from_nn(nn_xd, nn_yd, k)
            row[f"mknn_ridge_k{k}"] = mknn_from_nn(nn_md, nn_ym, k)
        rows.append(row)
        print(
            f"[{direction}] {name}: dense10={row['mknn_dense_k10']:.4f} "
            f"ridge10={row['mknn_ridge_k10']:.4f}",
            flush=True,
        )
    df = pd.DataFrame(rows).sort_values(["family", "log10_params"])
    stats_by_k = {
        k: slopes_and_T(df, f"mknn_dense_k{k}", f"mknn_ridge_k{k}") for k in k_eval
    }
    return df, stats_by_k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument("--pairs-yaml", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_root = (
        args.out_root.expanduser().resolve() if args.out_root else root / "outputs"
    )
    rev_dir = out_root / "reverse_direction_scaling"
    k_dir = out_root / "k_sensitivity"
    rev_dir.mkdir(parents=True, exist_ok=True)
    k_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_yaml is not None:
        pairs_path = args.pairs_yaml.expanduser().resolve()
    else:
        cands = [
            root / "tmp_ridge_geom/official_legacy_pairs.yaml",
            root
            / "configs/official_legacy_pairs.yaml",
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

    data: dict[str, dict] = {}
    n_ref = None
    for name in names:
        cfg = pairs[name]
        X1, X2 = load_pair_arrays(root, cfg, args.max_n, args.seed)
        if n_ref is None:
            n_ref = len(X1)
        assert len(X1) == n_ref, f"row mismatch {name}"
        params_m = float(
            cfg.get("approx_params_m", cfg.get("approx_params_millions", 1))
        )
        fam = FAM_ALIAS.get(str(cfg["family"]).lower(), str(cfg["family"]).lower())
        data[name] = {
            "X_hsc": X1.astype(np.float32),
            "X_leg": X2.astype(np.float32),
            "family": fam,
            "log10_params": math.log10(max(params_m * 1e6, 1.0)),
            "parameter_count": int(round(params_m * 1e6)),
            "size_name": cfg.get("size_name"),
        }
        print(
            f"loaded {name} d_leg={X2.shape[1]} d_hsc={X1.shape[1]} fam={fam}",
            flush=True,
        )

    n = int(n_ref)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"n={n} train={len(train_idx)} test={len(test_idx)}", flush=True)

    t0 = time.time()
    fwd_df, fwd_stats = run_direction(
        data,
        train_idx,
        test_idx,
        direction="L2H",
        k_eval=K_GRID,
        alpha=args.alpha,
        device=device,
        row_batch=args.row_batch,
    )
    fwd_df.to_csv(k_dir / "forward_scores_by_k.csv", index=False)

    rev_df, rev_stats = run_direction(
        data,
        train_idx,
        test_idx,
        direction="H2L",
        k_eval=K_GRID,
        alpha=args.alpha,
        device=device,
        row_batch=args.row_batch,
    )
    rev_out = rev_df[
        [
            "family",
            "model",
            "parameter_count",
            "log10_params",
            "mknn_dense_k10",
            "mknn_ridge_k10",
        ]
    ].rename(
        columns={
            "mknn_dense_k10": "raw_dense_mknn",
            "mknn_ridge_k10": "ridge_mknn",
        }
    )
    rev_out.to_csv(rev_dir / "reverse_scores.csv", index=False)
    rev_df.to_csv(rev_dir / "reverse_scores_by_k.csv", index=False)

    def binomial_sign_p(n_pos: int, n_fam: int) -> float:
        return sum(math.comb(n_fam, k) for k in range(n_pos, n_fam + 1)) / (2**n_fam)

    k_rows = []
    for k, st in fwd_stats.items():
        k_rows.append(
            {
                "k": k,
                "T": st["T"],
                "positive_families": st["n_positive"],
                "n_families": st["n_families"],
                "T_without_IJEPA": st["T_excl_ijepa"],
                "sign_p": binomial_sign_p(st["n_positive"], st["n_families"]),
                **{f"delta_beta_{fam}": st["delta_beta"].get(fam) for fam in FAMILY_ORDER},
            }
        )
    pd.DataFrame(k_rows).to_csv(k_dir / "k_scaling_summary.csv", index=False)

    fwd10 = fwd_stats[10]
    rev10 = rev_stats[10]
    fam_rows = []
    for fam in FAMILY_ORDER:
        d_f = fwd10["delta_beta"][fam]
        d_r = rev10["delta_beta"][fam]
        fam_rows.append(
            {
                "family": fam,
                "delta_beta_L2H": d_f,
                "delta_beta_H2L": d_r,
                "delta_beta_bi_mean": 0.5 * (d_f + d_r),
            }
        )
    pd.DataFrame(fam_rows).to_csv(
        rev_dir / "family_delta_beta_bidirectional.csv", index=False
    )

    def long_frame(df: pd.DataFrame, k: int) -> pd.DataFrame:
        rows = []
        for rec in df.to_dict(orient="records"):
            for method, col in (
                ("Dense", f"mknn_dense_k{k}"),
                ("Ridge", f"mknn_ridge_k{k}"),
            ):
                rows.append(
                    {
                        "family": rec["family"],
                        "log10_params": rec["log10_params"],
                        "method": method,
                        "mknn": rec[col],
                    }
                )
        return pd.DataFrame(rows)

    theta_fwd = pooled_theta(long_frame(fwd_df, 10))
    theta_rev = pooled_theta(long_frame(rev_df, 10))
    (k_dir / "pooled_regression_theta.json").write_text(
        json.dumps(theta_fwd, indent=2) + "\n"
    )
    (rev_dir / "pooled_regression_theta_reverse.json").write_text(
        json.dumps(theta_rev, indent=2) + "\n"
    )

    n_test = len(test_idx)
    summary = {
        "T_L2H": fwd10["T"],
        "T_H2L": rev10["T"],
        "T_bi_descriptive": float(np.mean([fwd10["T"], rev10["T"]])),
        "n_pos_L2H": fwd10["n_positive"],
        "n_pos_H2L": rev10["n_positive"],
        "T_L2H_excl_ijepa": fwd10["T_excl_ijepa"],
        "T_H2L_excl_ijepa": rev10["T_excl_ijepa"],
        "sign_p_L2H": binomial_sign_p(fwd10["n_positive"], fwd10["n_families"]),
        "sign_p_H2L_correct_4of5": binomial_sign_p(
            rev10["n_positive"], rev10["n_families"]
        ),
        # slopes_and_T reports 0.5**n_families, which is the 5/5 tail even
        # when only 4 families are positive. Kept so the released summary
        # can be compared with that earlier figure.
        "sign_p_H2L_script_bug_used_0.5**5": fwd10["sign_p_one_sided"],
        "delta_beta_L2H": fwd10["delta_beta"],
        "delta_beta_H2L": rev10["delta_beta"],
        "mean_dense_L2H_k10": float(fwd_df["mknn_dense_k10"].mean()),
        "mean_ridge_L2H_k10": float(fwd_df["mknn_ridge_k10"].mean()),
        "mean_dense_H2L_k10": float(rev_df["mknn_dense_k10"].mean()),
        "mean_ridge_H2L_k10": float(rev_df["mknn_ridge_k10"].mean()),
        "chance_mknn_k10": 10 / (n_test - 1),
        "elapsed_s": time.time() - t0,
    }
    (rev_dir / "bidirectional_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (rev_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "n": n,
                "n_train": int(len(train_idx)),
                "n_test": int(n_test),
                "seed": args.seed,
                "alpha": args.alpha,
                "k_grid": list(K_GRID),
                "pairs_yaml": str(pairs_path),
                "elapsed_s": summary["elapsed_s"],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("T_L2H", "T_H2L")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
