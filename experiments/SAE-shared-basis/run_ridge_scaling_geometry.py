#!/usr/bin/env python3
"""Ridge scaling amplification statistics + Dense→Dense map geometry.

Does not overwrite paper_alignment_controls outputs. Writes under
outputs/ridge_scaling_geometry/.

Primary pre-registered statistic:
  T = mean over families of (β_Dense+Ridge − β_Dense)

Split / Ridge recipe match run_alignment_controls.py:
  train_test_split(test_size=0.2, random_state=0), Legacy→HSC (col2→col1),
  StandardScaler(X,Y) + Ridge(alpha=1, fit_intercept=True), test-only gallery.
"""
from __future__ import annotations

import argparse
import json
import math
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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SIGMA_ACTIVE_FRAC = 1e-8
K_RESID = (10, 25, 50)
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
def mknn_per_query(
    A: np.ndarray, B: np.ndarray, k: int, device: torch.device, row_batch: int
) -> np.ndarray:
    ta = torch.as_tensor(np.ascontiguousarray(A), device=device)
    tb = torch.as_tensor(np.ascontiguousarray(B), device=device)
    nn1 = knn_cos(ta, k, row_batch).cpu().numpy()
    nn2 = knn_cos(tb, k, row_batch).cpu().numpy()
    n = len(nn1)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = len(set(nn1[i]) & set(nn2[i])) / k
    return out


def effective_linear_map(
    W: np.ndarray, x_sc: StandardScaler, y_sc: StandardScaler
) -> np.ndarray:
    """A = diag(σy) W diag(1/σx) — end-to-end linear map in original coords."""
    sx = np.asarray(x_sc.scale_, dtype=np.float64)
    sy = np.asarray(y_sc.scale_, dtype=np.float64)
    sx = np.where(sx > 0, sx, 1.0)
    sy = np.where(sy > 0, sy, 1.0)
    return (sy[:, None] * W) / sx[None, :]


def fit_ridge_full(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    alpha: float,
    y_train_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mapped_all, A_eff, intercept_scaled).

    A_eff is the end-to-end linear map after undoing StandardScaler
    (not Ridge.coef_ in standardized coordinates).
    """
    x_tr = x[train_idx]
    y_tr = y[train_idx] if y_train_override is None else y_train_override
    x_sc = StandardScaler().fit(x_tr)
    y_sc = StandardScaler().fit(y_tr)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr))
    mapped = y_sc.inverse_transform(ridge.predict(x_sc.transform(x))).astype(np.float32)
    W = np.asarray(ridge.coef_, dtype=np.float64)
    A = effective_linear_map(W, x_sc, y_sc)
    b_sc = np.asarray(ridge.intercept_, dtype=np.float64)
    return mapped, A, b_sc


def orthogonal_procrustes_map(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray
) -> np.ndarray:
    x_tr = x[train_idx]
    y_tr = y[train_idx]
    mx = x_tr.mean(axis=0)
    my = y_tr.mean(axis=0)
    d = min(x_tr.shape[1], y_tr.shape[1])
    Xc = x_tr[:, :d] - mx[:d]
    Yc = y_tr[:, :d] - my[:d]
    M = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    return (((x[:, :d] - mx[:d]) @ Q) + my[:d]).astype(np.float32)


def spectrum_metrics(sing: np.ndarray) -> dict[str, float]:
    s = np.asarray(sing, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return {
            "n_active": 0,
            "A_log": float("nan"),
            "kappa_raw": float("nan"),
            "kappa_95_5": float("nan"),
            "D_sim": float("nan"),
            "geom_mean": float("nan"),
            "stable_rank": float("nan"),
            "sigma_max": float("nan"),
            "sigma_min_active": float("nan"),
        }
    smax = float(s.max())
    active = s[s >= SIGMA_ACTIVE_FRAC * smax]
    if active.size == 0:
        active = s[:1]
    log_s = np.log(active)
    geom = float(np.exp(np.mean(log_s)))
    tilde = active / geom
    A_log = float(np.std(np.log(tilde), ddof=0))
    c = float(np.mean(active))
    D_sim = float(np.linalg.norm(active - c) / (np.linalg.norm(active) + 1e-300))
    q95, q05 = np.quantile(active, [0.95, 0.05])
    s2 = active**2
    return {
        "n_active": int(active.size),
        "A_log": A_log,
        "kappa_raw": float(active.max() / max(active.min(), 1e-300)),
        "kappa_95_5": float(q95 / max(q05, 1e-300)),
        "D_sim": D_sim,
        "geom_mean": geom,
        "stable_rank": float((s2.sum() ** 2) / max((s2**2).sum(), 1e-300)),
        "sigma_max": float(active.max()),
        "sigma_min_active": float(active.min()),
    }


def residual_locality_from_nn(nn: np.ndarray, resid: np.ndarray) -> float:
    R = resid / (np.linalg.norm(resid, axis=1, keepdims=True) + 1e-12)
    n = len(R)
    acc = 0.0
    for i in range(n):
        acc += float(np.mean(R[i] @ R[nn[i]].T))
    return acc / n


def family_slopes_from_scores(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        out[fam] = family_ols_slope(
            sub["log10_params"].to_numpy(float), sub[score_col].to_numpy(float)
        )
    return out


def adjacent_deltas(df: pd.DataFrame, score_col: str) -> list[float]:
    deltas: list[float] = []
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        deltas.extend(np.diff(sub[score_col].to_numpy(float)).tolist())
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / "platonic-universe")
    ap.add_argument(
        "--pairs-yaml",
        type=Path,
        default=None,
        help="defaults to <root>/experiments/.../official_legacy_pairs.yaml "
        "or workspace copy",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-n", type=int, default=16384)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--row-batch", type=int, default=2048)
    ap.add_argument("--b-perm", type=int, default=200)
    ap.add_argument("--b-boot", type=int, default=5000)
    ap.add_argument("--b-resid-null", type=int, default=200)
    ap.add_argument("--skip-perm", action="store_true")
    ap.add_argument("--skip-boot", action="store_true")
    ap.add_argument("--skip-resid", action="store_true")
    ap.add_argument("--skip-procrustes", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else root / "outputs" / "ridge_scaling_geometry"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "paper_working" / "figures"
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

    # Shared train/test from first pair (identical n=16384 object rows)
    X1_0, X2_0, _ = load_pair_arrays(root, pairs[names[0]], args.max_n, args.seed)
    n = len(X1_0)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    n_train, n_test = len(train_idx), len(test_idx)
    print(
        f"n={n} train={n_train} test={n_test} device={device} "
        f"B_perm={args.b_perm} B_boot={args.b_boot} pairs={len(names)}",
        flush=True,
    )

    data: dict[str, dict] = {}
    for name in names:
        cfg = pairs[name]
        X1, X2, _ = load_pair_arrays(root, cfg, args.max_n, args.seed)
        assert len(X1) == n, f"row count mismatch {name}: {len(X1)} vs {n}"
        params_m = float(cfg.get("approx_params_m", 1))
        data[name] = {
            "cfg": cfg,
            "X_leg": X2.astype(np.float32),
            "X_hsc": X1.astype(np.float32),
            "family": str(cfg["family"]),
            "params_m": params_m,
            "log10_params": math.log10(max(params_m * 1e6, 1.0)),
            "parameter_count": int(round(params_m * 1e6)),
            "size_name": cfg.get("size_name"),
        }
        print(f"loaded {name} d={X2.shape[1]} fam={cfg['family']}", flush=True)

    meta = {
        "seed": args.seed,
        "test_size": args.test_size,
        "n": n,
        "n_train": n_train,
        "n_test": n_test,
        "k": args.k,
        "ridge_alpha": args.alpha,
        "scaler": "StandardScaler on X and Y train",
        "direction": "Legacy→HSC (col2→col1)",
        "gallery": "test-only",
        "sigma_active_frac": SIGMA_ACTIVE_FRAC,
        "primary_statistic": "T = mean_F (β_Dense+Ridge_F - β_Dense_F)",
        "b_perm": args.b_perm,
        "b_boot": args.b_boot,
        "synchronized_permutation": True,
        "pairs_yaml": str(pairs_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    score_rows = []
    sing_rows = []
    geom_rows = []
    resid_rows = []
    per_query: dict[str, dict[str, np.ndarray]] = {}

    for name, d in data.items():
        X, Y = d["X_leg"], d["X_hsc"]
        mapped, A, b_sc = fit_ridge_full(X, Y, train_idx, alpha=args.alpha)
        resid = (Y - mapped)[test_idx]
        svals = np.linalg.svd(A, compute_uv=False)
        for i, s in enumerate(svals, start=1):
            sing_rows.append(
                {
                    "family": d["family"],
                    "model": name,
                    "parameter_count": d["parameter_count"],
                    "rank_index": i,
                    "singular_value": float(s),
                }
            )
        sm = spectrum_metrics(svals)
        geom_rows.append(
            {
                "family": d["family"],
                "model": name,
                "parameter_count": d["parameter_count"],
                "log10_params": d["log10_params"],
                "intercept_l2": float(np.linalg.norm(b_sc)),
                **sm,
            }
        )

        Xte, Yte, Mte = X[test_idx], Y[test_idx], mapped[test_idx]
        q_dense = mknn_per_query(Xte, Yte, args.k, device, args.row_batch)
        q_ridge = mknn_per_query(Mte, Yte, args.k, device, args.row_batch)
        m_dense, m_ridge = float(q_dense.mean()), float(q_ridge.mean())
        per_query[name] = {"dense": q_dense, "ridge": q_ridge}

        m_proc = float("nan")
        if not args.skip_procrustes:
            proc = orthogonal_procrustes_map(X, Y, train_idx)
            q_proc = mknn_per_query(proc[test_idx], Yte, args.k, device, args.row_batch)
            m_proc = float(q_proc.mean())
            per_query[name]["procrustes"] = q_proc

        score_rows.append(
            {
                "family": d["family"],
                "model": name,
                "size_name": d["size_name"],
                "parameter_count": d["parameter_count"],
                "log10_params": d["log10_params"],
                "mknn_dense": m_dense,
                "mknn_dense_ridge": m_ridge,
                "mknn_procrustes": m_proc,
                "lift_ridge": m_ridge - m_dense,
                "lift_proc": m_proc - m_dense if np.isfinite(m_proc) else float("nan"),
                "gap_ridge_minus_proc": (
                    m_ridge - m_proc if np.isfinite(m_proc) else float("nan")
                ),
            }
        )
        print(
            f"{name}: dense={m_dense:.4f} ridge={m_ridge:.4f} proc={m_proc:.4f} "
            f"A_log={sm['A_log']:.4f} D_sim={sm['D_sim']:.4f}",
            flush=True,
        )

        if not args.skip_resid:
            tx = torch.as_tensor(np.ascontiguousarray(Xte), device=device)
            for kk in K_RESID:
                nn = knn_cos(tx, kk, args.row_batch).cpu().numpy()
                r_obs = residual_locality_from_nn(nn, resid)
                rng_r = np.random.default_rng(args.seed + 17_000 + hash(name) % 10_000 + kk)
                null_vals = np.empty(args.b_resid_null, dtype=np.float64)
                for b in range(args.b_resid_null):
                    null_vals[b] = residual_locality_from_nn(
                        nn, resid[rng_r.permutation(n_test)]
                    )
                p_r = (1 + np.sum(null_vals >= r_obs)) / (args.b_resid_null + 1)
                resid_rows.append(
                    {
                        "family": d["family"],
                        "model": name,
                        "parameter_count": d["parameter_count"],
                        "log10_params": d["log10_params"],
                        "k": kk,
                        "R_local": r_obs,
                        "null_mean": float(null_vals.mean()),
                        "null_sd": float(null_vals.std()),
                        "null_p": float(p_r),
                    }
                )

    scores = pd.DataFrame(score_rows).sort_values(["family", "log10_params"])
    scores.to_csv(out_dir / "rung_scores.csv", index=False)
    pd.DataFrame(sing_rows).to_csv(out_dir / "ridge_singular_values.csv", index=False)
    geom = pd.DataFrame(geom_rows).sort_values(["family", "log10_params"])
    geom.to_csv(out_dir / "ridge_geometry_metrics.csv", index=False)
    if resid_rows:
        pd.DataFrame(resid_rows).to_csv(out_dir / "residual_locality.csv", index=False)
    if not args.skip_procrustes:
        scores[
            [
                "family",
                "model",
                "parameter_count",
                "mknn_dense",
                "mknn_procrustes",
                "mknn_dense_ridge",
            ]
        ].to_csv(out_dir / "procrustes_scores.csv", index=False)

    np.savez_compressed(
        out_dir / "per_query_mknn.npz",
        test_idx=test_idx,
        **{f"{name}__dense": per_query[name]["dense"] for name in per_query},
        **{f"{name}__ridge": per_query[name]["ridge"] for name in per_query},
    )

    beta_d = family_slopes_from_scores(scores, "mknn_dense")
    beta_r = family_slopes_from_scores(scores, "mknn_dense_ridge")
    delta_beta = {f: beta_r[f] - beta_d[f] for f in beta_d}
    T_real = float(np.mean(list(delta_beta.values())))
    adj_d = adjacent_deltas(scores, "mknn_dense")
    adj_r = adjacent_deltas(scores, "mknn_dense_ridge")
    D_align = [ar - ad for ar, ad in zip(adj_r, adj_d)]
    mean_D = float(np.mean(D_align))
    n_pos_D = int(np.sum(np.asarray(D_align) > 0))
    sign_p = (0.5) ** len(delta_beta)

    family_stats = pd.DataFrame(
        [
            {
                "family": f,
                "beta_dense": beta_d[f],
                "beta_dense_ridge": beta_r[f],
                "delta_beta": delta_beta[f],
                "n_rungs": int((scores["family"] == f).sum()),
                "two_point_only": bool((scores["family"] == f).sum() == 2),
            }
            for f in FAMILY_ORDER
            if f in delta_beta
        ]
    )
    family_stats.to_csv(out_dir / "family_delta_beta.csv", index=False)
    summary_real = {
        "T_real_mean_family_delta_beta": T_real,
        "median_family_delta_beta": float(np.median(list(delta_beta.values()))),
        "n_positive_family_delta_beta": int(sum(v > 0 for v in delta_beta.values())),
        "n_families": len(delta_beta),
        "sign_test_one_sided_p": sign_p,
        "mean_adjacent_D_align": mean_D,
        "n_positive_adjacent_D": n_pos_D,
        "n_adjacent": len(D_align),
        "delta_beta_by_family": delta_beta,
    }
    (out_dir / "real_scaling_summary.json").write_text(
        json.dumps(summary_real, indent=2) + "\n"
    )
    print(f"T_real={T_real:.6f} sign_p={sign_p} mean_D={mean_D:.6f}", flush=True)

    # Synchronized permutation null
    p_perm = float("nan")
    T_null = np.array([])
    if not args.skip_perm:
        print(f"Running synchronized perm null B={args.b_perm} ...", flush=True)
        train_Y = {name: data[name]["X_hsc"][train_idx] for name in data}
        train_X = {name: data[name]["X_leg"][train_idx] for name in data}
        test_X = {name: data[name]["X_leg"][test_idx] for name in data}
        test_Y = {name: data[name]["X_hsc"][test_idx] for name in data}
        # Precompute test HSC knn (unchanged under train shuffle)
        nn_hsc = {
            name: knn_cos(
                torch.as_tensor(test_Y[name], device=device), args.k, args.row_batch
            )
            for name in data
        }

        perm_rows = []
        for b in range(args.b_perm):
            rng_b = np.random.default_rng(10_000 + b)
            shuf = rng_b.permutation(n_train)
            rung_scores = []
            for name in data:
                x_tr = train_X[name]
                y_tr_s = train_Y[name][shuf]
                x_sc = StandardScaler().fit(x_tr)
                y_sc = StandardScaler().fit(y_tr_s)
                ridge = Ridge(alpha=args.alpha, fit_intercept=True)
                ridge.fit(x_sc.transform(x_tr), y_sc.transform(y_tr_s))
                m_te = y_sc.inverse_transform(
                    ridge.predict(x_sc.transform(test_X[name]))
                ).astype(np.float32)
                nn_m = knn_cos(
                    torch.as_tensor(m_te, device=device), args.k, args.row_batch
                )
                mknn_b = mknn(nn_m, nn_hsc[name], args.k)
                rung_scores.append(
                    {
                        "family": data[name]["family"],
                        "model": name,
                        "log10_params": data[name]["log10_params"],
                        "mknn": mknn_b,
                    }
                )
            sdf = pd.DataFrame(rung_scores)
            beta_s: dict[str, float] = {}
            for fam in FAMILY_ORDER:
                sub = sdf[sdf["family"] == fam].sort_values("log10_params")
                if len(sub) < 2:
                    continue
                beta_s[fam] = family_ols_slope(
                    sub["log10_params"].to_numpy(float), sub["mknn"].to_numpy(float)
                )
            dbeta = {f: beta_s[f] - beta_d[f] for f in beta_d}
            T_b = float(np.mean(list(dbeta.values())))
            adj_s: list[float] = []
            for fam in FAMILY_ORDER:
                sub = sdf[sdf["family"] == fam].sort_values("log10_params")
                if len(sub) < 2:
                    continue
                adj_s.extend(np.diff(sub["mknn"].to_numpy(float)).tolist())
            D_b = [a - d0 for a, d0 in zip(adj_s, adj_d)]
            perm_rows.append(
                {
                    "perm_id": b,
                    "T": T_b,
                    "median_delta_beta": float(np.median(list(dbeta.values()))),
                    "n_pos_family": int(sum(v > 0 for v in dbeta.values())),
                    "mean_D": float(np.mean(D_b)),
                    "n_pos_D": int(np.sum(np.asarray(D_b) > 0)),
                    **{f"delta_beta_{f}": dbeta[f] for f in dbeta},
                }
            )
            if (b + 1) % 20 == 0 or b == 0:
                print(f"  perm {b+1}/{args.b_perm} T_b={T_b:.5f}", flush=True)

        perm_df = pd.DataFrame(perm_rows)
        perm_df.to_csv(out_dir / "scaling_permutation_null.csv", index=False)
        T_null = perm_df["T"].to_numpy(float)
        p_perm = (1 + np.sum(T_null >= T_real)) / (args.b_perm + 1)
        perm_summary = pd.DataFrame(
            [
                {
                    "statistic": "T_mean_family_delta_beta",
                    "T_real": T_real,
                    "null_mean": float(T_null.mean()),
                    "null_sd": float(T_null.std()),
                    "null_p2.5": float(np.quantile(T_null, 0.025)),
                    "null_p50": float(np.quantile(T_null, 0.5)),
                    "null_p97.5": float(np.quantile(T_null, 0.975)),
                    "p_perm": float(p_perm),
                    "B": args.b_perm,
                    "mean_D_real": mean_D,
                    "mean_D_null_mean": float(perm_df["mean_D"].mean()),
                    "n_pos_D_real": n_pos_D,
                    "n_pos_D_null_mean": float(perm_df["n_pos_D"].mean()),
                }
            ]
        )
        perm_summary.to_csv(out_dir / "scaling_permutation_summary.csv", index=False)
        print(f"perm null: mean={T_null.mean():.5f} p={p_perm:.4g}", flush=True)

    # Object bootstrap
    boot_summary: dict = {}
    if not args.skip_boot:
        print(f"Object bootstrap B={args.b_boot} ...", flush=True)
        boot_T = np.empty(args.b_boot, dtype=np.float64)
        rng_boot = np.random.default_rng(args.seed + 42_000)
        name_list = list(per_query.keys())
        for b in range(args.b_boot):
            bi = rng_boot.integers(0, n_test, size=n_test)
            rows = [
                {
                    "family": data[name]["family"],
                    "model": name,
                    "log10_params": data[name]["log10_params"],
                    "mknn_dense": float(per_query[name]["dense"][bi].mean()),
                    "mknn_dense_ridge": float(per_query[name]["ridge"][bi].mean()),
                }
                for name in name_list
            ]
            bdf = pd.DataFrame(rows)
            bd = family_slopes_from_scores(bdf, "mknn_dense")
            br = family_slopes_from_scores(bdf, "mknn_dense_ridge")
            boot_T[b] = float(np.mean([br[f] - bd[f] for f in bd]))
            if (b + 1) % 1000 == 0:
                print(f"  boot {b+1}/{args.b_boot}", flush=True)
        boot_df = pd.DataFrame({"boot_id": np.arange(args.b_boot), "T": boot_T})
        boot_df.to_csv(out_dir / "object_bootstrap_scaling.csv", index=False)
        ci = np.quantile(boot_T, [0.025, 0.5, 0.975])
        boot_summary = {
            "T_real": T_real,
            "boot_mean": float(boot_T.mean()),
            "ci_2.5": float(ci[0]),
            "ci_50": float(ci[1]),
            "ci_97.5": float(ci[2]),
            "B": args.b_boot,
        }
        (out_dir / "object_bootstrap_summary.json").write_text(
            json.dumps(boot_summary, indent=2) + "\n"
        )
        print(f"boot CI [{ci[0]:.5f}, {ci[2]:.5f}]", flush=True)

    # Geometry vs size
    geom_trends = []
    for fam in FAMILY_ORDER:
        sub = geom[geom["family"] == fam].sort_values("log10_params")
        if len(sub) < 2:
            continue
        for metric in ["A_log", "D_sim", "kappa_95_5"]:
            slope = family_ols_slope(
                sub["log10_params"].to_numpy(float), sub[metric].to_numpy(float)
            )
            geom_trends.append(
                {"family": fam, "metric": metric, "slope_vs_log10P": slope}
            )
    pd.DataFrame(geom_trends).to_csv(out_dir / "geometry_vs_size_slopes.csv", index=False)

    merged = scores.merge(
        geom[["model", "A_log", "D_sim", "kappa_95_5"]], on="model", how="left"
    )
    corr_rows = []
    for metric in ["A_log", "D_sim", "kappa_95_5"]:
        rho, p = spearmanr(merged["lift_ridge"], merged[metric])
        corr_rows.append(
            {
                "scope": "all_rungs",
                "x": "lift_ridge",
                "y": metric,
                "spearman": float(rho),
                "p": float(p),
            }
        )
        for fam in FAMILY_ORDER:
            sub = merged[merged["family"] == fam]
            if len(sub) < 3:
                continue
            rho, p = spearmanr(sub["lift_ridge"], sub[metric])
            corr_rows.append(
                {
                    "scope": fam,
                    "x": "lift_ridge",
                    "y": metric,
                    "spearman": float(rho),
                    "p": float(p),
                }
            )
    pd.DataFrame(corr_rows).to_csv(out_dir / "lift_vs_anisotropy_spearman.csv", index=False)

    # Residual locality vs size slopes
    if resid_rows:
        rdf = pd.DataFrame(resid_rows)
        resid_trends = []
        for fam in FAMILY_ORDER:
            for kk in K_RESID:
                sub = rdf[(rdf["family"] == fam) & (rdf["k"] == kk)].sort_values(
                    "log10_params"
                )
                if len(sub) < 2:
                    continue
                resid_trends.append(
                    {
                        "family": fam,
                        "k": kk,
                        "slope_R_local_vs_log10P": family_ols_slope(
                            sub["log10_params"].to_numpy(float),
                            sub["R_local"].to_numpy(float),
                        ),
                        "mean_R_local": float(sub["R_local"].mean()),
                        "mean_null_p": float(sub["null_p"].mean()),
                    }
                )
        pd.DataFrame(resid_trends).to_csv(
            out_dir / "residual_locality_vs_size_slopes.csv", index=False
        )

    # Figures
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for ax, fam in zip(axes, FAMILY_ORDER):
        sub = scores[scores["family"] == fam].sort_values("log10_params")
        ax.plot(sub["log10_params"], sub["mknn_dense"], "o-", label="Dense", color="C0")
        ax.plot(
            sub["log10_params"],
            sub["mknn_dense_ridge"],
            "s-",
            label="Dense+Ridge",
            color="C1",
        )
        ax.set_title(FAMILY_LABEL.get(fam, fam))
        ax.set_xlabel(r"$\log_{10} P$")
    axes[0].set_ylabel("mean mKNN@10")
    axes[0].legend(fontsize=8)
    fig.suptitle("Dense vs Dense+Ridge size ladders (test-only gallery)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "ridge_scaling_amplification.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_scaling_amplification.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not args.skip_perm and len(T_null):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(T_null, bins=30, color="0.7", edgecolor="k", alpha=0.9, label="null T")
        ax.axvline(T_real, color="C3", lw=2, label=f"T_real={T_real:.4f}")
        ax.set_xlabel(r"$T$ = mean family $\Delta\beta$")
        ax.set_ylabel("count")
        ax.set_title(
            f"Synchronized shuffle-refit null (B={args.b_perm}, p={p_perm:.4g})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "ridge_scaling_perm_null.png", dpi=160, bbox_inches="tight")
        fig.savefig(out_dir / "fig_perm_null.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam in FAMILY_ORDER:
        sub = geom[geom["family"] == fam].sort_values("log10_params")
        if sub.empty:
            continue
        ax.plot(
            sub["log10_params"],
            sub["A_log"],
            "o-",
            label=FAMILY_LABEL.get(fam, fam),
        )
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(r"$A_{\log}=\mathrm{std}(\log\tilde\sigma)$")
    ax.set_title("Ridge map anisotropy vs model size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "ridge_scaling_anisotropy.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "fig_anisotropy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if resid_rows:
        rdf = pd.DataFrame(resid_rows)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for ax, kk in zip(axes, K_RESID):
            for fam in FAMILY_ORDER:
                sub = rdf[(rdf["family"] == fam) & (rdf["k"] == kk)].sort_values(
                    "log10_params"
                )
                if sub.empty:
                    continue
                ax.plot(
                    sub["log10_params"],
                    sub["R_local"],
                    "o-",
                    label=FAMILY_LABEL.get(fam, fam),
                )
            ax.set_title(f"k={kk}")
            ax.set_xlabel(r"$\log_{10} P$")
        axes[0].set_ylabel(r"$R_{\mathrm{local}}$")
        axes[0].legend(fontsize=7)
        fig.suptitle("Residual locality of global Dense→Dense Ridge", y=1.02)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "ridge_scaling_residual_locality.png", dpi=160, bbox_inches="tight"
        )
        fig.savefig(out_dir / "fig_residual_locality.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
