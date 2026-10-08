#!/usr/bin/env python3
"""Relational geometry model-size scaling (arXiv:2509.19453 methodology).

Primary statistic: within-family ordered adjacent size comparisons + exact
binomial test (p=0.5), matching The Platonic Universe (2509.19453).

Phases:
  analyze  — import dense/SAE/BSF caches, sign tests, lifts, figures, report
  oracle   — paired Ridge oracle mKNN / CKA / Spearman per ladder rung
  unpaired — DualEncoder unpaired recovery on richest ladders (Z=256)
  all      — analyze → oracle → unpaired

Outputs: outputs/relational_geometry_size_scaling/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import binomtest, spearmanr

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
_REPO = _HERE.parents[3]

for _p in (
    _REPO / "configs",
    _PKG / "analysis",
    _PKG / "alignment",
    _PKG / "bsf",
):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from _common import platonic_root, resolve_path, load_aligned_pair  # noqa: E402

# Corrected mKNN (must slice to k)
from sae_affine_basis_mknn_gpu import knn_cos, mknn  # noqa: E402

FAMILY_ORDER = ("convnext", "astropt", "dinov2", "vit", "ijepa")

# Cached method names → canonical representation labels
METHOD_MAP = {
    "dense": ("dense_cosine", "paper_full_catalog"),
    "sae": ("shared_best_basis_idf", "heldout_query_full_gallery"),
    "bsf": ("shared_best_basis_cosine", "heldout_query_full_gallery"),
}

PRIMARY_KS = (10, 100)
SENSITIVITY_KS = (5, 10, 20, 50, 100)
CACHE_KS = (10, 20, 50)  # present in existing size_scaling artifacts


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument(
        "--pairs-yaml",
        default="configs/official_legacy_pairs.yaml",
    )
    p.add_argument(
        "--sae-cache",
        default="outputs/universetbd_shared_basis_mknn_ks/size_scaling",
    )
    p.add_argument(
        "--bsf-cache",
        default="outputs/universetbd_shared_basis_mknn_ks/size_scaling_bsf",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/relational_geometry_size_scaling",
    )
    p.add_argument(
        "--phase",
        choices=("analyze", "oracle", "unpaired", "all"),
        default="analyze",
    )
    p.add_argument("--families", default=",".join(FAMILY_ORDER))
    p.add_argument(
        "--unpaired-families",
        default="convnext,dinov2",
        help="Richest ladders for Phase-2 unpaired (default: convnext,dinov2)",
    )
    p.add_argument("--ks", default="10,20,50")
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.2)
    # Unpaired protocol (fixed across sizes)
    p.add_argument("--unpaired-z", type=int, default=256)
    p.add_argument("--unpaired-seeds", type=int, default=2)
    p.add_argument("--unpaired-epochs", type=int, default=80)
    p.add_argument("--unpaired-hidden", type=int, default=512)
    p.add_argument("--unpaired-batch-size", type=int, default=256)
    p.add_argument("--unpaired-lr", type=float, default=1e-3)
    p.add_argument("--n-oracle-train", type=int, default=2500)
    p.add_argument("--n-a-train", type=int, default=5500)
    p.add_argument("--n-b-train", type=int, default=5500)
    p.add_argument("--n-geometry", type=int, default=512)
    p.add_argument("--reps", default="dense,sae_shared")
    p.add_argument("--skip-soft", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_pairs_yaml(path: Path) -> dict[str, dict]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _sae_available(root: Path, cfg: dict) -> bool:
    stem1 = Path(cfg["parquet1"]).stem
    stem2 = Path(cfg["parquet2"]).stem
    tags = (
        "F2048_k20_seed0",
        "F2048_k21_seed0",
        "F2048_k22_seed0",
        "F2048_k19_seed0",
        "F2048_k64_seed0",
    )
    for stem, col in ((stem1, cfg["col1"]), (stem2, cfg["col2"])):
        base = root / "outputs" / "sae" / stem / col
        if not base.is_dir():
            return False
        if not any((base / t / "model.pt").is_file() for t in tags) and not any(
            (p / "model.pt").is_file() for p in base.iterdir() if p.is_dir()
        ):
            return False
    return True


def _bsf_available(root: Path, cfg: dict) -> bool:
    stem = Path(cfg["parquet1"]).stem
    base = root / "outputs" / "bsf" / stem
    if not base.is_dir():
        return False
    for col in (cfg["col1"], cfg["col2"]):
        cdir = base / col
        if not cdir.is_dir():
            return False
        if not any((p / "model.pt").is_file() for p in cdir.rglob("G*_b*_k*") if p.is_dir()):
            return False
    return True


def _dense_available(root: Path, cfg: dict) -> bool:
    return resolve_path(root, cfg["parquet1"]).is_file() and resolve_path(
        root, cfg["parquet2"]
    ).is_file()


def build_manifest(root: Path, pairs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, cfg in pairs.items():
        dense_ok = _dense_available(root, cfg)
        sae_ok = dense_ok and _sae_available(root, cfg)
        bsf_ok = dense_ok and _bsf_available(root, cfg)
        reason = ""
        included = dense_ok
        if not dense_ok:
            reason = "missing_parquet"
        params_m = float(cfg.get("approx_params_m", float("nan")))
        # Probe embedding dim from first row only (avoid full load)
        emb_dim = float("nan")
        if dense_ok:
            try:
                import pyarrow.parquet as pq

                col = cfg["col1"]
                table = pq.read_table(
                    resolve_path(root, cfg["parquet1"]), columns=[col]
                )
                emb_dim = int(np.asarray(table.column(0)[0].as_py()).size)
            except Exception as exc:  # noqa: BLE001
                reason = f"dim_probe_failed:{exc}"
        rows.append(
            {
                "model": cfg.get("paper_model", name),
                "pair": name,
                "family": cfg.get("family", ""),
                "size_label": cfg.get("size_name", ""),
                "size_order": int(cfg.get("size_rank", -1)),
                "parameter_count": params_m * 1e6 if params_m == params_m else float("nan"),
                "log10_parameter_count": (
                    math.log10(params_m * 1e6) if params_m > 0 else float("nan")
                ),
                "approx_params_m": params_m,
                "embedding_dim": emb_dim,
                "dense_available": dense_ok,
                "sae_available": sae_ok,
                "bsf_available": bsf_ok,
                "unpaired_compatible": dense_ok,
                "included": included,
                "exclusion_reason": reason,
                "comparison_mode": "paper_same_size_cross_survey",
                "reference_policy": "same_size_other_survey",
                "reference_model": "paired_legacysurvey_same_size",
                "col1": cfg.get("col1", ""),
                "col2": cfg.get("col2", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "size_order"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Import caches → family tables
# ---------------------------------------------------------------------------


def _load_cache(path: Path) -> pd.DataFrame:
    pq = path / "mknn_by_size.parquet"
    if not pq.is_file():
        return pd.DataFrame()
    return pd.read_parquet(pq)


def import_representation_table(
    cache: pd.DataFrame,
    *,
    representation: str,
    method: str,
    protocol: str | None,
    families: list[str],
) -> pd.DataFrame:
    if cache.empty:
        return pd.DataFrame()
    sub = cache[cache["method"] == method].copy()
    if protocol and "protocol" in sub.columns:
        sub = sub[sub["protocol"] == protocol]
    sub = sub[sub["family"].isin(families)]
    if sub.empty:
        return sub
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "family": r["family"],
                "model": r.get("paper_model", ""),
                "pair": r.get("pair", ""),
                "size_label": r.get("size_name", ""),
                "size_order": int(r["size_rank"]),
                "parameter_count": float(r["approx_params_m"]) * 1e6,
                "log10_parameter_count": math.log10(float(r["approx_params_m"]) * 1e6),
                "approx_params_m": float(r["approx_params_m"]),
                "reference_model": "paired_legacysurvey_same_size",
                "comparison_mode": "paper_same_size_cross_survey",
                "representation": representation,
                "metric": "mknn",
                "k_or_keff": int(r["k"]),
                "score": float(r["mknn"]),
                "bootstrap_low": float("nan"),
                "bootstrap_high": float("nan"),
                "null_score": float("nan"),
                "seed_std": float("nan"),
                "n": int(r.get("n", -1)),
                "protocol": r.get("protocol", ""),
                "method_raw": method,
            }
        )
    return pd.DataFrame(rows)


def assert_mknn_bounds(df: pd.DataFrame, col: str = "score") -> None:
    if df.empty or col not in df.columns:
        return
    bad = df[(df[col] < -1e-9) | (df[col] > 1.0 + 1e-9)]
    if len(bad):
        raise AssertionError(
            f"mKNN out of [0,1]: min={bad[col].min()}, max={bad[col].max()}, n={len(bad)}"
        )


# ---------------------------------------------------------------------------
# Adjacent diffs, binomial, leave-one-out
# ---------------------------------------------------------------------------


def adjacent_differences(family_df: pd.DataFrame, metric_col: str = "score") -> pd.DataFrame:
    """Within-family ordered adjacent size steps."""
    rows = []
    keys = ["family", "representation", "metric", "k_or_keff", "reference_model"]
    for key, g in family_df.groupby(keys, dropna=False):
        g = g.sort_values("size_order")
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            delta = float(b[metric_col] - a[metric_col])
            rows.append(
                {
                    "family": a["family"],
                    "representation": a["representation"],
                    "metric": a["metric"],
                    "k_or_keff": a["k_or_keff"],
                    "reference_model": a["reference_model"],
                    "size_from": a["size_label"],
                    "size_to": b["size_label"],
                    "size_order_from": int(a["size_order"]),
                    "size_order_to": int(b["size_order"]),
                    "model_from": a["model"],
                    "model_to": b["model"],
                    "params_from": a["parameter_count"],
                    "params_to": b["parameter_count"],
                    "score_from": float(a[metric_col]),
                    "score_to": float(b[metric_col]),
                    "delta": delta,
                    "positive": bool(delta > 0),
                }
            )
    return pd.DataFrame(rows)


def binomial_tests(adj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if adj.empty:
        return pd.DataFrame()
    for (rep, metric, k), g in adj.groupby(["representation", "metric", "k_or_keff"]):
        n_pos = int(g["positive"].sum())
        n_tot = int(len(g))
        if n_tot == 0:
            continue
        two = binomtest(n_pos, n_tot, 0.5, alternative="two-sided")
        one = binomtest(n_pos, n_tot, 0.5, alternative="greater")
        rows.append(
            {
                "representation": rep,
                "metric": metric,
                "k_or_keff": k,
                "n_positive": n_pos,
                "n_total": n_tot,
                "fraction_positive": n_pos / n_tot,
                "mean_delta": float(g["delta"].mean()),
                "median_delta": float(g["delta"].median()),
                "one_sided_p": float(one.pvalue),
                "two_sided_p": float(two.pvalue),
                "scope": "all_families",
            }
        )
        # Per-family
        for fam, gf in g.groupby("family"):
            n_pos_f = int(gf["positive"].sum())
            n_tot_f = int(len(gf))
            two_f = binomtest(n_pos_f, n_tot_f, 0.5, alternative="two-sided")
            one_f = binomtest(n_pos_f, n_tot_f, 0.5, alternative="greater")
            rows.append(
                {
                    "representation": rep,
                    "metric": metric,
                    "k_or_keff": k,
                    "n_positive": n_pos_f,
                    "n_total": n_tot_f,
                    "fraction_positive": n_pos_f / n_tot_f,
                    "mean_delta": float(gf["delta"].mean()),
                    "median_delta": float(gf["delta"].median()),
                    "one_sided_p": float(one_f.pvalue),
                    "two_sided_p": float(two_f.pvalue),
                    "scope": f"family:{fam}",
                }
            )
    return pd.DataFrame(rows)


def leave_one_family_out(adj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if adj.empty:
        return pd.DataFrame()
    families = sorted(adj["family"].unique())
    for (rep, metric, k), g in adj.groupby(["representation", "metric", "k_or_keff"]):
        for drop in families:
            sub = g[g["family"] != drop]
            n_pos = int(sub["positive"].sum())
            n_tot = int(len(sub))
            if n_tot == 0:
                continue
            one = binomtest(n_pos, n_tot, 0.5, alternative="greater")
            two = binomtest(n_pos, n_tot, 0.5, alternative="two-sided")
            rows.append(
                {
                    "representation": rep,
                    "metric": metric,
                    "k_or_keff": k,
                    "dropped_family": drop,
                    "n_positive": n_pos,
                    "n_total": n_tot,
                    "fraction_positive": n_pos / n_tot,
                    "one_sided_p": float(one.pvalue),
                    "two_sided_p": float(two.pvalue),
                }
            )
    return pd.DataFrame(rows)


def family_correlations(family_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["family", "representation", "metric", "k_or_keff"]
    for key, g in family_df.groupby(keys):
        g = g.dropna(subset=["log10_parameter_count", "score"])
        if len(g) < 2:
            continue
        if len(g) == 2:
            delta = float(g.sort_values("size_order")["score"].iloc[-1] - g.sort_values("size_order")["score"].iloc[0])
            rows.append(
                {
                    "family": g["family"].iloc[0],
                    "representation": g["representation"].iloc[0],
                    "metric": g["metric"].iloc[0],
                    "k_or_keff": g["k_or_keff"].iloc[0],
                    "n_sizes": 2,
                    "spearman_logP": float("nan"),
                    "spearman_p": float("nan"),
                    "direction_delta": delta,
                    "note": "two_sizes_direction_only",
                }
            )
            continue
        sp = spearmanr(g["log10_parameter_count"], g["score"])
        rows.append(
            {
                "family": g["family"].iloc[0],
                "representation": g["representation"].iloc[0],
                "metric": g["metric"].iloc[0],
                "k_or_keff": g["k_or_keff"].iloc[0],
                "n_sizes": int(len(g)),
                "spearman_logP": float(sp.correlation) if sp.correlation is not None else float("nan"),
                "spearman_p": float(sp.pvalue) if sp.pvalue is not None else float("nan"),
                "direction_delta": float(
                    g.sort_values("size_order")["score"].iloc[-1]
                    - g.sort_values("size_order")["score"].iloc[0]
                ),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def representation_lift(
    dense: pd.DataFrame, other: pd.DataFrame, other_name: str
) -> pd.DataFrame:
    if dense.empty or other.empty:
        return pd.DataFrame()
    d = dense.rename(columns={"score": "dense_score"})
    o = other.rename(columns={"score": "other_score"})
    keys = ["family", "size_order", "k_or_keff", "reference_model"]
    m = d.merge(
        o[keys + ["other_score", "model", "size_label", "parameter_count", "log10_parameter_count"]],
        on=keys,
        how="inner",
        suffixes=("", "_o"),
    )
    if m.empty:
        return m
    m["lift"] = m["other_score"] - m["dense_score"]
    m["representation"] = other_name
    m["metric"] = "mknn_lift_vs_dense"
    m["score"] = m["lift"]
    return m


# ---------------------------------------------------------------------------
# Null via row-shuffle of neighbor index tables
# ---------------------------------------------------------------------------


@torch.no_grad()
def mknn_with_null(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    k: int,
    row_batch: int,
    n_null: int,
    seed: int,
) -> dict[str, float]:
    nn_a = knn_cos(A, k, row_batch)
    nn_b = knn_cos(B, k, row_batch)
    true = float(mknn(nn_a, nn_b, k))
    assert 0.0 - 1e-9 <= true <= 1.0 + 1e-9
    rng = np.random.default_rng(seed)
    nulls = []
    n = nn_b.shape[0]
    for _ in range(n_null):
        perm = torch.as_tensor(rng.permutation(n), device=nn_b.device)
        nulls.append(float(mknn(nn_a, nn_b[perm], k)))
    nulls_a = np.asarray(nulls, dtype=np.float64)
    mu, sd = float(nulls_a.mean()), float(nulls_a.std(ddof=1) if n_null > 1 else 0.0)
    z = (true - mu) / sd if sd > 1e-12 else float("nan")
    p = float(np.mean(nulls_a >= true))
    return {
        "mknn": true,
        "null_mean": mu,
        "null_std": sd,
        "true_minus_null": true - mu,
        "z_score": z,
        "empirical_p": p,
    }


# ---------------------------------------------------------------------------
# Oracle phase
# ---------------------------------------------------------------------------


def run_oracle_ladder(
    root: Path,
    pairs: dict[str, dict],
    families: list[str],
    *,
    ks: list[int],
    max_n: int,
    test_size: float,
    ridge_alpha: float,
    seed: int,
    device: torch.device,
    row_batch: int,
    n_null: int,
    out_dir: Path,
    force: bool,
) -> pd.DataFrame:
    import run_unpaired_universal_geometry as uug
    from sklearn.linear_model import Ridge

    out_pq = out_dir / "oracle" / "oracle_scaling_results.parquet"
    if out_pq.is_file() and not force:
        log(f"Reusing oracle cache {out_pq}")
        return pd.read_parquet(out_pq)

    rows: list[dict[str, Any]] = []
    for fam in families:
        ladder = sorted(
            [(n, c) for n, c in pairs.items() if c.get("family") == fam],
            key=lambda t: int(t[1].get("size_rank", 0)),
        )
        for name, cfg in ladder:
            log(f"[oracle] {name}")
            X1, X2 = load_aligned_pair(
                resolve_path(root, cfg["parquet1"]),
                cfg["col1"],
                resolve_path(root, cfg["parquet2"]),
                cfg["col2"],
                allow_truncate=True,
            )
            n_full = len(X1)
            n_cap = int(cfg.get("default_max_n", 0) or 0)
            n_use = max_n
            if n_cap > 0:
                n_use = min(n_use, n_cap) if n_use > 0 else n_cap
            rng = np.random.default_rng(seed)
            if n_use and n_full > n_use:
                sel = np.sort(rng.choice(n_full, size=n_use, replace=False))
                X1, X2 = X1[sel], X2[sel]
            n = len(X1)
            idx = np.arange(n)
            from sklearn.model_selection import train_test_split

            tr, te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
            tr, te = np.sort(tr), np.sort(te)
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
            ridge.fit(X1[tr], X2[tr])
            pred = ridge.predict(X1).astype(np.float32)
            pred_t = uug.to_torch(pred, device)
            true_t = uug.to_torch(X2.astype(np.float32), device)
            # Geometry on full catalog (paper-style) for mKNN
            for k in ks:
                if k >= n:
                    continue
                null_pack = mknn_with_null(
                    pred_t, true_t, k=k, row_batch=row_batch, n_null=n_null, seed=seed + k
                )
                rows.append(
                    {
                        "family": fam,
                        "model": cfg.get("paper_model", name),
                        "pair": name,
                        "size_label": cfg.get("size_name", ""),
                        "size_order": int(cfg.get("size_rank", -1)),
                        "parameter_count": float(cfg["approx_params_m"]) * 1e6,
                        "log10_parameter_count": math.log10(float(cfg["approx_params_m"]) * 1e6),
                        "approx_params_m": float(cfg["approx_params_m"]),
                        "reference_model": "paired_legacysurvey_same_size",
                        "comparison_mode": "paper_same_size_cross_survey",
                        "representation": "ridge_oracle",
                        "metric": "mknn",
                        "k_or_keff": int(k),
                        "score": null_pack["mknn"],
                        "null_score": null_pack["null_mean"],
                        "null_std": null_pack["null_std"],
                        "true_minus_null": null_pack["true_minus_null"],
                        "z_score": null_pack["z_score"],
                        "empirical_p": null_pack["empirical_p"],
                        "bootstrap_low": float("nan"),
                        "bootstrap_high": float("nan"),
                        "seed_std": float("nan"),
                        "n": n,
                        "n_test": int(len(te)),
                    }
                )
            # Identity + CKA / Spearman on held-out eval
            pred_te = uug.to_torch(pred[te], device)
            true_te = uug.to_torch(X2[te].astype(np.float32), device)
            id_m = uug.eval_translation(
                pred_te, true_te, row_batch=row_batch, mknn_ks=(10, 100)
            )
            geom_idx = te[: min(len(te), 512)]
            pred_g = uug.to_torch(pred[geom_idx], device)
            true_g = uug.to_torch(X2[geom_idx].astype(np.float32), device)
            # Translated↔true distance Spearman
            def _dvec(X: torch.Tensor) -> np.ndarray:
                Xn = uug.l2n_t(X).cpu().numpy()
                G = Xn @ Xn.T
                D = 1.0 - G
                iu = np.triu_indices(len(Xn), k=1)
                return D[iu]

            d_spear = uug.spearman_corr(_dvec(pred_g), _dvec(true_g))
            cka = uug.linear_cka(pred[te], X2[te].astype(np.float32))
            for metric_name, val in (
                ("top1", id_m["top1"]),
                ("mrr", id_m["mrr"]),
                ("cka", cka),
                ("distance_spearman", d_spear),
            ):
                rows.append(
                    {
                        "family": fam,
                        "model": cfg.get("paper_model", name),
                        "pair": name,
                        "size_label": cfg.get("size_name", ""),
                        "size_order": int(cfg.get("size_rank", -1)),
                        "parameter_count": float(cfg["approx_params_m"]) * 1e6,
                        "log10_parameter_count": math.log10(float(cfg["approx_params_m"]) * 1e6),
                        "approx_params_m": float(cfg["approx_params_m"]),
                        "reference_model": "paired_legacysurvey_same_size",
                        "comparison_mode": "paper_same_size_cross_survey",
                        "representation": "ridge_oracle",
                        "metric": metric_name,
                        "k_or_keff": -1,
                        "score": float(val) if val == val else float("nan"),
                        "null_score": float("nan"),
                        "bootstrap_low": float("nan"),
                        "bootstrap_high": float("nan"),
                        "seed_std": float("nan"),
                        "n": n,
                        "n_test": int(len(te)),
                    }
                )
            del X1, X2, pred, pred_t, true_t
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    ensure_dir(out_dir / "oracle")
    df.to_parquet(out_pq, index=False)
    df.to_csv(out_dir / "oracle" / "family_scaling_oracle.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Unpaired phase (fixed Z, richest ladders)
# ---------------------------------------------------------------------------


def run_unpaired_ladder(
    root: Path,
    pairs: dict[str, dict],
    families: list[str],
    *,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    import run_unpaired_universal_geometry as uug
    from sae_affine_basis_mknn_gpu import encode, load_sae
    from run_universetbd_shared_basis_mknn_ks import resolve_sae_dir

    out_pq = out_dir / "unpaired" / "unpaired_scaling_results.parquet"
    if out_pq.is_file() and not args.force:
        log(f"Reusing unpaired cache {out_pq}")
        return pd.read_parquet(out_pq)

    ensure_dir(out_dir / "unpaired")
    rows: list[dict[str, Any]] = []
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    z = int(args.unpaired_z)
    lw = uug.LossWeights(
        recon=1.0, mmd=1.0, cycle=1.0, geom=1.0
    )

    for fam in families:
        ladder = sorted(
            [(n, c) for n, c in pairs.items() if c.get("family") == fam],
            key=lambda t: int(t[1].get("size_rank", 0)),
        )
        for name, cfg in ladder:
            log(f"[unpaired] {name}")
            X1, X2 = load_aligned_pair(
                resolve_path(root, cfg["parquet1"]),
                cfg["col1"],
                resolve_path(root, cfg["parquet2"]),
                cfg["col2"],
                allow_truncate=True,
            )
            n_full = len(X1)
            n_cap = int(cfg.get("default_max_n", 0) or 0)
            n_use = args.max_n
            if n_cap > 0:
                n_use = min(n_use, n_cap) if n_use > 0 else n_cap
            rng = np.random.default_rng(args.seed)
            if n_use and n_full > n_use:
                sel = np.sort(rng.choice(n_full, size=n_use, replace=False))
                X1, X2 = X1[sel], X2[sel]
            n = len(X1)
            split = uug.build_split(
                n,
                args.n_oracle_train,
                args.n_a_train,
                args.n_b_train,
                seed=args.seed,
            )
            # Dense + optional SAE shared
            sae1 = resolve_sae_dir(root, cfg["parquet1"], cfg["col1"])
            sae2 = resolve_sae_dir(root, cfg["parquet2"], cfg["col2"])

            for rep in reps:
                if rep == "dense":
                    FA, FB = X1.astype(np.float32), X2.astype(np.float32)
                elif rep == "sae_shared":
                    if sae1 is None or sae2 is None:
                        log(f"  skip sae_shared (missing SAE) for {name}")
                        continue
                    b1, b2 = load_sae(sae1, device), load_sae(sae2, device)
                    C1, C2 = encode(b1, X1, device), encode(b2, X2, device)
                    bundle = uug.fit_ridge_map_b_to_a(
                        C1,
                        C2,
                        train_idx=split["paired_oracle_train"],
                        alpha=args.ridge_alpha,
                    )
                    FA = C1.astype(np.float32)
                    FB = uug.map_b_to_a(bundle, C2).astype(np.float32)
                    idf = uug.idf_weights(FA[split["A_train"]])
                    FA = (FA * idf[None, :]).astype(np.float32)
                    FB = (FB * idf[None, :]).astype(np.float32)
                else:
                    continue

                seed_rows = []
                for s_i in range(args.unpaired_seeds):
                    seed = args.seed + s_i
                    model, _curves, meta = uug.train_dual_encoder(
                        FA,
                        FB,
                        split["A_train"],
                        split["B_train"],
                        z_dim=z,
                        hidden=args.unpaired_hidden,
                        device=device,
                        epochs=args.unpaired_epochs,
                        batch_size=args.unpaired_batch_size,
                        lr=args.unpaired_lr,
                        val_frac=0.1,
                        weights=lw,
                        seed=seed,
                    )
                    ev = uug.evaluate_unpaired_model(
                        model,
                        FA,
                        FB,
                        split["paired_eval"],
                        device=device,
                        row_batch=args.row_batch,
                        n_geometry=args.n_geometry,
                        n_soft_query=256,
                        n_shuffles=args.n_null,
                        skip_soft=args.skip_soft,
                        seed=seed,
                    )
                    idm = ev["identity_A_to_B"]
                    geom = ev["geometry"]
                    seed_rows.append(
                        {
                            "seed": seed,
                            "mknn_k10": idm.get("mknn_k10"),
                            "mknn_k100": idm.get("mknn_k100"),
                            "mknn_k5": idm.get("mknn_k5"),
                            "mknn_k20": idm.get("mknn_k20"),
                            "mknn_k50": idm.get("mknn_k50"),
                            "top1": idm.get("top1"),
                            "top10": idm.get("top10"),
                            "mrr": idm.get("mrr"),
                            "median_rank": idm.get("median_rank"),
                            "cka": geom.get("cka_transA_trueB"),
                            "distance_spearman": geom.get("spearman_transA_vs_trueB"),
                            "n_params": meta["n_params"],
                            "best_epoch": meta["best_epoch"],
                        }
                    )
                    for kk in (5, 10, 20, 50, 100):
                        v = idm.get(f"mknn_k{kk}")
                        if v is not None and not (0.0 - 1e-6 <= float(v) <= 1.0 + 1e-6):
                            raise AssertionError(
                                f"mKNN@{kk}={v} out of [0,1] for {name}"
                            )

                sdf = pd.DataFrame(seed_rows)
                base = {
                    "family": fam,
                    "model": cfg.get("paper_model", name),
                    "pair": name,
                    "size_label": cfg.get("size_name", ""),
                    "size_order": int(cfg.get("size_rank", -1)),
                    "parameter_count": float(cfg["approx_params_m"]) * 1e6,
                    "log10_parameter_count": math.log10(float(cfg["approx_params_m"]) * 1e6),
                    "approx_params_m": float(cfg["approx_params_m"]),
                    "reference_model": "paired_legacysurvey_same_size",
                    "comparison_mode": "paper_same_size_cross_survey",
                    "representation": f"unpaired_{rep}",
                    "Z": z,
                    "hidden": args.unpaired_hidden,
                    "n_trainable": int(sdf["n_params"].mean()),
                    "input_dim_a": int(FA.shape[1]),
                    "input_dim_b": int(FB.shape[1]),
                    "n": n,
                    "n_common_ids": n,
                }
                for metric, col in (
                    ("mknn", "mknn_k10"),
                    ("mknn", "mknn_k100"),
                    ("mknn", "mknn_k5"),
                    ("mknn", "mknn_k20"),
                    ("mknn", "mknn_k50"),
                    ("cka", "cka"),
                    ("distance_spearman", "distance_spearman"),
                    ("top1", "top1"),
                    ("top10", "top10"),
                    ("mrr", "mrr"),
                    ("median_rank", "median_rank"),
                ):
                    k_or = {
                        "mknn_k10": 10,
                        "mknn_k100": 100,
                        "mknn_k5": 5,
                        "mknn_k20": 20,
                        "mknn_k50": 50,
                    }.get(col, -1)
                    rows.append(
                        {
                            **base,
                            "metric": metric if not col.startswith("mknn") else "mknn",
                            "k_or_keff": k_or if col.startswith("mknn") else -1,
                            "score": float(sdf[col].mean()),
                            "seed_std": float(sdf[col].std(ddof=0)),
                            "seed_best": float(sdf[col].max()),
                            "seed_worst": float(sdf[col].min()),
                            "null_score": float("nan"),
                            "bootstrap_low": float("nan"),
                            "bootstrap_high": float("nan"),
                        }
                    )
            del X1, X2
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_parquet(out_pq, index=False)
    df.to_csv(out_dir / "unpaired" / "family_scaling_unpaired.csv", index=False)
    return df


def recoverability_table(
    unpaired: pd.DataFrame, oracle: pd.DataFrame
) -> pd.DataFrame:
    """R = (M_unpaired - M_null) / (M_oracle - M_null); null≈0 for CKA/mKNN if missing."""
    if unpaired.empty or oracle.empty:
        return pd.DataFrame()
    rows = []
    o_mknn = oracle[(oracle["representation"] == "ridge_oracle") & (oracle["metric"] == "mknn")]
    o_other = oracle[
        (oracle["representation"] == "ridge_oracle")
        & (oracle["metric"].isin(["cka", "distance_spearman"]))
    ]
    for _, u in unpaired.iterrows():
        if u["metric"] == "mknn":
            o = o_mknn[
                (o_mknn["family"] == u["family"])
                & (o_mknn["size_order"] == u["size_order"])
                & (o_mknn["k_or_keff"] == u["k_or_keff"])
            ]
        else:
            o = o_other[
                (o_other["family"] == u["family"])
                & (o_other["size_order"] == u["size_order"])
                & (o_other["metric"] == u["metric"])
            ]
        if o.empty:
            continue
        o = o.iloc[0]
        m_u = float(u["score"])
        m_o = float(o["score"])
        m_n = float(o["null_score"]) if o.get("null_score", float("nan")) == o.get("null_score", float("nan")) else 0.0
        if not (m_n == m_n):
            m_n = 0.0
        denom = m_o - m_n
        r = (m_u - m_n) / denom if abs(denom) > 1e-8 else float("nan")
        rows.append(
            {
                "family": u["family"],
                "model": u["model"],
                "size_label": u["size_label"],
                "size_order": u["size_order"],
                "parameter_count": u["parameter_count"],
                "log10_parameter_count": u["log10_parameter_count"],
                "representation": u["representation"],
                "metric": u["metric"],
                "k_or_keff": u["k_or_keff"],
                "unpaired_score": m_u,
                "oracle_score": m_o,
                "null_score": m_n,
                "recoverability": r,
                "score": r,
                "reference_model": u["reference_model"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures + report
# ---------------------------------------------------------------------------


def _plot_family_curves(df: pd.DataFrame, title: str, path: Path, ylabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fam, g in df.groupby("family"):
        g = g.sort_values("log10_parameter_count")
        ax.plot(
            g["log10_parameter_count"],
            g["score"],
            marker="o",
            label=fam,
        )
    ax.set_xlabel(r"$\log_{10}$ parameter count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_phase1_figures(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    fig_dir = ensure_dir(out_dir / "combined" / "figures")
    for rep, key in (("dense", "dense"), ("sae", "sae"), ("bsf", "bsf")):
        df = tables.get(key, pd.DataFrame())
        if df.empty:
            continue
        for k in (10, 20, 50, 100):
            sub = df[(df["metric"] == "mknn") & (df["k_or_keff"] == k)]
            if sub.empty:
                continue
            _plot_family_curves(
                sub,
                f"{rep} mKNN@{k} vs model size",
                fig_dir / f"{rep}_mknn_k{k}_vs_size.png",
                f"mKNN@{k}",
            )
    # Overlay dense vs SAE vs BSF at k=10
    pieces = []
    for rep, key in (("dense", "dense"), ("sae", "sae"), ("bsf", "bsf")):
        df = tables.get(key, pd.DataFrame())
        if df.empty:
            continue
        sub = df[(df["metric"] == "mknn") & (df["k_or_keff"] == 10)].copy()
        sub["rep_label"] = rep
        pieces.append(sub)
    if pieces:
        allp = pd.concat(pieces, ignore_index=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for (fam, rep), g in allp.groupby(["family", "rep_label"]):
            g = g.sort_values("log10_parameter_count")
            ax.plot(
                g["log10_parameter_count"],
                g["score"],
                marker="o",
                label=f"{fam}/{rep}",
                alpha=0.85,
            )
        ax.set_xlabel(r"$\log_{10}$ parameter count")
        ax.set_ylabel("mKNN@10")
        ax.set_title("Dense vs SAE vs BSF scaling overlay (k=10)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "dense_sae_bsf_overlay_k10.png", dpi=140)
        plt.close(fig)

    # Fraction positive bars
    binom = tables.get("binomial", pd.DataFrame())
    if not binom.empty:
        sub = binom[(binom["scope"] == "all_families") & (binom["k_or_keff"].isin([10, 50]))]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            labels = [f"{r.representation}\nk={int(r.k_or_keff)}" for r in sub.itertuples()]
            ax.bar(range(len(sub)), sub["fraction_positive"], color="steelblue")
            ax.axhline(0.5, color="gray", ls="--")
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Fraction positive adjacent steps")
            ax.set_title("Paper-style size-step success rate")
            fig.tight_layout()
            fig.savefig(fig_dir / "fraction_positive_by_method.png", dpi=140)
            plt.close(fig)

    adj = tables.get("adjacent", pd.DataFrame())
    if not adj.empty:
        sub = adj[(adj["k_or_keff"] == 10) & (adj["metric"] == "mknn")]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for rep, g in sub.groupby("representation"):
                ax.hist(g["delta"], bins=12, alpha=0.45, label=rep)
            ax.axvline(0, color="black", ls="--")
            ax.set_xlabel(r"$\Delta$ mKNN@10 (larger − smaller)")
            ax.set_title("Adjacent size-step delta distribution")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "adjacent_delta_hist_k10.png", dpi=140)
            plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    config: dict,
    tables: dict[str, pd.DataFrame],
) -> None:
    binom = tables.get("binomial", pd.DataFrame())
    adj = tables.get("adjacent", pd.DataFrame())
    loo = tables.get("loo", pd.DataFrame())
    corr = tables.get("corr", pd.DataFrame())
    lift = tables.get("lift", pd.DataFrame())

    def _binom_line(rep: str, k: int) -> str:
        if binom.empty:
            return "_missing_"
        g = binom[
            (binom["representation"] == rep)
            & (binom["k_or_keff"] == k)
            & (binom["scope"] == "all_families")
            & (binom["metric"] == "mknn")
        ]
        if g.empty:
            return f"_no rows for {rep} k={k}_"
        r = g.iloc[0]
        return (
            f"{int(r.n_positive)}/{int(r.n_total)} positive "
            f"(frac={r.fraction_positive:.3f}); "
            f"one-sided p={r.one_sided_p:.4g}, two-sided p={r.two_sided_p:.4g}; "
            f"mean Δ={r.mean_delta:+.5f}"
        )

    lines = [
        "# Relational Geometry Size Scaling Report",
        "",
        "Methodology aligned with arXiv:2509.19453 (within-family adjacent size",
        "comparisons + exact binomial test against p=0.5).",
        "",
        "Primary comparison mode: **paper_same_size_cross_survey**",
        "(Legacy Survey ↔ HSC, same architecture size on both sides).",
        "",
        "## Config",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Answers",
        "",
        "### 1. Does dense mKNN size-scaling reproduce?",
        _binom_line("dense", 10),
        "",
        "### 2–3. Adjacent increases and binomial tests (dense k=10)",
        _binom_line("dense", 10),
        "",
        "### 4. Strongest / weakest families",
    ]
    if not corr.empty:
        d = corr[(corr["representation"] == "dense") & (corr["k_or_keff"] == 10)]
        if not d.empty:
            d2 = d.sort_values("direction_delta", ascending=False)
            lines.append(
                f"- Strongest first→last: {d2.iloc[0]['family']} "
                f"(Δ={d2.iloc[0]['direction_delta']:+.5f})"
            )
            lines.append(
                f"- Weakest first→last: {d2.iloc[-1]['family']} "
                f"(Δ={d2.iloc[-1]['direction_delta']:+.5f})"
            )
    lines += [
        "",
        "### 5. k=10 vs k=50/100",
        f"- dense k=10: {_binom_line('dense', 10)}",
        f"- dense k=50: {_binom_line('dense', 50)}",
        f"- dense k=100: {_binom_line('dense', 100)}",
        "",
        "### 6. Shared SAE vs dense scaling",
        f"- SAE k=10: {_binom_line('sae', 10)}",
        "",
        "### 7. Shared BSF vs dense scaling",
        f"- BSF k=10: {_binom_line('bsf', 10)}",
        "",
        "### 8. Does SAE/BSF lift itself scale?",
    ]
    if not lift.empty and "lift" in lift.columns:
        for rep in sorted(lift["representation"].unique()):
            g = lift[(lift["representation"] == rep) & (lift["k_or_keff"] == 10)]
            if g.empty:
                continue
            sp = spearmanr(g["log10_parameter_count"], g["lift"])
            lines.append(
                f"- {rep} lift@k10 vs logP Spearman="
                f"{sp.correlation if sp.correlation==sp.correlation else float('nan'):+.3f} "
                f"(p={sp.pvalue if sp.pvalue==sp.pvalue else float('nan'):.3g}); "
                f"mean lift={g['lift'].mean():+.5f}"
            )
    else:
        lines.append("_Lift table empty._")

    lines += [
        "",
        "### 9–15. Unpaired / oracle / recoverability",
        "_Filled when `--phase oracle` / `--phase unpaired` complete._",
        "",
        "### 16–17. Reference / leave-one-family-out",
        "Primary ladders use paper same-size cross-survey references (not pooled).",
    ]
    if not loo.empty:
        g = loo[(loo["representation"] == "dense") & (loo["k_or_keff"] == 10)]
        if not g.empty:
            lines.append("Leave-one-family-out (dense mKNN@10):")
            for _, r in g.iterrows():
                lines.append(
                    f"- drop {r.dropped_family}: {int(r.n_positive)}/{int(r.n_total)} "
                    f"pos, one-sided p={r.one_sided_p:.4g}"
                )
    lines += [
        "",
        "### 18. Curve shape",
        "See family Spearman / first→last deltas; many ladders are mild/plateau-ish.",
        "",
        "### 19. vs arXiv:2509.19453",
        "Paper reported crossmodal 28/33 positive steps (p≈3e-5). Compare our dense",
        "binomial counts above on the official Legacy↔HSC ladders.",
        "",
        "### 20. Strongest defensible statement",
        "_Update after oracle/unpaired phases; Phase-1: dense adjacent signs + SAE/BSF lifts._",
        "",
        "## Tables",
        "- `model_size_manifest.csv`",
        "- `dense/family_scaling_dense.csv`",
        "- `sae/family_scaling_sae.csv`",
        "- `bsf/family_scaling_bsf.csv`",
        "- `combined/adjacent_size_differences.csv`",
        "- `combined/binomial_scaling_tests.csv`",
        "- `combined/family_correlations.csv`",
        "- `combined/representation_lift_scaling.csv`",
        "- `combined/leave_one_family_out.csv`",
    ]
    (out_dir / "relational_geometry_size_scaling_report.md").write_text(
        "\n".join(lines) + "\n"
    )


def update_report_with_oracle_unpaired(
    out_dir: Path,
    oracle: pd.DataFrame,
    unpaired: pd.DataFrame,
    recover: pd.DataFrame,
) -> None:
    path = out_dir / "relational_geometry_size_scaling_report.md"
    if not path.is_file():
        return
    extra = ["", "## Oracle / unpaired update", ""]
    if not oracle.empty:
        om = oracle[(oracle["metric"] == "mknn") & (oracle["k_or_keff"] == 10)]
        adj = adjacent_differences(om.assign(representation="ridge_oracle", metric="mknn"))
        bt = binomial_tests(adj)
        if not bt.empty:
            g = bt[bt["scope"] == "all_families"]
            if not g.empty:
                r = g.iloc[0]
                extra.append(
                    f"- Oracle mKNN@10 adjacent: {int(r.n_positive)}/{int(r.n_total)} "
                    f"(one-sided p={r.one_sided_p:.4g})"
                )
    if not unpaired.empty:
        for rep in sorted(unpaired["representation"].unique()):
            um = unpaired[
                (unpaired["representation"] == rep)
                & (unpaired["metric"] == "mknn")
                & (unpaired["k_or_keff"] == 10)
            ]
            adj = adjacent_differences(um)
            bt = binomial_tests(adj)
            if not bt.empty:
                g = bt[bt["scope"] == "all_families"]
                if not g.empty:
                    r = g.iloc[0]
                    extra.append(
                        f"- {rep} mKNN@10 adjacent: {int(r.n_positive)}/{int(r.n_total)} "
                        f"(one-sided p={r.one_sided_p:.4g})"
                    )
    if not recover.empty:
        for metric in ("mknn", "cka"):
            g = recover[recover["metric"] == metric]
            if g.empty:
                continue
            if metric == "mknn":
                g = g[g["k_or_keff"] == 10]
            if len(g) >= 3:
                sp = spearmanr(g["log10_parameter_count"], g["recoverability"])
                extra.append(
                    f"- Recoverability {metric} vs logP Spearman="
                    f"{sp.correlation if sp.correlation==sp.correlation else float('nan'):+.3f}"
                )
    path.write_text(path.read_text() + "\n".join(extra) + "\n")


# ---------------------------------------------------------------------------
# Phase analyze
# ---------------------------------------------------------------------------


def phase_analyze(args: argparse.Namespace, root: Path, pairs: dict) -> dict[str, pd.DataFrame]:
    out = ensure_dir(resolve_path(root, args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    # Prefer absolute under platonic root
    if not str(out).startswith(str(root)):
        out = ensure_dir(root / args.out_dir)
    for sub in ("dense", "sae", "bsf", "unpaired", "oracle", "combined"):
        ensure_dir(out / sub)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    log("Building model_size_manifest …")
    manifest = build_manifest(root, pairs)
    manifest.to_csv(out / "model_size_manifest.csv", index=False)

    sae_cache = _load_cache(resolve_path(root, args.sae_cache))
    bsf_cache = _load_cache(resolve_path(root, args.bsf_cache))

    dense = import_representation_table(
        sae_cache,
        representation="dense",
        method=METHOD_MAP["dense"][0],
        protocol=METHOD_MAP["dense"][1],
        families=families,
    )
    sae = import_representation_table(
        sae_cache,
        representation="sae",
        method=METHOD_MAP["sae"][0],
        protocol=METHOD_MAP["sae"][1],
        families=families,
    )
    # BSF shared uses cosine primary
    bsf = import_representation_table(
        bsf_cache,
        representation="bsf",
        method=METHOD_MAP["bsf"][0],
        protocol=None,  # BSF cache may omit protocol or differ
        families=families,
    )
    if not bsf.empty and "protocol" in bsf_cache.columns:
        # Prefer heldout if present
        pass

    assert_mknn_bounds(dense)
    assert_mknn_bounds(sae)
    assert_mknn_bounds(bsf)

    dense.to_csv(out / "dense" / "family_scaling_dense.csv", index=False)
    dense.to_parquet(out / "dense" / "dense_scaling_results.parquet", index=False)
    sae.to_csv(out / "sae" / "family_scaling_sae.csv", index=False)
    sae.to_parquet(out / "sae" / "sae_scaling_results.parquet", index=False)
    bsf.to_csv(out / "bsf" / "family_scaling_bsf.csv", index=False)
    bsf.to_parquet(out / "bsf" / "bsf_scaling_results.parquet", index=False)

    combined = pd.concat([dense, sae, bsf], ignore_index=True)
    adj = adjacent_differences(combined)
    binom = binomial_tests(adj)
    loo = leave_one_family_out(adj)
    corr = family_correlations(combined)
    lift_sae = representation_lift(dense, sae, "sae")
    lift_bsf = representation_lift(dense, bsf, "bsf")
    lift = pd.concat([lift_sae, lift_bsf], ignore_index=True) if len(lift_sae) or len(lift_bsf) else pd.DataFrame()

    adj.to_csv(out / "combined" / "adjacent_size_differences.csv", index=False)
    binom.to_csv(out / "combined" / "binomial_scaling_tests.csv", index=False)
    corr.to_csv(out / "combined" / "family_correlations.csv", index=False)
    if not lift.empty:
        lift.to_csv(out / "combined" / "representation_lift_scaling.csv", index=False)
    loo.to_csv(out / "combined" / "leave_one_family_out.csv", index=False)

    tables = {
        "dense": dense,
        "sae": sae,
        "bsf": bsf,
        "adjacent": adj,
        "binomial": binom,
        "loo": loo,
        "corr": corr,
        "lift": lift,
    }
    make_phase1_figures(out, tables)
    config = {
        "phase": "analyze",
        "families": families,
        "sae_cache": args.sae_cache,
        "bsf_cache": args.bsf_cache,
        "comparison_mode": "paper_same_size_cross_survey",
        "method_map": METHOD_MAP,
        "cache_ks": list(CACHE_KS),
        "note": "Primary k=10; cache has k∈{10,20,50}. k=5/100 require recompute.",
    }
    (out / "config_analyze.json").write_text(json.dumps(config, indent=2))
    write_report(out, config=config, tables=tables)
    log(f"Phase analyze done → {out}")
    return {"out": out, **tables}  # type: ignore


def main() -> None:
    args = parse_args()
    root = platonic_root(args.platonic_root)
    yaml_path = resolve_path(root, args.pairs_yaml)
    if not yaml_path.is_file():
        # fall back to worktree copy
        alt = _REPO / args.pairs_yaml
        yaml_path = alt if alt.is_file() else yaml_path
    pairs = load_pairs_yaml(yaml_path)
    log(f"Platonic root: {root}")
    log(f"Pairs yaml: {yaml_path} ({len(pairs)} pairs)")

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu and args.phase in ("oracle", "unpaired", "all"):
        raise SystemExit("CUDA required for oracle/unpaired (pass --allow-cpu to override)")

    out_dir = root / args.out_dir
    ensure_dir(out_dir)

    if args.phase in ("analyze", "all"):
        phase_analyze(args, root, pairs)

    if args.phase in ("oracle", "all"):
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        ks = [int(x) for x in args.ks.split(",") if x.strip()]
        # Ensure primary ks present
        for k in (10, 100):
            if k not in ks:
                ks.append(k)
        oracle = run_oracle_ladder(
            root,
            pairs,
            families,
            ks=sorted(set(ks)),
            max_n=args.max_n,
            test_size=args.test_size,
            ridge_alpha=args.ridge_alpha,
            seed=args.seed,
            device=device,
            row_batch=args.row_batch,
            n_null=args.n_null,
            out_dir=out_dir,
            force=args.force,
        )
        assert_mknn_bounds(oracle[oracle["metric"] == "mknn"])
        adj = adjacent_differences(
            oracle[oracle["metric"] == "mknn"].assign(representation="ridge_oracle")
        )
        if not adj.empty:
            binomial_tests(adj).to_csv(
                out_dir / "oracle" / "binomial_oracle.csv", index=False
            )
            adj.to_csv(out_dir / "oracle" / "adjacent_oracle.csv", index=False)
        update_report_with_oracle_unpaired(out_dir, oracle, pd.DataFrame(), pd.DataFrame())

    if args.phase in ("unpaired", "all"):
        ufamilies = [f.strip() for f in args.unpaired_families.split(",") if f.strip()]
        unpaired = run_unpaired_ladder(
            root, pairs, ufamilies, args=args, device=device, out_dir=out_dir
        )
        assert_mknn_bounds(unpaired[unpaired["metric"] == "mknn"])
        oracle_pq = out_dir / "oracle" / "oracle_scaling_results.parquet"
        oracle = pd.read_parquet(oracle_pq) if oracle_pq.is_file() else pd.DataFrame()
        recover = recoverability_table(unpaired, oracle)
        if not recover.empty:
            recover.to_csv(out_dir / "combined" / "recoverability_scaling.csv", index=False)
            # recoverability adjacent + binomial
            rec_adj = adjacent_differences(
                recover.assign(metric="recoverability", representation=recover["representation"])
            )
            if not rec_adj.empty:
                binomial_tests(rec_adj).to_csv(
                    out_dir / "combined" / "binomial_recoverability.csv", index=False
                )
        # Unpaired figures
        fig_dir = ensure_dir(out_dir / "unpaired" / "figures")
        for k in (10, 100):
            sub = unpaired[(unpaired["metric"] == "mknn") & (unpaired["k_or_keff"] == k)]
            if not sub.empty:
                _plot_family_curves(
                    sub,
                    f"Unpaired mKNN@{k} vs size",
                    fig_dir / f"unpaired_mknn_k{k}_vs_size.png",
                    f"mKNN@{k}",
                )
        for metric, ylab in (("cka", "CKA"), ("distance_spearman", "distance Spearman")):
            sub = unpaired[unpaired["metric"] == metric]
            if not sub.empty:
                _plot_family_curves(
                    sub,
                    f"Unpaired {ylab} vs size",
                    fig_dir / f"unpaired_{metric}_vs_size.png",
                    ylab,
                )
        update_report_with_oracle_unpaired(out_dir, oracle, unpaired, recover)

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
