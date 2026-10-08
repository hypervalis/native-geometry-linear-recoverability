#!/usr/bin/env python3
"""Block-level bipartite matching with Vanilla BSF codes.

Train BSFs first (train_vanilla_bsf.py), then:
  - shared-basis Ridge on block-norm codes (G-dim, like SAE feature acts)
  - top-k / Hungarian sparsification of W
  - partner-set Jaccard consistency (are block subspaces shared?)

Compares side-by-side with the existing TopK SAE on the same rows.

Usage:
  python src/nglr/bsf/run_bsf_bipartite.py \\
      --src dinov3 --dst vit_base --platonic-root ~/platonic-universe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _shared import (  # noqa: E402
    MODELS,
    idf_weights,
    knn_graph,
    load_col,
    load_model_codes,
    load_sae,
    l2n,
    mknn,
    platonic_root,
    resolve_path,
    sae_dir,
    topk_rows,
)
from bsf_model import GrassmannianBSF, VanillaBSF  # noqa: E402


def bsf_run_dir(
    root: Path,
    model: str,
    *,
    n_blocks: int,
    block_dim: int,
    k_blocks: int,
    seed: int = 0,
    featurizer: str = "vanilla_bsf",
) -> Path:
    parquet, col = MODELS[model]
    if featurizer in ("grassmannian", "grassmannian_bsf"):
        tag = f"grassmannian_G{n_blocks}_b{block_dim}_k{k_blocks}_seed{seed}"
    elif featurizer in ("sae_init_frozen_bsf", "sae_init_frozenD"):
        tag = f"sae_init_frozenD_G{n_blocks}_b{block_dim}_k{k_blocks}_seed{seed}"
    elif featurizer in ("sae_init_bsf", "sae_init"):
        tag = f"sae_init_G{n_blocks}_b{block_dim}_k{k_blocks}_seed{seed}"
    else:
        tag = f"G{n_blocks}_b{block_dim}_k{k_blocks}_seed{seed}"
    return root / "outputs" / "bsf" / Path(parquet).stem / col / tag


def load_bsf(path: Path, device: torch.device) -> dict:
    cfg = json.loads((path / "config.json").read_text())
    sc = np.load(path / "scaler_stats.npz")
    kind = cfg.get("featurizer", "vanilla_bsf")
    ctor = GrassmannianBSF if "grassmannian" in kind else VanillaBSF
    model = ctor(
        cfg["dim"], cfg["n_blocks"], cfg["block_dim"], cfg["k_blocks"]
    ).to(device)
    model.load_state_dict(
        torch.load(path / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    if hasattr(model, "project_blocks_"):
        with torch.no_grad():
            model.project_blocks_()
    return {
        "model": model,
        "mean": sc["mean"].astype(np.float32),
        "scale": sc["scale"].astype(np.float32),
        "cfg": cfg,
    }


@torch.inference_mode()
def encode_block_norms(
    bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048
) -> np.ndarray:
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    for i in range(0, len(xs), bs):
        z = bundle["model"].encode(
            torch.as_tensor(xs[i : i + bs], device=device)
        )
        outs.append(bundle["model"].block_norms(z).cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def fit_eval_map(
    C_src: np.ndarray,
    C_dst: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    k_mknn: int,
    alpha: float = 1.0,
) -> dict:
    x_sc = StandardScaler().fit(C_src[tr])
    y_sc = StandardScaler().fit(C_dst[tr])
    Xs = x_sc.transform(C_src).astype(np.float64)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(Xs[tr], y_sc.transform(C_dst[tr]))
    W = ridge.coef_.T
    b = ridge.intercept_
    S = np.linalg.svd(W, compute_uv=False)
    stable_rank = float((S.sum() ** 2) / (S**2).sum())

    w_idf = idf_weights(C_dst[tr])
    g_true = knn_graph(C_dst[te] * w_idf[None], k_mknn)

    def eval_W(Wv: np.ndarray) -> float:
        mapped = y_sc.inverse_transform(Xs[te] @ Wv + b).astype(np.float32)
        return mknn(
            g_true, knn_graph(np.maximum(mapped, 0.0) * w_idf[None], k_mknn)
        )

    # partner Jaccard consistency on top-k graph
    def consistency(k_row: int, n_sample: int = 300) -> dict:
        sets = []
        live = []
        Wk = topk_rows(W, k_row)
        for i in range(Wk.shape[0]):
            cols = np.flatnonzero(Wk[i])
            if len(cols) == 0:
                sets.append(set())
                continue
            sets.append(set(cols.tolist()))
            live.append(i)
        live = np.asarray(live)
        rng = np.random.default_rng(0)
        sample = (
            live
            if len(live) <= n_sample
            else rng.choice(live, n_sample, replace=False)
        )

        def jac(a, b):
            if not a and not b:
                return 1.0
            return len(a & b) / max(len(a | b), 1)

        nearest = []
        n_ge_05 = 0
        for i, a in enumerate(sample):
            best = 0.0
            Sa = sets[int(a)]
            for a2 in sample:
                if a2 == a:
                    continue
                best = max(best, jac(Sa, sets[int(a2)]))
            nearest.append(best)
            if best >= 0.5:
                n_ge_05 += 1
        hashes = [frozenset(sets[int(i)]) for i in live]
        counts = Counter(hashes)
        return {
            "nearest_jaccard_median": float(np.median(nearest)),
            "nearest_jaccard_mean": float(np.mean(nearest)),
            "frac_with_jaccard_ge_0.5": float(n_ge_05 / max(len(sample), 1)),
            "n_unique_partner_sets": len(counts),
            "n_live": int(len(live)),
            "largest_exact_dup_group": int(max(counts.values())),
        }

    out = {
        "mknn_full": eval_W(W),
        "mknn_top1": eval_W(topk_rows(W, 1)),
        "mknn_top8": eval_W(topk_rows(W, 8)),
        "mknn_top16": eval_W(topk_rows(W, 16)),
        "mknn_top64": eval_W(topk_rows(W, min(64, W.shape[1]))),
        "stable_rank": stable_rank,
        "consistency_top8": consistency(8),
        "consistency_top16": consistency(16),
    }
    # Hungarian
    ri, ci = linear_sum_assignment(-np.abs(W))
    Wh = np.zeros_like(W)
    Wh[ri, ci] = W[ri, ci]
    out["mknn_hungarian"] = eval_W(Wh)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="dinov3", choices=sorted(MODELS))
    p.add_argument("--dst", default="vit_base", choices=sorted(MODELS))
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--n-blocks", type=int, default=512)
    p.add_argument("--block-dim", type=int, default=4)
    p.add_argument("--k-blocks", type=int, default=16)
    p.add_argument(
        "--featurizer",
        default="vanilla_bsf",
        choices=["vanilla_bsf", "grassmannian_bsf"],
    )
    p.add_argument("--bsf-seed", type=int, default=0)
    p.add_argument("--k-mknn", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-dir", default="outputs/bipartite_matching/bsf_bipartite"
    )
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    src_dir = bsf_run_dir(
        root,
        args.src,
        n_blocks=args.n_blocks,
        block_dim=args.block_dim,
        k_blocks=args.k_blocks,
        seed=args.bsf_seed,
        featurizer=args.featurizer,
    )
    dst_dir = bsf_run_dir(
        root,
        args.dst,
        n_blocks=args.n_blocks,
        block_dim=args.block_dim,
        k_blocks=args.k_blocks,
        seed=args.bsf_seed,
        featurizer=args.featurizer,
    )
    for d in (src_dir, dst_dir):
        if not (d / "model.pt").is_file():
            raise FileNotFoundError(
                f"Missing BSF at {d}. Train with train_vanilla_bsf.py first."
            )

    parquet0, col0 = MODELS[args.src]
    n_full = len(load_col(root / parquet0, col0))
    n = min(args.max_n, n_full) if args.max_n else n_full
    sel = np.sort(rng.choice(n_full, size=n, replace=False))

    X_src = load_col(root / MODELS[args.src][0], MODELS[args.src][1])[sel]
    X_dst = load_col(root / MODELS[args.dst][0], MODELS[args.dst][1])[sel]
    print("Encoding BSF block norms...", flush=True)
    B_src = encode_block_norms(load_bsf(src_dir, device), X_src, device)
    B_dst = encode_block_norms(load_bsf(dst_dir, device), X_dst, device)
    print(f"BSF codes: {B_src.shape} {B_dst.shape}", flush=True)

    print("Encoding SAE codes...", flush=True)
    _, C_src = load_model_codes(root, args.src, sel, device)
    _, C_dst = load_model_codes(root, args.dst, sel, device)

    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    tr, te = np.sort(tr), np.sort(te)

    # dense baseline
    dense = mknn(
        knn_graph(l2n(X_src[te]), args.k_mknn),
        knn_graph(l2n(X_dst[te]), args.k_mknn),
    )

    print("Fitting BSF block-norm shared basis...", flush=True)
    bsf_res = fit_eval_map(B_src, B_dst, tr, te, args.k_mknn)
    print("Fitting SAE shared basis...", flush=True)
    sae_res = fit_eval_map(C_src, C_dst, tr, te, args.k_mknn)

    # also reverse direction for BSF
    print("Fitting reverse BSF map...", flush=True)
    bsf_rev = fit_eval_map(B_dst, B_src, tr, te, args.k_mknn)

    results = {
        "pair": f"{args.src}->{args.dst}",
        "dense_cosine": dense,
        "bsf": bsf_res,
        "bsf_reverse": bsf_rev,
        "sae": sae_res,
        "bsf_cfg": {
            "n_blocks": args.n_blocks,
            "block_dim": args.block_dim,
            "k_blocks": args.k_blocks,
        },
        "elapsed_s": time.time() - t0,
    }

    def show(tag, r):
        print(
            f"  {tag}: full={r['mknn_full']:.4f} top1={r['mknn_top1']:.4f} "
            f"top8={r['mknn_top8']:.4f} top16={r['mknn_top16']:.4f} "
            f"hung={r['mknn_hungarian']:.4f} sr={r['stable_rank']:.0f}",
            flush=True,
        )
        c = r["consistency_top8"]
        print(
            f"         consistency@top8: nearestJ={c['nearest_jaccard_median']:.3f} "
            f"frac_J≥0.5={c['frac_with_jaccard_ge_0.5']:.1%} "
            f"unique={c['n_unique_partner_sets']}/{c['n_live']}",
            flush=True,
        )

    print(f"\ndense cosine mKNN={dense:.4f}", flush=True)
    show("BSF ", bsf_res)
    show("BSF←", bsf_rev)
    show("SAE ", sae_res)

    out_dir = resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.src}__{args.dst}_G{args.n_blocks}_b{args.block_dim}_k{args.k_blocks}"
    (out_dir / f"bsf_bipartite_{tag}.json").write_text(
        json.dumps({"args": vars(args), **results}, indent=2, default=str)
    )
    print(f"Wrote {out_dir}/bsf_bipartite_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
