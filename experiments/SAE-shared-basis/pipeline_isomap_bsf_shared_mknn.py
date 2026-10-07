#!/usr/bin/env python3
"""Isomap residual-elbow dims → BSF → shared-basis Ridge → mKNN.

Reuses Isomap dims from the SAE pipeline. For each pair:
  k_blocks = clamp(ceil(max(d1,d2)) + margin)
  train BSF (vanilla / grassmannian / sae_init) if missing
  Ridge shared basis on full signed codes, cosine mKNN both ways
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_BIP = _HERE.parent / "bipartite-matching"
for path in (_HERE, _BIP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _common import load_col, load_aligned_pair, platonic_root, resolve_path  # noqa: E402

# Prefer bipartite helpers when present
try:
    from run_bsf_bipartite import load_bsf  # noqa: E402
    from bsf_model import GrassmannianBSF, VanillaBSF  # noqa: E402
except ImportError:
    load_bsf = None  # type: ignore
    GrassmannianBSF = None  # type: ignore
    VanillaBSF = None  # type: ignore

PAIRS_YAML = _HERE / "compatible_pairs.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--pairs-yaml", type=Path, default=PAIRS_YAML)
    p.add_argument("--pairs", default=None)
    p.add_argument("--surveys", default=None)
    p.add_argument("--kinds", default=None)
    p.add_argument(
        "--dims-json",
        default="outputs/sae_shared_basis/pipeline_isomap_sae_shared_mknn_rvelbow/isomap_dims.json",
        help="Cached Isomap residual-elbow dims (from SAE pipeline)",
    )
    p.add_argument(
        "--featurizer",
        choices=("vanilla", "grassmannian", "sae_init"),
        default="vanilla",
    )
    p.add_argument("--n-blocks", type=int, default=512)
    p.add_argument("--block-dim", type=int, default=4)
    p.add_argument("--k-margin", type=int, default=8)
    p.add_argument("--k-min", type=int, default=8)
    p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--bsf-epochs", type=int, default=300)
    p.add_argument("--bsf-batch-size", type=int, default=256)
    p.add_argument("--bsf-patience", type=int, default=30)
    p.add_argument("--bsf-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mknn-k", type=int, default=10)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Held-out fraction for mKNN (Ridge fit on the complement). "
        "Use 0.8 for apples-to-apples large-eval protocol.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--phase",
        choices=("all", "train", "eval", "train+eval"),
        default="all",
    )
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-reeval", action="store_true")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: outputs/sae_shared_basis/pipeline_isomap_{featurizer}_bsf_shared_mknn",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_pairs(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text()) or {}
    return data


def filter_pairs(catalog, *, names, surveys, kinds):
    out = {}
    for name, cfg in catalog.items():
        if names is not None and name not in names:
            continue
        if surveys is not None and cfg.get("survey") not in surveys:
            continue
        if kinds is not None and cfg.get("kind") not in kinds:
            continue
        out[name] = cfg
    return out


def embed_key(parquet: Path, col: str) -> str:
    return f"{parquet.stem}::{col}"


def choose_k(d1: float, d2: float, *, margin: int, k_min: int, k_max: int) -> int:
    base = int(np.ceil(max(d1, d2)))
    return int(np.clip(base + margin, k_min, k_max))


def bsf_tag(featurizer: str, *, G: int, b: int, k: int, seed: int) -> str:
    if featurizer == "grassmannian":
        return f"grassmannian_G{G}_b{b}_k{k}_seed{seed}"
    if featurizer == "sae_init":
        return f"sae_init_G{G}_b{b}_k{k}_seed{seed}"
    return f"G{G}_b{b}_k{k}_seed{seed}"


def bsf_dir(
    root: Path,
    parquet: Path,
    col: str,
    *,
    featurizer: str,
    G: int,
    b: int,
    k: int,
    seed: int,
) -> Path:
    return root / "outputs" / "bsf" / parquet.stem / col / bsf_tag(
        featurizer, G=G, b=b, k=k, seed=seed
    )


def sae_dir_for(root: Path, parquet: Path, col: str, k: int, seed: int) -> Path:
    """Prefer elbow-matched TopK SAE; fall back to k64 if needed."""
    preferred = root / "outputs" / "sae" / parquet.stem / col / f"F2048_k{k}_seed{seed}"
    if (preferred / "model.pt").is_file():
        return preferred
    for alt_k in (64, 32, 20, 16):
        alt = root / "outputs" / "sae" / parquet.stem / col / f"F2048_k{alt_k}_seed{seed}"
        if (alt / "model.pt").is_file():
            return alt
    raise FileNotFoundError(
        f"No F2048 SAE for {parquet.stem}/{col} (wanted k={k})"
    )


def train_bsf(
    root: Path,
    parquet: Path,
    col: str,
    *,
    featurizer: str,
    G: int,
    b: int,
    k: int,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float,
    device: str,
    dry_run: bool,
) -> Path:
    out = bsf_dir(
        root, parquet, col, featurizer=featurizer, G=G, b=b, k=k, seed=seed
    )
    if (out / "model.pt").is_file():
        print(f"  BSF exists: {out}", flush=True)
        return out
    pq_arg = (
        str(parquet.relative_to(root))
        if str(parquet).startswith(str(root))
        else str(parquet)
    )
    bip = _BIP if (_BIP / "train_vanilla_bsf.py").is_file() else (
        root / "experiments" / "bipartite-matching"
    )
    py = str(root / ".venv" / "bin" / "python")
    if featurizer == "grassmannian":
        script = bip / "train_grassmannian_bsf.py"
        cmd = [
            py, str(script),
            "--parquet", pq_arg, "--column", col,
            "--n-blocks", str(G), "--block-dim", str(b), "--k-blocks", str(k),
            "--epochs", str(epochs), "--batch-size", str(batch_size),
            "--patience", str(patience), "--lr", str(lr), "--seed", str(seed),
            "--device", device, "--platonic-root", str(root), "--out-dir", "outputs/bsf",
        ]
    elif featurizer == "sae_init":
        script = bip / "train_bsf_from_sae.py"
        sae_path = sae_dir_for(root, parquet, col, k, seed)
        cmd = [
            py, str(script),
            "--parquet", pq_arg, "--column", col, "--sae-dir", str(sae_path),
            "--n-blocks", str(G), "--block-dim", str(b), "--k-blocks", str(k),
            "--epochs", str(epochs), "--batch-size", str(batch_size),
            "--patience", str(patience), "--lr", str(lr), "--seed", str(seed),
            "--device", device, "--platonic-root", str(root), "--out-dir", "outputs/bsf",
        ]
    else:
        script = bip / "train_vanilla_bsf.py"
        cmd = [
            py, str(script),
            "--parquet", pq_arg, "--column", col,
            "--n-blocks", str(G), "--block-dim", str(b), "--k-blocks", str(k),
            "--epochs", str(epochs), "--batch-size", str(batch_size),
            "--patience", str(patience), "--lr", str(lr), "--seed", str(seed),
            "--device", device, "--platonic-root", str(root), "--out-dir", "outputs/bsf",
        ]
    print("  TRAIN:", " ".join(cmd), flush=True)
    if dry_run:
        return out
    env = os.environ.copy()
    env["PLATONIC_ROOT"] = str(root)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, cwd=str(root), env=env, check=True)
    if not (out / "model.pt").is_file():
        raise RuntimeError(f"train finished but missing {out}/model.pt")
    return out


def load_bsf_generic(path: Path, device: torch.device) -> dict:
    if load_bsf is not None:
        return load_bsf(path, device)
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
def encode_full(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048):
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    for i in range(0, len(xs), bs):
        z = bundle["model"].encode(torch.as_tensor(xs[i : i + bs], device=device))
        outs.append(z.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def l2n(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


@torch.inference_mode()
def knn_cos(Z: torch.Tensor, k: int, row_batch: int = 256) -> torch.Tensor:
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
    a, b = nn1.cpu().numpy(), nn2.cpu().numpy()
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


def shared_basis_cosine(
    C_src: np.ndarray,
    C_dst: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    *,
    k: int,
    alpha: float,
    device: torch.device,
) -> float:
    x_sc = StandardScaler().fit(C_src[tr])
    y_sc = StandardScaler().fit(C_dst[tr])
    Xs = x_sc.transform(C_src).astype(np.float64)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(Xs[tr], y_sc.transform(C_dst[tr]))
    mapped = y_sc.inverse_transform(Xs[te] @ ridge.coef_.T + ridge.intercept_).astype(
        np.float32
    )
    true = C_dst[te]
    return mknn(
        knn_cos(torch.as_tensor(true, device=device), k),
        knn_cos(torch.as_tensor(mapped, device=device), k),
        k,
    )


def eval_pair(
    root: Path,
    *,
    pair_name: str,
    parquet1: Path,
    col1: str,
    parquet2: Path,
    col2: str,
    bsf1: Path,
    bsf2: Path,
    max_n: int,
    mknn_k: int,
    alpha: float,
    test_size: float,
    seed: int,
    device: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    if results_path.is_file():
        return json.loads(results_path.read_text())

    t0 = time.time()
    X1, X2 = load_aligned_pair(parquet1, col1, parquet2, col2)
    n_full = len(X1)
    n = min(max_n, n_full) if max_n else n_full
    rng = np.random.default_rng(seed)
    sel = (
        np.sort(rng.choice(n_full, size=n, replace=False))
        if n < n_full
        else np.arange(n_full)
    )
    X1, X2 = X1[sel], X2[sel]
    dev = torch.device(device)
    b1 = load_bsf_generic(bsf1, dev)
    b2 = load_bsf_generic(bsf2, dev)
    Z1 = encode_full(b1, X1, dev)
    Z2 = encode_full(b2, X2, dev)

    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=test_size, random_state=seed, shuffle=True
    )
    tr, te = np.sort(tr), np.sort(te)

    dense = mknn(
        knn_cos(torch.as_tensor(l2n(X1[te]), device=dev), mknn_k),
        knn_cos(torch.as_tensor(l2n(X2[te]), device=dev), mknn_k),
        mknn_k,
    )
    unmapped = mknn(
        knn_cos(torch.as_tensor(Z1[te], device=dev), mknn_k),
        knn_cos(torch.as_tensor(Z2[te], device=dev), mknn_k),
        mknn_k,
    )
    a_to_b = shared_basis_cosine(
        Z1, Z2, tr, te, k=mknn_k, alpha=alpha, device=dev
    )
    b_to_a = shared_basis_cosine(
        Z2, Z1, tr, te, k=mknn_k, alpha=alpha, device=dev
    )
    # relu_cosine variants (optional secondary)
    a_to_b_relu = shared_basis_cosine(
        np.maximum(Z1, 0),
        np.maximum(Z2, 0),
        tr,
        te,
        k=mknn_k,
        alpha=alpha,
        device=dev,
    )
    b_to_a_relu = shared_basis_cosine(
        np.maximum(Z2, 0),
        np.maximum(Z1, 0),
        tr,
        te,
        k=mknn_k,
        alpha=alpha,
        device=dev,
    )
    payload = {
        "pair": pair_name,
        "n": int(n),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_size": float(test_size),
        "protocol": (
            "Same held-out rows for dense / unmapped / shared cosine mKNN; "
            f"Ridge fit on train ({1.0 - test_size:.0%}), eval on test ({test_size:.0%}). "
            "Not Platonic-Universe Table-2 full-N ambient."
        ),
        "k_blocks": int(b1["cfg"]["k_blocks"]),
        "n_blocks": int(b1["cfg"]["n_blocks"]),
        "block_dim": int(b1["cfg"]["block_dim"]),
        "dense_cosine": float(dense),
        "bsf_unmapped_cosine": float(unmapped),
        "shared_1_to_2_cosine": float(a_to_b),
        "shared_2_to_1_cosine": float(b_to_a),
        "shared_best_cosine": float(max(a_to_b, b_to_a)),
        "shared_1_to_2_relu_cosine": float(a_to_b_relu),
        "shared_2_to_1_relu_cosine": float(b_to_a_relu),
        "shared_best_relu_cosine": float(max(a_to_b_relu, b_to_a_relu)),
        "elapsed_s": time.time() - t0,
        "col1": col1,
        "col2": col2,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    (out_dir / "results.md").write_text(
        f"# {pair_name}\n\n"
        f"- protocol: Ridge train {(1.0 - test_size):.0%} / mKNN test {test_size:.0%} "
        f"(n_train={len(tr)}, n_test={len(te)})\n"
        f"- dense={dense:.4f}\n"
        f"- bsf unmapped cosine={unmapped:.4f}\n"
        f"- shared best cosine={payload['shared_best_cosine']:.4f} "
        f"(1→2={a_to_b:.4f}, 2→1={b_to_a:.4f})\n"
        f"- shared best relu_cosine={payload['shared_best_relu_cosine']:.4f}\n"
    )
    return payload


def main() -> None:
    args = parse_args()
    root = platonic_root(args.platonic_root)
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/sae_shared_basis/pipeline_isomap_{args.featurizer}_bsf_shared_mknn"
        )
    out_root = resolve_path(root, args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    feat = args.featurizer
    print(f"Featurizer: {feat}", flush=True)

    catalog = load_pairs(args.pairs_yaml)
    names = [x.strip() for x in args.pairs.split(",")] if args.pairs else None
    surveys = [x.strip() for x in args.surveys.split(",")] if args.surveys else None
    kinds = [x.strip() for x in args.kinds.split(",")] if args.kinds else None
    pairs = filter_pairs(catalog, names=names, surveys=surveys, kinds=kinds)
    if not pairs:
        raise SystemExit("No pairs selected")

    dims_path = resolve_path(root, args.dims_json)
    if not dims_path.is_file():
        raise SystemExit(f"Missing Isomap dims cache: {dims_path}")
    dims = json.loads(dims_path.read_text())
    print(f"Loaded {len(dims)} Isomap dims from {dims_path}", flush=True)

    plan = []
    for name, cfg in pairs.items():
        p1 = resolve_path(root, cfg["parquet1"])
        p2 = resolve_path(root, cfg["parquet2"])
        c1, c2 = cfg["col1"], cfg["col2"]
        d1 = float(dims.get(embed_key(p1, c1), {}).get("d_primary", float("nan")))
        d2 = float(dims.get(embed_key(p2, c2), {}).get("d_primary", float("nan")))
        # prefer residual elbow if present
        e1 = dims.get(embed_key(p1, c1), {})
        e2 = dims.get(embed_key(p2, c2), {})
        if "d_residual_elbow" in e1:
            d1 = float(e1["d_residual_elbow"])
        if "d_residual_elbow" in e2:
            d2 = float(e2["d_residual_elbow"])
        if not (np.isfinite(d1) and np.isfinite(d2)):
            print(f"[plan] SKIP {name}: missing dims", flush=True)
            continue
        k_blocks = choose_k(
            d1, d2, margin=args.k_margin, k_min=args.k_min, k_max=args.k_max
        )
        plan.append(
            {
                "name": name,
                "kind": cfg.get("kind"),
                "survey": cfg.get("survey"),
                "featurizer": feat,
                "parquet1": str(p1),
                "col1": c1,
                "parquet2": str(p2),
                "col2": c2,
                "d1": d1,
                "d2": d2,
                "k_blocks": k_blocks,
                "max_n": int(cfg.get("default_max_n", 16384)),
                "bsf1": str(
                    bsf_dir(
                        root,
                        p1,
                        c1,
                        featurizer=feat,
                        G=args.n_blocks,
                        b=args.block_dim,
                        k=k_blocks,
                        seed=args.seed,
                    )
                ),
                "bsf2": str(
                    bsf_dir(
                        root,
                        p2,
                        c2,
                        featurizer=feat,
                        G=args.n_blocks,
                        b=args.block_dim,
                        k=k_blocks,
                        seed=args.seed,
                    )
                ),
            }
        )
    (out_root / "pair_plan.json").write_text(json.dumps(plan, indent=2))
    print(f"\nPair plan ({len(plan)}):", flush=True)
    for row in plan:
        print(
            f"  {row['name']}: d=({row['d1']:.0f},{row['d2']:.0f}) → k_blocks={row['k_blocks']}",
            flush=True,
        )

    do_train = args.phase in ("all", "train", "train+eval")
    do_eval = args.phase in ("all", "eval", "train+eval")

    if do_train:
        jobs = {}
        for row in plan:
            for side in (1, 2):
                pq = Path(row[f"parquet{side}"])
                col = row[f"col{side}"]
                k = row["k_blocks"]
                jobs[(str(pq), col, k)] = (pq, col, k)
        print(f"\nTraining up to {len(jobs)} {feat} BSFs...", flush=True)
        for pq_s, col, k in sorted(jobs.keys()):
            pq = Path(pq_s)
            dest = bsf_dir(
                root,
                pq,
                col,
                featurizer=feat,
                G=args.n_blocks,
                b=args.block_dim,
                k=k,
                seed=args.seed,
            )
            if (dest / "model.pt").is_file() and not args.force_retrain:
                print(f"  skip existing {dest.relative_to(root)}", flush=True)
                continue
            train_bsf(
                root,
                pq,
                col,
                featurizer=feat,
                G=args.n_blocks,
                b=args.block_dim,
                k=k,
                seed=args.seed,
                epochs=args.bsf_epochs,
                batch_size=args.bsf_batch_size,
                patience=args.bsf_patience,
                lr=args.bsf_lr,
                device=args.device,
                dry_run=args.dry_run,
            )

    summaries = []
    if do_eval:
        print(f"\nEvaluating {len(plan)} pairs...", flush=True)
        for row in plan:
            eval_dir = out_root / "evals" / f"{row['name']}_k{row['k_blocks']}"
            results_path = eval_dir / "results.json"
            if results_path.is_file() and not args.force_reeval:
                print(f"  skip existing eval {row['name']}", flush=True)
                payload = json.loads(results_path.read_text())
            else:
                if args.force_reeval and results_path.is_file():
                    results_path.unlink()
                if args.dry_run:
                    print(f"  DRY eval {row['name']}", flush=True)
                    continue
                try:
                    payload = eval_pair(
                        root,
                        pair_name=row["name"],
                        parquet1=Path(row["parquet1"]),
                        col1=row["col1"],
                        parquet2=Path(row["parquet2"]),
                        col2=row["col2"],
                        bsf1=Path(row["bsf1"]),
                        bsf2=Path(row["bsf2"]),
                        max_n=row["max_n"],
                        mknn_k=args.mknn_k,
                        alpha=args.ridge_alpha,
                        test_size=args.test_size,
                        seed=args.seed,
                        device=args.device,
                        out_dir=eval_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  EVAL FAILED {row['name']}: {exc}", flush=True)
                    summaries.append({**row, "error": str(exc)})
                    continue
            summaries.append({**row, **payload})
            print(
                f"  {row['name']}: dense={payload.get('dense_cosine'):.4f} "
                f"shared_best={payload.get('shared_best_cosine'):.4f}",
                flush=True,
            )

    payload = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "n_pairs": len(plan),
        "plan": plan,
        "summaries": summaries,
        "elapsed_s": time.time() - t_all,
    }
    (out_root / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Isomap → {feat} BSF shared-basis mKNN",
        "",
        f"Pairs: {len(plan)} | featurizer={feat} | G={args.n_blocks} b={args.block_dim} | "
        f"k_margin={args.k_margin} | test_size={args.test_size}",
        "",
        "Protocol: same held-out rows for dense / unmapped / shared cosine; "
        f"Ridge fit on train ({1.0 - args.test_size:.0%}), mKNN on test ({args.test_size:.0%}). "
        "Within-pipeline comparison only — not Platonic-Universe Table-2 full-N ambient.",
        "",
        "| pair | kind | d1 | d2 | k_blocks | n_train | n_test | dense | bsf_unmap | shared_best_cos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        if "error" in s:
            lines.append(
                f"| {s['name']} | {s.get('kind','')} | | | | | | ERROR | | |"
            )
            continue
        lines.append(
            f"| {s['name']} | {s.get('kind','')} | {s.get('d1', float('nan')):.0f} | "
            f"{s.get('d2', float('nan')):.0f} | {s.get('k_blocks','')} | "
            f"{s.get('n_train', '')} | {s.get('n_test', '')} | "
            f"{s.get('dense_cosine', float('nan')):.4f} | "
            f"{s.get('bsf_unmapped_cosine', float('nan')):.4f} | "
            f"{s.get('shared_best_cosine', float('nan')):.4f} |"
        )
    (out_root / "pipeline_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_root / 'pipeline_summary.md'} in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
