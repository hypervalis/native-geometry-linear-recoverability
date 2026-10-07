#!/usr/bin/env python3
"""Affine map between TopK SAE codes: express one model in the other's basis.

Fits C_src ≈ C_tgt @ W + b (Ridge) on a train split, so mapped target codes
live in the source SAE feature space. Then scores:
  - code reconstruction (MSE / R² / cosine) of source codes
  - binary active-set overlap
  - cross-model mKNN in the shared source basis: knn(C_src) vs knn(C_tgt@W+b)
  - baselines: unmapped SAE IDF mKNN, dense cosine
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_SAE_CANDIDATES = [
    Path(__file__).resolve().parent / "sae",
]
_SAE = next((p for p in _SAE_CANDIDATES if (p / "sae_model.py").is_file()), None)
if _SAE is None:
    raise FileNotFoundError("sae_model.py not found")
sys.path.insert(0, str(_SAE))

from sae_model import TopKSAE  # noqa: E402


def load_col(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    return np.vstack(table.column(0).to_pylist()).astype(np.float32)


def l2n(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.inference_mode()
def knn_cos(Z: torch.Tensor, k: int, row_batch: int = 256) -> torch.Tensor:
    Z = l2n(Z)
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
def mknn(nn1: torch.Tensor, nn2: torch.Tensor, k: int) -> float:
    a, b = nn1.cpu().numpy(), nn2.cpu().numpy()
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


def load_sae(sae_dir: Path, device: torch.device) -> dict:
    cfg = json.loads((sae_dir / "config.json").read_text())
    sc = np.load(sae_dir / "scaler_stats.npz")
    model = TopKSAE(cfg["dim"], cfg["feature_dim"], cfg["k"]).to(device)
    model.load_state_dict(
        torch.load(sae_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return {
        "model": model,
        "mean": sc["mean"].astype(np.float32),
        "scale": sc["scale"].astype(np.float32),
        "k": int(cfg["k"]),
        "feature_dim": int(cfg["feature_dim"]),
    }


@torch.inference_mode()
def encode(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    for i in range(0, len(xs), bs):
        _, z = bundle["model"](torch.as_tensor(xs[i : i + bs], device=device))
        outs.append(z.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def cosine_rowwise(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return float((an * bn).sum(axis=1).mean())


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    true_a = y_true > 0
    # top-k of prediction magnitudes
    kk = min(k, y_pred.shape[1])
    top = np.argpartition(-y_pred, kk - 1, axis=1)[:, :kk]
    pred_a = np.zeros_like(true_a)
    for i in range(len(y_pred)):
        pred_a[i, top[i]] = True
    tp = (true_a & pred_a).sum(axis=1).astype(np.float64)
    prec = tp / np.maximum(pred_a.sum(axis=1), 1)
    rec = tp / np.maximum(true_a.sum(axis=1), 1)
    union = (true_a | pred_a).sum(axis=1).astype(np.float64)
    jacc = tp / np.maximum(union, 1)
    return {
        "precision_at_k": float(prec.mean()),
        "recall_at_k": float(rec.mean()),
        "jaccard_at_k": float(jacc.mean()),
    }


def fit_affine_express_in_basis(
    codes_basis: np.ndarray,
    codes_other: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    """Express `other` in `basis` feature space: basis ≈ other @ W + b.

    Returns mapped other codes (test+train full via transform) and metrics.
    """
    x_tr = codes_other[train_idx]  # predictors
    y_tr = codes_basis[train_idx]  # targets = basis coords
    x_te = codes_other[test_idx]
    y_te = codes_basis[test_idx]

    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr)
    xs_tr = x_scaler.transform(x_tr)
    ys_tr = y_scaler.transform(y_tr)
    xs_te = x_scaler.transform(x_te)

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(xs_tr, ys_tr)
    # W maps standardized other → standardized basis; sklearn: y = X @ coef_.T + intercept_
    ys_hat_te = ridge.predict(xs_te)
    y_hat_te = y_scaler.inverse_transform(ys_hat_te)
    ys_hat_tr = ridge.predict(xs_tr)
    y_hat_tr = y_scaler.inverse_transform(ys_hat_tr)

    # Map all rows
    xs_all = x_scaler.transform(codes_other)
    y_hat_all = y_scaler.inverse_transform(ridge.predict(xs_all))

    def pack(y, yhat, split: str) -> dict:
        return {
            "split": split,
            "mse": float(mean_squared_error(y, yhat)),
            "r2": float(r2_score(y, yhat, multioutput="uniform_average")),
            "cosine": cosine_rowwise(y, yhat),
            "binary": binary_metrics(y, yhat, k=int((y > 0).sum(axis=1).mean())),
        }

    # Dense coefficient stats in original (unstandardized) coordinates approx via
    # effective map on scaled space: report singular spectrum of coef_
    svals = np.linalg.svd(ridge.coef_, compute_uv=False)
    return {
        "y_hat_all": y_hat_all.astype(np.float32),
        "train": pack(y_tr, y_hat_tr, "train"),
        "test": pack(y_te, y_hat_te, "test"),
        "alpha": alpha,
        "coef_svals_top10": [float(x) for x in svals[:10]],
        "coef_nuclear": float(svals.sum()),
        "coef_fro": float(np.linalg.norm(ridge.coef_)),
        "effective_rank_95": int(np.searchsorted(np.cumsum(svals) / svals.sum(), 0.95) + 1),
    }


def idf_np(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    df = (C > 0).sum(axis=0).astype(np.float64)
    return (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet1", default="data_hf/physics/vit_base_test.parquet")
    p.add_argument("--col1", default="vit_base_galaxies")
    p.add_argument("--parquet2", default="data_hf/physics/dinov3_vitb16_test.parquet")
    p.add_argument("--col2", default="dinov3_vitb16_galaxies")
    p.add_argument(
        "--sae1",
        default="outputs/sae/vit_base_test/vit_base_galaxies/F2048_k64_seed0",
    )
    p.add_argument(
        "--sae2",
        default="outputs/sae/dinov3_vitb16_test/dinov3_vitb16_galaxies/F2048_k64_seed0",
    )
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument(
        "--output-dir",
        default="outputs/sae_affine_basis/physics_vit_dino_n16k_F2048_k64",
    )
    args = p.parse_args()

    root = Path("/home/angus/platonic-universe")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    def R(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    X1 = load_col(R(args.parquet1), args.col1)
    X2 = load_col(R(args.parquet2), args.col2)
    n = min(len(X1), len(X2))
    X1, X2 = X1[:n], X2[:n]
    rng = np.random.default_rng(args.seed)
    if args.max_n and n > args.max_n:
        sel = np.sort(rng.choice(n, size=args.max_n, replace=False))
        X1, X2 = X1[sel], X2[sel]
        n = args.max_n
    print(f"n={n}", flush=True)

    b1 = load_sae(R(args.sae1), device)
    b2 = load_sae(R(args.sae2), device)
    print("Encoding SAE codes...", flush=True)
    C1 = encode(b1, X1, device)  # ViT basis
    C2 = encode(b2, X2, device)  # DINO
    print(f"codes {C1.shape} / {C2.shape}", flush=True)

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"train={len(train_idx)} test={len(test_idx)}", flush=True)

    # Direction A: express DINO in ViT SAE basis  (C1 ≈ C2 W + b)
    print("Fit: ViT_basis ≈ affine(DINO_codes)...", flush=True)
    dino_in_vit = fit_affine_express_in_basis(
        C1, C2, train_idx=train_idx, test_idx=test_idx, alpha=args.alpha
    )
    # Direction B: express ViT in DINO SAE basis
    print("Fit: DINO_basis ≈ affine(ViT_codes)...", flush=True)
    vit_in_dino = fit_affine_express_in_basis(
        C2, C1, train_idx=train_idx, test_idx=test_idx, alpha=args.alpha
    )

    for name, block in [("dino_in_vit", dino_in_vit), ("vit_in_dino", vit_in_dino)]:
        te = block["test"]
        print(
            f"  {name} test: mse={te['mse']:.4f} r2={te['r2']:.4f} cos={te['cosine']:.4f} "
            f"jacc@k={te['binary']['jaccard_at_k']:.4f}  "
            f"effrank95(W)={block['effective_rank_95']}",
            flush=True,
        )

    # mKNN evaluations on TEST split only (held-out affine)
    Z1 = torch.as_tensor(X1, device=device)
    Z2 = torch.as_tensor(X2, device=device)
    # L2 for dense
    Z1n = Z1 / Z1.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Z2n = Z2 / Z2.norm(dim=1, keepdim=True).clamp_min(1e-12)

    C1_t = torch.as_tensor(C1, device=device)
    C2_t = torch.as_tensor(C2, device=device)
    D_in_V = torch.as_tensor(dino_in_vit["y_hat_all"], device=device)
    V_in_D = torch.as_tensor(vit_in_dino["y_hat_all"], device=device)

    te = torch.as_tensor(test_idx, device=device, dtype=torch.long)
    rows = []

    def add(method: str, A: torch.Tensor, B: torch.Tensor) -> None:
        t0 = time.time()
        s = mknn(
            knn_cos(A[te], args.k, args.row_batch),
            knn_cos(B[te], args.k, args.row_batch),
            args.k,
        )
        rows.append({"method": method, "split": "test", "mknn": s, "sec": time.time() - t0})
        print(f"  mknn {method:<40} {s:.4f}", flush=True)

    add("dense_cosine", Z1n, Z2n)
    add("sae_codes_cosine", C1_t, C2_t)
    idf1, idf2 = idf_np(C1[train_idx]), idf_np(C2[train_idx])  # IDF from train only
    add(
        "sae_idf_cosine",
        C1_t * torch.as_tensor(idf1, device=device)[None, :],
        C2_t * torch.as_tensor(idf2, device=device)[None, :],
    )
    # Shared ViT basis: true ViT codes vs DINO mapped into ViT basis
    add("shared_vit_basis_cosine", C1_t, D_in_V)
    add(
        "shared_vit_basis_idf",
        C1_t * torch.as_tensor(idf1, device=device)[None, :],
        D_in_V * torch.as_tensor(idf1, device=device)[None, :],  # same basis IDF
    )
    # Shared DINO basis
    add("shared_dino_basis_cosine", C2_t, V_in_D)
    idf2_t = torch.as_tensor(idf2, device=device)
    add(
        "shared_dino_basis_idf",
        C2_t * idf2_t[None, :],
        V_in_D * idf2_t[None, :],
    )
    # Also: mapped-vs-mapped in one basis (both expressed?) — skip
    # Residual: how much of source neighborhood is recovered by mapped target alone
    # Compare knn(C1) vs knn(D_in_V) — same as shared_vit_basis

    out_dir = R(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # save maps compactly
    np.savez_compressed(
        out_dir / "mapped_codes_test.npz",
        test_idx=test_idx,
        train_idx=train_idx,
        dino_in_vit_test=dino_in_vit["y_hat_all"][test_idx],
        vit_in_dino_test=vit_in_dino["y_hat_all"][test_idx],
        c1_test=C1[test_idx],
        c2_test=C2[test_idx],
    )

    payload = {
        "meta": {
            "n": n,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "alpha": args.alpha,
            "k": args.k,
            "seed": args.seed,
            "sae_k": b1["k"],
            "feature_dim": b1["feature_dim"],
            "protocol": "basis ≈ other @ W + b  (Ridge on standardized codes)",
        },
        "dino_in_vit": {
            k: v
            for k, v in dino_in_vit.items()
            if k != "y_hat_all"
        },
        "vit_in_dino": {k: v for k, v in vit_in_dino.items() if k != "y_hat_all"},
        "mknn_rows": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Affine SAE-code basis transfer",
        "",
        f"- n={n}, train={len(train_idx)}, test={len(test_idx)}, Ridge α={args.alpha}",
        f"- TopK SAE F={b1['feature_dim']} k={b1['k']}",
        "- Map: `basis_codes ≈ other_codes @ W + b` (other expressed in basis space)",
        "",
        "## Code prediction (test)",
        "",
        "| direction | MSE | R² | cosine | Jaccard@L0 | eff-rank95(W) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in [("DINO→ViT basis", dino_in_vit), ("ViT→DINO basis", vit_in_dino)]:
        te = block["test"]
        lines.append(
            f"| {name} | {te['mse']:.4f} | {te['r2']:.4f} | {te['cosine']:.4f} | "
            f"{te['binary']['jaccard_at_k']:.4f} | {block['effective_rank_95']} |"
        )
    lines += [
        "",
        "## Cross-model mKNN on test (shared basis)",
        "",
        "| method | mknn |",
        "|---|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r['mknn']:.4f} |")
    best = max(rows, key=lambda r: r["mknn"])
    lines += [
        "",
        f"Best: `{best['method']}` mknn={best['mknn']:.4f}.",
        "",
        "Shared-basis rows compare neighborhoods of true basis codes vs the "
        "affine image of the other model’s codes in that same feature space.",
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir}", flush=True)
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
