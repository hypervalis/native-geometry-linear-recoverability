# SAE TopK sensitivity

## What K means here

**Inference-time TopK override**, not retraining.

Each of the 16 official Legacy↔HSC rungs has a single trained F=2048 seed=0
checkpoint with training TopK in {18,...,23} (heterogeneous across rungs; no
common training-time K across the full ladder). We load that checkpoint and set
`TopKSAE.k` before `encode`, so sparsity at evaluation varies while encoder /
decoder weights stay fixed.

This is **not** training-time TopK sensitivity.

## Available trained TopK (paper protocol preference)

Per-rung preferred tags (Legacy and HSC match):
{
  "official_legacy_cross_astropt_15m": {
    "train_k": 22,
    "tag": "F2048_k22_seed0"
  },
  "official_legacy_cross_astropt_95m": {
    "train_k": 19,
    "tag": "F2048_k19_seed0"
  },
  "official_legacy_cross_astropt_850m": {
    "train_k": 19,
    "tag": "F2048_k19_seed0"
  },
  "official_legacy_cross_convnext_nano": {
    "train_k": 21,
    "tag": "F2048_k21_seed0"
  },
  "official_legacy_cross_convnext_tiny": {
    "train_k": 21,
    "tag": "F2048_k21_seed0"
  },
  "official_legacy_cross_convnext_base": {
    "train_k": 21,
    "tag": "F2048_k21_seed0"
  },
  "official_legacy_cross_convnext_large": {
    "train_k": 20,
    "tag": "F2048_k20_seed0"
  },
  "official_legacy_cross_dino_small": {
    "train_k": 20,
    "tag": "F2048_k20_seed0"
  },
  "official_legacy_cross_dino_base": {
    "train_k": 22,
    "tag": "F2048_k22_seed0"
  },
  "official_legacy_cross_dino_large": {
    "train_k": 21,
    "tag": "F2048_k21_seed0"
  },
  "official_legacy_cross_dino_giant": {
    "train_k": 22,
    "tag": "F2048_k22_seed0"
  },
  "official_legacy_cross_vit_base": {
    "train_k": 20,
    "tag": "F2048_k20_seed0"
  },
  "official_legacy_cross_vit_large": {
    "train_k": 23,
    "tag": "F2048_k23_seed0"
  },
  "official_legacy_cross_vit_huge": {
    "train_k": 23,
    "tag": "F2048_k23_seed0"
  },
  "official_legacy_cross_ijepa_huge": {
    "train_k": 18,
    "tag": "F2048_k18_seed0"
  },
  "official_legacy_cross_ijepa_giant": {
    "train_k": 18,
    "tag": "F2048_k18_seed0"
  }
}

Independently trained common-K grids (e.g. F2048_K8/16/24/32 on all 16 rungs)
are **not** available.

## Chosen K grid

`[10, 20, 40]`

Reason: spans below / near / above the operating training range 18--23, and is
supported for all 16 rungs via inference-time override on the preferred
checkpoint. Same K is used for Legacy and HSC within each rung.

## Protocol (frozen)

- SAE width F=2048, seed=0 checkpoints
- Mapping direction Legacy → HSC (col2 → col1)
- n=16384, test_size=0.2, random_state=0
- train-only Ridge (α=1.0), StandardScaler on X and Y
- train-only IDF on HSC SAE codes
- mKNN k=10, self-exclusion
- **test-only gallery** (same as paper alignment controls)
- Dense / Dense+Ridge / native taken from `outputs/paper_alignment_controls/test_only_gallery_scores.csv` at k=10
  (same frozen split; not retuned per K)
- Ridge not retuned per K; K not chosen using held-out mKNN

## Missing rungs

None for K∈[10, 20, 40] — all 16 rungs evaluated at each K.

## Primary quantities

- `S_SAE = M_SAE+Ridge - M_Dense+Ridge`
- `Δβ_SAE,F = β_SAE+Ridge,F - β_Dense+Ridge,F` within family
- `T_SAE,K = mean_F Δβ_SAE,F,K`

Elapsed wall time: 289.4s on cuda.
