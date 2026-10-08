# Native Geometry vs. Linear Recoverability

Analysis code for **Native Geometry vs. Linear Recoverability: How Alignment Changes Platonic Scaling**.

The question is whether cross-survey correspondence between Legacy Survey and HSC embeddings scales with model size in the native cosine geometry, or only after a supervised affine map. On a frozen held-out split the code compares

- native cosine mutual \(k\)NN, \(M_{\mathrm{native}}\),
- the same mutual \(k\)NN after a train-only affine Ridge map, \(M_{\mathrm{linear}}\),

and the equal-family slope gap \(T = \tfrac{1}{5}\sum_F(\beta_{\mathrm{Dense+Ridge},F}-\beta_{\mathrm{Dense},F})\) of mKNN@10 against \(\log_{10} P\).

`outputs/` contains the tables behind the camera-ready figures and appendix checks. Raw embeddings and SAE/BSF checkpoints are not included.

## Layout

| Path | Role |
| --- | --- |
| `src/nglr/analysis/run_ridge_scaling_geometry.py` | Primary \(T\), object bootstrap, correspondence-shuffle null |
| `src/nglr/analysis/run_fixed_rank_scaling.py` | Independent PCA rank 256, \(T_{256}\) |
| `src/nglr/analysis/run_final_pca_rank_sweep.py` | Independent PCA ranks 128/256/512 |
| `src/nglr/analysis/run_raw_pca_control.py` | Pooled train-only PCA, no pairing |
| `src/nglr/analysis/run_singular_spectrum_analysis.py` | Singular spectrum of \(A_{\mathrm{eff}}\) |
| `src/nglr/analysis/run_anisotropy_concentration.py` | Anisotropy mass vs. functional mKNN lift |
| `src/nglr/analysis/run_translation_ablation.py` | Affine vs. linear-only ablation |
| `src/nglr/analysis/run_data_supported_distortion.py` | Local edge stretch \(\sigma_{\mathrm{local}}\) |
| `src/nglr/analysis/run_direction_k_robustness.py` | HSC→Legacy and \(k\in\{5,10,25,50\}\) |
| `src/nglr/analysis/run_ridge_alpha_robustness.py` | Ridge \(\alpha\) grid |
| `src/nglr/analysis/run_alignment_controls.py` | Dense+Ridge vs. SAE and BSF |
| `src/nglr/analysis/run_sae_topk_sensitivity.py` | Inference-time TopK \(K\in\{10,20,40\}\) |
| `src/nglr/analysis/run_cross_model_symmetry.py` | Same-Legacy cross-model control |
| `src/nglr/analysis/run_patch_residual_diagnostic.py` | Residual locality of the global Ridge map |
| `src/nglr/scaling/run_relational_geometry_size_scaling.py` | Unpaired DualEncoder control (appendix) |
| `src/nglr/alignment/run_unpaired_universal_geometry.py` | DualEncoder used by that control |
| `src/nglr/bsf/` | Block-sparse factorization (BSF) control |
| `src/nglr/plots/plot_*.py` | Camera-ready figures from the released tables |
| `configs/official_legacy_pairs.yaml` | Sixteen size rungs and parameter counts |

The importable package is `nglr` (`pip install -e .`). The scripts also run directly, without an install.

Frozen protocol, shared by the primary scripts: \(n=16384\), `test_size=0.2`, `random_state=0`, Legacy→HSC, train-only `StandardScaler` on both sides, `Ridge(alpha=1, fit_intercept=True)`, cosine mKNN with a test-only gallery.

## Rebuild the figures

From this directory, with the packages in `requirements.txt`:

```bash
python src/nglr/plots/plot_scaling_figure.py
python src/nglr/plots/plot_geometry_anisotropy_functional.py
python src/nglr/plots/plot_translation_ablation.py
python src/nglr/plots/plot_geometry_figure.py
python src/nglr/plots/plot_geometry_scale_diagnostics.py
python src/nglr/plots/plot_sae_topk_sensitivity.py
```

`plot_scaling_figure.py` is Figure 1, `plot_geometry_anisotropy_functional.py` is Figure 2, `plot_translation_ablation.py` is Figure 3, and `plot_sae_topk_sensitivity.py` is Figure 4. `plot_geometry_figure.py` and `plot_geometry_scale_diagnostics.py` are the \(D_{\mathrm{sim}}\) and \(\sigma_{\mathrm{local}}\) diagnostics from the geometry appendix.

Figures are written to `src/nglr/plots/figures/`. The affine-reconstruction identity used by the translation ablation is checked by

```bash
python -m pytest
```

## Recompute from embeddings

Set `PLATONIC_ROOT` to a checkout that contains the official UniverseTBD Legacy↔HSC embedding parquets at the paths in `configs/official_legacy_pairs.yaml` (row index is the object identity; `default_max_n` is 16384). SAE and BSF runs also need the corresponding checkpoints under `$PLATONIC_ROOT/outputs/sae` and `$PLATONIC_ROOT/outputs/bsf`. Those weights are not in this repository. GPU is expected; the scripts fall back to CPU when CUDA is absent.

Run from this directory. Primary statistic:

```bash
python src/nglr/analysis/run_ridge_scaling_geometry.py \
  --root "$PLATONIC_ROOT" \
  --pairs-yaml configs/official_legacy_pairs.yaml \
  --out-dir outputs/ridge_scaling_geometry
```

Fixed rank 256:

```bash
python src/nglr/analysis/run_fixed_rank_scaling.py \
  --root "$PLATONIC_ROOT" \
  --pairs-yaml configs/official_legacy_pairs.yaml \
  --out-dir outputs/fixed_rank_scaling
```

The other `run_*.py` scripts take the same `--root` / `--pairs-yaml` pattern where they load embeddings. `run_anisotropy_concentration.py` does not refit Ridge; it reads `outputs/ridge_scaling_geometry/effective_map_spectrum/`.

`run_sae_topk_sensitivity.py` reads Dense and Dense+Ridge scores from `outputs/paper_alignment_controls/test_only_gallery_scores.csv`, which `run_alignment_controls.py` writes. That scores file is not in the released tables; rerun the alignment control before the TopK sweep.
