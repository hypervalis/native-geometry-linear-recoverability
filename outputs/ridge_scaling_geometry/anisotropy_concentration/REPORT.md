# Anisotropy concentration of Dense+Ridge effective maps

## Protocol

- Source: `outputs/ridge_scaling_geometry/effective_map_spectrum`
- Map: $A_{\mathrm{eff}}=\mathrm{diag}(\sigma_y)W\,\mathrm{diag}(1/\sigma_x)$ (Legacy→HSC, α=1, frozen split)
- Singular values: active set with relative floor `eps=1e-08` × σ_max (same as existing $A_{\log}$)
- Log floor for numerical zeros: `1e-300`
- **No Ridge refit**

## A. How concentrated is anisotropy?

Across the 16 maps, 90% of centered log-singular-value anisotropy mass is carried by
**41.1–49.0%** of singular directions
(median **42.7%**; mean 43.7%).

Entropy-effective fraction $f_{\mathrm{eff}}^{\mathrm{anis}}$:
range 36.8–55.9% (median 42.7%).

## B. Does concentration change with model size?

Descriptive within-family OLS slopes of $f_{90}$ vs $\log_{10}P$
(negative ⇒ more concentrated with size):

| family | n | β_f90 | Δ first→last f90 | β_feff | Δ first→last feff |
|---|---:|---:|---:|---:|---:|
| astropt | 3 | +0.0018 | +0.0035 | +0.0131 | +0.0228 |
| convnext | 4 | -0.0011 | -0.0021 | -0.0118 | -0.0165 |
| dinov2 | 4 | -0.0033 | -0.0043 | -0.0096 | -0.0141 |
| vit | 3 | -0.0132 | -0.0122 | -0.0593 | -0.0517 |
| ijepa | 2 | -0.0577 | -0.0116 | -0.1146 | -0.0230 |

- Families with $β_{f90}<0$: **4/5**
- Families with $β_{feff}<0$: **4/5**

Pooled descriptive Spearman (n=16; not IID):
- $\rho(\log_{10}P, f_{90})$ = 0.128, $p$=0.636,
  $p_{\mathrm{perm}}$=0.911 (family-preserving)
- $\rho(\log_{10}P, f_{\mathrm{eff}})$ = 0.059, $p$=0.828,
  $p_{\mathrm{perm}}$=0.968

## C. Magnitude vs concentration

- $\rho(A_{\log}, f_{90})$ = 0.391, $p$=0.134
- $A_{\log}$ range: 1.759–3.153

Strong overall anisotropy does **not** imply dispersed anisotropy (they are separate axes).

## D. Anisotropy concentration vs functional lift concentration

Existing truncated-SVD / transfer complexity: fraction of directions recovering ≥90% of
Dense+Ridge mKNN lift (`k90_transfer_frac` from effective_map_spectrum).

- Lift $f_{90}$ range: 2.1–16.7%
  (median 4.8%)
- Anisotropy $f_{90}$ median 42.7% vs lift median
  4.8%
- Spearman $\rho(f_{90}^{\mathrm{anis}}, f_{90}^{\mathrm{lift}})$ =
  -0.101, $p$=0.709

These measure different things; do not equate spectral anisotropy mass with mKNN-useful subspace.

## Parity and numerics

- Max $|A_{\log}^{\mathrm{existing}}-A_{\log}^{\mathrm{recomputed}}|$ = 1.288e-14
- Inactive singular values (relative eps): 159 total across models
  (per-model max 154)
- Active values floored at 1e-300: 0

## Figures

- `figures/anisotropy_concentration_curves.png`
- `figures/anisotropy_concentration_vs_scale.png`
