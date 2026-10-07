# Cross-model same-dataset symmetry (Legacy only)

Exploratory first pass: PCA256 (train-only per model), Ridge(α=1), mKNN@10 test-only gallery, cross-family pairs only.

- **Pairs scored:** 101
- **Models:** 16
- **Runtime:** 50.5s on cuda

## Q1 — Native cross-model mKNN

Mean native mKNN@10 over 101 cross-family pairs: **0.0502** (median 0.0536; range [0.0222, 0.1343]).

Chance level (test gallery, k=10): **0.00305**.

## Q2 — Does Ridge increase correspondence in both directions?

Mean ridge mKNN A→B: **0.1036**; B→A: **0.1171**.

Pairs with ridge gain > 0 in A→B: **101/101**; B→A: **101/101**.

## Q3 — Directional gain symmetry

| Statistic | Mean | Median |
|-----------|------|--------|
| Gain A→B | 0.0533 | 0.0513 |
| Gain B→A | 0.0669 | 0.0634 |
| $G_{\rm sym}$ | 0.0601 | 0.0588 |
| $A_{\rm dir}$ | 0.0355 | 0.0350 |
| $A_{\rm rel}$ | 0.3323 | 0.2555 |

## Q4 — Does $G_{\rm sym}$ increase with scale?

### Approximate matched-scale bands (both models in band)

| band   | metric   |   n_pairs |      mean |    median |   ci95_lo |   ci95_hi |
|:-------|:---------|----------:|----------:|----------:|----------:|----------:|
| small  | native   |         5 | 0.0405188 | 0.0271895 | 0.0256515 | 0.055386  |
| small  | G_sym    |         5 | 0.0454593 | 0.0336131 | 0.0315563 | 0.0593622 |
| small  | A_dir    |         5 | 0.0244797 | 0.033964  | 0.011419  | 0.0375404 |
| medium | native   |         6 | 0.038226  | 0.0327739 | 0.02722   | 0.0548317 |
| medium | G_sym    |         6 | 0.0468238 | 0.0456591 | 0.0395713 | 0.0564642 |
| medium | A_dir    |         6 | 0.0341522 | 0.0329722 | 0.0187824 | 0.049466  |
| large  | native   |         1 | 0.0615807 | 0.0615807 | 0.0615807 | 0.0615807 |
| large  | G_sym    |         1 | 0.0545468 | 0.0545468 | 0.0545468 | 0.0545468 |
| large  | A_dir    |         1 | 0.02426   | 0.02426   | 0.02426   | 0.02426   |
| xlarge | native   |         9 | 0.0702268 | 0.063198  | 0.052439  | 0.0894687 |
| xlarge | G_sym    |         9 | 0.0828824 | 0.0764571 | 0.0700657 | 0.0957649 |
| xlarge | A_dir    |         9 | 0.053528  | 0.0594141 | 0.0398589 | 0.0663554 |

### All-pair OLS ($s_{\min}$, $s_{\rm gap}$)

- Native: β_min=0.01713 (SE 0.00425), β_gap=0.00732
- $G_{\rm sym}$: β_min=0.02610 (SE 0.00395), β_gap=0.01309

## Q5 — Does directional asymmetry decrease with scale?

- $A_{\rm dir}$: β_min=0.01674 (SE 0.00449), β_gap=0.00935 (SE 0.00455)

## Q6 — Larger→smaller vs smaller→larger

- Mean gain **large→small**: 0.0586 (99 ordered directions)
- Mean gain **small→large**: 0.0624 (99 ordered directions)
- Difference (large→small minus small→large): **-0.0038**

Within matched-scale bands (same band, ordered by relative size):

| band   |   mean_large_to_small |   mean_small_to_large |        diff |   n_directions |
|:-------|----------------------:|----------------------:|------------:|---------------:|
| small  |             0.0594751 |             0.0373665 |  0.0221086  |              4 |
| medium |             0.0271224 |             0.0665243 | -0.0394019  |              5 |
| large  |             0.0424168 |             0.0666768 | -0.02426    |              1 |
| xlarge |             0.0862544 |             0.0795104 |  0.00674397 |              9 |

## Q7 — Comparison with cross-survey asymmetry

Cross-survey Legacy↔HSC showed **directional asymmetry** (forward $T_{L\to H}$ stronger than reverse $T_{H\to L}$, family-heterogeneous). Same-dataset cross-model removes survey information differences; any remaining asymmetry reflects representation / training differences only.

Here mean $|G_{A\to B}-G_{B\to A}|$ = 0.0355 vs mean $G_{\rm sym}$ = 0.0601 (ratio 0.59). Do not compare raw mKNN magnitudes to cross-survey numbers (different task: cross-model vs cross-survey).

## Q8 — Verdict

### B

> Mutual recoverability increases with scale, but directional asymmetry persists.

## Outputs

- `outputs/cross_model_symmetry/cross_model_pair_scores.csv`
- `paper_working/figures/cross_model_*_heatmap.png`
- `paper_working/figures/cross_model_symmetry_summary.png`

