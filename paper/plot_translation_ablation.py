#!/usr/bin/env python3
"""Figure 3: translation ablation, from the released rung and family tables.

Same two panels as ``plot_ablation`` in ``run_translation_ablation.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNG = ROOT / "outputs/translation_ablation/translation_ablation_rung_scores.csv"
FAM = ROOT / "outputs/translation_ablation/family_delta_beta_translation.csv"
OUT = Path(__file__).resolve().parent / "figures" / "translation_ablation_summary.png"

FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
FAMILY_ORDER = list(FAMILY_LABEL)


def main() -> None:
    df = pd.read_csv(RUNG)
    fam = pd.read_csv(FAM).set_index("family")
    order = [f for f in FAMILY_ORDER if f in fam.index]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.scatter(df["gain_affine"], df["gain_linear_only"], s=36, c="#1f4e79", zorder=3)
    lim_lo = min(df["gain_affine"].min(), df["gain_linear_only"].min(), 0.0) - 0.002
    lim_hi = max(df["gain_affine"].max(), df["gain_linear_only"].max()) + 0.002
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.6)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"Affine gain $G_{\mathrm{affine}}$")
    ax.set_ylabel(r"Linear-only gain $G_A$")
    ax.set_title("Per-rung gain: affine vs linear-only")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    x = range(len(order))
    w = 0.35
    ax.bar(
        [i - w / 2 for i in x],
        [fam.loc[f, "delta_beta_affine"] for f in order],
        w,
        label=r"$\Delta\beta^{\mathrm{affine}}$",
        color="#1f4e79",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [fam.loc[f, "delta_beta_linear_only"] for f in order],
        w,
        label=r"$\Delta\beta^{A}$",
        color="#c44e52",
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in order], rotation=15)
    ax.set_ylabel(r"Family slope amplification $\Delta\beta_F$")
    ax.set_title(r"Family $\Delta\beta$: affine vs linear-only")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
