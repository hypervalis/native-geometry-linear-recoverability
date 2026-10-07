#!/usr/bin/env python3
"""Figure 1: ambient Dense vs Dense+Ridge and fixed-rank-256 scaling.

Canvas width matches NeurIPS \\textwidth so 10 pt labels stay ~body size
when included at \\linewidth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
RUNG = ROOT / "outputs/ridge_scaling_geometry/rung_scores.csv"
FIXED = ROOT / "outputs/fixed_rank_scaling/fixed_rank_scores.csv"

FAM = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
ORDER = list(FAM)
NATIVE = "#4C72B0"
ALIGN = "#DD8452"

FS = 10
FIGSIZE = (5.5, 3.55)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.titlesize": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": FS,
            "axes.titleweight": "normal",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    rung = pd.read_csv(RUNG)
    fr256 = pd.read_csv(FIXED)
    fr256 = fr256[fr256["rank"] == 256]

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE, sharey="row")

    for j, fam in enumerate(ORDER):
        ax = axes[0, j]
        sub = rung[rung.family == fam].sort_values("log10_params")
        ax.plot(
            sub.log10_params,
            sub.mknn_dense,
            "o-",
            ms=4.0,
            lw=1.35,
            color=NATIVE,
            label=r"$M_{\mathrm{Dense}}$",
        )
        ax.plot(
            sub.log10_params,
            sub.mknn_dense_ridge,
            "s-",
            ms=4.0,
            lw=1.35,
            color=ALIGN,
            label=r"$M_{\mathrm{Dense+Ridge}}$",
        )
        ax.set_title(FAM[fam], fontsize=FS, fontweight="normal")
        ax.tick_params(axis="both", labelsize=FS)
        ax.grid(True, alpha=0.28, lw=0.6)
        if j == 0:
            ax.set_ylabel("(a) Ambient\nmKNN@10", fontsize=FS)

        ax2 = axes[1, j]
        sub2 = fr256[fr256.family == fam].sort_values("log10_params")
        ax2.plot(
            sub2.log10_params,
            sub2.raw_pca_mknn,
            "o-",
            ms=4.0,
            lw=1.35,
            color=NATIVE,
            label=r"$M_{\mathrm{indPCA256}}$",
        )
        ax2.plot(
            sub2.log10_params,
            sub2.pca_ridge_mknn,
            "s-",
            ms=4.0,
            lw=1.35,
            color=ALIGN,
            label=r"$M_{\mathrm{indPCA256+Ridge}}$",
        )
        ax2.set_xlabel(r"$\log_{10} P$", fontsize=FS)
        ax2.tick_params(axis="both", labelsize=FS)
        ax2.grid(True, alpha=0.28, lw=0.6)
        if j == 0:
            ax2.set_ylabel("(b) Fixed rank 256\nmKNN@10", fontsize=FS)

    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        h0 + h1,
        l0 + l1,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=FS,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    fig.subplots_adjust(
        left=0.14, right=0.99, top=0.84, bottom=0.12, wspace=0.18, hspace=0.30
    )

    FIG.mkdir(parents=True, exist_ok=True)
    for name in ("scaling_dense_and_fixedrank.png", "scaling_dense_and_indpca256.png"):
        out = FIG / name
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
