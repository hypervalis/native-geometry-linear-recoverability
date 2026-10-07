#!/usr/bin/env python3
"""Appendix: anisotropy magnitude/concentration vs model scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PER = ROOT / "outputs/ridge_scaling_geometry/anisotropy_concentration/per_model_summary.csv"
LOCAL = ROOT / "outputs/data_supported_distortion/local_edge_summary.csv"
OUT = Path(__file__).resolve().parent / "figures" / "geometry_scale_diagnostics.png"
OUT_ROOT = ROOT / "figures" / "geometry_scale_diagnostics.png"

FAM_ORDER = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAM_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
FAM_COLOR = {
    "astropt": "#4C72B0",
    "convnext": "#DD8452",
    "dinov2": "#55A868",
    "vit": "#C44E52",
    "ijepa": "#8172B3",
}

FS = 9


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.titlesize": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS - 1,
            "ytick.labelsize": FS - 1,
            "legend.fontsize": 7,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    per = pd.read_csv(PER)
    local = pd.read_csv(LOCAL)
    local = local[local["k_edge"] == 10].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    panels = [
        (axes[0], per, "log10_P", "A_log", r"$A_{\log}$", "(a)"),
        (axes[1], per, "log10_P", "f90_anis", r"$f_{90}^{\mathrm{anis}}$", "(b)"),
        (axes[2], local, "log10_params", "sigma_local", r"$\sigma_{\mathrm{local}}$", "(c)"),
    ]
    for ax, df, xcol, ycol, ylab, title in panels:
        for fam in FAM_ORDER:
            sub = df[df["family"] == fam].sort_values(xcol)
            ax.plot(
                sub[xcol],
                sub[ycol],
                "o-",
                color=FAM_COLOR[fam],
                lw=1.3,
                markersize=4.5,
                label=FAM_LABEL[fam],
            )
        ax.set_xlabel(r"$\log_{10}P$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_ROOT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
