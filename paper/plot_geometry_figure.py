#!/usr/bin/env python3
"""Figure 2: D_sim and sigma_local vs model size (cached outputs only)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "outputs/ridge_scaling_geometry/singular_spectrum_summary.csv"
LOCAL = ROOT / "outputs/data_supported_distortion/local_edge_summary.csv"
OUT = Path(__file__).resolve().parent / "figures/geometry_distortion_summary.png"

FAMILY_STYLE = {
    "astropt": ("AstroPT", "#1f77b4"),
    "convnext": ("ConvNeXt", "#ff7f0e"),
    "dinov2": ("DINOv2", "#2ca02c"),
    "vit": ("ViT", "#d62728"),
    "ijepa": ("I-JEPA", "#9467bd"),
}


def _panel(ax, df, ycol, ylabel):
    for family, (label, color) in FAMILY_STYLE.items():
        sub = df[df["family"] == family].sort_values("log10_params")
        if sub.empty:
            continue
        ax.plot(
            sub["log10_params"],
            sub[ycol],
            "o-",
            color=color,
            label=label,
            linewidth=1.6,
            markersize=5,
        )
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)


def main():
    spec = pd.read_csv(SPEC)
    local = pd.read_csv(LOCAL)
    local = local[local["k_edge"] == 10].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    _panel(axes[0], spec, "D_sim", r"$D_{\mathrm{sim}}$")
    _panel(axes[1], local, "sigma_local", r"$\sigma_{\mathrm{local}}$")

    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes, fontsize=11, fontweight="bold")
    axes[1].text(-0.12, 1.04, "(b)", transform=axes[1].transAxes, fontsize=11, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
