#!/usr/bin/env python3
"""Main geometry figure: f90_lift vs f90_anis per map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANIS = ROOT / "outputs/ridge_scaling_geometry/anisotropy_concentration"
PER = ANIS / "per_model_summary.csv"
OUT = Path(__file__).resolve().parent / "figures" / "geometry_anisotropy_vs_functional.png"
OUT_ROOT = ROOT / "figures" / "geometry_anisotropy_vs_functional.png"

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
    rows = []
    for fam in FAM_ORDER:
        rows.append(per[per["family"] == fam].sort_values("log10_P"))
    per = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(3.4, 3.6), constrained_layout=True)

    y = np.arange(len(per))
    lift = per["fraction_directions_for_90pct_lift"].to_numpy() * 100
    anis = per["f90_anis"].to_numpy() * 100
    for i, (_, r) in enumerate(per.iterrows()):
        c = FAM_COLOR[r["family"]]
        ax.plot([lift[i], anis[i]], [i, i], color=c, lw=1.1, alpha=0.85, zorder=1)
        ax.plot(
            lift[i],
            i,
            marker="o",
            markersize=4.5,
            color=c,
            markeredgecolor="black",
            markeredgewidth=0.35,
            zorder=3,
        )
        ax.plot(
            anis[i],
            i,
            marker="s",
            markersize=4.0,
            color="white",
            markeredgecolor=c,
            markeredgewidth=1.2,
            zorder=3,
        )

    labels = [f"{FAM_LABEL[r['family']]} {r['size_name']}" for _, r in per.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(-0.8, len(per) - 0.2)
    ax.set_xlim(0, 55)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    ax.set_xlabel("Fraction of singular directions")
    ax.invert_yaxis()

    (h_lift,) = ax.plot(
        [],
        [],
        "o",
        color="0.3",
        markersize=4.5,
        markeredgecolor="black",
        markeredgewidth=0.35,
        label=r"$f_{90}^{\mathrm{lift}}$",
    )
    (h_anis,) = ax.plot(
        [],
        [],
        "s",
        color="white",
        markersize=4.0,
        markeredgecolor="0.3",
        markeredgewidth=1.2,
        label=r"$f_{90}^{\mathrm{anis}}$",
    )
    ax.legend(handles=[h_lift, h_anis], frameon=False, loc="lower right", fontsize=7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    fig.savefig(OUT_ROOT, dpi=220)
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"wrote {OUT_ROOT}")
    print(
        f"medians: lift={per['fraction_directions_for_90pct_lift'].median()*100:.1f}% "
        f"anis={per['f90_anis'].median()*100:.1f}%"
    )


if __name__ == "__main__":
    main()
