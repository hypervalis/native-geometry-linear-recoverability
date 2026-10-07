#!/usr/bin/env python3
"""Appendix figure: SAE inference-time TopK sensitivity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_robustness/sae_topk_sensitivity"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SUMMARY = OUT / "summary_by_topk.csv"

FAM_COLS = [
    ("delta_beta_astropt", "AstroPT"),
    ("delta_beta_convnext", "ConvNeXt"),
    ("delta_beta_dinov2", "DINOv2"),
    ("delta_beta_vit", "ViT"),
    ("delta_beta_ijepa", "I-JEPA"),
]

FS = 10
FIGSIZE = (5.5, 2.6)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.titlesize": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": 8,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    s = pd.read_csv(SUMMARY).sort_values("topk")
    xs = s["topk"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    ax = axes[0]
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.fill_between(
        xs,
        s["S_SAE_family_boot_lo"],
        s["S_SAE_family_boot_hi"],
        color="#4C72B0",
        alpha=0.2,
        zorder=1,
    )
    ax.plot(
        xs,
        s["mean_S_SAE"],
        "o-",
        color="#4C72B0",
        lw=1.5,
        markersize=5,
        zorder=2,
        label=r"mean $S_{\mathrm{SAE}}$",
    )
    ax.set_xticks(xs)
    ax.set_xlabel(r"SAE TopK (inference)")
    ax.set_ylabel(r"$M_{\mathrm{SAE+Ridge}}-M_{\mathrm{Dense+Ridge}}$")
    ax.set_title("(a)")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    for col, lab in FAM_COLS:
        ax.plot(
            xs,
            s[col],
            "o-",
            alpha=0.35,
            lw=1.0,
            markersize=3.5,
            label=lab,
            zorder=1,
        )
    ax.plot(
        xs,
        s["T_SAE"],
        "o-",
        color="black",
        lw=1.8,
        markersize=5,
        zorder=2,
        label=r"$T_{\mathrm{SAE}}$",
    )
    ax.set_xticks(xs)
    ax.set_xlabel(r"SAE TopK (inference)")
    ax.set_ylabel(r"$\Delta\beta^{\mathrm{SAE}}$ / $T_{\mathrm{SAE}}$")
    ax.set_title("(b)")
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_paper = FIG_DIR / "sae_topk_sensitivity.png"
    out_root = ROOT / "figures" / "sae_topk_sensitivity.png"
    out_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_paper, dpi=200)
    fig.savefig(out_root, dpi=200)
    fig.savefig(OUT / "sae_topk_sensitivity.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_paper}")
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
