#!/usr/bin/env python3
"""Plot group-mean B2B family traces (mean ± SE) from compare_b2b_groups.py outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIRM_WIN = (0.0, 0.8)
GROUP_ORDER = (
    ("TD", "TD", "#1f4e79"),
    ("dyslexia_normal_CAP", "Normal CAP", "#2a9d8f"),
    ("dyslexia_atypical_CAP", "Atypical CAP", "#e07a3d"),
)


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    times = np.array([float(c) for c in df.columns])
    return times, df.to_numpy(dtype=float)


def plot_group_means(in_dir: Path, families: tuple[str, ...], title: str, out_png: Path,
                     xlabel: str = "Time (s, onset-locked)") -> None:
    fig, axes = plt.subplots(
        1, len(families), figsize=(4.15 * len(families), 3.55), sharex=True, squeeze=False
    )
    fig.suptitle(title, fontsize=11, y=1.02)

    for ax, fam in zip(axes[0], families):
        plotted = False
        for gkey, glabel, color in GROUP_ORDER:
            path = in_dir / f"traces_primary_{fam}_{gkey}.csv"
            if not path.is_file():
                continue
            times, mat = load_trace(path)
            n = mat.shape[0]
            mu = mat.mean(axis=0)
            se = mat.std(axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(times, mu - se, mu + se, color=color, alpha=0.22, linewidth=0)
            ax.plot(times, mu, color=color, lw=1.6, label=f"{glabel} (n={n})")
            plotted = True
        if not plotted:
            raise SystemExit(f"No primary traces for family={fam} under {in_dir}")

        ax.axvspan(CONFIRM_WIN[0], CONFIRM_WIN[1], color="0.88", zorder=0)
        ax.axvline(0.0, color="0.45", ls="--", lw=0.8)
        ax.axhline(0.0, color="0.75", lw=0.6)
        ax.set_title(fam.capitalize(), fontsize=11)
        ax.set_xlim(times.min(), times.max())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0][0].set_ylabel("Family score (Σ diag H)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="best")
    for ax in axes[0]:
        ax.set_xlabel(xlabel)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_hedges_g(in_dir: Path, families: tuple[str, ...], out_png: Path) -> None:
    summary = in_dir / "group_comparison_summary.csv"
    if not summary.is_file():
        return
    df = pd.read_csv(summary)
    df = df.loc[df["analysis_set"] == "primary"].copy()
    if df.empty:
        return
    contrasts = list(df["contrast"])
    x = np.arange(len(families))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cmap = plt.cm.tab10
    for i, contrast in enumerate(contrasts):
        row = df.loc[df["contrast"] == contrast].iloc[0]
        gs = [float(row[f"hedges_g_{f}"]) for f in families]
        ax.bar(x + (i - (len(contrasts) - 1) / 2) * width, gs, width, label=contrast, color=cmap(i))
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.set_ylabel("Hedges g (confirm 0–0.8 s)")
    ax.set_title("Lexical B2B · window-mean effect sizes (descriptive)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def infer_families(in_dir: Path, requested: str | None) -> tuple[str, ...]:
    if requested:
        return tuple(x.strip() for x in requested.split(",") if x.strip())
    found = []
    for gkey, _, _ in GROUP_ORDER:
        for p in sorted(in_dir.glob(f"traces_primary_*_{gkey}.csv")):
            fam = p.name[len("traces_primary_") : -(len(gkey) + 5)]
            if fam not in found:
                found.append(fam)
        if found:
            break
    if not found:
        raise SystemExit(f"No traces_primary_*.csv under {in_dir}")
    return tuple(found)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot B2B v4 group-comparison traces")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--families", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="Time (s, onset-locked)")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    families = infer_families(in_dir, args.families or None)
    title = args.title or (
        "Lexical B2B onset · group mean ± SE · confirm 0–0.8 s shaded."
        if set(families) == {"duration", "frequency", "surprisal"}
        else "B2B · group mean ± SE · confirm 0–0.8 s shaded."
    )
    plots = in_dir / "plots"
    mean_png = plots / "group_means_by_family.png"
    plot_group_means(in_dir, families, title, mean_png, xlabel=args.xlabel)
    g_png = plots / "hedges_g_confirm.png"
    plot_hedges_g(in_dir, families, g_png)
    print(f"Wrote {mean_png}")
    if g_png.is_file():
        print(f"Wrote {g_png}")


if __name__ == "__main__":
    main()
