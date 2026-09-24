#!/usr/bin/env python3
"""Plots for the Storytime N400 revision. Reads saved tables; does not refit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
LMM = POOL / "n400_storytime_v1" / "revision" / "lmm"
AUDIT = POOL / "n400_storytime_v1" / "revision" / "audit"
UNFOLD = POOL / "n400_storytime_v1" / "unfold"
SECOND = POOL / "n400_storytime_v1" / "revision" / "unfold_second_level"
POOL_MEDIA = POOL / "n400_storytime_v1" / "revision" / "plots"
STORE_MEDIA = Path(
    "/run/user/245046/cursor_agent_stores/bc-e43b7bb7-443a-4118-b8d5-2145d2516ab7/files/media/n400-storytime-revision"
)
POOL_MEDIA.mkdir(parents=True, exist_ok=True)

GROUP_COLOR = {
    "TD": "#0072B2",
    "dyslexia_normal_CAP": "#E69F00",
    "dyslexia_atypical_CAP": "#D55E00",
}
GROUP_LABEL = {
    "TD": "TD",
    "dyslexia_normal_CAP": "nCAP",
    "dyslexia_atypical_CAP": "aCAP",
}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    path = POOL_MEDIA / name
    fig.savefig(path, dpi=140)
    print("wrote", path, path.stat().st_size)
    if STORE_MEDIA.parent.is_dir():
        STORE_MEDIA.mkdir(parents=True, exist_ok=True)
        fig.savefig(STORE_MEDIA / name, dpi=140)
        print("wrote", STORE_MEDIA / name)
    plt.close(fig)


def plot_coefficients() -> None:
    coef = pd.read_csv(LMM / "core_coef.csv")
    want = ["sur_z", "sur_z:age_c", "sur_z:g1", "sur_z:g2"]
    labels = [
        "surprisal",
        "surprisal × age (per year)",
        "surprisal × g1\nTD − mean(nCAP, aCAP)",
        "surprisal × g2\nnCAP − aCAP",
    ]
    sub = coef.set_index("term").loc[want]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(want))[::-1]
    est = sub["Estimate"].to_numpy()
    lo = sub["ci95_low"].to_numpy()
    hi = sub["ci95_high"].to_numpy()
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color="#222222", capsize=3)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("CP amplitude per 1 SD (95% CI, random-slope model)")
    ax.set_title("Core model")
    _save(fig, "core_coefficients.png")


def plot_simple_slopes() -> None:
    ss = pd.read_csv(LMM / "core_simple_slopes.csv")
    # label like surprisal|TD|age_c=0
    parts = ss["label"].str.extract(r"surprisal\|(?P<group>[^|]+)\|age_c=(?P<age>-?\d+)")
    ss = pd.concat([ss, parts], axis=1)
    ss["age"] = ss["age"].astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for g, sub in ss.groupby("group"):
        sub = sub.sort_values("age")
        ax.errorbar(
            sub["age"],
            sub["estimate"],
            yerr=[sub["estimate"] - sub["ci95_low"], sub["ci95_high"] - sub["estimate"]],
            fmt="-o",
            color=GROUP_COLOR.get(g, "black"),
            label=GROUP_LABEL.get(g, g),
            capsize=3,
        )
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["mean − 1 year", "mean age", "mean + 1 year"])
    ax.set_ylabel("Surprisal slope (CP amplitude per 1 SD)")
    ax.set_title("Per-group simple slopes")
    ax.legend(frameon=False)
    _save(fig, "simple_slopes_by_age.png")


def plot_reliability() -> None:
    blup = pd.read_csv(LMM / "splithalf_blups.csv")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for g, sub in blup.groupby("group"):
        ax.scatter(
            sub["blup_s1"],
            sub["blup_s2"],
            s=28,
            color=GROUP_COLOR.get(g, "black"),
            label=GROUP_LABEL.get(g, g),
            alpha=0.9,
        )
    lims = np.concatenate([blup["blup_s1"].to_numpy(), blup["blup_s2"].to_numpy()])
    lo, hi = np.nanmin(lims), np.nanmax(lims)
    pad = 0.05 * (hi - lo + 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888888", lw=0.8)
    rel = pd.read_csv(LMM / "splithalf_reliability.csv").iloc[0]
    ax.set_xlabel("Section 1 surprisal BLUP")
    ax.set_ylabel("Section 2 surprisal BLUP")
    ax.set_title(
        f"Split-half r = {rel.pearson_r:.2f}, Spearman–Brown = {rel.spearman_brown:.2f}"
    )
    ax.legend(frameon=False)
    _save(fig, "splithalf_reliability.png")


def plot_region() -> None:
    coef = pd.read_csv(LMM / "region_coef.csv")
    # Three-way names vary with formula order. Keep terms that mention sur_z and region and a group code.
    key = coef["term"].astype(str)
    mask = key.str.contains("sur_z") & key.str.contains("region")
    sub = coef.loc[mask].copy()
    if sub.empty:
        print("region plot: no matching terms")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(sub))[::-1]
    est = sub["Estimate"].to_numpy()
    lo = sub["ci95_low"].to_numpy()
    hi = sub["ci95_high"].to_numpy()
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color="#222222", capsize=3)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["term"])
    ax.set_xlabel("Amplitude per 1 SD (region coded CP +1/2, Fz −1/2)")
    ax.set_title("Surprisal × region × group")
    _save(fig, "region_by_group.png")


def plot_unfold() -> None:
    paths = sorted(UNFOLD.glob("*/rerp.npz"))
    if not paths:
        print("unfold plot: no rerp.npz")
        return
    waves = []
    times = None
    for p in paths:
        z = np.load(p)
        times = z["times"]
        waves.append(z["sur_cp"])
    mean = np.mean(np.stack(waves, axis=0), axis=0)
    # Fixed-window ERP: subject-equal grand average from the audit topography (CP mean of Pz, Cz, Oz).
    topo = np.load(AUDIT / "topography_grand.npz")
    ch = list(topo["channels"])
    grand = topo["grand"]
    cp = grand[[ch.index(c) for c in ("Pz", "Cz", "Oz")]].mean(axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)
    axes[0].plot(topo["times"], cp, color="#222222")
    axes[0].axvspan(0.35, 0.55, color="#bbbbbb", alpha=0.4)
    axes[0].axhline(0, color="#888888", lw=0.6)
    axes[0].set_ylabel("Fixed-window ERP (CP)")
    axes[0].set_title("Content-word ERP vs Unfold surprisal rERP")
    axes[1].plot(times, mean, color="#0072B2")
    axes[1].axvspan(0.35, 0.55, color="#bbbbbb", alpha=0.4)
    axes[1].axhline(0, color="#888888", lw=0.6)
    core = LMM / "core_coef.csv"
    if core.is_file():
        coef = pd.read_csv(core)
        hit = coef.loc[coef.term == "sur_z"]
        if len(hit):
            b = float(hit.Estimate.iloc[0])
            axes[1].hlines(b, 0.35, 0.55, colors="#D55E00", lw=2, label="fixed-window LMM surprisal")
            axes[1].legend(frameon=False, loc="lower right")
    axes[1].set_ylabel("Surprisal rERP (CP)")
    axes[1].set_xlabel("Time from word onset (s)")
    _save(fig, "unfold_vs_fixed_window.png")


def main() -> None:
    if (LMM / "core_coef.csv").is_file():
        plot_coefficients()
        plot_simple_slopes()
    if (LMM / "splithalf_blups.csv").is_file():
        plot_reliability()
    if (LMM / "region_coef.csv").is_file():
        plot_region()
    if any(UNFOLD.glob("*/rerp.npz")):
        plot_unfold()
    print("plots done", POOL_MEDIA)


if __name__ == "__main__":
    main()
