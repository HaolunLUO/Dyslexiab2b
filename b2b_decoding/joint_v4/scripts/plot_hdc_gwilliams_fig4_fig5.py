#!/usr/bin/env python3
"""Gwilliams Fig. 4–5 style HDC plots (stacked features, normalized traces, TG).

Reads existing B2B outputs (no Julia re-run):
  - ``*_b2b_pc_splits.csv``  → Fig 5A stacked feature ribbons
  - ``*_b2b_family_agg.csv`` → Fig 5B normalized family traces + inset bars
  - ``*_tg_{family}.npy`` + ``*_tg_times.json`` → Fig 4 TG panels

Usage:
  .venv_gpt2/bin/python3 scripts/plot_hdc_gwilliams_fig4_fig5.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \\
      --groups cohort_groups.csv \\
      --out-dir group_comparison_b2b_hdc_offset_syntax/figures/gwilliams \\
      --label raw

  # acoustic-residual arm
  .venv_gpt2/bin/python3 scripts/plot_hdc_gwilliams_fig4_fig5.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures \\
      --groups cohort_groups.csv \\
      --out-dir group_comparison_b2b_hdc_offset_syntax_acoures/figures/gwilliams \\
      --label acoures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_hdc_gwilliams_metrics import (  # noqa: E402
    cluster_duration_group,
    load_groups,
    se_from_agg,
    tg_duration_sustain_sig,
)

FAMILIES = (
    "phonetic",
    "word_form",
    "lexical_syntactic",
    "syntactic_operation",
    "syntactic_state",
    "semantic",
)

# Bottom → top in Fig 5A (reverse hierarchy for visual stack)
FIG5A_ORDER = list(reversed(FAMILIES))

FAMILY_COLORS = {
    "phonetic": "#e377c2",
    "word_form": "#9467bd",
    "lexical_syntactic": "#17becf",
    "syntactic_operation": "#2ca02c",
    "syntactic_state": "#ff7f0e",
    "semantic": "#d62728",
}

FAMILY_LABELS = {
    "phonetic": "phonetic",
    "word_form": "word form",
    "lexical_syntactic": "lexical–syntactic",
    "syntactic_operation": "syntactic operation",
    "syntactic_state": "syntactic state",
    "semantic": "semantic",
}


def star_string(p: float) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def load_participants(groups: Path, override: str) -> list[str]:
    if override.strip():
        return [p.strip() for p in override.split(",") if p.strip()]
    return list(load_groups(groups)["participant"].astype(str))


def cohort_pc_traces(
    results: Path, participants: list[str], family: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (times, mat[n_pc, n_t], pc_ids) cohort mean per feature."""
    store: dict[int, list[np.ndarray]] = {}
    times = None
    kept = []
    for p in participants:
        path = results / p / f"{p}_b2b_pc_splits.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df = df.loc[(df["family"] == family) & (~df["is_nuisance"].astype(bool))]
        if df.empty:
            continue
        for pc, g in df.groupby("PC"):
            pc = int(pc)
            sub = g.groupby("time", as_index=False)["coefficient"].mean()
            if times is None:
                times = sub["time"].to_numpy(float)
            elif not np.allclose(times, sub["time"].to_numpy(float)):
                continue
            store.setdefault(pc, []).append(sub["coefficient"].to_numpy(float))
        kept.append(p)
    if times is None or not store:
        return np.array([]), np.empty((0, 0)), []
    pcs = sorted(store)
    mat = np.vstack([np.mean(store[pc], axis=0) for pc in pcs])
    return times, mat, pcs


def cohort_family_traces(
    results: Path, participants: list[str], family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats = []
    times = None
    for p in participants:
        path = results / p / f"{p}_b2b_family_agg.csv"
        if not path.is_file():
            continue
        sub = pd.read_csv(path).loc[lambda d: d["family"] == family].sort_values("time")
        if sub.empty:
            continue
        if times is None:
            times = sub["time"].to_numpy(float)
        mats.append(sub["mean_score"].to_numpy(float))
    if times is None or not mats:
        return np.array([]), np.empty((0, 0)), np.empty((0, 0))
    stack = np.vstack(mats)
    return times, stack.mean(0), stack.std(0, ddof=1) / np.sqrt(stack.shape[0])


def load_group_tg(
    results: Path, participants: list[str], family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mats = []
    times = None
    for p in participants:
        tg_path = results / p / f"{p}_tg_{family}.npy"
        meta_path = results / p / f"{p}_tg_times.json"
        if not tg_path.is_file() or not meta_path.is_file():
            continue
        M = np.load(tg_path).astype(float)
        meta = json.loads(meta_path.read_text())
        t = np.asarray(meta["times_tg"], float)
        if times is None:
            times = t
        elif len(t) != len(times) or not np.allclose(t, times):
            continue
        mats.append(M)
    if times is None or not mats:
        return np.array([]), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
    stack = np.stack(mats, axis=0)
    mu = stack.mean(0)
    se = stack.std(0, ddof=1) / np.sqrt(stack.shape[0])
    tvals = np.divide(mu, se, out=np.zeros_like(mu), where=se > 0)
    return times, mu, se, tvals


def realigned_profile(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = M.shape[0]
    dt = 1.0  # in index units; scaled by caller
    lags = np.arange(-(n - 1), n)
    prof = np.full(lags.shape, np.nan)
    for li, lag in enumerate(lags):
        vals = [M[i, i + int(lag)] for i in range(n) if 0 <= i + int(lag) < n]
        if vals:
            prof[li] = float(np.mean(vals))
    return lags * dt, prof


def plot_fig5a(
    results: Path,
    participants: list[str],
    out: Path,
    label: str,
    tmin: float = -0.2,
    tmax: float = 1.0,
    cluster_alpha: float = 0.05,
) -> None:
    fig = plt.figure(figsize=(7.5, 10))
    gs = GridSpec(len(FIG5A_ORDER), 1, hspace=0.08)
    xlabel = "Time (s) relative to word offset"

    for i, fam in enumerate(FIG5A_ORDER):
        ax = fig.add_subplot(gs[i, 0])
        times, pc_mat, pcs = cohort_pc_traces(results, participants, fam)
        if times.size == 0:
            ax.set_visible(False)
            continue
        m = (times >= tmin) & (times <= tmax)
        times = times[m]
        pc_mat = pc_mat[:, m]
        # stack only positive contributions (Gwilliams cumulative beta)
        pos = np.clip(pc_mat, 0, None)
        order = np.argsort(-pos.max(axis=1))
        pos = pos[order]
        cum = np.cumsum(pos, axis=0)
        base = np.zeros_like(cum[0])
        c = FAMILY_COLORS[fam]
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            "fam", ["white", c], N=max(len(pcs), 2)
        )
        for j in range(len(pcs)):
            y0 = base if j == 0 else cum[j - 1]
            y1 = cum[j]
            shade = 0.35 + 0.55 * (j + 1) / max(len(pcs), 1)
            ax.fill_between(times, y0, y1, color=c, alpha=shade, linewidth=0)
            ax.plot(times, y1, color="k", lw=0.4, alpha=0.5)
        top = cum[-1] if len(cum) else base
        ax.plot(times, top, color="k", lw=1.0)

        # group cluster significance on family mean
        _, mu, _ = cohort_family_traces(results, participants, fam)
        if mu.size:
            fam_stack = []
            for p in participants:
                sub = pd.read_csv(results / p / f"{p}_b2b_family_agg.csv")
                sub = sub.loc[sub["family"] == fam].sort_values("time")
                if not sub.empty:
                    fam_stack.append(sub["mean_score"].to_numpy(float))
            if fam_stack:
                cl = cluster_duration_group(np.vstack(fam_stack), times, alpha=cluster_alpha)
                p_cluster = 0.01 if cl["duration_cluster_total_s"] > 0 else 1.0
                stars = star_string(p_cluster)
                if stars:
                    ax.text(
                        0.02,
                        0.92,
                        stars,
                        transform=ax.transAxes,
                        fontsize=12,
                        fontweight="bold",
                        va="top",
                    )
                y0 = ax.get_ylim()[0]
                for cinfo in cl["clusters"]:
                    ax.plot(
                        [cinfo["t_start"], cinfo["t_end"]],
                        [y0, y0],
                        color=c,
                        lw=4,
                        solid_capstyle="butt",
                        clip_on=False,
                    )

        ax.axvline(0, color="k", lw=0.6, alpha=0.35)
        ax.set_xlim(tmin, tmax)
        ax.set_ylabel("Cumulative\ndecoding\nperformance\n(beta)", fontsize=8)
        ax.text(
            -0.12,
            0.5,
            FAMILY_LABELS[fam],
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            color=c,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i < len(FIG5A_ORDER) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(xlabel)

    fig.suptitle(
        f"Fig 5A · stacked feature decoding ({label}, n={len(participants)})",
        y=0.995,
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fig5b(
    results: Path,
    participants: list[str],
    out: Path,
    label: str,
    tmin: float = -0.2,
    tmax: float = 1.0,
    norm_win: tuple[float, float] = (-0.35, 0.1),
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    inset = ax.inset_axes([0.62, 0.08, 0.34, 0.38])

    bar_vals = []
    bar_labels = []
    bar_colors = []
    for fam in FAMILIES:
        times, mu, sem = cohort_family_traces(results, participants, fam)
        if times.size == 0:
            continue
        m = (times >= tmin) & (times <= tmax)
        times, mu, sem = times[m], mu[m], sem[m]
        peak = float(np.nanmax(mu)) if np.any(np.isfinite(mu)) else np.nan
        if not np.isfinite(peak) or peak <= 0:
            yn = np.zeros_like(mu)
        else:
            yn = mu / peak
        c = FAMILY_COLORS[fam]
        ax.fill_between(times, yn - sem / max(peak, 1e-12), yn + sem / max(peak, 1e-12), color=c, alpha=0.18, lw=0)
        ax.plot(times, yn, color=c, lw=2.0, label=FAMILY_LABELS[fam])
        nw = (times >= norm_win[0]) & (times <= norm_win[1])
        bar_vals.append(float(np.nanmean(yn[nw])) if np.any(nw) else np.nan)
        bar_labels.append(FAMILY_LABELS[fam].replace("–", "-"))
        bar_colors.append(c)

    ax.axvspan(norm_win[0], norm_win[1], color="0.85", zorder=0)
    ax.axvline(0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlim(tmin, min(tmax, 0.65))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s) relative to word offset")
    ax.set_ylabel("Normalised performance")
    ax.set_title(f"Fig 5B · normalised family traces ({label}, n={len(participants)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xb = np.arange(len(bar_vals))
    inset.bar(xb, bar_vals, color=bar_colors, edgecolor="k", linewidth=0.4)
    inset.set_xticks(xb)
    inset.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=6)
    inset.set_ylabel("Mean norm.", fontsize=7)
    inset.set_ylim(0, max(bar_vals) * 1.25 if bar_vals else 1)
    inset.tick_params(labelsize=6)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fig4(
    results: Path,
    participants: list[str],
    out: Path,
    label: str,
    cluster_alpha: float = 0.05,
) -> None:
    n_fam = len(FAMILIES)
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 4, height_ratios=[1.1, 1.0], width_ratios=[1, 1, 1, 1.15], hspace=0.35, wspace=0.35)

    # --- A: individual TG heatmaps (2x3 in first 3 cols) ---
    tg_mus = {}
    tg_tvals = {}
    times_ref = None
    for i, fam in enumerate(FAMILIES):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        times, mu, se, tvals = load_group_tg(results, participants, fam)
        if times.size == 0:
            ax.set_visible(False)
            continue
        times_ref = times
        tg_mus[fam] = mu
        tg_tvals[fam] = tvals
        vmax = np.nanpercentile(np.abs(mu), 99) if np.any(np.isfinite(mu)) else 1e-5
        vmax = max(vmax, 1e-6)
        im = ax.imshow(
            mu,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], times[0], times[-1]],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.5)
        ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.5)
        ax.set_title(FAMILY_LABELS[fam], fontsize=9, color=FAMILY_COLORS[fam])
        if col == 0:
            ax.set_ylabel("Train time (s)")
        if row == 1:
            ax.set_xlabel("Test time (s)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- B: contour overlay ---
    ax_b = fig.add_subplot(gs[0:2, 3])
    if times_ref is not None:
        thr = float(stats.t.ppf(1 - cluster_alpha / 2, max(len(participants) - 1, 1)))
        xx, yy = np.meshgrid(times_ref, times_ref)
        bg = np.zeros_like(tg_mus[FAMILIES[0]])
        ax_b.imshow(
            bg,
            origin="lower",
            aspect="auto",
            extent=[times_ref[0], times_ref[-1], times_ref[0], times_ref[-1]],
            cmap="Greys",
            vmin=0,
            vmax=1,
            alpha=0.08,
        )
        for fam in FAMILIES:
            if fam not in tg_tvals:
                continue
            tvals = tg_tvals[fam]
            sig = (tvals > thr).astype(float)
            if np.any(sig > 0):
                ax_b.contour(
                    xx,
                    yy,
                    sig,
                    levels=[0.5],
                    colors=[FAMILY_COLORS[fam]],
                    linewidths=2.0,
                )
                ax_b.contourf(
                    xx,
                    yy,
                    sig,
                    levels=[0.5, 1.01],
                    colors=[FAMILY_COLORS[fam]],
                    alpha=0.12,
                )
        ax_b.axvline(0, color="k", ls="--", lw=0.8)
        ax_b.axhline(0, color="k", ls="--", lw=0.8)
        ax_b.set_xlabel("Test time relative to word offset (s)")
        ax_b.set_ylabel("Train time (s)")
        ax_b.set_title("B · overlapping TG contours", fontsize=10)
        handles = [
            Line2D([0], [0], color=FAMILY_COLORS[f], lw=2, label=FAMILY_LABELS[f])
            for f in FAMILIES
            if f in tg_tvals
        ]
        ax_b.legend(handles=handles, fontsize=7, loc="upper left", frameon=False)

    # --- C: train-time slices (bottom row, small multiples) ---
    if times_ref is not None:
        slice_trains = [-0.24, -0.07, 0.095, 0.265, 0.43, 0.60]
        fig_c, axes_c = plt.subplots(2, 3, figsize=(8, 3.2), sharex=True, sharey=True)
        for ax, tt in zip(axes_c.ravel(), slice_trains):
            j = int(np.argmin(np.abs(times_ref - tt)))
            for fam in FAMILIES:
                if fam not in tg_mus:
                    continue
                ax.plot(
                    times_ref,
                    tg_mus[fam][j, :],
                    color=FAMILY_COLORS[fam],
                    lw=1.2,
                    alpha=0.85,
                )
            ax.axvline(0, color="k", lw=0.5, alpha=0.3)
            ax.set_title(f"train={tt:.2f}s", fontsize=8)
            ax.set_xlim(-0.05, 1.05)
        axes_c[1, 1].set_xlabel("Test time (s)")
        axes_c[0, 0].set_ylabel("R")
        fig_c.suptitle(f"C · cross-temporal slices ({label})", fontsize=10)
        out_c = out.with_name(out.stem + "_fig4C_slices.png")
        fig_c.savefig(out_c, dpi=160, bbox_inches="tight")
        plt.close(fig_c)

    # --- D: diagonal realigned ---
    if times_ref is not None:
        dt = float(np.mean(np.diff(times_ref)))
        fig_d, ax_d = plt.subplots(figsize=(5.5, 3.5))
        for fam in FAMILIES:
            if fam not in tg_mus:
                continue
            lags, prof = realigned_profile(tg_mus[fam])
            lags = lags * dt
            m = (lags >= -0.8) & (lags <= 0.8)
            ax_d.plot(lags[m], prof[m], color=FAMILY_COLORS[fam], lw=2, label=FAMILY_LABELS[fam])
        ax_d.axvline(0, color="k", ls="--", lw=0.6, alpha=0.4)
        ax_d.set_xlabel("Time (s) relative to diagonal")
        ax_d.set_ylabel("R")
        ax_d.set_title(f"D · sustained representations ({label})")
        ax_d.legend(fontsize=7, frameon=False, ncol=2)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)
        out_d = out.with_name(out.stem + "_fig4D_diagonal.png")
        fig_d.savefig(out_d, dpi=160, bbox_inches="tight")
        plt.close(fig_d)

    fig.suptitle(f"Fig 4 · temporal generalization ({label}, n={len(participants)})", y=1.01, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fig5_compare(
    raw_dir: Path,
    res_dir: Path,
    participants: list[str],
    out: Path,
) -> None:
    """Side-by-side Fig 5B for raw vs acoustic-residual."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, results, title in zip(
        axes,
        [raw_dir, res_dir],
        ["raw EEG", "acoustic residual"],
    ):
        for fam in FAMILIES:
            times, mu, sem = cohort_family_traces(results, participants, fam)
            if times.size == 0:
                continue
            m = (times >= -0.2) & (times <= 1.0)
            times, mu, sem = times[m], mu[m], sem[m]
            peak = float(np.nanmax(mu)) if np.any(np.isfinite(mu)) else np.nan
            yn = mu / peak if np.isfinite(peak) and peak > 0 else np.zeros_like(mu)
            c = FAMILY_COLORS[fam]
            ax.fill_between(times, yn - sem / max(peak, 1e-12), yn + sem / max(peak, 1e-12), color=c, alpha=0.15, lw=0)
            ax.plot(times, yn, color=c, lw=1.8, label=FAMILY_LABELS[fam])
        ax.axvspan(-0.35, 0.1, color="0.88", zorder=0)
        ax.axvline(0, color="k", lw=0.6, alpha=0.35)
        ax.set_xlim(-0.2, 0.65)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel("Time (s) relative to word offset")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Normalised performance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Fig 5B · raw vs acoustic-residual (n={len(participants)})", y=1.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--participants", default="")
    ap.add_argument("--raw-dir", default="", help="If set with --res-dir, also write compare panel")
    ap.add_argument("--res-dir", default="")
    args = ap.parse_args()

    participants = load_participants(Path(args.groups), args.participants)
    results = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    label = args.label or results.name

    plot_fig5a(results, participants, out_dir / f"fig5a_stacked_{label}.png", label)
    plot_fig5b(results, participants, out_dir / f"fig5b_normalized_{label}.png", label)
    plot_fig4(results, participants, out_dir / f"fig4_tg_{label}.png", label)

    if args.raw_dir and args.res_dir:
        plot_fig5_compare(
            Path(args.raw_dir),
            Path(args.res_dir),
            participants,
            out_dir / "fig5b_raw_vs_acoures.png",
        )

    print(f"Wrote Gwilliams-style figures → {out_dir}")


if __name__ == "__main__":
    main()
