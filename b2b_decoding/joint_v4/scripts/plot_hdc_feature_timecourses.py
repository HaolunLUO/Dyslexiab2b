#!/usr/bin/env python3
"""Gwilliams-style multi-panel per-feature decoding time courses.

Aggregates ``*_b2b_pc_splits.csv`` across subjects (mean over splits within
subject, then mean ± SEM across subjects) and plots selected features in an
8-panel grid colored by linguistic family.

Usage:
  .venv_gpt2/bin/python3 scripts/plot_hdc_feature_timecourses.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \\
      --groups cohort_groups.csv \\
      --out group_comparison_b2b_hdc_offset_syntax/figures/gwilliams_feature_timecourses.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Panel layout mirrors Gwilliams Fig. 3 (subset of features per family).
# (family, PC 1-based, display label)
PANELS: list[tuple[str, list[tuple[str, int, str]], str]] = [
    (
        "phonetic",
        [
            ("phonetic", 3, "Nasal"),
            ("phonetic", 1, "Voicing"),
            ("phonetic", 12, "Low vowel"),
        ],
        "#d627a8",
    ),
    (
        "word_form",
        [
            ("word_form", 3, "#Syllables"),
            ("word_form", 2, "#Phonemes"),
            ("word_form", 4, "#Morphemes"),
        ],
        "#9467bd",
    ),
    (
        "word_form",
        [
            ("word_form", 1, "Word frequency"),
            ("word_form", 5, "Phon/syll"),
        ],
        "#7b4b9a",
    ),
    (
        "lexical_syntactic",
        [
            ("lexical_syntactic", 1, "Adj."),
            ("lexical_syntactic", 7, "Noun"),
            ("lexical_syntactic", 13, "Verb"),
        ],
        "#1f77b4",
    ),
    (
        "lexical_syntactic",
        [
            ("lexical_syntactic", 4, "Coord-conj"),
            ("lexical_syntactic", 5, "Determiner"),
            ("lexical_syntactic", 10, "Pronoun"),
        ],
        "#17becf",
    ),
    (
        "syntactic_operation",
        [
            ("syntactic_operation", 2, "Closing nodes"),
            ("syntactic_operation", 1, "Opening nodes"),
        ],
        "#2ca02c",
    ),
    (
        "syntactic_state",
        [
            ("syntactic_state", 3, "Depth"),
            ("syntactic_state", 1, "Open nodes"),
        ],
        "#ff7f0e",
    ),
    (
        "semantic",
        [
            ("semantic", 1, "GloVe PC1"),
            ("semantic", 2, "GloVe PC2"),
            ("semantic", 3, "GloVe PC3"),
        ],
        "#d62728",
    ),
]

FAMILY_HEADER = [
    ("phonetic", "#d627a8"),
    ("word form", "#9467bd"),
    ("lexical–syntactic", "#1f77b4"),
    ("syntactic operation", "#2ca02c"),
    ("syntactic state", "#ff7f0e"),
    ("semantic", "#d62728"),
]


def load_groups(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return list(df.loc[df["include_primary"].astype(bool), "participant"].astype(str))


def needed_keys() -> set[tuple[str, int]]:
    return {(fam, pc) for _, feats, _ in PANELS for fam, pc, _ in feats}


def accumulate_subject(
    path: Path,
    keys: set[tuple[str, int]],
    store: dict[tuple[str, int], list[np.ndarray]],
    times_ref: list[np.ndarray],
) -> None:
    usecols = ["time", "family", "PC", "coefficient", "is_nuisance"]
    df = pd.read_csv(path, usecols=usecols)
    if "is_nuisance" in df.columns:
        df = df.loc[~df["is_nuisance"].astype(bool)]
    # mean over splits within subject
    g = (
        df.groupby(["family", "PC", "time"], as_index=False)["coefficient"]
        .mean()
    )
    if not times_ref:
        times_ref.append(np.sort(g["time"].unique()))
    times = times_ref[0]
    for fam, pc in keys:
        sub = g.loc[(g["family"] == fam) & (g["PC"] == pc)].sort_values("time")
        if sub.empty:
            continue
        y = sub["coefficient"].to_numpy(float)
        t = sub["time"].to_numpy(float)
        if len(t) != len(times) or not np.allclose(t, times):
            # interpolate onto reference grid
            y = np.interp(times, t, y)
        store.setdefault((fam, pc), []).append(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply coefficients for display (default 1)",
    )
    args = ap.parse_args()
    results = Path(args.results_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    participants = load_groups(Path(args.groups))
    keys = needed_keys()
    store: dict[tuple[str, int], list[np.ndarray]] = {}
    times_ref: list[np.ndarray] = []
    n_ok = 0
    for p in participants:
        path = results / p / f"{p}_b2b_pc_splits.csv"
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        accumulate_subject(path, keys, store, times_ref)
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  loaded {n_ok}/{len(participants)}")
    print(f"Loaded {n_ok} subjects")
    times = times_ref[0]

    # group mean ± SEM
    stats: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, mats in store.items():
        M = np.vstack(mats) * args.scale
        mu = M.mean(0)
        se = M.std(0, ddof=1) / np.sqrt(M.shape[0])
        stats[key] = (mu, se)

    # --- figure ---
    fig = plt.figure(figsize=(11.5, 10.5))
    # top family labels
    for i, (lab, col) in enumerate(FAMILY_HEADER):
        fig.text(
            0.08 + i * 0.155,
            0.965,
            lab,
            color=col,
            fontsize=11,
            fontweight="bold",
            ha="left",
        )

    axes = []
    for i in range(8):
        ax = fig.add_subplot(4, 2, i + 1)
        axes.append(ax)

    # y-limits: syntax panels get larger scale like Gwilliams
    for ax_i, (fam_tag, feats, base_color) in enumerate(PANELS):
        ax = axes[ax_i]
        n_lines = len(feats)
        # shade of base color for each line
        for j, (fam, pc, label) in enumerate(feats):
            mu, se = stats[(fam, pc)]
            # darker → lighter within panel
            alpha_line = 0.55 + 0.45 * (j / max(n_lines - 1, 1))
            color = base_color
            lw = 2.2 if j == 0 else 1.5
            ax.plot(times, mu, color=color, lw=lw, alpha=alpha_line, label=label)
            ax.fill_between(
                times, mu - se, mu + se, color=color, alpha=0.12 + 0.06 * j, lw=0
            )
        ax.axhline(0, color="0.45", ls="--", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlim(float(times.min()), float(times.max()))
        if ax_i >= 6 or fam_tag in ("syntactic_operation", "syntactic_state"):
            # leave autoscaled but ensure room; Gwilliams used larger y for syntax
            pass
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        if ax_i % 2 == 0:
            ax.set_ylabel("Decoding performance\n(diag $H$)", fontsize=8)
        if ax_i >= 6:
            ax.set_xlabel("Time (s) relative to word offset", fontsize=9)
        else:
            ax.set_xticklabels([])
        if ax_i == 0:
            ax.text(
                0.02,
                0.05,
                "chance",
                transform=ax.transAxes,
                fontsize=8,
                color="0.4",
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    # Match Gwilliams: enlarge y-scale on syntax panels if needed
    for ax_i in (5, 6):  # operation, state
        ax = axes[ax_i]
        ymin, ymax = ax.get_ylim()
        # if peaks are small, still fine; if large relative to others, keep auto
        ax.set_ylim(min(ymin, -0.05 * abs(ymax)), ymax * 1.05)

    fig.suptitle(
        f"Feature decoding time courses · B2B diag(H) · n={n_ok} · offset-locked",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06, hspace=0.22, wspace=0.18)
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")

    # also dump means for reuse
    rows = []
    for (fam, pc), (mu, se) in stats.items():
        for t, m, s in zip(times, mu, se):
            rows.append(
                {"family": fam, "PC": pc, "time": float(t), "mean": float(m), "se": float(s)}
            )
    csv_out = out.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
