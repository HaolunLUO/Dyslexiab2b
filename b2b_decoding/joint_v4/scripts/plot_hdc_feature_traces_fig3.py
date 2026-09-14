#!/usr/bin/env python3
"""Gwilliams Fig. 3–style per-feature decoding traces (cohort mean ± SEM).

Reads ``*_b2b_pc_splits.csv`` (passthrough: PC index = feature column order),
maps to feature names, and draws a multi-panel grid with family colors.

Usage:
  .venv_gpt2/bin/python3 scripts/plot_hdc_feature_traces_fig3.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \\
      --groups cohort_groups.csv \\
      --out group_comparison_b2b_hdc_offset_syntax/figures/feature_traces_fig3_style.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FEATURE_ROOT = Path(
    "/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared"
    "/_shared_wordlocked_features/section_001"
)

FAMILY_KEYS = {
    "phonetic": "hdc_phonetic",
    "word_form": "hdc_word_form",
    "lexical_syntactic": "hdc_lexical_syntactic",
    "syntactic_operation": "hdc_syntactic_operation",
    "syntactic_state": "hdc_syntactic_state",
    "semantic": "hdc_semantic",
    "syntax_proxy": "hdc_syntax_proxy",
}

FAMILY_COLORS = {
    "phonetic": "#e377c2",
    "word_form": "#9467bd",
    "lexical_syntactic": "#17becf",
    "syntactic_operation": "#2ca02c",
    "syntactic_state": "#8c564b",
    "semantic": "#d62728",
    "syntax_proxy": "#2ca02c",
}

# Pretty labels approximating Gwilliams naming
LABEL_MAP = {
    "ph_voiced": "Voicing",
    "ph_fricative": "Fricative",
    "ph_nasal": "Nasal",
    "ph_plosive": "Plosive",
    "ph_approximant": "Approximant",
    "ph_vowel": "Vowel",
    "ph_labial": "Labial",
    "ph_coronal": "Coronal",
    "ph_velar": "Velar",
    "ph_glottal": "Glottal",
    "ph_central": "Central vowel",
    "ph_low": "Low vowel",
    "ph_mid": "Mid vowel",
    "ph_high": "High vowel",
    "logfreq": "Word frequency",
    "wf_n_phonemes": "#Phonemes",
    "wf_n_syllables": "#Syllables",
    "wf_n_morphemes": "#Morphemes",
    "wf_phonemes_per_syll": "Phon/syll",
    "pos_ADJ": "Adj.",
    "pos_ADP": "Adp.",
    "pos_ADV": "Adverb",
    "pos_CCONJ": "Coord-conj",
    "pos_DET": "Determiner",
    "pos_INTJ": "Intj.",
    "pos_NOUN": "Noun",
    "pos_NUM": "Num.",
    "pos_PART": "Particle",
    "pos_PRON": "Pronoun",
    "pos_PROPN": "Proper noun",
    "pos_SCONJ": "Subord-conj",
    "pos_VERB": "Verb",
    "pos_X": "Other POS",
    "n_opening": "Opening nodes",
    "n_closing": "Closing nodes",
    "sentence_end": "Sentence end",
    "n_open_nodes": "Open nodes",
    "depth": "Depth",
    "depth_m1": "Depth −1",
    "depth_p1": "Depth +1",
    "linear_order": "Linear order",
    "order_m1": "Order −1",
    "order_p1": "Order +1",
    "top_down": "Top-down",
    "bottom_up": "Bottom-up",
    "left_corner": "Left-corner",
}
for i in range(1, 11):
    LABEL_MAP[f"glove_pc{i:02d}"] = f"GloVe PC{i}"

# Panel layout: (title family color key, list of raw feature names)
# Mirrors Gwilliams Fig. 3 structure with our Chinese feature inventory.
DEFAULT_PANELS = [
    ("phonetic", ["ph_nasal", "ph_voiced", "ph_low"]),
    ("word_form", ["wf_n_syllables", "wf_n_phonemes", "wf_n_morphemes"]),
    ("word_form", ["logfreq", "wf_phonemes_per_syll"]),
    ("lexical_syntactic", ["pos_NOUN", "pos_ADJ", "pos_VERB"]),
    ("lexical_syntactic", ["pos_CCONJ", "pos_DET", "pos_PRON"]),
    ("syntactic_operation", ["n_closing", "n_opening", "sentence_end"]),
    ("syntactic_state", ["depth", "n_open_nodes", "linear_order"]),
    ("semantic", ["glove_pc01", "glove_pc02", "glove_pc03"]),
]


def load_feature_names() -> dict[str, list[str]]:
    out = {}
    for fam, key in FAMILY_KEYS.items():
        path = FEATURE_ROOT / f"X_word_{key}_feature_names.txt"
        if path.is_file():
            out[fam] = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return out


def pretty(name: str) -> str:
    return LABEL_MAP.get(name, name)


def load_groups(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return list(df.loc[df["include_primary"].astype(bool), "participant"].astype(str))


def subject_feature_traces(path: Path, families: list[str]) -> pd.DataFrame:
    """Mean coefficient across splits → rows of time × family × PC."""
    usecols = ["time", "family", "PC", "coefficient", "is_nuisance"]
    df = pd.read_csv(path, usecols=usecols)
    df = df.loc[~df["is_nuisance"].astype(bool) & df["family"].isin(families)]
    g = (
        df.groupby(["family", "PC", "time"], as_index=False)["coefficient"]
        .mean()
    )
    return g


def cohort_mean_sem(
    results: Path, participants: list[str], families: list[str]
) -> tuple[np.ndarray, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]]:
    """Return times and dict (family, pc) -> (mean, sem) over subjects."""
    cubes: dict[tuple[str, int], list[np.ndarray]] = {}
    times = None
    n_ok = 0
    for p in participants:
        path = results / p / f"{p}_b2b_pc_splits.csv"
        if not path.is_file():
            continue
        g = subject_feature_traces(path, families)
        if times is None:
            times = np.sort(g["time"].unique())
        for (fam, pc), sub in g.groupby(["family", "PC"]):
            sub = sub.sort_values("time")
            y = sub["coefficient"].to_numpy(float)
            # align to global times
            if len(y) != len(times) or not np.allclose(sub["time"].to_numpy(float), times):
                y = np.interp(times, sub["time"].to_numpy(float), y)
            cubes.setdefault((str(fam), int(pc)), []).append(y)
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  loaded {n_ok}/{len(participants)}", flush=True)
    assert times is not None
    out = {}
    for key, arrs in cubes.items():
        M = np.vstack(arrs)
        mu = M.mean(0)
        se = M.std(0, ddof=1) / np.sqrt(M.shape[0])
        out[key] = (mu, se)
    print(f"cohort n={n_ok} features={len(out)}")
    return times, out


def resolve_panels(
    panels: list[tuple[str, list[str]]],
    names_by_fam: dict[str, list[str]],
    available_fams: set[str],
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Map feature names to (PC 1-indexed, display name); drop missing."""
    resolved = []
    for fam, feats in panels:
        if fam not in available_fams or fam not in names_by_fam:
            continue
        name_list = names_by_fam[fam]
        items = []
        for feat in feats:
            if feat not in name_list:
                continue
            pc = name_list.index(feat) + 1
            items.append((pc, pretty(feat)))
        if items:
            resolved.append((fam, items))
    return resolved


def plot_fig3(
    times: np.ndarray,
    traces: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    panels: list[tuple[str, list[tuple[int, str]]]],
    out: Path,
    scale: float,
    title: str,
) -> None:
    n = len(panels)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(
        nrows, 2, figsize=(11.5, 2.35 * nrows + 0.8), sharex=True, constrained_layout=False
    )
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.07, hspace=0.35, wspace=0.22)

    # Top family legend
    legend_fams = []
    for fam, _ in panels:
        if fam not in legend_fams:
            legend_fams.append(fam)
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[f], lw=3, label=f.replace("_", " "))
        for f in legend_fams
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.98),
    )

    for i, (fam, items) in enumerate(panels):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        color = FAMILY_COLORS[fam]
        peaks = []
        for j, (pc, lab) in enumerate(items):
            key = (fam, pc)
            if key not in traces:
                continue
            mu, se = traces[key]
            y = mu * scale
            e = se * scale
            lw = 2.4 if j == 0 else 1.4
            ax.plot(times, y, color=color, lw=lw, solid_capstyle="round")
            ax.fill_between(times, y - e, y + e, color=color, alpha=0.18, lw=0)
            # label near peak
            k = int(np.nanargmax(np.abs(y)))
            peaks.append((times[k], y[k], lab, j))
        ax.axhline(0.0, color="0.55", ls="--", lw=0.9)
        ax.axvline(0.0, color="0.75", ls=":", lw=0.8)
        ax.set_xlim(float(times[0]), float(times[-1]))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # annotate labels staggered
        if peaks:
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin if ymax > ymin else 1.0
            for t, y, lab, j in peaks:
                ax.text(
                    t,
                    y + (0.04 + 0.03 * j) * span * np.sign(y if y != 0 else 1),
                    lab,
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="bottom" if y >= 0 else "top",
                    fontweight="bold" if j == 0 else "normal",
                )
        if r == nrows - 1:
            ax.set_xlabel("Time (s) relative to word offset")
        if c == 0:
            ax.set_ylabel(f"Decoding (×{scale:g})")
        if i == 0:
            ax.annotate(
                "chance",
                xy=(times[2], 0),
                xytext=(times[2], ax.get_ylim()[1] * 0.55),
                fontsize=8,
                color="0.4",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
            )

    # hide unused axes
    for j in range(n, nrows * 2):
        r, c = divmod(j, 2)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1e4, help="Multiply betas for display")
    ap.add_argument(
        "--cache",
        default="",
        help="Optional .npz cache for cohort mean/SEM traces",
    )
    args = ap.parse_args()
    results = Path(args.results_dir)
    participants = load_groups(Path(args.groups))
    names_by_fam = load_feature_names()

    # Discover families from first subject
    man = json.loads((results / participants[0] / f"{participants[0]}_run_manifest.json").read_text())
    families = list(man.get("family_names", list(FAMILY_KEYS)))
    panels = resolve_panels(DEFAULT_PANELS, names_by_fam, set(families))
    if not panels:
        raise SystemExit("No panels could be resolved for available families")

    cache = Path(args.cache) if args.cache else results / "_cache_feature_traces_fig3.npz"
    if cache.is_file():
        print(f"loading cache {cache}")
        z = np.load(cache, allow_pickle=True)
        times = z["times"]
        keys = [tuple(x) for x in z["keys"]]
        traces = {
            (str(fam), int(pc)): (z[f"mu_{fam}_{pc}"], z[f"se_{fam}_{pc}"])
            for fam, pc in keys
        }
    else:
        print("aggregating per-feature traces across cohort (slow once)…")
        times, traces = cohort_mean_sem(results, participants, families)
        payload = {"times": times, "keys": np.array(list(traces.keys()), dtype=object)}
        for (fam, pc), (mu, se) in traces.items():
            payload[f"mu_{fam}_{pc}"] = mu
            payload[f"se_{fam}_{pc}"] = se
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **payload)
        print(f"cached {cache}")

    title = (
        "Per-feature decoding (cohort mean ± SEM) · offset-locked · "
        f"n={len(participants)} · scale ×{args.scale:g}"
    )
    plot_fig3(times, traces, panels, Path(args.out), args.scale, title)

    # also dump which features were plotted
    meta = {
        "panels": [
            {"family": fam, "features": [lab for _, lab in items], "pcs": [pc for pc, _ in items]}
            for fam, items in panels
        ],
        "scale": args.scale,
        "n_subjects": len(participants),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
