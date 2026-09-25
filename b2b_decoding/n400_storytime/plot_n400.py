#!/usr/bin/env python3
"""Grand-average ERPs, coefficient forest, and per-subject slope distributions."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import n400_common as C

GROUPS = ["TD", "dyslexia_normal_CAP", "dyslexia_atypical_CAP"]
GROUP_LABEL = {
    "TD": "TD",
    "dyslexia_normal_CAP": "nCAP",
    "dyslexia_atypical_CAP": "aCAP",
}
TERTILE_COLOR = {"low": "#4C78A8", "mid": "#B0B0B0", "high": "#E45756"}


def _subject_wave_means(lmm: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Return times and subject x tertile mean waveforms for CP and Fz."""
    times = None
    bag: dict[tuple[str, str], list[np.ndarray]] = {}
    bag_fz: dict[tuple[str, str], list[np.ndarray]] = {}
    for subject, sub in lmm.groupby("subject"):
        group = sub["group"].iloc[0]
        npz = np.load(C.OUT_ROOT / subject / "timecourse_kept.npz")
        if times is None:
            times = npz["times"].astype(float)
        key = pd.DataFrame(
            {"section": npz["section"].astype(int), "word_index": npz["word_index"].astype(int)}
        )
        key["row"] = np.arange(len(key))
        merged = sub.merge(key, on=["section", "word_index"], how="inner")
        cp = npz["cp"]
        fz = npz["fz"]
        for tert, chunk in merged.groupby("sur_tertile", observed=True):
            ix = chunk["row"].to_numpy(int)
            if ix.size == 0:
                continue
            bag.setdefault((group, str(tert)), []).append(cp[ix].mean(axis=0))
            bag_fz.setdefault((group, str(tert)), []).append(fz[ix].mean(axis=0))
    return times, {"cp": bag, "fz": bag_fz}


def plot_erps(lmm: pd.DataFrame, out: Path) -> None:
    times, waves = _subject_wave_means(lmm)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey="row")
    for row, roi in enumerate(("cp", "fz")):
        for col, group in enumerate(GROUPS):
            ax = axes[row, col]
            for tert in ("low", "mid", "high"):
                arr = np.vstack(waves[roi].get((group, tert), [np.full(len(times), np.nan)]))
                if not np.isfinite(arr).any():
                    continue
                mu = np.nanmean(arr, axis=0)
                se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.isfinite(arr[:, 0]).sum())
                ax.plot(times, mu, color=TERTILE_COLOR[tert], label=f"{tert} surprisal", lw=1.6)
                ax.fill_between(times, mu - se, mu + se, color=TERTILE_COLOR[tert], alpha=0.18, lw=0)
            ax.axvline(0, color="k", lw=0.6)
            ax.axvspan(0.35, 0.55, color="0.85", zorder=0)
            ax.set_title(f"{GROUP_LABEL[group]}  {roi.upper()}")
            if row == 1:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Amplitude (as stored)")
            if row == 0 and col == 2:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Grand-average ERP by GPT2-CN surprisal tertile", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(lmm_dir: Path, out: Path) -> None:
    frames = []
    for roi, name in (("CP", "coef_full_cp.csv"), ("Fz", "coef_full_fz.csv")):
        df = pd.read_csv(lmm_dir / name)
        df["roi"] = roi
        frames.append(df)
    coef = pd.concat(frames, ignore_index=True)
    terms = ["sur_z", "freq_z", "assoc_z"]
    labels = ["surprisal", "frequency", "association"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ypos = np.arange(len(terms))
    for roi, dodge, color in (("CP", -0.12, "#1F4E79"), ("Fz", 0.12, "#C47B2B")):
        sub = coef.loc[coef["roi"] == roi].set_index("term")
        est = np.array([sub.loc[t, "Estimate"] for t in terms])
        lo = np.array([sub.loc[t, "ci95_low"] for t in terms])
        hi = np.array([sub.loc[t, "ci95_high"] for t in terms])
        ax.errorbar(
            est, ypos + dodge, xerr=[est - lo, hi - est],
            fmt="o", color=color, label=roi, ms=6, lw=1.4, capsize=3,
        )
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient (amplitude per 1 SD, 95% Wald CI)")
    ax.set_title("Full model, all children")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def subject_slopes(lmm: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "sur_z", "freq_z", "assoc_z", "duration_z", "position_z",
        "sur_m1_z", "sur_p1_z", "freq_m1_z", "freq_p1_z", "assoc_m1_z", "assoc_p1_z",
    ]
    rows = []
    for subject, sub in lmm.groupby("subject"):
        y = sub["dv_cp"].to_numpy(float)
        x_cols = predictors.copy()
        X = [np.ones(len(sub))]
        if sub["section"].nunique() > 1:
            X.append((sub["section"].to_numpy(int) == 2).astype(float))
            x_cols = ["section2", *x_cols]
        for p in predictors:
            X.append(sub[p].to_numpy(float))
        X = np.column_stack(X)
        if len(sub) < X.shape[1] + 5 or not np.isfinite(X).all() or not np.isfinite(y).all():
            slope = np.nan
        else:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            slope = float(beta[x_cols.index("sur_z") + 1])  # +1 intercept
        rows.append(
            {
                "subject": subject,
                "group": sub["group"].iloc[0],
                "n_trials": int(len(sub)),
                "sur_slope_cp": slope,
            }
        )
    return pd.DataFrame(rows)


def plot_slopes(slopes: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    data = [slopes.loc[slopes["group"] == g, "sur_slope_cp"].dropna().to_numpy() for g in GROUPS]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_alpha(0.7)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(np.arange(1, 4))
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    ax.set_ylabel("Within-subject surprisal slope (CP)")
    ax.set_title("Per-subject partial slopes")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    media = Path(
        "/run/user/245046/cursor_agent_stores/bc-e43b7bb7-443a-4118-b8d5-2145d2516ab7/files/media/n400-storytime"
    )
    media.mkdir(parents=True, exist_ok=True)
    lmm = pd.read_csv(C.STIM_DIR / "lmm_input.csv")
    plot_erps(lmm, media / "erp_surprisal_tertile.png")
    plot_coefficients(C.STIM_DIR / "lmm", media / "coefficients_cp_fz.png")
    slopes = subject_slopes(lmm)
    slopes.to_csv(C.STIM_DIR / "subject_slopes_cp.csv", index=False)
    plot_slopes(slopes, media / "subject_surprisal_slopes.png")
    for p in media.glob("*.png"):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
