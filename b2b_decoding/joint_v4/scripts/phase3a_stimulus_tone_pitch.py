#!/usr/bin/env python3
"""Phase 3a stimulus-side (no EEG): lag-1 tone transitions and tone(n) ~ pitch(n-1).

Decides how the tone-category null is worded. Does not edit features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
OUT = Path(__file__).resolve().parents[1] / "phase3a"
TONES = ("T1", "T2", "T3", "T4")
PITCH_NAMES = [
    "f0_mean", "f0_std", "f0_range",
    "f0_t00", "f0_t01", "f0_t02", "f0_t03", "f0_t04",
    "f0_t05", "f0_t06", "f0_t07", "f0_t08", "f0_t09",
]


def load_section(sid: int):
    d = SHARED / f"section_{sid:03d}"
    tone = np.load(d / "X_word_tone_v3.npy")
    pitch = np.load(d / "X_word_pitch.npy")
    return tone, pitch


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tones, pitches = [], []
    for sid in (1, 2):
        t, p = load_section(sid)
        tones.append(t)
        pitches.append(p)
        # story-internal lag-1 only
    trans = np.zeros((4, 4), dtype=float)
    n_lag = 0
    rows = []
    for tone, pitch in zip(tones, pitches):
        onehot = tone[:, :4]
        lab = onehot.argmax(axis=1)
        valid = onehot.sum(axis=1) > 0.5
        for i in range(1, len(lab)):
            if not (valid[i] and valid[i - 1]):
                continue
            trans[lab[i - 1], lab[i]] += 1
            n_lag += 1
            rec = {f"tone_{k+1}": onehot[i, k] for k in range(4)}
            rec["tone_dev"] = tone[i, 4]
            rec["tone_dev_valid"] = tone[i, 5]
            for j, name in enumerate(PITCH_NAMES):
                rec[f"prev_{name}"] = pitch[i - 1, j]
            rec["prev_tone"] = lab[i - 1] + 1
            rec["tone"] = lab[i] + 1
            rows.append(rec)

    trans_p = trans / trans.sum(axis=1, keepdims=True)
    trans_df = pd.DataFrame(trans_p, index=TONES, columns=TONES)
    trans_df.to_csv(OUT / "tone_lag1_transition.csv")
    counts = pd.DataFrame(trans, index=TONES, columns=TONES)
    counts.to_csv(OUT / "tone_lag1_counts.csv")

    df = pd.DataFrame(rows)
    pred = [c for c in df.columns if c.startswith("prev_f0_")]
    coef_rows = []
    for k, name in enumerate(TONES):
        y = df[f"tone_{k+1}"].to_numpy(float)
        X = np.column_stack([np.ones(len(df)), df[pred].to_numpy(float)])
        # OLS; report R² and max |β| on pitch predictors
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        coef_rows.append(dict(
            outcome=name, r2=r2, intercept=beta[0],
            **{pred[j]: beta[j + 1] for j in range(len(pred))},
        ))
    coef = pd.DataFrame(coef_rows)
    coef.to_csv(OUT / "tone_n_on_pitch_nminus1.csv", index=False)

    # How much of tone one-hots is linearly predictable from previous pitch?
    r2_mean = float(coef["r2"].mean())
    # Self-transition vs uniform
    self_p = float(np.mean(np.diag(trans_p)))
    # Sandhi-ish: T3→T2 rate
    t3_t2 = float(trans_p[2, 1])

    md = OUT / "stimulus_tone_pitch.md"
    md.write_text(
        "# Phase 3a stimulus-side — tone vs previous pitch\n\n"
        f"Tokens with valid lag-1 pair: **{n_lag}** (both stories, story-internal).\n\n"
        "## Lag-1 tone transition (row = tone n−1)\n\n"
        + trans_df.round(3).to_string()
        + "\n\n"
        f"Mean self-transition: {self_p:.3f}. T3→T2: {t3_t2:.3f}.\n\n"
        "## tone(n) one-hots ~ pitch(n−1) (13-D OLS)\n\n"
        + coef[["outcome", "r2"]].round(3).to_string(index=False)
        + f"\n\nMean R²: **{r2_mean:.3f}**.\n\n"
        "## Wording for the tone-category null\n\n"
        + (
            "Previous-word pitch linearly predicts current-tone category at "
            f"R² ≈ {r2_mean:.2f}. "
            + (
                "The tone family is not a near-copy of lag-1 F0; a tone-category "
                "null is about category encoding after contemporaneous pitch is "
                "partialled in the joint B2B, not about missing F0 context.\n"
                if r2_mean < 0.25
                else "Current-tone one-hots are substantially linear in previous "
                "F0. A tone-category null must be worded as: unique variance "
                "after pitch (including lag-0 contour) is already in the model; "
                "do not claim the stimulus lacks F0→tone structure.\n"
            )
        )
    )
    print(md.read_text())


if __name__ == "__main__":
    main()
