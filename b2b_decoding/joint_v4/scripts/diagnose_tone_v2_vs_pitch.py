#!/usr/bin/env python3
"""Combined-arm design diagnostic: tone_v2 one-hots vs pitch mean-F0.

Reports correlations and the condition number of the concatenated
standardized design (envelope_v2 + pitch + tone_v2 + offset + freq + surprisal).
Does not touch EEG.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POOL = Path("/orcd/pool/005/haolun52")
SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "envelope_v2",
    "pitch",
    "tone_v2",
    "offset",
    "lexical_frequency",
    "gpt2cn_surprisal",
)


def _load(key: str) -> np.ndarray:
    blocks = []
    for sid in (1, 2):
        p = SHARED / f"section_{sid:03d}" / f"X_word_{key}.npy"
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        blocks.append(np.load(p))
    return np.vstack(blocks)


def _z(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - X.mean(axis=0)) / sd


def main() -> None:
    fam = {k: _load(k) for k in KEYS}
    pitch = fam["pitch"]
    tone = fam["tone_v2"]
    # pitch cols: f0_mean, f0_std, f0_range, f0_t00...
    names_p = ["f0_mean", "f0_std", "f0_range"] + [f"f0_t{k:02d}" for k in range(10)]
    names_t = ["tone_1", "tone_2", "tone_3", "tone_4", "tone_dev", "tone_dev_valid"]

    print(f"n_words={tone.shape[0]}  pitch={pitch.shape}  tone_v2={tone.shape}")
    print("corr(tone one-hot / tone_dev, pitch columns):")
    corr = {}
    for i, tn in enumerate(names_t):
        row = {}
        for j, pn in enumerate(names_p[:4]):  # mean/std/range + t00
            r = float(np.corrcoef(tone[:, i], pitch[:, j])[0, 1])
            row[pn] = r
        corr[tn] = row
        print(
            f"  {tn:16s}  f0_mean={row['f0_mean']:+.3f}  "
            f"f0_std={row['f0_std']:+.3f}  f0_range={row['f0_range']:+.3f}  "
            f"f0_t00={row['f0_t00']:+.3f}"
        )

    X = np.hstack([_z(fam[k]) for k in KEYS])
    # cond of X'X and of X
    xtx = X.T @ X
    cond_x = float(np.linalg.cond(X))
    cond_xtx = float(np.linalg.cond(xtx))
    print(f"combined design  n={X.shape[0]} p={X.shape[1]}")
    print(f"  cond(X)    = {cond_x:.3f}")
    print(f"  cond(X'X)  = {cond_xtx:.3f}")
    # family-block cond
    for k in KEYS:
        c = float(np.linalg.cond(_z(fam[k])))
        print(f"  cond({k:20s}) = {c:.3f}  dim={fam[k].shape[1]}")

    out = {
        "n_words": int(tone.shape[0]),
        "corr_tone_vs_pitch": corr,
        "cond_X": cond_x,
        "cond_XtX": cond_xtx,
        "dims": {k: int(fam[k].shape[1]) for k in KEYS},
        "note": (
            "Substantial corr(T1/T4, f0_mean)>0 and corr(T2/T3, f0_mean)<0 is expected. "
            "Worry only if cond(X) is huge (near-collinear)."
        ),
    }
    path = ROOT / "plots" / "tone_v2_vs_pitch_design.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
