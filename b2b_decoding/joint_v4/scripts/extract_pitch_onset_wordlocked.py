#!/usr/bin/env python3
"""Word-locked pitch (log-F0) and acoustic onset/offset features.

Pitch and envelope come from the existing 100 Hz acoustic cache used by
regress_out_acoustic_mne.py (Hilbert log1p envelope + Praat/librosa log-F0).

Word onset is the Di Liberto-style half-wave-rectified envelope derivative,
plus inter-onset interval. Word offset is the matching energy-drop
(−d(env)/dt) at word offset plus inter-offset interval. Frequency /
GPT2CN surprisal / envelope_v2 are already on disk and are not rewritten.

Usage:
  .venv_gpt2/bin/python scripts/extract_pitch_onset_wordlocked.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
DEFAULT_CACHE = (
    POOL / "extracted_sections_wordlocked_shared_acoustic_residual" / "_acoustic_features_cache"
)
FS = 100.0
N_FRAMES = 10
FRAME_HZ = 50.0
SECTIONS = (1, 2)
EXPECTED = {1: 1753, 2: 1800}


def _load_cache(cache_dir: Path, section: int) -> tuple[np.ndarray, np.ndarray]:
    path = cache_dir / f"section_{section:03d}__fs{int(FS)}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing acoustic cache {path}. Run regress_out_acoustic_mne.py once "
            "to build section_*__fs100.npz."
        )
    with np.load(path) as z:
        env = np.asarray(z["envelope"], dtype=np.float64).ravel()
        log_f0 = np.asarray(z["log_f0"], dtype=np.float64).ravel()
    n = min(env.size, log_f0.size)
    return env[:n], log_f0[:n]


def _load_words(shared: Path, section: int) -> pd.DataFrame:
    csv_path = shared / f"section_{section:03d}" / "word_timing_relative.csv"
    df = pd.read_csv(csv_path)
    n_exp = EXPECTED[section]
    if len(df) != n_exp:
        raise SystemExit(f"section {section}: {len(df)} words != {n_exp}")
    if "onset_relative" not in df.columns:
        df["onset_relative"] = df["onset_sample"].astype(float) / FS
    if "offset_relative" not in df.columns:
        df["offset_relative"] = df["offset_sample"].astype(float) / FS
    return df


def _sample_frames(x: np.ndarray, onset: float) -> np.ndarray:
    out = np.zeros(N_FRAMES, dtype=np.float64)
    n = x.size
    for k in range(N_FRAMES):
        idx = int(round((onset + k / FRAME_HZ) * FS))
        idx = max(0, min(n - 1, idx))
        out[k] = float(x[idx])
    return out


def word_pitch_features(log_f0: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = ["f0_mean", "f0_std", "f0_range"] + [f"f0_t{k:02d}" for k in range(N_FRAMES)]
    n = len(df)
    X = np.zeros((n, len(names)), dtype=np.float64)
    n_f0 = log_f0.size
    for i, row in df.iterrows():
        a = max(0, min(n_f0, int(round(float(row["onset_relative"]) * FS))))
        b = max(a + 1, min(n_f0, int(round(float(row["offset_relative"]) * FS))))
        seg = log_f0[a:b]
        X[i, 0] = float(seg.mean())
        X[i, 1] = float(seg.std(ddof=0))
        X[i, 2] = float(seg.max() - seg.min())
        X[i, 3:] = _sample_frames(log_f0, float(row["onset_relative"]))
    return X, names


def word_onset_features(env: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Half-wave rectified d(env)/dt plus ISI (s since previous word onset)."""
    denv = np.diff(env, prepend=env[:1]) * FS
    onset = np.maximum(denv, 0.0)
    names = ["onset_at_t0", "onset_max_50ms", "onset_mean_50ms", "isi"]
    n = len(df)
    X = np.zeros((n, len(names)), dtype=np.float64)
    n_on = onset.size
    win = max(1, int(round(0.05 * FS)))
    onsets = df["onset_relative"].to_numpy(dtype=float)
    for i, t0 in enumerate(onsets):
        a = max(0, min(n_on - 1, int(round(t0 * FS))))
        b = max(a + 1, min(n_on, a + win))
        seg = onset[a:b]
        X[i, 0] = float(onset[a])
        X[i, 1] = float(seg.max())
        X[i, 2] = float(seg.mean())
        X[i, 3] = 0.0 if i == 0 else float(max(onsets[i] - onsets[i - 1], 0.0))
    return X, names


def word_offset_features(env: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Half-wave rectified −d(env)/dt at word offset plus inter-offset interval."""
    denv = np.diff(env, prepend=env[:1]) * FS
    drop = np.maximum(-denv, 0.0)
    names = ["offset_at_t0", "offset_max_50ms", "offset_mean_50ms", "isi"]
    n = len(df)
    X = np.zeros((n, len(names)), dtype=np.float64)
    n_on = drop.size
    win = max(1, int(round(0.05 * FS)))
    offsets = df["offset_relative"].to_numpy(dtype=float)
    for i, t0 in enumerate(offsets):
        a = max(0, min(n_on - 1, int(round(t0 * FS))))
        b = max(a + 1, min(n_on, a + win))
        seg = drop[a:b]
        X[i, 0] = float(drop[a])
        X[i, 1] = float(seg.max())
        X[i, 2] = float(seg.mean())
        X[i, 3] = 0.0 if i == 0 else float(max(offsets[i] - offsets[i - 1], 0.0))
    return X, names


def _write_feature(shared: Path, section: int, key: str, X: np.ndarray, names: list[str], extra: dict) -> None:
    out_dir = shared / f"section_{section:03d}"
    npy = out_dir / f"X_word_{key}.npy"
    np.save(npy, X.astype(np.float64))
    (out_dir / f"X_word_{key}_feature_names.txt").write_text("\n".join(names) + "\n")
    meta = {
        "feature": key,
        "section_id": section,
        "n_words": int(X.shape[0]),
        "n_dim": int(X.shape[1]),
        "fs": FS,
        "n_onset_frames": N_FRAMES,
        "onset_frame_hz": FRAME_HZ,
        "mean": X.mean(axis=0).tolist(),
        "std": X.std(axis=0, ddof=1).tolist(),
        **extra,
    }
    (out_dir / f"X_word_{key}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {npy}  shape={X.shape}  mean0={X[:, 0].mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    print(f"shared={args.shared}")
    print(f"cache ={args.cache}")
    for sid in SECTIONS:
        env, log_f0 = _load_cache(args.cache, sid)
        df = _load_words(args.shared, sid)
        print(f"section {sid}: words={len(df)}  env={env.shape}  f0={log_f0.shape}")
        Xp, pnames = word_pitch_features(log_f0, df)
        Xo, onames = word_onset_features(env, df)
        Xf, fnames = word_offset_features(env, df)
        if not np.isfinite(Xp).all():
            raise SystemExit(f"non-finite pitch features in section {sid}")
        if not np.isfinite(Xo).all():
            raise SystemExit(f"non-finite onset features in section {sid}")
        if not np.isfinite(Xf).all():
            raise SystemExit(f"non-finite offset features in section {sid}")
        _write_feature(args.shared, sid, "pitch", Xp, pnames, {"source": "log_f0 cache"})
        _write_feature(
            args.shared,
            sid,
            "word_onset",
            Xo,
            onames,
            {"source": "half-wave d(envelope)/dt + ISI"},
        )
        _write_feature(
            args.shared,
            sid,
            "offset",
            Xf,
            fnames,
            {"source": "half-wave -d(envelope)/dt at offset + inter-offset ISI"},
        )
    print("Done.")


if __name__ == "__main__":
    main()
