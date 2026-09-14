#!/usr/bin/env python3
"""Rebuild section F0 with an explicit voicing mask (v2).

Writes a *new* cache; the filled-mean v1 files (section_XXX__fs100.npz) are
never overwritten so existing pitch arms keep reproducing.

  section_XXX__f0_voicing_fs100_v2.npz
      f0_hz     Hz, NaN where unvoiced
      voiced    bool, from the tracker (Praat voiced / pYIN p>=0.5)
      log_f0    ln(f0_hz), NaN where unvoiced
      fs        100.0
      backend   "praat" | "librosa_pyin"

Usage:
  .venv_gpt2/bin/python scripts/extract_f0_voicing_v2.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
DEFAULT_CACHE = (
    POOL / "extracted_sections_wordlocked_shared_acoustic_residual" / "_acoustic_features_cache"
)
FS_TARGET = 100.0
PITCH_FMIN, PITCH_FMAX = 75.0, 400.0
SECTIONS = (1, 2)


def resolve_wav(shared: Path, section: int) -> Path:
    candidates = [
        shared / f"section_{section:03d}" / f"task-lppCN_section_{section}.wav",
        shared / f"section_{section:03d}" / f"task-lppCN_section_{section:03d}.wav",
        POOL / f"task-lppCN_section_{section}.wav",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise SystemExit("No WAV for section "
                     f"{section}. Tried:\n  " + "\n  ".join(str(c) for c in candidates))


def load_audio(path: Path) -> tuple[np.ndarray, float]:
    x, fs = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), float(fs)


def _resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from scipy import signal
    n_out = int(round(x.size * fs_out / fs_in))
    return signal.resample(x, n_out)


def _map_to_grid(
    times: np.ndarray,
    values: np.ndarray,
    n_target: int,
    fs: float,
    *,
    voiced: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour map of tracker frames onto a regular fs grid.

    Frames farther than half a grid step, or marked unvoiced, become NaN.
    """
    t_grid = np.arange(n_target, dtype=np.float64) / fs
    f0_hz = np.full(n_target, np.nan, dtype=np.float64)
    v_out = np.zeros(n_target, dtype=bool)
    if times.size == 0:
        return f0_hz, v_out
    idx = np.searchsorted(times, t_grid, side="left")
    idx = np.clip(idx, 1, times.size - 1)
    left = idx - 1
    choose_right = np.abs(times[idx] - t_grid) < np.abs(times[left] - t_grid)
    nearest = np.where(choose_right, idx, left)
    dist = np.abs(times[nearest] - t_grid)
    ok = dist <= (0.5 / fs + 1e-9)
    vals = values[nearest]
    if voiced is None:
        is_v = np.isfinite(vals) & (vals > 0)
    else:
        is_v = voiced[nearest] & np.isfinite(vals) & (vals > 0)
    keep = ok & is_v
    f0_hz[keep] = vals[keep]
    v_out[keep] = True
    return f0_hz, v_out


def pitch_praat(audio: np.ndarray, fs_audio: float, fs_target: float) -> tuple[np.ndarray, np.ndarray]:
    import parselmouth
    snd = parselmouth.Sound(audio, sampling_frequency=fs_audio)
    pitch = snd.to_pitch(
        time_step=1.0 / fs_target,
        pitch_floor=PITCH_FMIN,
        pitch_ceiling=PITCH_FMAX,
    )
    times = np.asarray(pitch.xs(), dtype=np.float64)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    n_target = int(round(float(snd.xmax) * fs_target))
    return _map_to_grid(times, f0, n_target, fs_target, voiced=f0 > 0)


def pitch_pyin(audio: np.ndarray, fs_audio: float, fs_target: float) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    fs_pyin = 16000.0
    if abs(fs_audio - fs_pyin) > 1.0:
        audio_ds = _resample_1d(audio, fs_audio, fs_pyin)
    else:
        audio_ds = audio
        fs_pyin = fs_audio
    hop = max(1, int(round(fs_pyin / fs_target)))
    print(f"  [pitch] librosa.pyin @ {fs_pyin:g} Hz  hop={hop}  n={audio_ds.size}", flush=True)
    f0, _voiced_flag, voiced_prob = librosa.pyin(
        audio_ds,
        fmin=PITCH_FMIN,
        fmax=PITCH_FMAX,
        sr=fs_pyin,
        hop_length=hop,
        frame_length=2048,
        fill_na=np.nan,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    if voiced_prob is None:
        voiced = np.isfinite(f0) & (f0 > 0)
    else:
        voiced = (np.asarray(voiced_prob, dtype=np.float64) >= 0.5) & np.isfinite(f0) & (f0 > 0)
    n_target = int(round(audio.size / fs_audio * fs_target))
    times = np.arange(f0.size, dtype=np.float64) * (hop / fs_pyin)
    return _map_to_grid(times, f0, n_target, fs_target, voiced=voiced)


def extract_section(shared: Path, cache: Path, section: int, *, overwrite: bool) -> Path:
    out = cache / f"section_{section:03d}__f0_voicing_fs100_v2.npz"
    old = cache / f"section_{section:03d}__fs{int(FS_TARGET)}.npz"
    if old.is_file():
        print(f"  leaving v1 cache untouched: {old.name}")
    if out.is_file() and not overwrite:
        print(f"  exists, skip: {out}")
        return out

    wav = resolve_wav(shared, section)
    audio, fs_audio = load_audio(wav)
    print(f"  [audio] {wav.name}  ({audio.size / fs_audio:.1f}s @ {fs_audio:g} Hz)", flush=True)

    backend = None
    try:
        import parselmouth  # noqa: F401
        f0_hz, voiced = pitch_praat(audio, fs_audio, FS_TARGET)
        backend = "praat"
    except ImportError:
        f0_hz, voiced = pitch_pyin(audio, fs_audio, FS_TARGET)
        backend = "librosa_pyin"

    log_f0 = np.where(voiced & (f0_hz > 0), np.log(f0_hz), np.nan)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        f0_hz=f0_hz.astype(np.float32),
        voiced=voiced,
        log_f0=log_f0.astype(np.float32),
        fs=np.array(FS_TARGET),
        backend=np.array(backend),
    )
    n = f0_hz.size
    n_v = int(voiced.sum())
    finite = f0_hz[voiced]
    print(
        f"  [cache] wrote {out.name}  backend={backend}  T={n}  "
        f"voiced={n_v} ({n_v / max(n, 1):.3f})  "
        f"F0 Hz {float(np.min(finite)):.1f}–{float(np.max(finite)):.1f}",
        flush=True,
    )
    meta = {
        "section": section,
        "path": str(out),
        "backend": backend,
        "n": n,
        "n_voiced": n_v,
        "f0_hz_min": float(np.min(finite)) if finite.size else None,
        "f0_hz_max": float(np.max(finite)) if finite.size else None,
        "note": "NaN = unvoiced; v1 section_XXX__fs100.npz not modified",
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    print(f"shared={args.shared}")
    print(f"cache ={args.cache}")
    for sid in SECTIONS:
        extract_section(args.shared, args.cache, sid, overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
