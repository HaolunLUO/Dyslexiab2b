#!/usr/bin/env python3
"""Gwilliams-style envelope + pitch TRF residualization (MNE ReceptiveField).

Matches SI §1.6 of Gwilliams et al. PNAS 2025:
  - predictors: broadband envelope + log-F0
  - MNE ReceptiveField with TimeDelayingRidge (laplacian), 10 log-spaced alphas
  - lags = preceding 200 ms (tmin=-0.2, tmax=0)
  - demean/scale before fit; 3 contiguous folds for alpha selection
  - predictions transformed back to original EEG units, then subtracted

Writes a new extractor tree with residual EEG and a symlink to the shared
wordlocked features so HDC B2B can run unchanged against the residual root.

Usage:
  .venv_gpt2/bin/python3 scripts/regress_out_acoustic_mne.py --participants RN109
  .venv_gpt2/bin/python3 scripts/regress_out_acoustic_mne.py --cohort-csv cohort_groups.csv
  .venv_gpt2/bin/python3 scripts/regress_out_acoustic_mne.py --cohort-index 1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import hilbert

import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge

mne.set_log_level("WARNING")

try:
    import librosa

    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False

try:
    import parselmouth

    HAVE_PRAAT = True
except ImportError:
    HAVE_PRAAT = False

POOL = Path("/home/haolun52/orcd/pool")
DEFAULT_SRC = POOL / "extracted_sections_wordlocked_shared"
DEFAULT_OUT = POOL / "extracted_sections_wordlocked_shared_acoustic_residual"
DEFAULT_COHORT = Path(__file__).resolve().parent.parent / "cohort_groups.csv"

FS_TARGET = 100.0
TRF_TMIN = -0.2
TRF_TMAX = 0.0
N_ALPHAS = 10
ALPHAS = np.logspace(-6, 6, N_ALPHAS)
N_FOLDS = 3
PITCH_FMIN, PITCH_FMAX = 75.0, 400.0
SECTIONS = (1, 2)


def load_cohort_participants(path: Path) -> List[str]:
    import csv

    out: List[str] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("include_primary", "0")).strip() in ("1", "true", "True"):
                out.append(str(row["participant"]).strip())
    if not out:
        raise SystemExit(f"No primary participants in {path}")
    return out


def resolve_audio(shared: Path, section: int) -> Path:
    p = shared / f"section_{section:03d}" / f"task-lppCN_section_{section}.wav"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_audio(path: Path) -> Tuple[np.ndarray, float]:
    x, fs = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), float(fs)


def _resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    n_out = int(round(x.size * fs_out / fs_in))
    return signal.resample(x, n_out)


def envelope_hilbert(audio: np.ndarray, fs_audio: float, fs_target: float) -> np.ndarray:
    """Broadband Hilbert envelope, log1p-compressed, at fs_target."""
    env = np.abs(hilbert(audio))
    env = _resample_1d(env, fs_audio, fs_target)
    return np.log1p(np.maximum(env, 0.0))


def pitch_librosa(audio: np.ndarray, fs_audio: float, fs_target: float) -> np.ndarray:
    # Downsample before pyin — full-rate 44.1 kHz pyin on ~10 min audio is
    # prohibitively slow on login nodes; 16 kHz is enough for F0 in 75–400 Hz.
    fs_pyin = 16000.0
    if abs(fs_audio - fs_pyin) > 1.0:
        audio_ds = _resample_1d(audio, fs_audio, fs_pyin)
    else:
        audio_ds = audio
        fs_pyin = fs_audio
    hop = max(1, int(round(fs_pyin / fs_target)))
    print(f"  [pitch] librosa.pyin @ {fs_pyin:g} Hz  hop={hop}  n={audio_ds.size}", flush=True)
    f0, _, _ = librosa.pyin(
        audio_ds,
        fmin=PITCH_FMIN,
        fmax=PITCH_FMAX,
        sr=fs_pyin,
        hop_length=hop,
        frame_length=2048,
    )
    log_f0 = np.where(np.isfinite(f0) & (f0 > 0), np.log(f0), np.nan)
    mean_lf0 = float(np.nanmean(log_f0)) if np.any(np.isfinite(log_f0)) else 0.0
    log_f0 = np.where(np.isfinite(log_f0), log_f0, mean_lf0)
    n_target = int(round(audio.size / fs_audio * fs_target))
    if log_f0.size < n_target:
        pad = np.full(n_target - log_f0.size, mean_lf0)
        log_f0 = np.concatenate([log_f0, pad])
    return log_f0[:n_target].astype(np.float64)


def pitch_praat(audio: np.ndarray, fs_audio: float, fs_target: float) -> np.ndarray:
    snd = parselmouth.Sound(audio, sampling_frequency=fs_audio)
    pitch = snd.to_pitch(
        time_step=1.0 / fs_target,
        pitch_floor=PITCH_FMIN,
        pitch_ceiling=PITCH_FMAX,
    )
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    log_f0 = np.where(voiced, np.log(np.maximum(f0, 1e-3)), np.nan)
    mean_lf0 = float(np.nanmean(log_f0)) if voiced.any() else 0.0
    log_f0 = np.where(np.isfinite(log_f0), log_f0, mean_lf0)
    n_target = int(round(audio.size / fs_audio * fs_target))
    if log_f0.size < n_target:
        log_f0 = np.concatenate([log_f0, np.full(n_target - log_f0.size, mean_lf0)])
    return log_f0[:n_target].astype(np.float64)


def get_acoustic_features(
    shared: Path, cache_dir: Path, section: int
) -> Dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"section_{section:03d}__fs{int(FS_TARGET)}.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            return {k: z[k].astype(np.float64) for k in ("envelope", "log_f0")}

    audio_path = resolve_audio(shared, section)
    audio, fs_audio = load_audio(audio_path)
    print(f"  [audio] {audio_path.name}  ({audio.size / fs_audio:.1f}s @ {fs_audio:g} Hz)")

    env = envelope_hilbert(audio, fs_audio, FS_TARGET)
    if HAVE_PRAAT:
        log_f0 = pitch_praat(audio, fs_audio, FS_TARGET)
        backend = "praat"
    elif HAVE_LIBROSA:
        log_f0 = pitch_librosa(audio, fs_audio, FS_TARGET)
        backend = "librosa"
    else:
        raise ImportError("Need parselmouth or librosa for pitch extraction")

    n = min(env.size, log_f0.size)
    env, log_f0 = env[:n], log_f0[:n]
    np.savez_compressed(
        cache_path,
        envelope=env.astype(np.float32),
        log_f0=log_f0.astype(np.float32),
        pitch_backend=np.array(backend),
    )
    print(f"  [cache] wrote {cache_path.name}  pitch={backend}  T={n}")
    return {"envelope": env, "log_f0": log_f0}


def _eeg_CT(eeg: np.ndarray) -> np.ndarray:
    """Return (n_channels, n_times)."""
    if eeg.ndim != 2:
        raise ValueError(f"EEG must be 2-D, got {eeg.shape}")
    return eeg if eeg.shape[0] < eeg.shape[1] else eeg.T


def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd, mu, sd


def _fold_slices(n: int, n_folds: int = N_FOLDS) -> List[slice]:
    edges = np.round(np.linspace(0, n, n_folds + 1)).astype(int)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]


def fit_residualise_mne(
    eeg_CT: np.ndarray, feats: Dict[str, np.ndarray], sfreq: float
) -> Tuple[np.ndarray, dict]:
    """Fit TRF on z-scored data; return residual in original EEG units."""
    n_ch, n_t = eeg_CT.shape
    X = np.column_stack([feats["envelope"][:n_t], feats["log_f0"][:n_t]])
    if X.shape[0] < n_t:
        pad = np.tile(X[-1:], (n_t - X.shape[0], 1))
        X = np.vstack([X, pad])
    X = X[:n_t]

    Xz, _, _ = _zscore_cols(X)
    Y = eeg_CT.T  # (T, C)
    Yz, y_mu, y_sd = _zscore_cols(Y)

    n_lag = max(1, int(round(abs(TRF_TMIN) * sfreq)))
    folds = _fold_slices(n_t, N_FOLDS)
    fold_scores = np.zeros((N_ALPHAS, N_FOLDS))
    for fi, te in enumerate(folds):
        train_parts = [folds[j] for j in range(N_FOLDS) if j != fi]
        # Concatenate train blocks with a zero gap so lag windows do not
        # bleed across the held-out region (boolean masking would).
        X_blocks, Y_blocks = [], []
        for bi, p in enumerate(train_parts):
            if bi:
                X_blocks.append(np.zeros((n_lag, Xz.shape[1]), dtype=Xz.dtype))
                Y_blocks.append(np.zeros((n_lag, Yz.shape[1]), dtype=Yz.dtype))
            X_blocks.append(Xz[p])
            Y_blocks.append(Yz[p])
        Xtr = np.vstack(X_blocks)
        Ytr = np.vstack(Y_blocks)
        Xte, Yte = Xz[te], Yz[te]
        if Xtr.shape[0] < 50 or Xte.shape[0] < 20:
            continue
        for ai, alpha in enumerate(ALPHAS):
            est = TimeDelayingRidge(
                TRF_TMIN, TRF_TMAX, sfreq, alpha=float(alpha), reg_type="laplacian"
            )
            rf = ReceptiveField(
                TRF_TMIN,
                TRF_TMAX,
                sfreq,
                feature_names=["envelope", "log_f0"],
                estimator=est,
                scoring="corrcoef",
            )
            rf.fit(Xtr, Ytr)
            sc = np.asarray(rf.score(Xte, Yte), float)
            fold_scores[ai, fi] = float(np.nanmean(sc))

    mean_scores = np.nanmean(fold_scores, axis=1)
    if np.all(~np.isfinite(mean_scores)):
        best_i = len(ALPHAS) // 2
    else:
        best_i = int(np.nanargmax(mean_scores))
    best_alpha = float(ALPHAS[best_i])

    est = TimeDelayingRidge(
        TRF_TMIN, TRF_TMAX, sfreq, alpha=best_alpha, reg_type="laplacian"
    )
    rf = ReceptiveField(
        TRF_TMIN,
        TRF_TMAX,
        sfreq,
        feature_names=["envelope", "log_f0"],
        estimator=est,
        scoring="corrcoef",
    )
    rf.fit(Xz, Yz)
    Yhat_z = np.asarray(rf.predict(Xz), float)
    if Yhat_z.ndim == 1:
        Yhat_z = Yhat_z[:, None]
    # Edge samples lack full lag support; zero the prediction there so residual
    # ≈ original EEG. MNE stores this as a slice (not a bool mask).
    if hasattr(rf, "valid_samples_") and rf.valid_samples_ is not None:
        vs = rf.valid_samples_
        mask = np.zeros(n_t, dtype=bool)
        if isinstance(vs, slice):
            mask[vs] = True
        else:
            vs_arr = np.asarray(vs)
            if vs_arr.dtype == bool and vs_arr.shape == (n_t,):
                mask = vs_arr
            elif vs_arr.ndim == 1:
                idx = vs_arr.astype(int)
                mask[idx] = True
        if Yhat_z.shape[0] == n_t:
            Yhat_z = np.where(mask[:, None], Yhat_z, 0.0)

    Yhat = Yhat_z * y_sd + y_mu  # back to original EEG units
    residual_TC = Y - Yhat
    residual_CT = residual_TC.T

    var_tot = np.var(Y, axis=0) + 1e-12
    var_res = np.var(residual_TC, axis=0)
    prop = 1.0 - var_res / var_tot
    info = {
        "best_alpha": best_alpha,
        "cv_scores_by_alpha": mean_scores.tolist(),
        "alphas": ALPHAS.tolist(),
        "mean_var_explained": float(np.mean(prop)),
        "prop_var_explained_per_channel": prop.astype(np.float64),
        "tmin": TRF_TMIN,
        "tmax": TRF_TMAX,
        "predictors": ["envelope", "log_f0"],
        "n_folds": N_FOLDS,
        "reg_type": "laplacian",
    }
    return residual_CT, info


def ensure_shared_symlink(src: Path, out: Path) -> None:
    shared_src = src / "_shared_wordlocked_features"
    shared_dst = out / "_shared_wordlocked_features"
    if not shared_src.is_dir():
        raise FileNotFoundError(shared_src)
    out.mkdir(parents=True, exist_ok=True)
    if shared_dst.is_symlink() or shared_dst.exists():
        if shared_dst.is_symlink() and shared_dst.resolve() == shared_src.resolve():
            pass
        elif shared_dst.is_symlink() or shared_dst.is_file():
            shared_dst.unlink()
            shared_dst.symlink_to(shared_src, target_is_directory=True)
        else:
            raise FileExistsError(
                f"{shared_dst} exists and is not the expected symlink; remove it"
            )
    else:
        shared_dst.symlink_to(shared_src, target_is_directory=True)

    # Tree syntax file lives at extractor root (not under shared features).
    tree_src = src / "lppCN_tree.txt"
    tree_dst = out / "lppCN_tree.txt"
    if tree_src.is_file():
        if tree_dst.is_symlink() or tree_dst.is_file():
            if tree_dst.is_symlink() and tree_dst.resolve() == tree_src.resolve():
                return
            tree_dst.unlink()
        elif tree_dst.exists():
            raise FileExistsError(f"{tree_dst} exists and is not a symlink/file")
        tree_dst.symlink_to(tree_src)



def _copy_aux(src_sec: Path, dst_sec: Path) -> None:
    dst_sec.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "word_events.csv", "train_keep_mask.npy"):
        p = src_sec / name
        if p.is_file():
            shutil.copy2(p, dst_sec / name)


def process_section(
    participant: str,
    section: int,
    src_root: Path,
    out_root: Path,
    cache_dir: Path,
    force: bool,
) -> Optional[dict]:
    src = src_root / participant / f"section_{section:03d}"
    dst = out_root / participant / f"section_{section:03d}"
    if not src.is_dir():
        print(f"  [skip] missing {src}")
        return None
    out_eeg = dst / "eeg_data.npy"
    if out_eeg.is_file() and not force:
        print(f"  [skip] {participant} sec{section} already residualized")
        meta = json.loads((dst / "metadata.json").read_text())
        return meta.get("acoustic_residualised")

    meta = json.loads((src / "metadata.json").read_text())
    fs_eeg = float(meta.get("sfreq", FS_TARGET))
    if abs(fs_eeg - FS_TARGET) > 1e-6:
        raise ValueError(f"{participant} sec{section}: sfreq={fs_eeg}, expected {FS_TARGET}")

    eeg = np.load(src / "eeg_data.npy").astype(np.float64)
    eeg_CT = _eeg_CT(eeg)
    feats = get_acoustic_features(src_root / "_shared_wordlocked_features", cache_dir, section)

    residual_CT, info = fit_residualise_mne(eeg_CT, feats, FS_TARGET)
    if residual_CT.shape != eeg_CT.shape:
        raise RuntimeError(
            f"shape mismatch residual {residual_CT.shape} vs eeg {eeg_CT.shape}"
        )

    _copy_aux(src, dst)
    # Preserve original orientation (channels, times) if that was the on-disk layout
    if eeg.shape == eeg_CT.shape:
        to_save = residual_CT
    else:
        to_save = residual_CT.T
    np.save(out_eeg, to_save.astype(np.float32))
    np.savez_compressed(
        dst / "_acoustic_trf_diagnostics.npz",
        prop_var_explained=info["prop_var_explained_per_channel"],
        best_alpha=info["best_alpha"],
        cv_scores_by_alpha=np.asarray(info["cv_scores_by_alpha"]),
        alphas=np.asarray(info["alphas"]),
    )
    meta_out = dict(meta)
    meta_out["acoustic_residualised"] = {
        "method": "mne_receptive_field_laplacian_ridge",
        "predictors": info["predictors"],
        "tmin": info["tmin"],
        "tmax": info["tmax"],
        "best_alpha": info["best_alpha"],
        "n_folds": info["n_folds"],
        "mean_var_explained_by_acoustic": info["mean_var_explained"],
        "source_eeg": str(src / "eeg_data.npy"),
    }
    (dst / "metadata.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(
        f"  ✓ {participant} sec{section}: "
        f"var explained = {info['mean_var_explained'] * 100:.2f}%  "
        f"α={info['best_alpha']:.2e}  shape={to_save.shape}"
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument(
        "--cohort-index",
        type=int,
        default=None,
        help="1-based index into primary cohort (for Slurm array)",
    )
    ap.add_argument("--sections", type=int, nargs="*", default=list(SECTIONS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not HAVE_PRAAT and not HAVE_LIBROSA:
        raise SystemExit("Install librosa (or parselmouth) for F0 extraction")

    if args.participants:
        participants = list(args.participants)
    else:
        participants = load_cohort_participants(args.cohort_csv)
        if args.cohort_index is not None:
            i = args.cohort_index
            if not (1 <= i <= len(participants)):
                raise SystemExit(f"cohort-index {i} out of 1..{len(participants)}")
            participants = [participants[i - 1]]

    ensure_shared_symlink(args.src, args.out)
    cache_dir = args.out / "_acoustic_features_cache"

    # Pre-cache acoustics once
    for s in args.sections:
        print(f"\n--- acoustic features section {s} ---")
        get_acoustic_features(args.src / "_shared_wordlocked_features", cache_dir, s)

    summaries = []
    for pi, pid in enumerate(participants, 1):
        print(f"\n[{pi}/{len(participants)}] {pid}")
        for s in args.sections:
            try:
                info = process_section(
                    pid, s, args.src, args.out, cache_dir, force=args.force
                )
                if info is not None:
                    summaries.append(
                        {
                            "participant": pid,
                            "section": s,
                            "mean_var_explained": info.get(
                                "mean_var_explained",
                                info.get("mean_var_explained_by_acoustic"),
                            ),
                            "best_alpha": info.get("best_alpha"),
                        }
                    )
            except Exception as e:
                print(f"  ! {pid} sec{s} failed: {e}")
                import traceback

                traceback.print_exc()

    summ_path = args.out / "_residualization_summary.json"
    # merge with existing if array jobs
    existing = []
    if summ_path.is_file():
        try:
            existing = json.loads(summ_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    by_key = {(r["participant"], r["section"]): r for r in existing if isinstance(r, dict)}
    for r in summaries:
        by_key[(r["participant"], r["section"])] = r
    merged = [by_key[k] for k in sorted(by_key)]
    summ_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nDone. Residual EEG → {args.out}")
    print(f"Summary → {summ_path}  ({len(merged)} section rows)")


if __name__ == "__main__":
    main()
