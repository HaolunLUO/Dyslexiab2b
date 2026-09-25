#!/usr/bin/env python3
"""Epoch as-stored EEG at content-word onsets and save single-trial N400 DVs.

No bandpass and no re-reference. Artifact rule is per-subject:
peak-to-peak of the baseline-corrected CP-ROI mean > median + 5 * MAD.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import n400_common as C


def _bad_overlaps(bad, start: int, end: int, n_times: int, sfreq: float) -> bool:
    if bad is None:
        return False
    if isinstance(bad, (str, int, float)):
        raise SystemExit(f"unrecognized bad_intervals entry: {bad!r}")
    if len(bad) == 0:
        return False
    t0 = start / sfreq
    t1 = end / sfreq
    for item in bad:
        if isinstance(item, dict):
            a = item.get("start", item.get("t0", item.get("onset")))
            b = item.get("end", item.get("t1", item.get("offset")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            a, b = item[0], item[1]
        else:
            raise SystemExit(f"unrecognized bad_intervals entry: {item!r}")
        a = float(a)
        b = float(b)
        # Sample indices are larger than the section duration in seconds.
        if max(abs(a), abs(b)) > (n_times / sfreq) * 2:
            a, b = a / sfreq, b / sfreq
        lo, hi = (a, b) if a <= b else (b, a)
        if t0 < hi and t1 > lo:
            return True
    return False


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def check_channels(subject: str) -> dict:
    meta_path = C.SHARED / subject / "section_001" / "metadata.json"
    eeg_path = C.SHARED / subject / "section_001" / "eeg_data.npy"
    meta = json.loads(meta_path.read_text())
    eeg = np.load(eeg_path, mmap_mode="r")
    if "channel_names" in meta:
        raise SystemExit(
            "metadata.json now has channel_names; confirm they match "
            "WORDLOCKED_CHAN_65 before using the hardcoded montage."
        )
    info = C.confirm_montage(int(eeg.shape[0]))
    info.update(
        {
            "subject_checked": subject,
            "eeg_shape": [int(eeg.shape[0]), int(eeg.shape[1])],
            "sfreq": meta.get("sfreq"),
            "substitution": None,
            "source": "joint_v4/eeg_band.jl WORDLOCKED_CHAN_65 (metadata has no channel_names)",
        }
    )
    C.STIM_DIR.mkdir(parents=True, exist_ok=True)
    (C.STIM_DIR / "channel_check.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("n_ch", "index", "eeg_shape", "sfreq", "substitution")}))
    return info


def epoch_subject(subject: str, force: bool = False) -> None:
    out_dir = C.OUT_ROOT / subject
    done = out_dir / "trials.csv"
    if done.is_file() and not force:
        print(f"{subject} exists, skip", flush=True)
        return
    word_path = C.STIM_DIR / "word_table.csv"
    if not word_path.is_file():
        raise SystemExit(f"word table missing (run build_word_table.py first): {word_path}")
    words = pd.read_csv(word_path)
    content = words.loc[words["content"]].copy()
    times = C.epoch_times()
    n_t = len(times)
    base_ix = np.where(times < -1e-8)[0]
    if base_ix.size == 0:
        raise SystemExit("empty baseline window")
    pre = int(round(-C.TMIN * C.SFREQ))
    if pre != int(base_ix.size):
        raise SystemExit(f"baseline samples {base_ix.size} != {pre}")

    # Montage check on this subject's EEG before any epoch is kept.
    eeg0 = np.load(C.SHARED / subject / "section_001" / "eeg_data.npy", mmap_mode="r")
    montage = C.confirm_montage(int(eeg0.shape[0]))
    cp_ix = [montage["index"][ch] for ch in C.CP_CHANNELS]
    fz_ix = montage["index"][C.FZ_CHANNEL]
    del eeg0

    records = []
    cp_waves = []
    fz_waves = []
    wave_keys = []

    for section in C.SECTIONS:
        sec = f"section_{section:03d}"
        sec_dir = C.SHARED / subject / sec
        meta = json.loads((sec_dir / "metadata.json").read_text())
        sfreq = float(meta["sfreq"])
        if abs(sfreq - C.SFREQ) > 1e-6:
            raise SystemExit(f"{subject} {sec} sfreq {sfreq}")
        eeg = np.load(sec_dir / "eeg_data.npy")
        if eeg.ndim != 2 or eeg.shape[0] != 65:
            raise SystemExit(f"{subject} {sec} eeg shape {eeg.shape}")
        mask = np.load(sec_dir / "train_keep_mask.npy").astype(bool).ravel()
        if mask.shape[0] != eeg.shape[1]:
            raise SystemExit(f"{subject} {sec} mask {mask.shape} vs eeg {eeg.shape}")
        bad = meta.get("bad_intervals") or []
        sub = content.loc[content["section"] == section]
        n_samples = eeg.shape[1]
        for rec in sub.itertuples(index=False):
            onset = int(rec.onset_sample)
            start = onset - pre
            end = start + n_t
            row = {
                "subject": subject,
                "section": section,
                "word_index": int(rec.word_index),
                "item_id": rec.item_id,
                "keep": 0,
                "reject_reason": "",
                "dv_cp": np.nan,
                "dv_fz": np.nan,
                "ptp_cp": np.nan,
            }
            if start < 0 or end > n_samples:
                row["reject_reason"] = "boundary"
                records.append(row)
                continue
            if not bool(mask[start:end].all()):
                row["reject_reason"] = "mask"
                records.append(row)
                continue
            if _bad_overlaps(bad, start, end, n_samples, sfreq):
                row["reject_reason"] = "bad_interval"
                records.append(row)
                continue
            ep = eeg[:, start:end].astype(np.float64)
            cp = ep[cp_ix].mean(axis=0)
            fz = ep[fz_ix]
            cp = cp - cp[base_ix].mean()
            fz = fz - fz[base_ix].mean()
            ptp = float(cp.max() - cp.min())
            row["ptp_cp"] = ptp
            row["reject_reason"] = "candidate"
            records.append(row)
            cp_waves.append(cp.astype(np.float32))
            fz_waves.append(fz.astype(np.float32))
            wave_keys.append((section, int(rec.word_index)))

    trials = pd.DataFrame.from_records(records).reset_index(drop=True)
    cand = trials["reject_reason"] == "candidate"
    ptp = trials.loc[cand, "ptp_cp"].to_numpy(float)
    if ptp.size == 0:
        raise SystemExit(f"{subject}: no candidate epochs")
    med = float(np.median(ptp))
    mad = _mad(ptp)
    if mad == 0:
        thr = float("inf")
        thr_note = "mad_zero_no_amplitude_reject"
    else:
        thr = med + 5.0 * mad
        thr_note = "median_plus_5mad"
    art = cand & (trials["ptp_cp"] > thr)
    trials.loc[art, "reject_reason"] = "artifact"
    trials.loc[cand & ~art, "reject_reason"] = "keep"
    trials.loc[cand & ~art, "keep"] = 1

    win = (times >= C.N400_WIN[0] - 1e-8) & (times <= C.N400_WIN[1] + 1e-8)
    row_of = {
        (int(r.section), int(r.word_index)): i
        for i, r in enumerate(trials.itertuples(index=False))
    }
    dv_cp = np.full(len(trials), np.nan)
    dv_fz = np.full(len(trials), np.nan)
    keep_cp = []
    keep_fz = []
    keep_sec = []
    keep_wix = []
    for (sec, wix), cp, fz in zip(wave_keys, cp_waves, fz_waves):
        rowi = row_of[(sec, wix)]
        if int(trials.at[rowi, "keep"]) != 1:
            continue
        dv_cp[rowi] = float(cp[win].mean())
        dv_fz[rowi] = float(fz[win].mean())
        keep_cp.append(cp)
        keep_fz.append(fz)
        keep_sec.append(sec)
        keep_wix.append(wix)
    trials["dv_cp"] = dv_cp
    trials["dv_fz"] = dv_fz

    out_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out_dir / "trials.csv", index=False)
    np.savez_compressed(
        out_dir / "timecourse_kept.npz",
        times=times.astype(np.float32),
        section=np.asarray(keep_sec, dtype=np.int16),
        word_index=np.asarray(keep_wix, dtype=np.int32),
        cp=np.vstack(keep_cp) if keep_cp else np.zeros((0, n_t), np.float32),
        fz=np.vstack(keep_fz) if keep_fz else np.zeros((0, n_t), np.float32),
    )
    summary = {
        "subject": subject,
        "n_content": int(len(trials)),
        "n_boundary": int((trials.reject_reason == "boundary").sum()),
        "n_mask": int((trials.reject_reason == "mask").sum()),
        "n_bad_interval": int((trials.reject_reason == "bad_interval").sum()),
        "n_artifact": int((trials.reject_reason == "artifact").sum()),
        "n_kept": int(trials.keep.sum()),
        "reject_rate_artifact_among_candidates": float(art.sum() / cand.sum()),
        "ptp_median": med,
        "ptp_mad": mad,
        "ptp_threshold": None if not np.isfinite(thr) else thr,
        "threshold_rule": thr_note,
        "cp_channels": list(C.CP_CHANNELS),
        "fz_channel": C.FZ_CHANNEL,
        "channel_index": montage["index"],
        "n_times": n_t,
        "tmin": C.TMIN,
        "tmax": C.TMAX,
        "baseline": list(C.BASELINE),
        "n400_window_s": list(C.N400_WIN),
        "n400_n_samples": int(win.sum()),
        "eeg_filter": "none (as stored)",
        "rereference": "none",
    }
    (out_dir / "reject_log.json").write_text(json.dumps(summary, indent=2))
    print(
        f"{subject} kept {summary['n_kept']}/{summary['n_content']} "
        f"artifact {summary['n_artifact']} thr {summary['ptp_threshold']}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.check_only:
        check_channels(args.subject or "RN102")
        return
    if not args.subject:
        raise SystemExit("--subject is required")
    epoch_subject(args.subject, force=args.force)


if __name__ == "__main__":
    main()
