#!/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
"""Recompute 350–550 ms means at Fz, Cz, Pz, Oz from as-stored EEG.

Reuses the V1 keep mask. Reapplies the V1 artifact rule and records any
disagreement. Does not bandpass and does not fit a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
SHARED = POOL / "extracted_sections_wordlocked_shared"
V1 = POOL / "n400_storytime_v1"
WORD = V1 / "v1_strict" / "stimulus" / "word_table.csv"
OUT = V1 / "v1_strict" / "trials"
SFREQ = 100.0
TMIN = -0.2
TMAX = 1.0
N400 = (0.35, 0.55)
CP = ("Pz", "Cz", "Oz")
CHANNELS = (
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT7", "FT8", "FT9", "FT10",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "TP9", "TP10",
    "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
)
IDX = {name: i for i, name in enumerate(CHANNELS)}
PREDS = ("freq", "assoc", "cloze")
TERT_COL = {"freq": "tert_freq", "assoc": "tert_assoc", "cloze": "tert_cloze"}


def epoch_times() -> np.ndarray:
    n = int(round((TMAX - TMIN) * SFREQ)) + 1
    return np.round(TMIN + np.arange(n) / SFREQ, 5)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _bad_overlaps(bad, start: int, end: int, n_times: int, sfreq: float) -> bool:
    if not bad:
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
        if max(abs(a), abs(b)) > (n_times / sfreq) * 2:
            a, b = a / sfreq, b / sfreq
        lo, hi = (a, b) if a <= b else (b, a)
        if t0 < hi and t1 > lo:
            return True
    return False


def extract_subject(subject: str) -> None:
    out_dir = OUT / subject
    done = out_dir / "dv.csv"
    fig_path = out_dir / "fig.npz"
    if done.is_file() and fig_path.is_file():
        print(f"{subject} exists, skip", flush=True)
        return
    word = pd.read_csv(WORD)
    content = word.loc[word["content"].astype(bool)].copy()
    old = pd.read_csv(V1 / subject / "trials.csv")
    old["section"] = old["section"].astype(int)
    old["word_index"] = old["word_index"].astype(int)
    old_keep = {
        (int(r.section), int(r.word_index)): int(r.keep)
        for r in old.itertuples(index=False)
    }
    old_dv = {
        (int(r.section), int(r.word_index)): (float(r.dv_cp), float(r.dv_fz))
        for r in old.itertuples(index=False)
        if int(r.keep) == 1
    }
    groups = pd.read_csv(
        POOL / "dyslexia_natualistics_listing" / "b2b_decoding" / "joint_v4" / "cohort_groups.csv"
    )
    groups = groups.loc[groups["include_primary"].astype(int) == 1]
    grow = groups.loc[groups["participant"].astype(str) == subject]
    if len(grow) != 1:
        raise SystemExit(f"{subject} group rows {len(grow)}")
    group = str(grow["group"].iloc[0])

    times = epoch_times()
    n_t = len(times)
    base_ix = np.where(times < -1e-8)[0]
    win = (times >= N400[0] - 1e-8) & (times <= N400[1] + 1e-8)
    if int(win.sum()) != 21 or base_ix.size != 20:
        raise SystemExit(f"window/baseline samples {int(win.sum())} {base_ix.size}")
    pre = int(round(-TMIN * SFREQ))
    cp_ix = [IDX[ch] for ch in CP]
    fz_i, cz_i, pz_i, oz_i = IDX["Fz"], IDX["Cz"], IDX["Pz"], IDX["Oz"]

    # tertile lookup
    tert = {}
    for rec in word.itertuples(index=False):
        if not bool(rec.in_model):
            continue
        tert[(int(rec.section), int(rec.word_index))] = (
            int(rec.tert_freq), int(rec.tert_assoc), int(rec.tert_cloze)
        )

    records = []
    ptp_rows = []
    fz_sum = np.zeros((3, 3, n_t), np.float64)
    pz_sum = np.zeros((3, 3, n_t), np.float64)
    wave_n = np.zeros((3, 3), np.int32)
    topo_low = np.zeros((3, 65), np.float64)
    topo_high = np.zeros((3, 65), np.float64)
    topo_n = np.zeros((3, 2), np.int32)

    for section in (1, 2):
        sec = f"section_{section:03d}"
        sec_dir = SHARED / subject / sec
        meta = json.loads((sec_dir / "metadata.json").read_text())
        if abs(float(meta["sfreq"]) - SFREQ) > 1e-6:
            raise SystemExit(f"{subject} {sec} sfreq")
        if "channel_names" in meta:
            raise SystemExit(f"{subject} metadata now has channel_names")
        eeg = np.load(sec_dir / "eeg_data.npy")
        if eeg.shape[0] != 65:
            raise SystemExit(f"{subject} {sec} n_ch {eeg.shape[0]}")
        mask = np.load(sec_dir / "train_keep_mask.npy").astype(bool).ravel()
        if mask.shape[0] != eeg.shape[1]:
            raise SystemExit(f"{subject} {sec} mask length")
        bad = meta.get("bad_intervals") or []
        sub = content.loc[content["section"] == section]
        n_samples = eeg.shape[1]
        for rec in sub.itertuples(index=False):
            key = (section, int(rec.word_index))
            if key not in old_keep:
                raise SystemExit(f"{subject} missing V1 trial {key}")
            onset = int(rec.onset_sample)
            start = onset - pre
            end = start + n_t
            reason = "candidate"
            if start < 0 or end > n_samples:
                reason = "boundary"
            elif not bool(mask[start:end].all()):
                reason = "mask"
            elif _bad_overlaps(bad, start, end, n_samples, SFREQ):
                reason = "bad_interval"
            if reason != "candidate":
                records.append((key, reason, np.nan))
                continue
            ep = eeg[:, start:end].astype(np.float64)
            cp = ep[cp_ix].mean(axis=0)
            cp = cp - cp[base_ix].mean()
            ptp = float(cp.max() - cp.min())
            records.append((key, "candidate", ptp))
            ptp_rows.append((key, ptp, ep))

    cand_ptp = np.array([p for _k, reason, p in records if reason == "candidate"], float)
    if cand_ptp.size == 0:
        raise SystemExit(f"{subject}: no candidate epochs")
    med = float(np.median(cand_ptp))
    mad = _mad(cand_ptp)
    if mad == 0:
        thr = float("inf")
        rule = "mad_zero_no_amplitude_reject"
    else:
        thr = med + 5.0 * mad
        rule = "median_plus_5mad"
    recomputed_keep = {}
    for key, reason, ptp in records:
        if reason != "candidate":
            recomputed_keep[key] = 0
        elif ptp > thr:
            recomputed_keep[key] = 0
        else:
            recomputed_keep[key] = 1
    n_mismatch = sum(recomputed_keep[k] != old_keep[k] for k in old_keep)
    # Stored V1 keep is the analysis mask. The recomputed rule is a check.
    use_keep = old_keep

    kept_ep = {key: ep for key, _ptp, ep in ptp_rows if use_keep.get(key, 0) == 1}
    rows = []
    max_cp_err = 0.0
    max_fz_err = 0.0
    n_compared = 0
    for key, ep in kept_ep.items():
        base = ep[:, base_ix].mean(axis=1, keepdims=True)
        ep_b = ep - base
        dv = {
            "dv_fz": float(ep_b[fz_i, win].mean()),
            "dv_cz": float(ep_b[cz_i, win].mean()),
            "dv_pz": float(ep_b[pz_i, win].mean()),
            "dv_oz": float(ep_b[oz_i, win].mean()),
        }
        dv["dv_cp"] = float(np.mean([dv["dv_pz"], dv["dv_cz"], dv["dv_oz"]]))
        if key in old_dv:
            old_cp, old_fz = old_dv[key]
            if np.isfinite(old_cp):
                max_cp_err = max(max_cp_err, abs(dv["dv_cp"] - old_cp))
                max_fz_err = max(max_fz_err, abs(dv["dv_fz"] - old_fz))
                n_compared += 1
        rows.append({"section": key[0], "word_index": key[1], **dv})
        if key not in tert:
            continue
        tf, ta, tc = tert[key]
        bins = (tf, ta, tc)
        fz_wave = ep_b[fz_i]
        pz_wave = ep_b[pz_i]
        topo = ep_b[:, win].mean(axis=1)
        for p_i, b in enumerate(bins):
            fz_sum[p_i, b] += fz_wave
            pz_sum[p_i, b] += pz_wave
            wave_n[p_i, b] += 1
            if b == 0:
                topo_low[p_i] += topo
                topo_n[p_i, 0] += 1
            elif b == 2:
                topo_high[p_i] += topo
                topo_n[p_i, 1] += 1

    n_v1_keep = int(sum(use_keep.values()))
    if len(rows) != n_v1_keep:
        raise SystemExit(f"{subject}: wrote {len(rows)} DVs but V1 kept {n_v1_keep}")
    if n_compared == 0:
        raise SystemExit(f"{subject}: no DV comparison against V1")
    if max_cp_err > 1e-4 or max_fz_err > 1e-4:
        raise SystemExit(
            f"{subject}: DV mismatch vs V1 cp {max_cp_err} fz {max_fz_err}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(done, index=False)
    np.savez_compressed(
        fig_path,
        times=times.astype(np.float64),
        fz_sum=fz_sum,
        pz_sum=pz_sum,
        wave_n=wave_n,
        topo_low_sum=topo_low,
        topo_high_sum=topo_high,
        topo_n=topo_n,
        group=np.array(group),
        preds=np.array(PREDS),
    )
    qc = {
        "subject": subject,
        "group": group,
        "n_kept_v1": int(sum(use_keep.values())),
        "n_dv_rows": len(rows),
        "n_keep_mismatch_vs_recomputed_rule": int(n_mismatch),
        "artifact_rule": rule,
        "ptp_median": med,
        "ptp_mad": mad,
        "ptp_threshold": None if not np.isfinite(thr) else thr,
        "max_abs_err_dv_cp": max_cp_err,
        "max_abs_err_dv_fz": max_fz_err,
        "n_compared": n_compared,
        "eeg_filter": "none (as stored)",
    }
    (out_dir / "extract_qc.json").write_text(json.dumps(qc, indent=2))
    print(
        f"{subject} kept {qc['n_kept_v1']} mismatch {n_mismatch} "
        f"cp_err {max_cp_err:.3g} fz_err {max_fz_err:.3g}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+")
    args = ap.parse_args()
    if not WORD.is_file():
        raise SystemExit(f"run build_word_table.py first: {WORD}")
    for subject in args.subjects:
        extract_subject(subject)


if __name__ == "__main__":
    main()
