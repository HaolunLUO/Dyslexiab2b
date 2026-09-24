#!/usr/bin/env python3
"""Join kept trials to the word table, z-score predictors, attach group and age."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import n400_common as C

Z_COLS = [
    "duration",
    "position",
    "surprisal",
    "log_frequency",
    "association",
    "sur_m1",
    "sur_p1",
    "freq_m1",
    "freq_p1",
    "assoc_m1",
    "assoc_p1",
]
Z_NAMES = {
    "duration": "duration_z",
    "position": "position_z",
    "surprisal": "sur_z",
    "log_frequency": "freq_z",
    "association": "assoc_z",
    "sur_m1": "sur_m1_z",
    "sur_p1": "sur_p1_z",
    "freq_m1": "freq_m1_z",
    "freq_p1": "freq_p1_z",
    "assoc_m1": "assoc_m1_z",
    "assoc_p1": "assoc_p1_z",
}


def main() -> None:
    subjects = C.load_subjects()
    missing = [s for s in subjects if not (C.OUT_ROOT / s / "trials.csv").is_file()]
    if missing:
        raise SystemExit(f"missing trials for {len(missing)} subjects, e.g. {missing[:5]}")
    words = pd.read_csv(C.STIM_DIR / "word_table.csv")
    pred_cols = ["section", "word_index", "item_id", "word", "analysis_complete", *Z_COLS]
    words = words[pred_cols]
    frames = []
    reject_rows = []
    for s in subjects:
        tr = pd.read_csv(C.OUT_ROOT / s / "trials.csv")
        tr["subject"] = s
        frames.append(tr)
        log = json.loads((C.OUT_ROOT / s / "reject_log.json").read_text())
        reject_rows.append(log)
    trials = pd.concat(frames, ignore_index=True)
    groups = C.load_groups().rename(columns={"participant": "subject"})
    ages = C.load_ages().rename(columns={"participant": "subject"})
    trials = trials.merge(words, on=["section", "word_index", "item_id"], how="left")
    trials = trials.merge(groups, on="subject", how="left")
    trials = trials.merge(ages, on="subject", how="left")
    if trials["group"].isna().any():
        raise SystemExit("group missing after merge")
    if trials["analysis_complete"].isna().any():
        raise SystemExit("word-table merge failed for some trials")

    scaler = {}
    base = words.loc[words["analysis_complete"]].copy()
    for col in Z_COLS:
        mu = float(base[col].mean())
        sd = float(base[col].std(ddof=0))
        if sd <= 0:
            raise SystemExit(f"zero sd for {col}")
        scaler[col] = {"mean": mu, "sd": sd}
        trials[Z_NAMES[col]] = (trials[col] - mu) / sd
    (C.STIM_DIR / "predictor_scaler.json").write_text(json.dumps(scaler, indent=2))

    kept = trials.loc[trials["keep"] == 1].copy()
    lmm = kept.loc[kept["analysis_complete"]].copy()
    # Stimulus-level surprisal tertiles on the complete-case items.
    items = base[["item_id", "surprisal"]].drop_duplicates()
    items["sur_tertile"] = pd.qcut(items["surprisal"], 3, labels=["low", "mid", "high"])
    lmm = lmm.merge(items[["item_id", "sur_tertile"]], on="item_id", how="left")

    reject = pd.DataFrame(reject_rows).merge(groups, on="subject", how="left")
    trials_path = C.STIM_DIR / "trials_all.csv"
    lmm_path = C.STIM_DIR / "lmm_input.csv"
    # trials_all is large; keep the columns needed for QC and slopes.
    keep_cols = [
        "subject", "group", "age", "section", "word_index", "item_id", "word",
        "keep", "reject_reason", "dv_cp", "dv_fz", "ptp_cp", "analysis_complete",
        *Z_COLS, *Z_NAMES.values(),
    ]
    trials[keep_cols].to_csv(trials_path, index=False)
    lmm_cols = keep_cols + ["sur_tertile"]
    lmm[lmm_cols].to_csv(lmm_path, index=False)
    reject.to_csv(C.STIM_DIR / "reject_summary.csv", index=False)

    age_ok = lmm.dropna(subset=["age"])
    summary = {
        "n_subjects_epoch": int(lmm["subject"].nunique()),
        "n_trials_kept": int(len(kept)),
        "n_trials_lmm": int(len(lmm)),
        "n_items_lmm": int(lmm["item_id"].nunique()),
        "n_subjects_with_age_in_lmm": int(age_ok["subject"].nunique()),
        "n_trials_age_complete": int(len(age_ok)),
        "subjects_missing_age": sorted(set(lmm.loc[lmm["age"].isna(), "subject"])),
        "group_n_subjects": lmm.groupby("group")["subject"].nunique().to_dict(),
        "group_n_trials": lmm.groupby("group").size().to_dict(),
        "reject_rate_by_group": reject.groupby("group")["reject_rate_artifact_among_candidates"].mean().to_dict(),
        "median_kept_by_group": reject.groupby("group")["n_kept"].median().to_dict(),
    }
    (C.STIM_DIR / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
