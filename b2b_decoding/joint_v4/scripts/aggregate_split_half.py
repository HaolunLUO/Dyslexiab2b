#!/usr/bin/env python3
"""Phase 2: ICC(2,1) + Spearman–Brown on split-half null-subtracted scores.

After 63/63: python scripts/aggregate_split_half.py --frame onset
Writes reliability/icc_splithalf.csv and Gate 2 labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
HERE = Path(__file__).resolve().parents[1]


def icc21(x: np.ndarray, y: np.ndarray) -> float:
    """Shrout & Fleiss ICC(2,1) for two ratings per subject."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        return float("nan")
    M = np.column_stack([x, y])
    grand = M.mean()
    k = 2
    row_means = M.mean(axis=1)
    col_means = M.mean(axis=0)
    ss_r = k * np.sum((row_means - grand) ** 2)
    ss_c = n * np.sum((col_means - grand) ** 2)
    ss_t = np.sum((M - grand) ** 2)
    ss_e = ss_t - ss_r - ss_c
    msr = ss_r / (n - 1)
    msc = ss_c / (k - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den else float("nan")


def spearman_brown(icc: float) -> float:
    if not np.isfinite(icc):
        return float("nan")
    return float(2 * icc / (1 + icc)) if icc != -1 else float("nan")


def gate_label(sb: float) -> str:
    if not np.isfinite(sb):
        return "unknown"
    if sb >= 0.5:
        return "id_and_classifier"
    if sb >= 0.3:
        return "group_mean_only"
    return "excluded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=("onset", "offset"))
    ap.add_argument("--split-dir", type=Path)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--cohort", type=Path, default=HERE / "cohort_groups.csv")
    ap.add_argument("--out", type=Path, default=HERE / "reliability" / "icc_splithalf.csv")
    args = ap.parse_args()
    defaults = {
        "onset": (
            POOL / "encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_splithalf",
            HERE / "existence" / "scores_onset.csv",
        ),
        "offset": (
            POOL / "encoding_results_b2b_mne_ica_v1_envpitch_tonev3_woffset_tmin05_splithalf",
            HERE / "existence" / "scores_offset.csv",
        ),
    }
    split_dir = args.split_dir or defaults[args.frame][0]
    scores_path = args.scores or defaults[args.frame][1]
    floors = pd.read_csv(scores_path)
    floors = floors[floors["column"] == 0][["participant", "family", "null_median"]]
    cohort = pd.read_csv(args.cohort)
    parts = [r["participant"] for _, r in cohort.iterrows()
             if str(r["include_primary"]) in ("1", "True", "true")]

    rows = []
    for p in parts:
        path = split_dir / p / f"{p}_splithalf_scores.csv"
        if not path.is_file():
            print("missing", path)
            continue
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        raise SystemExit("no split-half files")
    raw = pd.concat(rows, ignore_index=True)
    raw = raw.merge(floors, on=["participant", "family"], how="left")
    raw["nullsub"] = raw["confirm_mean"] - raw["null_median"]

    metrics = [
        ("nullsub", "confirm_nullsub"),
        ("peak_lat", "peak_latency"),
        ("fa50_lat", "fa50_latency"),
    ]
    out_rows = []
    for (fam, sub), g in raw.groupby(["family", "subfamily"]):
        wide = g.pivot_table(index="participant", columns="split",
                             values=["nullsub", "peak_lat", "fa50_lat"], aggfunc="mean")
        for src, name in metrics:
            try:
                x = wide[(src, "odd")].to_numpy()
                y = wide[(src, "even")].to_numpy()
            except KeyError:
                continue
            icc = icc21(x, y)
            sb = spearman_brown(icc)
            out_rows.append(dict(
                frame=args.frame, family=fam, subfamily=sub, metric=name,
                n=int(np.isfinite(x).sum()),
                icc21=icc, spearman_brown=sb, gate=gate_label(sb),
            ))
    out = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        prev = pd.read_csv(args.out)
        prev = prev[prev["frame"] != args.frame]
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(args.out, index=False)
    show = out[out["frame"] == args.frame]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
