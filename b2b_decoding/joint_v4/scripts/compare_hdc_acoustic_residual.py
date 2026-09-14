#!/usr/bin/env python3
"""Compare HDC syntax family decoding: raw EEG vs acoustic-residualized EEG.

Loads cohort-mean family traces from two result dirs and reports per-family
peak (0–0.8 s), group cluster duration, and % change after residualization.
Families whose cluster duration drops >50% are flagged as acoustically confounded.

Usage:
  .venv_gpt2/bin/python3 scripts/compare_hdc_acoustic_residual.py \\
      --raw-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \\
      --res-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures \\
      --groups cohort_groups.csv \\
      --out-dir group_comparison_b2b_hdc_offset_syntax_acoures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse cluster helper from sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_hdc_gwilliams_metrics import (  # noqa: E402
    cluster_duration_group,
    load_groups,
)

FAMILIES = (
    "phonetic",
    "word_form",
    "lexical_syntactic",
    "syntactic_operation",
    "syntactic_state",
    "semantic",
)
CONFIRM = (0.0, 0.8)
DROP_FRAC = 0.50


def load_family_mats(results: Path, participants: list[str], families: list[str]):
    mats = {f: [] for f in families}
    kept = []
    times = None
    missing = []
    for p in participants:
        path = results / p / f"{p}_b2b_family_agg.csv"
        if not path.is_file():
            missing.append(p)
            continue
        df = pd.read_csv(path)
        row_ok = True
        fam_vecs = {}
        for fam in families:
            sub = df.loc[df["family"] == fam].sort_values("time")
            if sub.empty:
                missing.append(f"{p}:{fam}")
                row_ok = False
                break
            if times is None:
                times = sub["time"].to_numpy(float)
            fam_vecs[fam] = sub["mean_score"].to_numpy(float)
        if not row_ok:
            continue
        for fam in families:
            mats[fam].append(fam_vecs[fam])
        kept.append(p)
    stacked = {
        f: np.vstack(v) if v else np.empty((0, 0)) for f, v in mats.items()
    }
    return stacked, times, missing, kept


def peak_in_window(mu: np.ndarray, times: np.ndarray, win=CONFIRM) -> tuple[float, float]:
    m = (times >= win[0]) & (times <= win[1])
    if not np.any(m):
        return float("nan"), float("nan")
    i = int(np.nanargmax(mu[m]))
    t_idx = np.where(m)[0][i]
    return float(mu[t_idx]), float(times[t_idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--res-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--cluster-alpha", type=float, default=0.05)
    ap.add_argument(
        "--min-subjects",
        type=int,
        default=5,
        help="Min subjects with family_agg in BOTH dirs to score a family",
    )
    ap.add_argument(
        "--participants",
        default="",
        help="Optional comma-separated participant override (else primary cohort)",
    )
    args = ap.parse_args()

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    groups = load_groups(Path(args.groups))
    if args.participants.strip():
        participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    else:
        participants = list(groups["participant"].astype(str))
    raw_dir = Path(args.raw_dir)
    res_dir = Path(args.res_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_mats, times, miss_raw, kept_raw = load_family_mats(raw_dir, participants, families)
    res_mats, times_r, miss_res, kept_res = load_family_mats(res_dir, participants, families)
    if times is None or times_r is None:
        raise SystemExit("No family_agg found in one or both dirs")
    if len(times) != len(times_r) or not np.allclose(times, times_r):
        raise SystemExit("Time grids differ between raw and residual dirs")

    common = [p for p in participants if p in set(kept_raw) and p in set(kept_res)]
    raw_idx = {p: i for i, p in enumerate(kept_raw)}
    res_idx = {p: i for i, p in enumerate(kept_res)}

    rows = []
    for fam in families:
        if len(common) < args.min_subjects:
            print(
                f"[warn] skipping {fam}: "
                f"n_common={len(common)} (need ≥{args.min_subjects})"
            )
            continue
        raw_stack = np.vstack([raw_mats[fam][raw_idx[p]] for p in common])
        res_stack = np.vstack([res_mats[fam][res_idx[p]] for p in common])
        mu_raw = raw_stack.mean(0)
        mu_res = res_stack.mean(0)
        pk_raw, t_raw = peak_in_window(mu_raw, times)
        pk_res, t_res = peak_in_window(mu_res, times)
        cl_raw = cluster_duration_group(raw_stack, times, alpha=args.cluster_alpha)
        cl_res = cluster_duration_group(res_stack, times, alpha=args.cluster_alpha)
        dur_raw = cl_raw["duration_cluster_total_s"]
        dur_res = cl_res["duration_cluster_total_s"]
        if dur_raw > 1e-9:
            pct = 100.0 * (dur_res - dur_raw) / dur_raw
        else:
            pct = float("nan") if dur_res == 0 else float("inf")
        confounded = bool(np.isfinite(pct) and pct < -100.0 * DROP_FRAC)
        rows.append(
            {
                "family": fam,
                "n_raw": int(raw_mats[fam].shape[0]),
                "n_res": int(res_mats[fam].shape[0]),
                "n_compared": int(len(common)),
                "peak_raw": pk_raw,
                "peak_time_raw_s": t_raw,
                "peak_res": pk_res,
                "peak_time_res_s": t_res,
                "peak_pct_change": (
                    100.0 * (pk_res - pk_raw) / abs(pk_raw) if abs(pk_raw) > 1e-12 else np.nan
                ),
                "duration_cluster_raw_s": dur_raw,
                "duration_cluster_res_s": dur_res,
                "duration_pct_change": pct,
                "acoustically_confounded": confounded,
                "n_clusters_raw": cl_raw["n_clusters"],
                "n_clusters_res": cl_res["n_clusters"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "acoustic_residual_comparison.csv", index=False)
    manifest = {
        "raw_dir": str(raw_dir),
        "res_dir": str(res_dir),
        "families": families,
        "confirm_win_s": list(CONFIRM),
        "drop_frac_threshold": DROP_FRAC,
        "cluster_alpha": args.cluster_alpha,
        "min_subjects": args.min_subjects,
        "participants": participants,
        "common_participants": common,
        "missing_raw": miss_raw[:50],
        "missing_res": miss_res[:50],
        "n_missing_raw": len(miss_raw),
        "n_missing_res": len(miss_res),
        "note": (
            "acoustically_confounded=True when residual cluster duration drops "
            f"> {int(DROP_FRAC * 100)}% relative to raw EEG"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("=== Raw vs acoustic-residual HDC ===")
    if df.empty:
        print("(no families met --min-subjects; see warnings above)")
        print(f"Wrote empty {out / 'acoustic_residual_comparison.csv'}")
        return
    print(df.to_string(index=False))
    flagged = df.loc[df["acoustically_confounded"], "family"].tolist()
    if flagged:
        print(f"\nFlagged as acoustically confounded: {flagged}")
    else:
        print("\nNo family dropped >50% cluster duration after residualization.")
    print(f"Wrote {out / 'acoustic_residual_comparison.csv'}")


if __name__ == "__main__":
    main()
