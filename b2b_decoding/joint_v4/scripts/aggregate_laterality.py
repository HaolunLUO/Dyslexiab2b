#!/usr/bin/env python3
"""Phase 3c laterality on delta-envelope null-subtracted scores.

LI = (R - L) / (R + L) using confirm-window family-trace nullsub.
C1 stays on all-channel delta envelope; this file is laterality only.

  python scripts/aggregate_laterality.py --frame onset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
HERE = Path(__file__).resolve().parents[1)


def _load_env(scores: Path) -> pd.DataFrame:
    df = pd.read_csv(scores)
    env = df[(df["family"] == "envelope") & (df["column"] == 0)].copy()
    return env[["participant", "nullsub", "observed", "null_median"]].rename(
        columns={
            "nullsub": "env_nullsub",
            "observed": "env_obs",
            "null_median": "env_null_med",
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="onset", choices=("onset", "offset"))
    ap.add_argument("--left-scores", type=Path)
    ap.add_argument("--right-scores", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cohort", type=Path, default=HERE / "cohort_groups.csv")
    args = ap.parse_args()
    tag = "onset_tmin03" if args.frame == "onset" else "woffset_tmin05"
    left = args.left_scores or HERE / "band_delta_left_temporal" / f"scores_{args.frame}.csv"
    right = args.right_scores or HERE / "band_delta_right_temporal" / f"scores_{args.frame}.csv"
    if not left.is_file():
        left = HERE / "band_delta" / f"scores_{args.frame}_left.csv"
    out = args.out or HERE / "laterality" / f"li_delta_{args.frame}.csv"

    if not left.is_file() or not right.is_file():
        raise SystemExit(
            f"Need left and right score tables.\n  left={left}\n  right={right}\n"
            "Run aggregate_existence_nulls.py into band_delta_left_temporal / "
            "band_delta_right_temporal first."
        )

    L = _load_env(left).rename(columns=lambda c: c if c == "participant" else f"L_{c}")
    R = _load_env(right).rename(columns=lambda c: c if c == "participant" else f"R_{c}")
    m = L.merge(R, on="participant", how="inner")
    num = m["R_env_nullsub"] - m["L_env_nullsub"]
    den = m["R_env_nullsub"] + m["L_env_nullsub"]
    m["LI"] = np.where(np.abs(den) > 1e-12, num / den, np.nan)

    cohort = pd.read_csv(args.cohort)
    keep = cohort[cohort["include_primary"].astype(str).isin(("1", "True", "true"))]
    m = m.merge(keep[["participant", "group"]], on="participant", how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(f"wrote {out}  n={len(m)}  LI finite={int(np.isfinite(m['LI']).sum())}")
    print(m.groupby("group")["LI"].describe())


if __name__ == "__main__":
    main()
