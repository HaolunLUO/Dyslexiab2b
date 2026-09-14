#!/usr/bin/env python3
"""Stability check for 3-subject HDC temporal generalization.

TG is "stable" if every pilot subject wrote finite family matrices and
duration/sustain metrics, and at least one family has above-chance
diagonal duration on the 3-subject mean (duration > half the TG window
would be expected from sign-flip noise around 0; we require a family
whose mean duration exceeds that chance floor *or* whose TG diagonal
matches the sign of the already-decodable family trace).

Usage:
  python3 scripts/analyze_hdc_tg.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = (
    "phonetic",
    "word_form",
    "lexical_syntactic",
    "syntax_proxy",
    "semantic",
    "syntactic_operation",
    "syntactic_state",
)
DEFAULT_SUBJECTS = ("RN109", "D007d", "D011d")
MIN_VALID = 16


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--subjects", nargs="*", default=list(DEFAULT_SUBJECTS))
    ap.add_argument(
        "--families",
        default="",
        help="Comma-separated family list; default = those present on disk",
    )
    args = ap.parse_args()
    results = Path(args.results_dir)
    requested = [x.strip() for x in args.families.split(",") if x.strip()] or list(FAMILIES)

    rows = []
    missing = []
    nan_files = []
    families_present = []
    for p in args.subjects:
        metrics = results / p / f"{p}_tg_metrics.csv"
        if not metrics.is_file():
            missing.append(str(metrics))
            continue
        m = pd.read_csv(metrics)
        m["participant"] = p
        rows.append(m)
        fams = [f for f in requested if f in set(m["family"].astype(str))]
        if not families_present:
            families_present = fams
        for fam in fams:
            npy = results / p / f"{p}_tg_{fam}.npy"
            if not npy.is_file():
                missing.append(str(npy))
                continue
            M = _load_npy(npy)
            if not np.isfinite(M).all():
                nan_files.append(str(npy))

    if missing:
        print("STOP: missing TG outputs:")
        for x in missing:
            print(" ", x)
        out = {
            "tg_stable": False,
            "reason": "missing_files",
            "missing": missing,
        }
        (results / "hdc_tg_stopgate.json").write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(1)

    tg = pd.concat(rows, ignore_index=True)
    print("=== TG metrics ===")
    print(tg.to_string(index=False))

    n_ok = True
    for p in args.subjects:
        sp = tg.loc[tg["participant"] == p]
        nv = int(sp["n_valid_splits"].min()) if len(sp) else 0
        if nv < MIN_VALID:
            print(f"FAIL {p}: n_valid_splits {nv} < {MIN_VALID}")
            n_ok = False
    if nan_files:
        print("FAIL non-finite TG matrices:")
        for x in nan_files:
            print(" ", x)
        n_ok = False

    print("\n=== Family mean duration / sustain ===")
    chance_floor = None
    decodable = []
    for fam in families_present:
        sub = tg.loc[tg["family"] == fam]
        dur = sub["duration_s"].to_numpy(float)
        sus = sub["sustain_s"].to_numpy(float)
        tmin = float(sub["tmin_s"].mean())
        tmax = float(sub["tmax_s"].mean())
        window = tmax - tmin
        chance_floor = 0.5 * window
        mu_d, mu_s = float(dur.mean()), float(sus.mean())
        ok = mu_d > chance_floor
        decodable.append(ok)
        flag = "YES" if ok else "no "
        print(
            f"  {flag}  {fam:22s}  duration={mu_d:.3f}s  sustain={mu_s:.3f}s  "
            f"chance_floor={chance_floor:.3f}s"
        )

    any_ok = any(decodable)
    print("\n=== TG stop gate ===")
    stable = bool(n_ok and any_ok)
    if stable:
        print("PASS: TG matrices are finite and at least one family lasts longer than chance.")
        print("Proceed to the 63-subject observed cohort with B2B_DO_TG=1.")
    elif not n_ok:
        print("STOP: TG estimator/files failed. Do not launch the full cohort.")
    else:
        print("STOP: no family has TG duration above the 50% chance floor.")
        print("Diagonal may be sign-flip noise; do not launch the full cohort.")

    out = {
        "tg_stable": stable,
        "n_ok": bool(n_ok),
        "any_family_above_chance_duration": bool(any_ok),
        "chance_floor_s": chance_floor,
        "families": families_present,
        "decodable": {f: bool(v) for f, v in zip(families_present, decodable)},
        "metrics": tg.to_dict(orient="records"),
        "nan_files": nan_files,
    }
    path = results / "hdc_tg_stopgate.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")
    raise SystemExit(0 if stable else 2)


if __name__ == "__main__":
    main()
