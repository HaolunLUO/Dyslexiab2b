#!/usr/bin/env python3
"""Stop-gate analysis for the 3-subject HDC pilot.

Looks at family mean traces in 0–0.8 s and reports whether any family
has a contiguous above-chance window. Also prints κ / ridge-boundary
diagnostics from per-subject QC.

Usage:
  python3 scripts/analyze_hdc_pilot.py \\
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
)
CONFIRM = (0.0, 0.8)
DEFAULT_SUBJECTS = ("RN109", "D007d", "D011d")
KAPPA_WARN = 32.0
KAPPA_INVALID = 85.0
MIN_VALID = 16
MIN_RUN = 5  # contiguous samples (~50 ms at 100 Hz)


def _load_agg(results: Path, p: str) -> pd.DataFrame:
    path = results / p / f"{p}_b2b_family_agg.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return pd.read_csv(path)


def _qc(results: Path, p: str) -> dict:
    path = results / p / f"{p}_qc_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--subjects", nargs="*", default=list(DEFAULT_SUBJECTS))
    args = ap.parse_args()
    results = Path(args.results_dir)

    frames = []
    qc_rows = []
    for p in args.subjects:
        df = _load_agg(results, p)
        df["participant"] = p
        frames.append(df)
        q = _qc(results, p)
        qc_rows.append(
            {
                "participant": p,
                "n_valid_partitions": q.get("n_valid_partitions"),
                "max_kappa": q.get("max_kappa"),
                "any_lambda_boundary": q.get("any_lambda_boundary"),
                "any_invalid": q.get("any_invalid"),
                "n_usable": q.get("n_usable"),
                "keep_frac": q.get("keep_frac"),
            }
        )

    qc_df = pd.DataFrame(qc_rows)
    print("=== QC / estimator gates ===")
    print(qc_df.to_string(index=False))
    kappa_ok = True
    for _, r in qc_df.iterrows():
        nv = r["n_valid_partitions"]
        mk = r["max_kappa"]
        if nv is not None and nv < MIN_VALID:
            print(f"FAIL {r['participant']}: valid partitions {nv} < {MIN_VALID}")
            kappa_ok = False
        if mk is not None and mk > KAPPA_INVALID:
            print(f"FAIL {r['participant']}: max κ {mk:.1f} > {KAPPA_INVALID}")
            kappa_ok = False
        elif mk is not None and mk > KAPPA_WARN:
            print(f"WARN {r['participant']}: max κ {mk:.1f} > {KAPPA_WARN}")
        if r["any_lambda_boundary"]:
            print(f"WARN {r['participant']}: ridge α on grid boundary")
        if r["any_invalid"]:
            print(f"FAIL {r['participant']}: invalid H rank")
            kappa_ok = False

    all_df = pd.concat(frames, ignore_index=True)
    print("\n=== Family decoding (confirm window 0–0.8 s) ===")
    decodable = []
    for fam in FAMILIES:
        sub = all_df.loc[all_df["family"] == fam]
        times = np.sort(sub["time"].unique())
        mats = []
        for p in args.subjects:
            sp = sub.loc[sub["participant"] == p].sort_values("time")
            mats.append(sp["mean_score"].to_numpy())
        M = np.vstack(mats)
        mu = M.mean(axis=0)
        se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0]) if M.shape[0] > 1 else np.zeros_like(mu)
        win = (times >= CONFIRM[0]) & (times <= CONFIRM[1])
        mu_w, se_w, t_w = mu[win], se[win], times[win]
        # one-sample vs 0 using 3-subject SE; also contiguous mu>0 run
        tstat = np.divide(mu_w, se_w, out=np.zeros_like(mu_w), where=se_w > 1e-12)
        above = (mu_w > 0) & (tstat > 1.0)
        run = longest_run(above)
        peak = float(mu_w.max()) if mu_w.size else float("nan")
        t_peak = float(t_w[int(np.argmax(mu_w))]) if mu_w.size else float("nan")
        mean_win = float(mu_w.mean()) if mu_w.size else float("nan")
        ok = run >= MIN_RUN and mean_win > 0
        decodable.append(ok)
        flag = "YES" if ok else "no "
        print(
            f"  {flag}  {fam:20s}  mean={mean_win:+.4f}  peak={peak:+.4f} @ {t_peak:.2f}s  "
            f"contiguous(mu>0 & t>1)={run} samples"
        )

    any_ok = any(decodable)
    print("\n=== Stop gate ===")
    if kappa_ok and any_ok:
        print("PASS: estimator gates OK and at least one family is decodable.")
        print("Proceed to temporal generalization (B2B_DO_TG=1) and, if TG is stable, the full cohort.")
    elif not kappa_ok:
        print("STOP: estimator gates failed (κ / rank / partitions). Do not launch TG.")
    else:
        print("STOP: no family has a contiguous above-chance window in 0–0.8 s.")
        print("HDC is not identifiable at this EEG SNR on the 3-subject pilot.")

    out = results / "hdc_pilot_stopgate.json"
    out.write_text(
        json.dumps(
            {
                "kappa_ok": bool(kappa_ok),
                "any_family_decodable": bool(any_ok),
                "families": list(FAMILIES),
                "decodable": {f: bool(v) for f, v in zip(FAMILIES, decodable)},
                "qc": qc_df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
