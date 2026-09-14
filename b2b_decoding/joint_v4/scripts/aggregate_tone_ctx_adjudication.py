#!/usr/bin/env python3
"""Adjudication scores after existence + split-half finish.

Group T = mean_i confirm unique mean_trace.
T*_b = mean_i shift-b unique mean_trace (independent k per subject).
p = (1 + #{T* >= T}) / (B + 1), one-sided positive.
Holm on M2 and M3 only. M1 is the positive-control null, not in Holm.
SB = 2r/(1+r) on odd/even 60 s confirm scores. Gate SB>=0.30.
No TD/DD or behavior here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
POOL = Path("/orcd/pool/005/haolun52")
CONFIRM = (0.0, 0.8)
HOLM = ("evidence", "resid")
ARM_FAM = {"M1": "pitch", "M2": "evidence", "M3": "resid"}


def confirm_obs(obs_dir: Path, p: str, fam: str) -> float:
    agg = pd.read_csv(obs_dir / p / f"{p}_b2b_family_agg.csv")
    sub = agg[(agg["family"] == fam) & (agg["time"] >= CONFIRM[0]) & (agg["time"] <= CONFIRM[1])]
    return float(sub["mean_trace"].mean())


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def spearman_brown(r: float) -> float:
    if not np.isfinite(r) or r <= -1:
        return float("nan")
    return float(2 * r / (1 + r))


def pearson(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, default=HERE / "cohort_groups.csv")
    ap.add_argument("--out", type=Path, default=HERE / "tone_ctx_v1" / "adjudication")
    args = ap.parse_args()
    cohort = pd.read_csv(args.cohort)
    parts = list(cohort.loc[cohort["include_primary"].astype(str).isin(("1", "True", "true")), "participant"])
    args.out.mkdir(parents=True, exist_ok=True)
    exist_p = {}
    sb = {}
    rows = []
    for arm, fam in ARM_FAM.items():
        obs_dir = POOL / f"encoding_results_b2b_tone_ctx_v1_{arm}_nucleus_offset"
        exist_dir = POOL / f"encoding_results_b2b_tone_ctx_v1_{arm}_nucleus_offset_existence_nulls"
        split_dir = POOL / f"encoding_results_b2b_tone_ctx_v1_{arm}_nucleus_offset_splithalf"
        obs = []
        nulls = []
        odd, even = [], []
        floors = []
        for p in parts:
            cpath = exist_dir / p / f"{p}_existence_confirm.csv"
            if not cpath.is_file():
                continue
            conf = pd.read_csv(cpath)
            fam_null = conf[(conf["family"] == fam) & (conf["column"] == 0)]["confirm_mean"].to_numpy(float)
            t = confirm_obs(obs_dir, p, fam)
            obs.append(t)
            nulls.append(fam_null)
            floors.append(float(np.nanmedian(fam_null)))
            sh = split_dir / p / f"{p}_splithalf_scores.csv"
            if sh.is_file():
                sdf = pd.read_csv(sh)
                sdf = sdf[(sdf["family"] == fam) & (sdf["subfamily"] == "all")]
                o = sdf.loc[sdf["split"] == "odd", "confirm_mean"]
                e = sdf.loc[sdf["split"] == "even", "confirm_mean"]
                if len(o) and len(e):
                    odd.append(float(o.iloc[0]))
                    even.append(float(e.iloc[0]))
        if not obs:
            print(f"{arm} {fam}: missing existence outputs")
            continue
        B = min(len(x) for x in nulls)
        T = float(np.mean(obs))
        Tstar = np.array([np.mean([n[b] for n in nulls]) for b in range(B)])
        p_mc = float((1.0 + np.sum(Tstar >= T)) / (B + 1))
        exist_p[fam] = p_mc
        r = pearson(odd, even) if odd else float("nan")
        sb[fam] = {"r": r, "SB": spearman_brown(r), "n_split": len(odd)}
        for p, t, fl in zip(parts[:len(obs)], obs, floors):
            rows.append(dict(arm=arm, family=fam, participant=p,
                             observed=t, null_median=fl, nullsub=t - fl))
        print(f"{arm} {fam}: T={T:.6g}  B={B}  p_one_sided={p_mc:.4g}  "
              f"SB={sb[fam]['SB'] if np.isfinite(sb[fam]['SB']) else 'NA'}")
    adj = holm({k: exist_p[k] for k in HOLM if k in exist_p})
    gate = []
    for fam, p0 in exist_p.items():
        p_h = adj.get(fam, p0 if fam not in HOLM else float("nan"))
        s = sb.get(fam, {})
        sbv = s.get("SB", float("nan"))
        exists = bool(fam in HOLM and p_h < 0.05) if fam in HOLM else bool(p0 < 0.05)
        reliable = bool(np.isfinite(sbv) and sbv >= 0.30)
        action = (
            "group_model_later" if exists and reliable else
            "group_existence_only" if exists and not reliable else
            "stop_inferential" if fam in HOLM else
            "m1_control_" + ("pass" if exists else "FAIL_AUDIT")
        )
        gate.append(dict(
            family=fam, p_one_sided=p0, p_holm=p_h if fam in HOLM else None,
            exists=exists, SB=sbv, r_odd_even=s.get("r"),
            reliable=reliable, action=action,
        ))
    pd.DataFrame(rows).to_csv(args.out / "subject_scores.csv", index=False)
    pd.DataFrame(gate).to_csv(args.out / "gate.csv", index=False)
    (args.out / "gate.json").write_text(json.dumps({
        "exist_p": exist_p, "holm": adj, "reliability": sb, "gate": gate,
        "note": "No TD/DD. Holm only on evidence and resid.",
    }, indent=2) + "\n")
    print(pd.DataFrame(gate).to_string(index=False))


if __name__ == "__main__":
    main()
