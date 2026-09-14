#!/usr/bin/env python3
"""Aggregate Phase 1 existence nulls after 63/63 subjects finish.

Writes (under joint_v4/existence/ by default):
  nulls_{frame}.parquet   subject × family × column × shift (+ confirm_mean)
  scores_{frame}.csv      observed, null median/SD/p95, nullsub, z
  floors_by_group.csv     per-subject floors + TD vs DD Mann–Whitney
  gate1_{frame}.csv       sign-flip existence table (family × frame)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FAMILIES = ("envelope", "pitch", "tone", "offset", "frequency", "surprisal")
CONFIRM = (0.0, 0.8)
POOL = Path("/orcd/pool/005/haolun52")
HERE = Path(__file__).resolve().parents[1]


def _load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["include_primary"].astype(str).isin(("1", "True", "true"))].copy()
    df["dd"] = df["group"].ne("TD")
    return df


def _confirm_mean_obs(agg: pd.DataFrame, family: str, t0=CONFIRM[0], t1=CONFIRM[1]) -> float:
    sub = agg[(agg["family"] == family) & (agg["time"] >= t0) & (agg["time"] <= t1)]
    return float(sub["mean_trace"].mean())


def _sign_flip_p(x: np.ndarray, n_perm: int = 10_000, seed: int = 42) -> float:
    """Two-sided sign-flip test of mean(x) != 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null = np.abs((signs * x).mean(axis=1))
    return float((np.count_nonzero(null >= obs) + 1) / (n_perm + 1))


def _ica_n_removed(extractor: Path, participant: str) -> float:
    meta = extractor / participant / "section_001" / "metadata.json"
    if not meta.is_file():
        return float("nan")
    obj = json.loads(meta.read_text())
    excl = obj.get("ica_exclude") or []
    return float(len(excl))


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        csv_path = path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False)
        print(f"parquet unavailable; wrote {csv_path}")


def aggregate_frame(frame: str, exist_dir: Path, obs_dir: Path, cohort: pd.DataFrame,
                    extractor: Path, out_dir: Path, n_perm: int = 10_000) -> None:
    parts = list(cohort["participant"])
    score_rows = []
    null_rows = []
    floor_rows = []
    fam_nullsub = {f: [] for f in FAMILIES}

    for p in parts:
        conf_path = exist_dir / p / f"{p}_existence_confirm.csv"
        if not conf_path.is_file():
            print(f"missing {conf_path}")
            continue
        conf = pd.read_csv(conf_path)
        agg = pd.read_csv(obs_dir / p / f"{p}_b2b_family_agg.csv")
        qc_path = obs_dir / p / f"{p}_qc_summary.json"
        qc = json.loads(qc_path.read_text()) if qc_path.is_file() else {}
        n_usable = qc.get("n_usable", float("nan"))
        n_ica = _ica_n_removed(extractor, p)
        group = cohort.loc[cohort["participant"] == p, "group"].iloc[0]

        for fam in FAMILIES:
            fam_null = conf[(conf["family"] == fam) & (conf["column"] == 0)]["confirm_mean"].to_numpy()
            obs = _confirm_mean_obs(agg, fam)
            med = float(np.nanmedian(fam_null))
            mu = float(np.nanmean(fam_null))
            sd = float(np.nanstd(fam_null, ddof=1)) if np.isfinite(fam_null).sum() > 1 else float("nan")
            p95 = float(np.nanpercentile(fam_null, 95))
            nullsub = obs - med
            z = (obs - mu) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
            fam_nullsub[fam].append(nullsub)
            score_rows.append(dict(
                participant=p, frame=frame, family=fam, column=0, feat_name=f"{fam}_trace",
                observed=obs, null_median=med, null_mean=mu, null_sd=sd,
                null_p95=p95, nullsub=nullsub, z=z,
            ))
            floor_rows.append(dict(
                participant=p, group=group, frame=frame, family=fam,
                null_median=med, null_sd=sd, n_usable=n_usable,
                n_ica_removed=n_ica,
            ))
            for _, row in conf[conf["family"] == fam].iterrows():
                null_rows.append(dict(
                    participant=p, frame=frame, family=fam,
                    column=int(row["column"]), feat_name=row["feat_name"],
                    shift=int(row["shift"]), confirm_mean=float(row["confirm_mean"]),
                ))

            cols = conf[(conf["family"] == fam) & (conf["column"] != 0)]
            for feat_name, g in cols.groupby("feat_name"):
                nv = g["confirm_mean"].to_numpy()
                score_rows.append(dict(
                    participant=p, frame=frame, family=fam,
                    column=int(g["column"].iloc[0]), feat_name=feat_name,
                    observed=float("nan"),
                    null_median=float(np.nanmedian(nv)),
                    null_mean=float(np.nanmean(nv)),
                    null_sd=float(np.nanstd(nv, ddof=1)) if np.isfinite(nv).sum() > 1 else float("nan"),
                    null_p95=float(np.nanpercentile(nv, 95)),
                    nullsub=float("nan"), z=float("nan"),
                ))

        pc_path = obs_dir / p / f"{p}_b2b_pc_splits.csv"
        if pc_path.is_file():
            pc = pd.read_csv(pc_path, usecols=["direction", "time", "family", "PC", "coefficient"])
            pc = pc[(pc["direction"] == 0) & (pc["time"] >= CONFIRM[0]) & (pc["time"] <= CONFIRM[1])]
            obs_by = pc.groupby(["family", "PC"], sort=False)["coefficient"].mean()
            lookup = {(r["family"], r["column"]): r for r in score_rows
                      if r["participant"] == p and r["column"] != 0}
            for (fam, col), obs_c in obs_by.items():
                r = lookup.get((fam, int(col)))
                if r is None:
                    continue
                r["observed"] = float(obs_c)
                if math.isfinite(r["null_median"]):
                    r["nullsub"] = float(obs_c) - r["null_median"]
                if r["null_sd"] and math.isfinite(r["null_sd"]) and r["null_sd"] > 0:
                    r["z"] = (float(obs_c) - r["null_mean"]) / r["null_sd"]

    scores = pd.DataFrame(score_rows)
    floors = pd.DataFrame(floor_rows)
    nulls = pd.DataFrame(null_rows)

    gate_rows = []
    for fam in FAMILIES:
        xs = np.asarray(fam_nullsub[fam], dtype=float)
        p_exist = _sign_flip_p(xs, n_perm=n_perm)
        gate_rows.append(dict(
            frame=frame, family=fam, n=int(np.isfinite(xs).sum()),
            mean_nullsub=float(np.nanmean(xs)) if xs.size else float("nan"),
            p_signflip=p_exist,
            exists=bool(p_exist < 0.05) if math.isfinite(p_exist) else False,
        ))
    gate = pd.DataFrame(gate_rows)

    # TD vs DD floors on family-level null median / SD (pooled across families
    # would be wrong; test envelope as the floor covariate candidate, plus each family).
    mw_rows = []
    floors["dd"] = floors["group"].ne("TD")
    for fam in FAMILIES:
        sub = floors[floors["family"] == fam]
        td = sub.loc[~sub["dd"], "null_median"].to_numpy()
        dd = sub.loc[sub["dd"], "null_median"].to_numpy()
        if len(td) and len(dd):
            u, p_med = stats.mannwhitneyu(td, dd, alternative="two-sided")
            u2, p_sd = stats.mannwhitneyu(
                sub.loc[~sub["dd"], "null_sd"].dropna(),
                sub.loc[sub["dd"], "null_sd"].dropna(),
                alternative="two-sided",
            )
        else:
            p_med = p_sd = float("nan")
        mw_rows.append(dict(frame=frame, family=fam, mw_p_null_median=p_med, mw_p_null_sd=p_sd))
    mw = pd.DataFrame(mw_rows)
    floors = floors.merge(mw, on=["frame", "family"], how="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(nulls, out_dir / f"nulls_{frame}.parquet")
    scores.to_csv(out_dir / f"scores_{frame}.csv", index=False)
    gate.to_csv(out_dir / f"gate1_{frame}.csv", index=False)
    floors.to_csv(out_dir / f"floors_{frame}.csv", index=False)

    # Family-level full null traces for Phase 4 (subject × shift × time × family)
    traces = []
    times = None
    kept = []
    for p in parts:
        tpath = exist_dir / p / f"{p}_existence_family_trace.npy"
        tspath = exist_dir / p / f"{p}_existence_times.npy"
        if not tpath.is_file():
            continue
        traces.append(np.load(tpath))
        kept.append(p)
        if times is None and tspath.is_file():
            times = np.load(tspath)
    if traces:
        np.savez_compressed(
            out_dir / f"null_traces_{frame}.npz",
            traces=np.stack(traces, axis=0),
            participants=np.array(kept),
            families=np.array(FAMILIES),
            times=times if times is not None else np.array([]),
        )

    print(gate.to_string(index=False))
    print(mw.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=("onset", "offset"))
    ap.add_argument("--exist-dir", type=Path)
    ap.add_argument("--obs-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=HERE / "existence")
    ap.add_argument("--cohort", type=Path, default=HERE / "cohort_groups.csv")
    ap.add_argument("--extractor", type=Path,
                    default=POOL / "extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b")
    args = ap.parse_args()
    defaults = {
        "onset": (
            POOL / "encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_existence_nulls",
            POOL / "encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_passthrough",
        ),
        "offset": (
            POOL / "encoding_results_b2b_mne_ica_v1_envpitch_tonev3_woffset_tmin05_existence_nulls",
            POOL / "encoding_results_b2b_mne_ica_v1_envpitch_tonev3_woffset_tmin05_passthrough",
        ),
    }
    exist_dir = args.exist_dir or defaults[args.frame][0]
    obs_dir = args.obs_dir or defaults[args.frame][1]
    cohort = _load_cohort(args.cohort)
    aggregate_frame(args.frame, exist_dir, obs_dir, cohort, args.extractor, args.out_dir)
    # Combined floors file if both frames present
    f_on = args.out_dir / "floors_onset.csv"
    f_off = args.out_dir / "floors_offset.csv"
    if f_on.is_file() and f_off.is_file():
        pd.concat([pd.read_csv(f_on), pd.read_csv(f_off)], ignore_index=True).to_csv(
            args.out_dir / "floors_by_group.csv", index=False
        )
    elif f_on.is_file():
        pd.read_csv(f_on).to_csv(args.out_dir / "floors_by_group.csv", index=False)
    elif f_off.is_file():
        pd.read_csv(f_off).to_csv(args.out_dir / "floors_by_group.csv", index=False)
    g_on = args.out_dir / "gate1_onset.csv"
    g_off = args.out_dir / "gate1_offset.csv"
    if g_on.is_file() and g_off.is_file():
        pd.concat([pd.read_csv(g_on), pd.read_csv(g_off)], ignore_index=True).to_csv(
            args.out_dir / "gate1.csv", index=False
        )


if __name__ == "__main__":
    main()
