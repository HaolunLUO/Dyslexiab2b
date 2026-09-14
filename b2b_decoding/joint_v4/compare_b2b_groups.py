#!/usr/bin/env python3
"""
compare_b2b_groups.py — joint B2B v4 group analysis.

Representation-existence tests use participant stimulus-shift nulls
(from per-subject null_traces). Group differences use subject-label
permutations. Both use one max-cluster statistic across time × families.

Usage:
  python compare_b2b_groups.py \\
      --results-dir /path/to/encoding_results_b2b_largev2_onset_joint_k8 \\
      --groups cohort_groups.csv \\
      --out-dir group_comparison_b2b_largev2_onset_joint_k8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FAMILIES = ("acoustic", "speech", "language")
CONFIRM_WIN = (0.0, 0.8)
NEGCTRL_WIN = (-0.2, 0.0)
EXPECTED_GROUPS = {
    "TD": 24,
    "dyslexia_normal_CAP": 19,
    "dyslexia_atypical_CAP": 20,
}
PRIMARY_N = 63
ROBUST_KEEP_FRAC = 0.5


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_groups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"participant", "group", "include_primary"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"groups CSV missing columns: {missing}")
    df = df.loc[df["include_primary"].astype(bool)].copy()
    if df["participant"].duplicated().any():
        raise SystemExit("Duplicate participants in group table")
    counts = df["group"].value_counts().to_dict()
    for g, n in EXPECTED_GROUPS.items():
        if counts.get(g, 0) != n:
            raise SystemExit(f"Group {g}: got {counts.get(g, 0)}, expected {n}")
    if len(df) != PRIMARY_N:
        raise SystemExit(f"Expected {PRIMARY_N} participants, got {len(df)}")
    # Pooled dyslexia must be 39
    n_dys = counts.get("dyslexia_normal_CAP", 0) + counts.get("dyslexia_atypical_CAP", 0)
    if n_dys != 39:
        raise SystemExit(f"Pooled dyslexia count {n_dys} != 39")
    return df


def validate_results(results_dir: str, groups: pd.DataFrame, cohort_hash: str) -> list[str]:
    ok = []
    errors = []
    for p in groups["participant"]:
        part = Path(results_dir) / p
        man = part / f"{p}_run_manifest.json"
        agg = part / f"{p}_b2b_family_agg.csv"
        if not man.is_file():
            errors.append(f"{p}: missing run_manifest.json")
            continue
        if not agg.is_file():
            errors.append(f"{p}: missing b2b_family_agg.csv")
            continue
        with open(man) as f:
            meta = json.load(f)
        if meta.get("participant") not in (None, p) and meta.get("participant") != p:
            errors.append(f"{p}: manifest participant mismatch")
            continue
        stored = meta.get("cohort_hash") or meta.get("group_table_hash")
        if stored and stored != cohort_hash:
            errors.append(f"{p}: cohort hash mismatch ({stored} != {cohort_hash})")
            continue
        ok.append(p)
    if errors:
        print("Validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} participants failed validation")
    if len(ok) != PRIMARY_N:
        raise SystemExit(f"Expected {PRIMARY_N} validated participants, got {len(ok)}")
    return ok


def load_family_matrix(
    results_dir: str,
    participants: list[str],
    family: str,
    score_col: str = "mean_trace",
):
    """Load subject × time family scores.

    Default ``mean_trace`` = Σ diag(H) within the family (Gwilliams/King-style
    aggregate). ``mean_score`` = trace/K is kept as an optional sensitivity.
    ``mean_z`` = mean_score / split_sd (per-subject noise-normalized; low-SNR).
    """
    rows = {}
    ref_times = None
    for p in participants:
        path = Path(results_dir) / p / f"{p}_b2b_family_agg.csv"
        df = pd.read_csv(path)
        sub = df.loc[df["family"] == family].sort_values("time")
        if sub.empty:
            raise SystemExit(f"{p}: family '{family}' missing")
        t = sub["time"].to_numpy(dtype=float)
        if score_col == "mean_z":
            if "mean_score" not in sub.columns or "split_sd" not in sub.columns:
                raise SystemExit(
                    f"{p}: mean_z needs mean_score and split_sd in {path.name}"
                )
            score = sub["mean_score"].to_numpy(dtype=float)
            sd = sub["split_sd"].to_numpy(dtype=float)
            y = np.divide(
                score,
                sd,
                out=np.zeros_like(score),
                where=np.isfinite(sd) & (sd > 1e-12),
            )
        else:
            if score_col not in sub.columns:
                raise SystemExit(
                    f"{p}: missing score column '{score_col}' in {path.name}. "
                    "Use --score-col mean_score for older mean-normalized runs."
                )
            y = sub[score_col].to_numpy(dtype=float)
        if ref_times is None:
            ref_times = t
        else:
            if len(t) != len(ref_times) or not np.allclose(t, ref_times):
                raise SystemExit(
                    f"Time axis mismatch for {p} family={family}; "
                    "refusing silent interpolation"
                )
        rows[p] = y
    mat = np.vstack([rows[p] for p in participants])
    return ref_times, mat


def load_null_cube(results_dir: str, participants: list[str], family: str, times: np.ndarray):
    """
    Load stimulus-shift null traces: (n_subj, n_null, n_times).
    Requires each participant's *_null_traces.csv.
    """
    cubes = []
    n_null_ref = None
    for p in participants:
        path = Path(results_dir) / p / f"{p}_null_traces.csv"
        if not path.is_file():
            raise SystemExit(f"{p}: missing null_traces.csv (needed for existence tests)")
        df = pd.read_csv(path)
        sub = df.loc[df["family"] == family]
        reps = sorted(sub["null_rep"].unique())
        if n_null_ref is None:
            n_null_ref = len(reps)
        elif len(reps) != n_null_ref:
            raise SystemExit(
                f"{p}: null rep count {len(reps)} != {n_null_ref}"
            )
        # Detect duplicate (rep, time)
        key_counts = sub.groupby(["null_rep", "time"]).size()
        if (key_counts > 1).any():
            raise SystemExit(f"{p}: duplicate null (rep,time) rows for family={family}")
        mat = np.full((n_null_ref, len(times)), np.nan)
        for i, r in enumerate(reps):
            row = sub.loc[sub["null_rep"] == r].sort_values("time")
            t = row["time"].to_numpy(float)
            if len(t) != len(times) or not np.allclose(t, times):
                raise SystemExit(f"{p}: null time axis mismatch family={family} rep={r}")
            mat[i] = row["family_score"].to_numpy(float)
        if not np.isfinite(mat).all():
            raise SystemExit(f"{p}: non-finite null traces family={family}")
        cubes.append(mat)
    return np.stack(cubes, axis=0)  # n_subj × n_null × n_times


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if a.ndim > 1:
        a = a.mean(axis=1)
        b = b.mean(axis=1)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((na - 1) * sa**2 + (nb - 1) * sb**2) / max(na + nb - 2, 1)
    sp = np.sqrt(sp2) if sp2 > 0 else np.nan
    d = (ma - mb) / sp if sp and np.isfinite(sp) else np.nan
    J = 1.0 - 3.0 / (4 * (na + nb) - 9) if (na + nb) > 2 else 1.0
    return float(J * d)


def bootstrap_ci(a, b, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    diffs = []
    for _ in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        diffs.append(aa.mean() - bb.mean())
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])


def cluster_mass_1d(tvals, threshold):
    mask = np.abs(tvals) > threshold
    clusters = []
    i = 0
    n = len(tvals)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        mass = float(np.sum(np.abs(tvals[i:j])))
        clusters.append((i, j, mass))
        i = j
    return clusters


def max_cluster_mass(tvals, threshold):
    cl = cluster_mass_1d(tvals, threshold)
    return max((c[2] for c in cl), default=0.0)


def max_cluster_across_families(t_by_family: dict[str, np.ndarray], threshold: float) -> float:
    """One max-cluster statistic across time and all families."""
    return max(max_cluster_mass(t_by_family[f], threshold) for f in t_by_family)


def stimulus_shift_existence(
    mats: dict[str, np.ndarray],
    null_cubes: dict[str, np.ndarray],
    times: np.ndarray,
    win=CONFIRM_WIN,
):
    """
    Representation-existence via participant stimulus-shift nulls.

    For each null replicate r, form the group-mean trace from each subject's
    r-th stimulus-shifted score; compute max cluster mass across families;
    compare observed max cluster to this null distribution.
    """
    idx = np.where(window_mask(times, win))[0]
    families = list(mats.keys())
    n_subj = mats[families[0]].shape[0]
    n_null = null_cubes[families[0]].shape[1]
    thr = stats.t.ppf(1 - 0.025, n_subj - 1)

    def t_from_mat(M):
        Y = M[:, idx]
        mu = Y.mean(0)
        se = Y.std(0, ddof=1) / np.sqrt(n_subj)
        return np.divide(mu, se, out=np.zeros_like(mu), where=se > 0)

    t_obs = {f: t_from_mat(mats[f]) for f in families}
    obs_max = max_cluster_across_families(t_obs, thr)
    obs_by_f = {f: max_cluster_mass(t_obs[f], thr) for f in families}

    null_max = np.zeros(n_null)
    for r in range(n_null):
        t_null = {}
        for f in families:
            # null_cubes[f]: n_subj × n_null × n_times
            M = null_cubes[f][:, r, :]
            t_null[f] = t_from_mat(M)
        null_max[r] = max_cluster_across_families(t_null, thr)

    p_corr = (np.sum(null_max >= obs_max) + 1) / (n_null + 1)
    return {
        "obs_max_cluster_mass": float(obs_max),
        "obs_mass_by_family": {f: float(v) for f, v in obs_by_f.items()},
        "p_cluster_corrected": float(p_corr),
        "n_null": int(n_null),
        "null_method": "participant_stimulus_shift",
    }


def label_perm_contrast_multifamily(
    mats_A: dict[str, np.ndarray],
    mats_B: dict[str, np.ndarray],
    times: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
    win=CONFIRM_WIN,
):
    """
    Two-group label permutation with ONE max-cluster statistic across
    time and all three families.
    """
    rng = np.random.default_rng(seed)
    idx = np.where(window_mask(times, win))[0]
    families = list(FAMILIES)
    nA = mats_A[families[0]].shape[0]
    nB = mats_B[families[0]].shape[0]
    df = nA + nB - 2
    thr = stats.t.ppf(1 - 0.025, df)

    def t_trace(X, Y):
        mx, my = X.mean(0), Y.mean(0)
        sx = X.std(0, ddof=1)
        sy = Y.std(0, ddof=1)
        sp = np.sqrt(((nA - 1) * sx**2 + (nB - 1) * sy**2) / max(df, 1))
        se = sp * np.sqrt(1 / nA + 1 / nB)
        return np.divide(mx - my, se, out=np.zeros_like(mx), where=se > 0)

    t_obs = {}
    clusters_by_f = {}
    for f in families:
        A_w = mats_A[f][:, idx]
        B_w = mats_B[f][:, idx]
        t = t_trace(A_w, B_w)
        t_obs[f] = t
        clusters_by_f[f] = cluster_mass_1d(t, thr)

    obs_mass = max_cluster_across_families(t_obs, thr)

    # Pool subjects for label shuffle (same shuffle applied to all families)
    pooled = {f: np.vstack([mats_A[f][:, idx], mats_B[f][:, idx]]) for f in families}
    null = np.zeros(n_perm)
    for i in range(n_perm):
        order = rng.permutation(nA + nB)
        t_null = {}
        for f in families:
            P = pooled[f][order]
            t_null[f] = t_trace(P[:nA], P[nA:])
        null[i] = max_cluster_across_families(t_null, thr)
    p = (np.sum(null >= obs_mass) + 1) / (n_perm + 1)

    # Per-family descriptive effects (window mean) — inference is joint
    per_family = {}
    for f in families:
        a_mean = mats_A[f][:, idx].mean(1)
        b_mean = mats_B[f][:, idx].mean(1)
        g = hedges_g(a_mean, b_mean)
        ci = bootstrap_ci(a_mean, b_mean, seed=seed)
        full_clusters = []
        for i0, i1, mass in clusters_by_f[f]:
            full_clusters.append(
                {
                    "t_start": float(times[idx[i0]]),
                    "t_end": float(times[idx[min(i1 - 1, len(idx) - 1)]]),
                    "mass": mass,
                }
            )
        per_family[f] = {
            "hedges_g": g,
            "bootstrap_ci_95": ci,
            "mean_A": float(a_mean.mean()),
            "mean_B": float(b_mean.mean()),
            "clusters": full_clusters,
            "obs_cluster_mass": float(max_cluster_mass(t_obs[f], thr)),
        }

    return {
        "p_cluster_joint": float(p),
        "obs_max_mass_joint": float(obs_mass),
        "n_A": nA,
        "n_B": nB,
        "n_perm": n_perm,
        "null_method": "subject_label_permutation",
        "correction": "max_cluster_across_time_and_families",
        "per_family": per_family,
    }


def three_group_omnibus(mats_by_group, times, n_perm=10000, seed=42, win=CONFIRM_WIN):
    """Secondary F-based omnibus via label permutation on window means (per family).

    mats_by_group: group_name -> {family -> (n_subj, n_times)}
    """
    rng = np.random.default_rng(seed)
    idx = np.where(window_mask(times, win))[0]
    out = {}
    group_names = list(mats_by_group.keys())
    for fam in FAMILIES:
        labels = []
        data = []
        for gi, g in enumerate(group_names):
            y = mats_by_group[g][fam][:, idx].mean(1)
            data.append(y)
            labels.extend([gi] * len(y))
        y_all = np.concatenate(data)
        lab = np.asarray(labels)

        def f_stat(y, lab_):
            groups = [y[lab_ == i] for i in np.unique(lab_)]
            return float(stats.f_oneway(*groups).statistic)

        obs = f_stat(y_all, lab)
        null = np.zeros(n_perm)
        lab_work = lab.copy()
        for i in range(n_perm):
            rng.shuffle(lab_work)
            null[i] = f_stat(y_all, lab_work)
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        out[fam] = {"F": obs, "p": float(p)}
    return out


def parse_exclude_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    out = []
    seen = set()
    for tok in raw.replace(" ", ",").split(","):
        p = tok.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def run_analysis(results_dir, groups_path, out_dir, n_perm=10000, seed=42,
                 skip_existence_if_no_nulls=False, exclude=None, families=None,
                 score_col: str = "mean_trace"):
    global FAMILIES
    if families:
        FAMILIES = tuple(families)
    os.makedirs(out_dir, exist_ok=True)
    groups = load_groups(groups_path)
    chash = sha256_file(groups_path)
    participants = validate_results(results_dir, groups, chash)
    print(f"Validated {len(participants)} participants")
    print(f"group_table_hash={chash}")
    label = {
        "mean_trace": "Σ diag(H)",
        "mean_score": "trace/K",
        "mean_z": "mean_score/split_sd",
    }.get(score_col, score_col)
    print(f"score_col={score_col}  ({label})")

    exclude = parse_exclude_list(exclude) if isinstance(exclude, str) else list(exclude or [])
    if exclude:
        unknown = [p for p in exclude if p not in participants]
        if unknown:
            raise SystemExit(f"--exclude names not in primary cohort: {unknown}")
        participants = [p for p in participants if p not in set(exclude)]
        groups = groups.loc[groups["participant"].isin(participants)].copy()
        print(f"Excluded {len(exclude)} participants: {', '.join(exclude)}")
        print(f"Remaining N={len(participants)}")

    keep_frac = {}
    for p in participants:
        qpath = Path(results_dir) / p / f"{p}_qc_summary.csv"
        if qpath.is_file():
            q = pd.read_csv(qpath)
            keep_frac[p] = float(q["keep_frac"].iloc[0])
        else:
            keep_frac[p] = np.nan

    gmap = dict(zip(groups["participant"], groups["group"]))
    td = [p for p in participants if gmap[p] == "TD"]
    dn = [p for p in participants if gmap[p] == "dyslexia_normal_CAP"]
    da = [p for p in participants if gmap[p] == "dyslexia_atypical_CAP"]
    dys = dn + da
    if not exclude:
        if len(td) != 24 or len(dys) != 39 or len(dn) != 19 or len(da) != 20:
            raise SystemExit(
                f"Contrast counts wrong: TD={len(td)} dys={len(dys)} "
                f"normalCAP={len(dn)} atypicalCAP={len(da)}"
            )
    else:
        if min(len(td), len(dn), len(da)) < 2:
            raise SystemExit(
                f"Too few remaining after exclude: TD={len(td)} "
                f"normalCAP={len(dn)} atypicalCAP={len(da)}"
            )
        print(
            f"Post-exclusion counts: TD={len(td)} dys={len(dys)} "
            f"normalCAP={len(dn)} atypicalCAP={len(da)}"
        )

    def subset(ps, min_keep=None):
        if min_keep is None:
            return ps
        return [p for p in ps if keep_frac.get(p, 0) >= min_keep]

    summaries = []
    for robust, tag in ((False, "primary"), (True, "robust_keep50")):
        min_keep = ROBUST_KEEP_FRAC if robust else None
        td_s, dn_s, da_s = subset(td, min_keep), subset(dn, min_keep), subset(da, min_keep)
        dys_s = dn_s + da_s
        if robust and len(td_s) + len(dys_s) < 40:
            print(f"[{tag}] too few participants after keep filter; skipping")
            continue
        print(
            f"\n=== Analysis set: {tag}  "
            f"TD={len(td_s)} normCAP={len(dn_s)} atypCAP={len(da_s)} ==="
        )

        times = None
        fam_mats_all = {}
        for fam in FAMILIES:
            t, M_td = load_family_matrix(results_dir, td_s, fam, score_col=score_col)
            _, M_dn = load_family_matrix(results_dir, dn_s, fam, score_col=score_col)
            _, M_da = load_family_matrix(results_dir, da_s, fam, score_col=score_col)
            _, M_dys = load_family_matrix(results_dir, dys_s, fam, score_col=score_col)
            times = t
            fam_mats_all[fam] = {
                "TD": M_td,
                "dyslexia_normal_CAP": M_dn,
                "dyslexia_atypical_CAP": M_da,
                "dyslexia_pooled": M_dys,
            }
            for label, ps, M in (
                ("TD", td_s, M_td),
                ("dyslexia_normal_CAP", dn_s, M_dn),
                ("dyslexia_atypical_CAP", da_s, M_da),
            ):
                df = pd.DataFrame(M, index=ps, columns=times)
                df.index.name = "participant"
                df.to_csv(Path(out_dir) / f"traces_{tag}_{fam}_{label}.csv")

        # Representation existence via stimulus-shift nulls
        mats_pool = {
            fam: np.vstack(
                [fam_mats_all[fam]["TD"], fam_mats_all[fam]["dyslexia_pooled"]]
            )
            for fam in FAMILIES
        }
        pool_ps = td_s + dys_s
        try:
            null_cubes = {
                fam: load_null_cube(results_dir, pool_ps, fam, times) for fam in FAMILIES
            }
            uniq = stimulus_shift_existence(mats_pool, null_cubes, times)
        except SystemExit as e:
            if skip_existence_if_no_nulls:
                print(f"[{tag}] skipping existence test: {e}")
                uniq = {"skipped": True, "reason": str(e)}
            else:
                raise

        with open(Path(out_dir) / f"unique_family_{tag}.json", "w") as f:
            json.dump(uniq, f, indent=2)

        # Primary contrast: TD vs pooled dyslexia (joint across families)
        mats_td = {f: fam_mats_all[f]["TD"] for f in FAMILIES}
        mats_dys = {f: fam_mats_all[f]["dyslexia_pooled"] for f in FAMILIES}
        mats_dn = {f: fam_mats_all[f]["dyslexia_normal_CAP"] for f in FAMILIES}
        mats_da = {f: fam_mats_all[f]["dyslexia_atypical_CAP"] for f in FAMILIES}

        primary = {
            "TD_vs_dyslexia_pooled": label_perm_contrast_multifamily(
                mats_td, mats_dys, times, n_perm=n_perm, seed=seed
            ),
        }
        # Secondary: normal-CAP vs atypical-CAP
        secondary = {
            "dyslexia_normal_vs_atypical_CAP": label_perm_contrast_multifamily(
                mats_dn, mats_da, times, n_perm=n_perm, seed=seed + 1
            ),
            "TD_vs_dyslexia_normal_CAP": label_perm_contrast_multifamily(
                mats_td, mats_dn, times, n_perm=n_perm, seed=seed + 2
            ),
            "TD_vs_dyslexia_atypical_CAP": label_perm_contrast_multifamily(
                mats_td, mats_da, times, n_perm=n_perm, seed=seed + 3
            ),
            "omnibus_3group": three_group_omnibus(
                {
                    "TD": mats_td,
                    "dyslexia_normal_CAP": mats_dn,
                    "dyslexia_atypical_CAP": mats_da,
                },
                times,
                n_perm=n_perm,
                seed=seed + 4,
            ),
        }

        # Use full pre-onset span from the fitted traces when longer than default.
        negctrl_win = (float(np.min(times)), 0.0) if float(np.min(times)) < NEGCTRL_WIN[0] - 1e-9 else NEGCTRL_WIN
        neg_idx = window_mask(times, negctrl_win)
        neg_ctrl = {
            fam: {
                "mean_abs_TD": float(np.mean(np.abs(mats_td[fam][:, neg_idx]))),
                "mean_abs_dys": float(np.mean(np.abs(mats_dys[fam][:, neg_idx]))),
            }
            for fam in FAMILIES
        }

        payload = {
            "analysis_set": tag,
            "n_TD": len(td_s),
            "n_dyslexia_pooled": len(dys_s),
            "n_dyslexia_normal_CAP": len(dn_s),
            "n_dyslexia_atypical_CAP": len(da_s),
            "excluded_participants": exclude,
            "confirm_win_s": list(CONFIRM_WIN),
            "negctrl_win_s": list(negctrl_win),
            "primary": primary,
            "secondary": secondary,
            "neg_control": neg_ctrl,
            "representation_existence": uniq,
            "cohort_hash": chash,
        }
        with open(Path(out_dir) / f"group_comparison_{tag}_joint.json", "w") as f:
            json.dump(payload, f, indent=2)

        for contrast_name, res in {**primary, **{k: v for k, v in secondary.items() if k != "omnibus_3group"}}.items():
            row = {
                "analysis_set": tag,
                "contrast": contrast_name,
                "p_cluster_joint": res["p_cluster_joint"],
                "obs_max_mass_joint": res["obs_max_mass_joint"],
                "n_A": res["n_A"],
                "n_B": res["n_B"],
            }
            for fam in FAMILIES:
                row[f"hedges_g_{fam}"] = res["per_family"][fam]["hedges_g"]
            summaries.append(row)
            g_vals = "/".join(f"{res['per_family'][f]['hedges_g']:.3f}" for f in FAMILIES)
            fam_tag = "/".join(FAMILIES)
            print(
                f"  {contrast_name}: joint p={res['p_cluster_joint']:.4f} "
                f"mass={res['obs_max_mass_joint']:.3f} "
                f"g({fam_tag})={g_vals}"
            )

    pd.DataFrame(summaries).to_csv(
        Path(out_dir) / "group_comparison_summary.csv", index=False
    )

    manifest = {
        "results_dir": results_dir,
        "groups": groups_path,
        "cohort_hash": chash,
        "score_col": score_col,
        "score_definition": (
            "sum of family diag(H) = mean_trace"
            if score_col == "mean_trace"
            else "mean of family diag(H) = mean_score = trace/K"
        ),
        "confirm_win_s": list(CONFIRM_WIN),
        "negctrl_win_s": list(NEGCTRL_WIN),
        "n_perm": n_perm,
        "seed": seed,
        "excluded_participants": exclude,
        "n_remaining": len(participants),
        "n_TD": len(td),
        "n_dyslexia_pooled": len(dys),
        "n_dyslexia_normal_CAP": len(dn),
        "n_dyslexia_atypical_CAP": len(da),
        "primary_contrast": (
            f"TD_{len(td)}_vs_dyslexia_pooled_{len(dys)}"
            if exclude else "TD_24_vs_dyslexia_pooled_39"
        ),
        "secondary_contrast": (
            f"dyslexia_normal_CAP_{len(dn)}_vs_atypical_CAP_{len(da)}"
            if exclude else "dyslexia_normal_CAP_19_vs_atypical_CAP_20"
        ),
        "existence_null": "participant_stimulus_shift",
        "group_null": "subject_label_permutation",
        "statistic": "max_cluster_across_time_and_families",
    }
    with open(Path(out_dir) / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote results under {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Joint B2B v4 group analysis")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-existence-if-no-nulls",
        action="store_true",
        help="Allow observed-only group contrasts when null_traces are absent",
    )
    ap.add_argument(
        "--exclude",
        default="",
        help="Comma-separated participant IDs to drop after validating the primary 63",
    )
    ap.add_argument(
        "--families",
        default="acoustic,speech,language",
        help="Comma-separated family names matching b2b_family_agg.csv "
             "(lexical arm: duration,frequency,surprisal)",
    )
    ap.add_argument(
        "--score-col",
        default="mean_trace",
        choices=("mean_trace", "mean_score", "mean_z"),
        help="Family aggregate: mean_trace=Σ diag(H) [default, paper-style]; "
             "mean_score=trace/K; mean_z=mean_score/split_sd (SNR-normalized)",
    )
    args = ap.parse_args()
    families = tuple(x.strip() for x in args.families.split(",") if x.strip())
    run_analysis(
        args.results_dir,
        args.groups,
        args.out_dir,
        n_perm=args.n_perm,
        seed=args.seed,
        skip_existence_if_no_nulls=args.skip_existence_if_no_nulls,
        exclude=args.exclude,
        families=families,
        score_col=args.score_col,
    )


if __name__ == "__main__":
    main()
