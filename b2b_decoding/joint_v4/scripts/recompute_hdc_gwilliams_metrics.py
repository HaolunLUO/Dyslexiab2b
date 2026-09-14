#!/usr/bin/env python3
"""Recompute Gwilliams-style HDC duration / sustain / hierarchy on existing outputs.

Uses:
  - ``*_b2b_family_agg.csv`` for diagonal duration + normalized rise (full 100 Hz)
  - ``*_tg_{family}.npy`` + ``*_tg_times.json`` for TG duration / sustain
  - Subject threshold: mean_score > z * SE, SE = split_sd / sqrt(n_valid_splits)
  - Group duration: one-sample t vs 0 + contiguous |t|>t_crit cluster extent
  - Hierarchy: rank correlations; primary sustain set excludes semantic
  - TD vs dyslexia: label permutation on subject-level metrics

Does not re-run Julia B2B.

Usage:
  .venv_gpt2/bin/python3 scripts/recompute_hdc_gwilliams_metrics.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot \\
      --groups cohort_groups.csv \\
      --out-dir group_comparison_b2b_hdc_offset_pilot/gwilliams_metrics \\
      --families phonetic,word_form,lexical_syntactic,syntax_proxy,semantic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Hierarchy ranks for primary sustain correlation (Gwilliams Fig. 5D table;
# semantic omitted).
PRIMARY_SUSTAIN_EXCLUDE = frozenset({"semantic"})


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def perm_p(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> float:
    obs = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(float(pooled[:na].mean() - pooled[na:].mean()))
        ge += int(d >= obs)
    return (ge + 1) / (n_perm + 1)


def load_groups(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["include_primary"].astype(bool)].copy()


def longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def total_true(mask: np.ndarray) -> int:
    return int(np.sum(mask.astype(bool)))


def se_from_agg(sub: pd.DataFrame) -> np.ndarray:
    sd = sub["split_sd"].to_numpy(float)
    n = sub["n_valid_splits"].to_numpy(float)
    n = np.maximum(n, 1.0)
    return sd / np.sqrt(n)


def halfmax_onset(times: np.ndarray, y: np.ndarray) -> float:
    """First time normalized score crosses 0.5 (NaN if never positive)."""
    y = np.asarray(y, float)
    times = np.asarray(times, float)
    ymax = float(np.nanmax(y)) if np.any(np.isfinite(y)) else float("nan")
    if not np.isfinite(ymax) or ymax <= 0:
        return float("nan")
    yn = y / ymax
    hits = np.where(yn >= 0.5)[0]
    if hits.size == 0:
        return float("nan")
    return float(times[hits[0]])


def pre_offset_norm_mean(times: np.ndarray, y: np.ndarray, tmin: float = -0.4) -> float:
    y = np.asarray(y, float)
    times = np.asarray(times, float)
    ymax = float(np.nanmax(y)) if np.any(np.isfinite(y)) else float("nan")
    if not np.isfinite(ymax) or ymax <= 0:
        return float("nan")
    yn = y / ymax
    m = (times >= tmin) & (times <= 0.0) & np.isfinite(yn)
    if not np.any(m):
        return float("nan")
    return float(np.mean(yn[m]))


def duration_from_trace(
    times: np.ndarray, y: np.ndarray, se: np.ndarray, z: float
) -> dict:
    times = np.asarray(times, float)
    y = np.asarray(y, float)
    se = np.asarray(se, float)
    dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
    thresh = z * se
    above = np.isfinite(y) & np.isfinite(thresh) & (y > thresh)
    return {
        "duration_longest_s": longest_true_run(above) * dt,
        "duration_total_s": total_true(above) * dt,
        "n_above": int(total_true(above)),
        "dt_s": dt,
        "onset_halfmax_s": halfmax_onset(times, y),
        "pre_offset_norm_mean": pre_offset_norm_mean(times, y),
        "peak_time_s": float(times[int(np.nanargmax(y))]) if np.any(np.isfinite(y)) else float("nan"),
        "peak_score": float(np.nanmax(y)) if np.any(np.isfinite(y)) else float("nan"),
    }


def tg_duration_sustain_sig(
    M: np.ndarray, times: np.ndarray, se_tg: np.ndarray, z: float
) -> dict:
    """Duration = longest above-chance diagonal; sustain = mean row width above chance."""
    M = np.asarray(M, float)
    times = np.asarray(times, float)
    se_tg = np.asarray(se_tg, float)
    n = M.shape[0]
    dt = float(np.mean(np.diff(times))) if n > 1 else 0.0
    thresh = z * se_tg
    diagv = np.array([M[i, i] for i in range(n)], float)
    diag_above = np.isfinite(diagv) & np.isfinite(thresh) & (diagv > thresh)

    widths = []
    for i in range(n):
        row_thresh = thresh[i] if np.isfinite(thresh[i]) else z * np.nanmedian(se_tg)
        row_above = np.isfinite(M[i, :]) & (M[i, :] > row_thresh)
        if not np.any(row_above):
            continue
        # Prefer rows that are themselves decodable on the diagonal
        if diag_above[i]:
            widths.append(float(np.sum(row_above)) * dt)
    if not widths:
        # fallback: any row with above-chance cells
        for i in range(n):
            row_thresh = thresh[i] if np.isfinite(thresh[i]) else z * np.nanmedian(se_tg)
            row_above = np.isfinite(M[i, :]) & (M[i, :] > row_thresh)
            if np.any(row_above):
                widths.append(float(np.sum(row_above)) * dt)

    # Diagonal-realigned profile: lag k -> mean M[i, i+k]
    max_lag = n - 1
    lags = np.arange(-max_lag, max_lag + 1)
    realigned = np.full(lags.shape, np.nan)
    for li, lag in enumerate(lags):
        vals = []
        for i in range(n):
            j = i + int(lag)
            if 0 <= j < n and np.isfinite(M[i, j]):
                vals.append(M[i, j])
        if vals:
            realigned[li] = float(np.mean(vals))

    return {
        "duration_longest_s": longest_true_run(diag_above) * dt,
        "duration_total_s": total_true(diag_above) * dt,
        "sustain_s": float(np.mean(widths)) if widths else 0.0,
        "sustain_n_rows": int(len(widths)),
        "dt_s": dt,
        "n_diag_above": int(total_true(diag_above)),
        "lags_s": lags * dt,
        "realigned": realigned,
    }


def align_se_to_times(agg_times: np.ndarray, se: np.ndarray, times_tg: np.ndarray) -> np.ndarray:
    """Nearest-neighbor SE from full-resolution agg onto TG grid."""
    out = np.empty(len(times_tg), float)
    for i, t in enumerate(times_tg):
        j = int(np.argmin(np.abs(agg_times - t)))
        out[i] = se[j]
    return out


def cluster_duration_group(
    mat: np.ndarray, times: np.ndarray, alpha: float = 0.05
) -> dict:
    """One-sample t vs 0; duration = total time in positive clusters with |t|>t_crit."""
    n_subj, n_t = mat.shape
    mu = mat.mean(0)
    se = mat.std(0, ddof=1) / np.sqrt(n_subj)
    tvals = np.divide(mu, se, out=np.zeros_like(mu), where=se > 0)
    thr = float(stats.t.ppf(1 - alpha / 2, max(n_subj - 1, 1)))
    dt = float(np.mean(np.diff(times))) if n_t > 1 else 0.0
    mask = tvals > thr  # positive decoding only (Gwilliams-style above chance)
    clusters = []
    i = 0
    while i < n_t:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n_t and mask[j]:
            j += 1
        clusters.append(
            {
                "t_start": float(times[i]),
                "t_end": float(times[j - 1]),
                "duration_s": (j - i) * dt,
                "mean_t": float(np.mean(tvals[i:j])),
            }
        )
        i = j
    total = float(sum(c["duration_s"] for c in clusters))
    longest = float(max((c["duration_s"] for c in clusters), default=0.0))
    return {
        "duration_cluster_total_s": total,
        "duration_cluster_longest_s": longest,
        "n_clusters": len(clusters),
        "t_crit": thr,
        "clusters": clusters,
        "tvals": tvals,
        "times": times,
    }


def spearman_or_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return float("nan"), float("nan")
    r_p, p_p = stats.pearsonr(x[m], y[m])
    return float(r_p), float(p_p)


def discover_families(results: Path, participants: list[str], requested: list[str]) -> list[str]:
    if requested:
        return requested
    # Prefer families present on first subject with TG
    for p in participants:
        man = results / p / f"{p}_tg_times.json"
        if man.is_file():
            meta = json.loads(man.read_text())
            fams = meta.get("families")
            if fams:
                return list(fams)
        metrics = results / p / f"{p}_tg_metrics.csv"
        if metrics.is_file():
            return list(pd.read_csv(metrics)["family"].astype(str).unique())
    raise SystemExit("Could not discover families")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", default="", help="Comma-separated hierarchical order")
    ap.add_argument("--z", type=float, default=1.0, help="Subject threshold in SE units")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cluster-alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for group t critical value",
    )
    args = ap.parse_args()

    results = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = load_groups(Path(args.groups))
    participants = list(groups["participant"].astype(str))
    gmap = dict(zip(groups["participant"].astype(str), groups["group"].astype(str)))

    requested = [x.strip() for x in args.families.split(",") if x.strip()]
    families = discover_families(results, participants, requested)
    rank = {f: i + 1 for i, f in enumerate(families)}

    subj_rows = []
    realigned_store: dict[str, list[np.ndarray]] = {f: [] for f in families}
    lags_ref: dict[str, np.ndarray] = {}
    diag_mats: dict[str, list[np.ndarray]] = {f: [] for f in families}
    diag_times: np.ndarray | None = None
    tg_diag_mats: dict[str, list[np.ndarray]] = {f: [] for f in families}
    tg_times_ref: np.ndarray | None = None

    missing = []
    for p in participants:
        agg_path = results / p / f"{p}_b2b_family_agg.csv"
        tg_meta_path = results / p / f"{p}_tg_times.json"
        if not agg_path.is_file():
            missing.append(f"{p}: missing family_agg")
            continue
        agg = pd.read_csv(agg_path)
        times_tg = None
        if tg_meta_path.is_file():
            meta = json.loads(tg_meta_path.read_text())
            times_tg = np.asarray(meta["times_tg"], float)
            if tg_times_ref is None:
                tg_times_ref = times_tg

        for fam in families:
            sub = agg.loc[agg["family"] == fam].sort_values("time")
            if sub.empty:
                missing.append(f"{p}: missing family {fam} in agg")
                continue
            times = sub["time"].to_numpy(float)
            y = sub["mean_score"].to_numpy(float)
            se = se_from_agg(sub)
            if diag_times is None:
                diag_times = times
            diag_mats[fam].append(y)

            d_agg = duration_from_trace(times, y, se, args.z)
            subj_rows.append(
                {
                    "participant": p,
                    "group": gmap.get(p, ""),
                    "family": fam,
                    "hierarchy_rank": rank[fam],
                    "metric_source": "family_agg",
                    "duration_s": d_agg["duration_longest_s"],
                    "duration_total_s": d_agg["duration_total_s"],
                    "sustain_s": np.nan,
                    "onset_halfmax_s": d_agg["onset_halfmax_s"],
                    "pre_offset_norm_mean": d_agg["pre_offset_norm_mean"],
                    "peak_time_s": d_agg["peak_time_s"],
                    "peak_score": d_agg["peak_score"],
                    "z": args.z,
                }
            )

            tg_path = results / p / f"{p}_tg_{fam}.npy"
            if tg_path.is_file() and times_tg is not None:
                M = np.load(tg_path)
                se_tg = align_se_to_times(times, se, times_tg)
                tg = tg_duration_sustain_sig(M, times_tg, se_tg, args.z)
                subj_rows.append(
                    {
                        "participant": p,
                        "group": gmap.get(p, ""),
                        "family": fam,
                        "hierarchy_rank": rank[fam],
                        "metric_source": "tg",
                        "duration_s": tg["duration_longest_s"],
                        "duration_total_s": tg["duration_total_s"],
                        "sustain_s": tg["sustain_s"],
                        "sustain_n_rows": tg["sustain_n_rows"],
                        "n_diag_above": tg["n_diag_above"],
                        "onset_halfmax_s": d_agg["onset_halfmax_s"],
                        "pre_offset_norm_mean": d_agg["pre_offset_norm_mean"],
                        "peak_time_s": d_agg["peak_time_s"],
                        "peak_score": d_agg["peak_score"],
                        "z": args.z,
                    }
                )
                realigned_store[fam].append(tg["realigned"])
                lags_ref[fam] = tg["lags_s"]
                tg_diag_mats[fam].append(
                    np.array([M[i, i] for i in range(M.shape[0])], float)
                )

    if missing:
        (out / "missing.txt").write_text("\n".join(missing) + "\n")
        print(f"Warnings: {len(missing)} missing items (see missing.txt)")

    sdf = pd.DataFrame(subj_rows)
    sdf["dyslexia_pooled"] = sdf["group"].ne("TD")
    sdf.to_csv(out / "subject_metrics.csv", index=False)

    # --- Group cluster duration on family_agg diagonals ---
    cluster_rows = []
    cluster_detail = {}
    if diag_times is not None:
        for fam in families:
            if len(diag_mats[fam]) < 5:
                continue
            mat = np.vstack(diag_mats[fam])
            # align participant order to those who contributed
            res = cluster_duration_group(mat, diag_times, alpha=args.cluster_alpha)
            cluster_rows.append(
                {
                    "family": fam,
                    "hierarchy_rank": rank[fam],
                    "source": "family_agg",
                    "duration_cluster_total_s": res["duration_cluster_total_s"],
                    "duration_cluster_longest_s": res["duration_cluster_longest_s"],
                    "n_clusters": res["n_clusters"],
                    "t_crit": res["t_crit"],
                    "n_subjects": int(mat.shape[0]),
                }
            )
            cluster_detail[fam] = {
                "clusters": res["clusters"],
                "tvals": res["tvals"].tolist(),
                "times": res["times"].tolist(),
            }
        # TG diagonal group clusters
        if tg_times_ref is not None:
            for fam in families:
                if len(tg_diag_mats[fam]) < 5:
                    continue
                mat = np.vstack(tg_diag_mats[fam])
                res = cluster_duration_group(mat, tg_times_ref, alpha=args.cluster_alpha)
                cluster_rows.append(
                    {
                        "family": fam,
                        "hierarchy_rank": rank[fam],
                        "source": "tg_diagonal",
                        "duration_cluster_total_s": res["duration_cluster_total_s"],
                        "duration_cluster_longest_s": res["duration_cluster_longest_s"],
                        "n_clusters": res["n_clusters"],
                        "t_crit": res["t_crit"],
                        "n_subjects": int(mat.shape[0]),
                    }
                )
    cdf = pd.DataFrame(cluster_rows)
    cdf.to_csv(out / "group_cluster_duration.csv", index=False)
    (out / "group_cluster_detail.json").write_text(json.dumps(cluster_detail, indent=2) + "\n")

    # --- Diagonal-realigned group means (Fig. 5D style) ---
    realigned_rows = []
    for fam in families:
        arrs = realigned_store[fam]
        if not arrs:
            continue
        # pad/truncate to common length
        L = min(a.shape[0] for a in arrs)
        stack = np.vstack([a[:L] for a in arrs])
        lags = lags_ref[fam][:L]
        mu = np.nanmean(stack, axis=0)
        se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
        for lag, m, s in zip(lags, mu, se):
            realigned_rows.append(
                {
                    "family": fam,
                    "hierarchy_rank": rank[fam],
                    "lag_s": float(lag),
                    "mean": float(m),
                    "se": float(s),
                    "n": int(stack.shape[0]),
                }
            )
    rdf = pd.DataFrame(realigned_rows)
    rdf.to_csv(out / "tg_diagonal_realigned_group.csv", index=False)

    # --- Hierarchy correlations (subject-level r, then group mean / vs 0) ---
    corr_rows = []
    sets = {
        "all": families,
        "no_semantic": [f for f in families if f not in PRIMARY_SUSTAIN_EXCLUDE],
        "no_semantic_no_state": [
            f
            for f in families
            if f not in PRIMARY_SUSTAIN_EXCLUDE and f != "syntactic_state"
        ],
    }
    metrics_for_corr = [
        ("tg", "duration_s"),
        ("tg", "sustain_s"),
        ("family_agg", "duration_s"),
        ("family_agg", "onset_halfmax_s"),
        ("family_agg", "pre_offset_norm_mean"),
    ]
    for set_name, fams in sets.items():
        if len(fams) < 3:
            continue
        for src, metric in metrics_for_corr:
            for p, sp in sdf.loc[sdf["metric_source"] == src].groupby("participant"):
                sp = sp.set_index("family").reindex(fams)
                if sp[metric].notna().sum() < 3:
                    continue
                ranks = np.array([rank[f] for f in fams], float)
                y = sp[metric].to_numpy(float)
                r, _ = spearman_or_pearson(ranks, y)
                corr_rows.append(
                    {
                        "participant": p,
                        "group": gmap.get(p, ""),
                        "family_set": set_name,
                        "metric_source": src,
                        "metric": metric,
                        "r_rank": r,
                        "n_families": int(np.sum(np.isfinite(y))),
                    }
                )
    cordf = pd.DataFrame(corr_rows)
    cordf.to_csv(out / "hdc_rank_correlations_sig.csv", index=False)

    # Second-level: mean r vs 0 (permutation sign-flip) + TD vs dys
    hier_summary = []
    rng = np.random.default_rng(args.seed)
    for (set_name, src, metric), sub in cordf.groupby(["family_set", "metric_source", "metric"]):
        rs = sub["r_rank"].to_numpy(float)
        rs = rs[np.isfinite(rs)]
        if len(rs) < 5:
            continue
        obs = float(np.mean(rs))
        # sign-flip null
        ge = 0
        for _ in range(args.n_perm):
            flips = rng.choice([-1.0, 1.0], size=len(rs))
            ge += int(abs(float(np.mean(rs * flips))) >= abs(obs))
        p0 = (ge + 1) / (args.n_perm + 1)
        td = sub.loc[sub["group"] == "TD", "r_rank"].to_numpy(float)
        dys = sub.loc[sub["group"] != "TD", "r_rank"].to_numpy(float)
        td, dys = td[np.isfinite(td)], dys[np.isfinite(dys)]
        hier_summary.append(
            {
                "family_set": set_name,
                "metric_source": src,
                "metric": metric,
                "mean_r": obs,
                "p_vs_0": p0,
                "mean_r_TD": float(td.mean()) if len(td) else np.nan,
                "mean_r_dyslexia": float(dys.mean()) if len(dys) else np.nan,
                "hedges_g_TD_vs_dys": hedges_g(td, dys) if len(td) > 2 and len(dys) > 2 else np.nan,
                "p_TD_vs_dys": perm_p(td, dys, args.n_perm, args.seed)
                if len(td) > 2 and len(dys) > 2
                else np.nan,
                "n_TD": int(len(td)),
                "n_dyslexia": int(len(dys)),
            }
        )
    # Also family-mean metric vs rank (group-level point estimate like Gwilliams r)
    for set_name, fams in sets.items():
        if len(fams) < 3:
            continue
        for src, metric in metrics_for_corr:
            sub = sdf.loc[sdf["metric_source"] == src]
            means = []
            ranks = []
            for fam in fams:
                vals = sub.loc[sub["family"] == fam, metric].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                means.append(float(vals.mean()))
                ranks.append(float(rank[fam]))
            r, p = spearman_or_pearson(np.array(ranks), np.array(means))
            hier_summary.append(
                {
                    "family_set": set_name,
                    "metric_source": src,
                    "metric": f"groupmean_{metric}",
                    "mean_r": r,
                    "p_vs_0": p,
                    "mean_r_TD": np.nan,
                    "mean_r_dyslexia": np.nan,
                    "hedges_g_TD_vs_dys": np.nan,
                    "p_TD_vs_dys": np.nan,
                    "n_TD": np.nan,
                    "n_dyslexia": np.nan,
                }
            )
    hdf = pd.DataFrame(hier_summary)
    hdf.to_csv(out / "hdc_hierarchy_summary.csv", index=False)

    # --- TD vs dyslexia on subject duration/sustain ---
    group_rows = []
    for src in ("tg", "family_agg"):
        sub = sdf.loc[sdf["metric_source"] == src]
        metrics = ["duration_s"]
        if src == "tg":
            metrics.append("sustain_s")
        else:
            metrics.extend(["onset_halfmax_s", "pre_offset_norm_mean"])
        for fam in families:
            fsub = sub.loc[sub["family"] == fam]
            td = fsub.loc[fsub["group"] == "TD"]
            dys = fsub.loc[fsub["group"] != "TD"]
            for metric in metrics:
                a = td[metric].to_numpy(float)
                b = dys[metric].to_numpy(float)
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) < 3 or len(b) < 3:
                    continue
                group_rows.append(
                    {
                        "family": fam,
                        "hierarchy_rank": rank[fam],
                        "metric_source": src,
                        "metric": metric,
                        "mean_TD": float(a.mean()),
                        "mean_dyslexia": float(b.mean()),
                        "hedges_g": hedges_g(a, b),
                        "p_perm": perm_p(a, b, args.n_perm, args.seed),
                        "n_TD": int(len(a)),
                        "n_dyslexia": int(len(b)),
                    }
                )
    gdf = pd.DataFrame(group_rows)
    gdf.to_csv(out / "hdc_duration_sustain_group_sig.csv", index=False)

    manifest = {
        "results_dir": str(results),
        "families": families,
        "hierarchy_rank": rank,
        "z_se_threshold": args.z,
        "cluster_alpha": args.cluster_alpha,
        "n_perm": args.n_perm,
        "n_subjects": int(sdf["participant"].nunique()),
        "primary_sustain_exclude": sorted(PRIMARY_SUSTAIN_EXCLUDE),
        "notes": [
            "Subject above-chance: score > z * (split_sd/sqrt(n_valid_splits))",
            "duration_s = longest contiguous above-chance run",
            "sustain_s = mean TG row width above chance (rows with above-chance diagonal preferred)",
            "Group cluster duration: one-sample t vs 0, positive clusters |t|>t_crit",
            "Primary hierarchy sustain set excludes semantic (Gwilliams Fig. 5D)",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("=== Group cluster duration ===")
    if len(cdf):
        print(cdf.to_string(index=False))
    print("\n=== Hierarchy summary (subject-mean r and group-mean) ===")
    if len(hdf):
        show = hdf.copy()
        print(show.to_string(index=False))
    print("\n=== TD vs dyslexia (sig metrics) ===")
    if len(gdf):
        print(gdf.to_string(index=False))
    print(f"\nWrote outputs under {out}")


if __name__ == "__main__":
    main()
