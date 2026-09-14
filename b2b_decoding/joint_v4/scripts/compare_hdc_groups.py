#!/usr/bin/env python3
"""HDC group analysis: family traces + duration/sustain (TG) vs TD/dyslexia.

1. Delegates family-score traces to compare_b2b_groups.py (observed-only OK).
2. If *_tg_metrics.csv exist, permutation-tests duration and sustain.

Usage:
  python3 scripts/compare_hdc_groups.py \\
      --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot \\
      --groups cohort_groups.csv \\
      --out-dir group_comparison_b2b_hdc_offset_pilot
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FAMILIES = (
    "phonetic",
    "word_form",
    "lexical_syntactic",
    "syntax_proxy",
    "semantic",
)


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
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(pooled[:na].mean() - pooled[na:].mean())
        ge += int(d >= obs)
    return (ge + 1) / (n_perm + 1)


def load_groups(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["include_primary"].astype(bool)].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated family names in hierarchical order",
    )
    ap.add_argument(
        "--score-col",
        default="mean_score",
        choices=("mean_score", "mean_trace", "mean_z"),
        help="HDC default mean_score = mean diag(H) within family (Gwilliams); "
             "mean_trace = Σ diag(H); mean_z = mean_score/split_sd (SNR-normalized)",
    )
    args = ap.parse_args()
    FAMILIES = tuple(x.strip() for x in args.families.split(",") if x.strip())
    HIERARCHY_RANK = {f: i + 1 for i, f in enumerate(FAMILIES)}
    results = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        str(here / "compare_b2b_groups.py"),
        "--results-dir", str(results),
        "--groups", args.groups,
        "--out-dir", str(out),
        "--n-perm", str(args.n_perm),
        "--seed", str(args.seed),
        "--skip-existence-if-no-nulls",
        "--families", ",".join(FAMILIES),
        "--score-col", args.score_col,
    ]
    print("Running", " ".join(cmd))
    subprocess.run(cmd, check=False)

    groups = load_groups(Path(args.groups))
    rows = []
    for _, r in groups.iterrows():
        p = r["participant"]
        path = results / p / f"{p}_tg_metrics.csv"
        if not path.is_file():
            continue
        m = pd.read_csv(path)
        m["participant"] = p
        m["group"] = r["group"]
        rows.append(m)
    if not rows:
        print("No TG metrics found; skipping duration/sustain tests.")
        return
    tg = pd.concat(rows, ignore_index=True)
    tg["dyslexia_pooled"] = tg["group"].ne("TD")
    tg.to_csv(out / "tg_metrics_all.csv", index=False)

    summary = []
    for fam in FAMILIES:
        sub = tg.loc[tg["family"] == fam]
        td = sub.loc[sub["group"] == "TD"]
        dys = sub.loc[sub["dyslexia_pooled"]]
        for metric in ("duration_s", "sustain_s"):
            a = td[metric].to_numpy()
            b = dys[metric].to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            summary.append(
                {
                    "family": fam,
                    "hierarchy_rank": HIERARCHY_RANK[fam],
                    "metric": metric,
                    "mean_TD": float(a.mean()),
                    "mean_dyslexia": float(b.mean()),
                    "hedges_g": hedges_g(a, b),
                    "p_perm": perm_p(a, b, args.n_perm, args.seed),
                    "n_TD": int(len(a)),
                    "n_dyslexia": int(len(b)),
                }
            )
    # Hierarchy correlation: family rank vs mean duration/sustain within subject, then group.
    corr_rows = []
    for p, sp in tg.groupby("participant"):
        sp = sp.set_index("family").reindex(FAMILIES).dropna()
        if len(sp) < 3:
            continue
        ranks = np.array([HIERARCHY_RANK[f] for f in sp.index], float)
        grp = sp["group"].iloc[0]
        for metric in ("duration_s", "sustain_s"):
            if sp[metric].std(ddof=0) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(ranks, sp[metric].to_numpy())[0, 1])
            corr_rows.append(
                {"participant": p, "group": grp, "metric": metric, "r_rank": r}
            )
    if corr_rows:
        cdf = pd.DataFrame(corr_rows)
        cdf.to_csv(out / "hdc_rank_correlations.csv", index=False)
        for metric, sub in cdf.groupby("metric"):
            a = sub.loc[sub["group"] == "TD", "r_rank"].to_numpy()
            b = sub.loc[sub["group"] != "TD", "r_rank"].to_numpy()
            summary.append(
                {
                    "family": "ALL",
                    "hierarchy_rank": np.nan,
                    "metric": f"r_rank_{metric}",
                    "mean_TD": float(a.mean()) if len(a) else np.nan,
                    "mean_dyslexia": float(b.mean()) if len(b) else np.nan,
                    "hedges_g": hedges_g(a, b) if len(a) > 2 and len(b) > 2 else np.nan,
                    "p_perm": perm_p(a, b, args.n_perm, args.seed)
                    if len(a) > 2 and len(b) > 2
                    else np.nan,
                    "n_TD": int(len(a)),
                    "n_dyslexia": int(len(b)),
                }
            )
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "hdc_duration_sustain_group.csv", index=False)
    print(sdf.to_string(index=False))
    (out / "hdc_group_manifest.json").write_text(
        json.dumps({"families": list(FAMILIES), "n_tg_subjects": int(tg["participant"].nunique())}, indent=2)
        + "\n"
    )

    # Significance-gated Gwilliams-style metrics (duration/sustain/hierarchy)
    gw_out = out / "gwilliams_metrics"
    gw_script = here / "scripts" / "recompute_hdc_gwilliams_metrics.py"
    if gw_script.is_file():
        gw_cmd = [
            sys.executable,
            str(gw_script),
            "--results-dir", str(results),
            "--groups", args.groups,
            "--out-dir", str(gw_out),
            "--families", ",".join(FAMILIES),
            "--n-perm", str(args.n_perm),
            "--seed", str(args.seed),
        ]
        print("Running", " ".join(gw_cmd))
        subprocess.run(gw_cmd, check=False)


if __name__ == "__main__":
    main()
