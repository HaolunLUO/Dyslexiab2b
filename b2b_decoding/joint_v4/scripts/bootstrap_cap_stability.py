#!/usr/bin/env python3
"""Subject-level bootstrap / split-half / leave-one-out stability for CAP contrast.

Resamples subjects only (no Julia refits). Uses the same joint cluster statistic
as compare_b2b_groups.label_perm_contrast_multifamily.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import compare_b2b_groups as cbg  # noqa: E402

HDC_FAMILIES = (
    "phonetic",
    "word_form",
    "lexical_syntactic",
    "syntactic_operation",
    "syntactic_state",
    "semantic",
)


def load_group_mats(results_dir: str, groups: pd.DataFrame, score_col: str):
    cbg.FAMILIES = HDC_FAMILIES
    times = None
    mats = {}
    for gname in ("dyslexia_normal_CAP", "dyslexia_atypical_CAP"):
        ps = groups.loc[groups["group"] == gname, "participant"].tolist()
        fam_mats = {}
        for fam in HDC_FAMILIES:
            t, M = cbg.load_family_matrix(results_dir, ps, fam, score_col=score_col)
            fam_mats[fam] = M
            times = t if times is None else times
        mats[gname] = {"participants": ps, "mats": fam_mats}
    return mats, times


def subset_mats(group_blob: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    return {f: group_blob["mats"][f][idx] for f in HDC_FAMILIES}


def run_contrast(mats_A, mats_B, times, n_perm: int, seed: int) -> dict:
    cbg.FAMILIES = HDC_FAMILIES
    return cbg.label_perm_contrast_multifamily(
        mats_A, mats_B, times, n_perm=n_perm, seed=seed, win=cbg.CONFIRM_WIN
    )


def hedges_from_result(res: dict) -> dict[str, float]:
    return {f: float(res["per_family"][f]["hedges_g"]) for f in HDC_FAMILIES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20",
    )
    ap.add_argument(
        "--groups",
        default=str(HERE / "cohort_groups.csv"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(HERE / "validation_cap_dss40_bin20" / "bootstrap"),
    )
    ap.add_argument("--score-col", default="mean_z", choices=("mean_z", "mean_score"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm-boot", type=int, default=1000)
    ap.add_argument("--n-split", type=int, default=200)
    ap.add_argument("--n-perm-split", type=int, default=500)
    ap.add_argument("--n-perm-loo", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-boot", action="store_true")
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    ap.add_argument("--time-one", action="store_true", help="Time one contrast then exit")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(args.groups)
    groups = groups.loc[groups["include_primary"].astype(bool)].copy()

    mats, times = load_group_mats(args.results_dir, groups, args.score_col)
    nA = len(mats["dyslexia_normal_CAP"]["participants"])
    nB = len(mats["dyslexia_atypical_CAP"]["participants"])
    print(f"Loaded CAP mats: normal={nA} atypical={nB} score={args.score_col}")

    # Full-sample reference
    t0 = time.time()
    full = run_contrast(
        mats["dyslexia_normal_CAP"]["mats"],
        mats["dyslexia_atypical_CAP"]["mats"],
        times,
        n_perm=args.n_perm_boot,
        seed=args.seed,
    )
    dt = time.time() - t0
    print(f"Full-sample p={full['p_cluster_joint']:.4f} mass={full['obs_max_mass_joint']:.3f}  ({dt:.1f}s for n_perm={args.n_perm_boot})")
    if args.time_one:
        print(json.dumps({"seconds_per_contrast": dt, "n_perm": args.n_perm_boot}))
        return

    summary = {
        "score_col": args.score_col,
        "full_sample": {
            "p_cluster_joint": full["p_cluster_joint"],
            "obs_max_mass_joint": full["obs_max_mass_joint"],
            "hedges_g": hedges_from_result(full),
        },
        "seconds_per_contrast_nperm": {str(args.n_perm_boot): dt},
    }

    rng = np.random.default_rng(args.seed)

    # ---- Bootstrap ----
    if not args.skip_boot:
        boot_rows = []
        print(f"Bootstrap B={args.n_boot} n_perm={args.n_perm_boot} …")
        for b in range(args.n_boot):
            idxA = rng.integers(0, nA, nA)
            idxB = rng.integers(0, nB, nB)
            res = run_contrast(
                subset_mats(mats["dyslexia_normal_CAP"], idxA),
                subset_mats(mats["dyslexia_atypical_CAP"], idxB),
                times,
                n_perm=args.n_perm_boot,
                seed=args.seed + 1000 + b,
            )
            row = {
                "boot": b,
                "p_cluster_joint": res["p_cluster_joint"],
                "obs_max_mass_joint": res["obs_max_mass_joint"],
                "sig_05": int(res["p_cluster_joint"] < 0.05),
            }
            for f, g in hedges_from_result(res).items():
                row[f"g_{f}"] = g
            boot_rows.append(row)
            if (b + 1) % 25 == 0 or b == 0:
                ps = [r["p_cluster_joint"] for r in boot_rows]
                print(
                    f"  boot {b+1}/{args.n_boot}  median_p={np.median(ps):.4f}  "
                    f"frac_sig={np.mean([r['sig_05'] for r in boot_rows]):.2f}"
                )
        boot = pd.DataFrame(boot_rows)
        boot.to_csv(out / f"bootstrap_{args.score_col}.csv", index=False)
        ps = boot["p_cluster_joint"].to_numpy()
        summary["bootstrap"] = {
            "B": args.n_boot,
            "n_perm": args.n_perm_boot,
            "median_p": float(np.median(ps)),
            "mean_p": float(np.mean(ps)),
            "p_ci95": [float(np.quantile(ps, 0.025)), float(np.quantile(ps, 0.975))],
            "frac_p_lt_05": float(np.mean(ps < 0.05)),
            "frac_p_lt_01": float(np.mean(ps < 0.01)),
            "median_mass": float(np.median(boot["obs_max_mass_joint"])),
        }
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.hist(ps, bins=30, color="#4C78A8", edgecolor="white")
        ax.axvline(0.05, color="#E45756", ls="--", label="α=0.05")
        ax.axvline(np.median(ps), color="#72B7B2", ls="-", label=f"median={np.median(ps):.3f}")
        ax.set_xlabel("joint cluster p (bootstrap)")
        ax.set_ylabel("count")
        ax.set_title(f"Subject bootstrap ({args.score_col})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"bootstrap_p_hist_{args.score_col}.png", dpi=140)
        plt.close(fig)

    # ---- Split-half ----
    if not args.skip_split:
        split_rows = []
        print(f"Split-half S={args.n_split} n_perm={args.n_perm_split} …")
        for s in range(args.n_split):
            # stratified 50/50 within each CAP group
            permA = rng.permutation(nA)
            permB = rng.permutation(nB)
            halfA = max(nA // 2, 2)
            halfB = max(nB // 2, 2)
            for half, idxA, idxB in [
                (1, permA[:halfA], permB[:halfB]),
                (2, permA[halfA:], permB[halfB:]),
            ]:
                res = run_contrast(
                    subset_mats(mats["dyslexia_normal_CAP"], idxA),
                    subset_mats(mats["dyslexia_atypical_CAP"], idxB),
                    times,
                    n_perm=args.n_perm_split,
                    seed=args.seed + 5000 + s * 2 + half,
                )
                gdict = hedges_from_result(res)
                split_rows.append(
                    {
                        "split": s,
                        "half": half,
                        "n_A": len(idxA),
                        "n_B": len(idxB),
                        "p_cluster_joint": res["p_cluster_joint"],
                        "obs_max_mass_joint": res["obs_max_mass_joint"],
                        "g_syntactic_state": gdict["syntactic_state"],
                        "g_lexical_syntactic": gdict["lexical_syntactic"],
                        "g_mean": float(np.nanmean(list(gdict.values()))),
                    }
                )
            if (s + 1) % 20 == 0 or s == 0:
                print(f"  split {s+1}/{args.n_split}")
        split = pd.DataFrame(split_rows)
        split.to_csv(out / f"split_half_{args.score_col}.csv", index=False)
        # consistency: correlation of mean g across halves
        wide = split.pivot(index="split", columns="half", values="g_mean")
        if 1 in wide.columns and 2 in wide.columns:
            r = float(np.corrcoef(wide[1], wide[2])[0, 1]) if len(wide) > 2 else np.nan
        else:
            r = np.nan
        same_sign = float(np.mean(np.sign(wide[1]) == np.sign(wide[2]))) if len(wide) else np.nan
        summary["split_half"] = {
            "n_splits": args.n_split,
            "n_perm": args.n_perm_split,
            "corr_mean_g_half1_half2": r,
            "frac_same_sign_mean_g": same_sign,
            "median_p_half1": float(np.median(split.loc[split["half"] == 1, "p_cluster_joint"])),
            "median_p_half2": float(np.median(split.loc[split["half"] == 2, "p_cluster_joint"])),
            "frac_sig_either_half": float(
                split.groupby("split")["p_cluster_joint"].min().lt(0.05).mean()
            ),
        }

    # ---- Leave-one-out ----
    if not args.skip_loo:
        loo_rows = []
        print(f"LOO over {nA + nB} subjects n_perm={args.n_perm_loo} …")
        for group_key, n in [("dyslexia_normal_CAP", nA), ("dyslexia_atypical_CAP", nB)]:
            ps = mats[group_key]["participants"]
            for i in range(n):
                keep = np.ones(n, dtype=bool)
                keep[i] = False
                other = "dyslexia_atypical_CAP" if group_key == "dyslexia_normal_CAP" else "dyslexia_normal_CAP"
                if group_key == "dyslexia_normal_CAP":
                    mats_A = subset_mats(mats[group_key], np.where(keep)[0])
                    mats_B = mats[other]["mats"]
                else:
                    mats_A = mats[other]["mats"]
                    mats_B = subset_mats(mats[group_key], np.where(keep)[0])
                res = run_contrast(
                    mats_A, mats_B, times, n_perm=args.n_perm_loo, seed=args.seed + 9000 + i
                )
                loo_rows.append(
                    {
                        "left_out": ps[i],
                        "left_out_group": group_key,
                        "p_cluster_joint": res["p_cluster_joint"],
                        "obs_max_mass_joint": res["obs_max_mass_joint"],
                        "delta_p": res["p_cluster_joint"] - full["p_cluster_joint"],
                        "delta_mass": res["obs_max_mass_joint"] - full["obs_max_mass_joint"],
                    }
                )
                if (len(loo_rows) % 10) == 0:
                    print(f"  LOO {len(loo_rows)}/{nA+nB}")
        loo = pd.DataFrame(loo_rows).sort_values("delta_p", ascending=False)
        loo.to_csv(out / f"loo_{args.score_col}.csv", index=False)
        summary["loo"] = {
            "n_perm": args.n_perm_loo,
            "max_delta_p": float(loo["delta_p"].max()),
            "min_delta_p": float(loo["delta_p"].min()),
            "most_influential_raise_p": loo.iloc[0][["left_out", "left_out_group", "p_cluster_joint", "delta_p"]].to_dict(),
            "most_influential_lower_p": loo.iloc[-1][["left_out", "left_out_group", "p_cluster_joint", "delta_p"]].to_dict(),
            "frac_still_sig_05": float((loo["p_cluster_joint"] < 0.05).mean()),
        }
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(range(len(loo)), loo["p_cluster_joint"].to_numpy(), color="#4C78A8")
        ax.axhline(0.05, color="#E45756", ls="--")
        ax.set_xlabel("leave-one-out index (sorted by Δp)")
        ax.set_ylabel("joint cluster p")
        ax.set_title(f"LOO stability ({args.score_col})")
        fig.tight_layout()
        fig.savefig(out / f"loo_p_{args.score_col}.png", dpi=140)
        plt.close(fig)

    (out / f"stability_summary_{args.score_col}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
