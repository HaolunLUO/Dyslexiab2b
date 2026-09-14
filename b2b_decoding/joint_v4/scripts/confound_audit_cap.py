#!/usr/bin/env python3
"""Confound audit for CAP-subtype B2B finding (dss40_bin20 mean_z arm).

Compares normal vs atypical CAP (and TD contrasts) on EEG QC, demographics,
reading battery, and DSS-benefit vs residual noise.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONFIRM_WIN = (0.0, 0.8)
HDC_FAMILIES = (
    "phonetic",
    "word_form",
    "lexical_syntactic",
    "syntactic_operation",
    "syntactic_state",
    "semantic",
)
READING_ITEMS = [
    "识字量",
    "汉字阅读流畅度",
    "语音意识-音节",
    "语音意识-音位",
    "语音意识-声调",
    "快速命名-数字",
    "快速命名-物体",
    "快速命名-颜色",
    "快速命名-混合",
    "数字分听-自由回忆",
    "噪声间隙（识别门限）",
]


def parse_age(raw) -> float:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return np.nan
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return np.nan
    s = s.replace("；", ";").replace("：", ":").replace(",", ".")
    m = re.match(r"^(\d+)\s*[;:]\s*(\d+)$", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / 12.0
    m = re.match(r"^0*(\d+)\s*[;:]\s*0*(\d+)$", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / 12.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1)
    sp = np.sqrt(sp2) if sp2 > 0 else np.nan
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    J = 1.0 - 3.0 / (4 * (na + nb) - 9) if (na + nb) > 2 else 1.0
    return float(J * d)


def perm_p_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        order = rng.permutation(len(pooled))
        d = abs(pooled[order][:na].mean() - pooled[order][na:].mean())
        ge += int(d >= obs)
    return (ge + 1) / (n_perm + 1)


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)


def load_behaviour_wide(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    idcol, itemcol, valcol = "编号", "测试项目", "测试结果"
    df[valcol] = pd.to_numeric(df[valcol], errors="coerce")
    # strip item names (汉字阅读流畅度 has trailing space in sheet)
    df[itemcol] = df[itemcol].astype(str).str.strip()
    meta = (
        df.groupby(idcol)
        .agg(
            beh_group=("组别", "first"),
            sex=("性别", "first"),
            grade=("年级", "first"),
            age_raw=("年龄", "first"),
        )
        .reset_index()
    )
    meta["age_years"] = meta["age_raw"].apply(parse_age)
    meta = meta.drop(columns="age_raw")
    wide = df.pivot_table(index=idcol, columns=itemcol, values=valcol, aggfunc="mean")
    wide = wide.reset_index()
    out = meta.merge(wide, on=idcol)
    out = out.rename(columns={idcol: "participant"})
    # normalize reading column names with trailing spaces already stripped via item strip
    return out


def window_mean_score(agg: pd.DataFrame, family: str, score_col: str) -> float:
    sub = agg.loc[agg["family"] == family].copy()
    if sub.empty:
        return np.nan
    t = sub["time"].to_numpy(float)
    m = (t >= CONFIRM_WIN[0]) & (t <= CONFIRM_WIN[1])
    if score_col == "mean_z":
        y = sub["mean_score"].to_numpy(float) / np.maximum(sub["split_sd"].to_numpy(float), 1e-12)
    else:
        y = sub[score_col].to_numpy(float)
    y = y[m]
    y = y[np.isfinite(y)]
    return float(np.mean(y)) if len(y) else np.nan


def mean_split_sd(agg: pd.DataFrame) -> float:
    t = agg["time"].to_numpy(float)
    m = (t >= CONFIRM_WIN[0]) & (t <= CONFIRM_WIN[1])
    sd = agg.loc[m, "split_sd"].to_numpy(float)
    sd = sd[np.isfinite(sd)]
    return float(np.mean(sd)) if len(sd) else np.nan


def load_dss_spectrum(path: Path) -> dict:
    if not path.is_file():
        return {"dss_eig_sum": np.nan, "dss_eig1": np.nan, "dss_n_comp": np.nan}
    meta = json.loads(path.read_text())
    eigs = meta.get("spatial_eigenvalues") or []
    eigs = [float(x) for x in eigs if np.isfinite(float(x))]
    return {
        "dss_eig_sum": float(sum(eigs)) if eigs else np.nan,
        "dss_eig1": float(eigs[0]) if eigs else np.nan,
        "dss_n_comp": float(len(eigs)),
        "dss_var_explained": float(meta.get("spatial_var_explained", np.nan)),
    }


def contrast_rows(df: pd.DataFrame, metric: str, contrasts: list[tuple[str, str, str]], n_perm: int):
    rows = []
    for name, ga, gb in contrasts:
        a = df.loc[df["group"] == ga, metric].to_numpy(float)
        b = df.loc[df["group"] == gb, metric].to_numpy(float)
        a_ok = a[np.isfinite(a)]
        b_ok = b[np.isfinite(b)]
        rows.append(
            {
                "metric": metric,
                "contrast": name,
                "n_A": int(len(a_ok)),
                "n_B": int(len(b_ok)),
                "mean_A": float(np.mean(a_ok)) if len(a_ok) else np.nan,
                "mean_B": float(np.mean(b_ok)) if len(b_ok) else np.nan,
                "hedges_g": hedges_g(a_ok, b_ok),
                "welch_p": welch_p(a_ok, b_ok),
                "perm_p": perm_p_diff(a_ok, b_ok, n_perm=n_perm),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20",
    )
    ap.add_argument(
        "--baseline-results-dir",
        default="/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc",
    )
    ap.add_argument(
        "--qc-csv",
        default="/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc/_qc/participant_qc.csv",
    )
    ap.add_argument(
        "--groups",
        default="/home/haolun52/orcd/pool/dyslexia_natualistics_listing/b2b_decoding/joint_v4/cohort_groups.csv",
    )
    ap.add_argument(
        "--behavior",
        default="/home/haolun52/orcd/pool/dyslexia_natualistics_listing/behavioral_data.xlsx",
    )
    ap.add_argument(
        "--out-dir",
        default="validation_cap_dss40_bin20",
    )
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    groups = pd.read_csv(args.groups)
    groups = groups.loc[groups["include_primary"].astype(bool)].copy()
    qc = pd.read_csv(args.qc_csv)
    if "group" not in qc.columns:
        qc = qc.merge(groups[["participant", "group"]], on="participant", how="left")

    beh = load_behaviour_wide(Path(args.behavior))
    # Match reading columns that may have been normalized
    beh_cols = {c.strip(): c for c in beh.columns}
    for item in list(READING_ITEMS):
        if item not in beh.columns and item in beh_cols:
            pass
        elif item not in beh.columns:
            # try fuzzy: strip all whitespace variants
            for c in beh.columns:
                if str(c).strip() == item:
                    beh = beh.rename(columns={c: item})
                    break

    rows = []
    results = Path(args.results_dir)
    baseline = Path(args.baseline_results_dir)
    for _, gr in groups.iterrows():
        p = gr["participant"]
        g = gr["group"]
        agg_path = results / p / f"{p}_b2b_family_agg.csv"
        base_path = baseline / p / f"{p}_b2b_family_agg.csv"
        row = {"participant": p, "group": g}
        q = qc.loc[qc["participant"] == p]
        if len(q):
            for col in (
                "n_usable",
                "n_rejected",
                "keep_frac",
                "mean_residual_noise",
                "median_residual_noise",
                "mean_channel_var",
                "n_valid_partitions",
            ):
                if col in q.columns:
                    row[col] = float(q.iloc[0][col])
        if agg_path.is_file():
            agg = pd.read_csv(agg_path)
            row["mean_split_sd"] = mean_split_sd(agg)
            for fam in HDC_FAMILIES:
                row[f"mean_z_{fam}"] = window_mean_score(agg, fam, "mean_z")
                row[f"mean_score_{fam}"] = window_mean_score(agg, fam, "mean_score")
            # joint mean_z across families (confirm window)
            row["mean_z_joint"] = float(
                np.nanmean([row[f"mean_z_{fam}"] for fam in HDC_FAMILIES])
            )
        if base_path.is_file():
            bagg = pd.read_csv(base_path)
            for fam in HDC_FAMILIES:
                row[f"base_mean_z_{fam}"] = window_mean_score(bagg, fam, "mean_z")
            row["base_mean_z_joint"] = float(
                np.nanmean([row.get(f"base_mean_z_{fam}", np.nan) for fam in HDC_FAMILIES])
            )
            row["dss_benefit_joint"] = row.get("mean_z_joint", np.nan) - row.get(
                "base_mean_z_joint", np.nan
            )
        row.update(load_dss_spectrum(results / p / f"{p}_snr_preprocess.json"))
        b = beh.loc[beh["participant"] == p]
        row["has_behavior"] = bool(len(b))
        if len(b):
            bb = b.iloc[0]
            row["sex"] = bb.get("sex")
            row["grade"] = bb.get("grade")
            row["age_years"] = float(bb.get("age_years")) if pd.notna(bb.get("age_years")) else np.nan
            row["sex_male"] = 1.0 if str(bb.get("sex")) == "男" else (0.0 if str(bb.get("sex")) == "女" else np.nan)
            for item in READING_ITEMS:
                if item in bb.index:
                    row[item] = float(bb[item]) if pd.notna(bb[item]) else np.nan
        rows.append(row)

    subj = pd.DataFrame(rows)
    subj.to_csv(out / "subject_table.csv", index=False)

    missing = subj.loc[~subj["has_behavior"], ["participant", "group"]]
    missing.to_csv(out / "behavior_missing.csv", index=False)

    contrasts = [
        ("dyslexia_normal_vs_atypical_CAP", "dyslexia_normal_CAP", "dyslexia_atypical_CAP"),
        ("TD_vs_dyslexia_pooled", "TD", "dyslexia_pooled"),  # handled specially
        ("TD_vs_dyslexia_normal_CAP", "TD", "dyslexia_normal_CAP"),
        ("TD_vs_dyslexia_atypical_CAP", "TD", "dyslexia_atypical_CAP"),
    ]

    # pooled helper column
    subj2 = subj.copy()
    # For TD vs pooled, create temporary label
    metrics_qc = [
        "n_usable",
        "keep_frac",
        "mean_residual_noise",
        "mean_channel_var",
        "mean_split_sd",
        "dss_eig1",
        "dss_eig_sum",
        "dss_benefit_joint",
        "age_years",
        "sex_male",
    ] + [i for i in READING_ITEMS if i in subj.columns]

    audit_rows = []
    for metric in metrics_qc:
        if metric not in subj.columns:
            continue
        # CAP contrast
        audit_rows.extend(
            contrast_rows(
                subj,
                metric,
                [("dyslexia_normal_vs_atypical_CAP", "dyslexia_normal_CAP", "dyslexia_atypical_CAP")],
                args.n_perm,
            )
        )
        # TD vs each
        for name, ga, gb in [
            ("TD_vs_dyslexia_normal_CAP", "TD", "dyslexia_normal_CAP"),
            ("TD_vs_dyslexia_atypical_CAP", "TD", "dyslexia_atypical_CAP"),
        ]:
            audit_rows.extend(contrast_rows(subj, metric, [(name, ga, gb)], args.n_perm))
        # TD vs pooled
        a = subj.loc[subj["group"] == "TD", metric].to_numpy(float)
        b = subj.loc[subj["group"].str.startswith("dyslexia_"), metric].to_numpy(float)
        a_ok, b_ok = a[np.isfinite(a)], b[np.isfinite(b)]
        audit_rows.append(
            {
                "metric": metric,
                "contrast": "TD_vs_dyslexia_pooled",
                "n_A": int(len(a_ok)),
                "n_B": int(len(b_ok)),
                "mean_A": float(np.mean(a_ok)) if len(a_ok) else np.nan,
                "mean_B": float(np.mean(b_ok)) if len(b_ok) else np.nan,
                "hedges_g": hedges_g(a_ok, b_ok),
                "welch_p": welch_p(a_ok, b_ok),
                "perm_p": perm_p_diff(a_ok, b_ok, n_perm=args.n_perm, seed=args.seed),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "confound_audit.csv", index=False)

    # Critical: noise vs DSS benefit
    dys = subj.loc[subj["group"].str.startswith("dyslexia_")].copy()
    noise_dss = {}
    if "mean_residual_noise" in dys.columns and "dss_benefit_joint" in dys.columns:
        pair = dys[["mean_residual_noise", "dss_benefit_joint", "group"]].dropna()
        if len(pair) >= 8:
            r, p = stats.spearmanr(pair["mean_residual_noise"], pair["dss_benefit_joint"])
            noise_dss = {
                "n": int(len(pair)),
                "spearman_r": float(r),
                "spearman_p": float(p),
                "note": "Does noisier EEG get larger DSS mean_z gain? Positive r would support artifact concern.",
            }
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for gname, marker in [
                ("dyslexia_normal_CAP", "o"),
                ("dyslexia_atypical_CAP", "s"),
            ]:
                sub = pair.loc[pair["group"] == gname]
                ax.scatter(
                    sub["mean_residual_noise"],
                    sub["dss_benefit_joint"],
                    label=gname.replace("dyslexia_", ""),
                    marker=marker,
                    alpha=0.8,
                )
            ax.set_xlabel("mean residual noise (QC)")
            ax.set_ylabel("DSS benefit (mean_z joint − baseline)")
            ax.set_title(f"Noise vs DSS benefit (ρ={r:.2f}, p={p:.3f}, n={len(pair)})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "plots" / "noise_vs_dss_benefit.png", dpi=140)
            plt.close(fig)

    # Brain–behavior among dyslexia with behavior
    bb_rows = []
    decode_cols = ["mean_z_syntactic_state", "mean_z_lexical_syntactic", "mean_z_joint"]
    for dcol in decode_cols:
        if dcol not in dys.columns:
            continue
        for item in READING_ITEMS:
            if item not in dys.columns:
                continue
            pair = dys[[dcol, item]].dropna()
            if len(pair) < 6:
                r, p, n = np.nan, np.nan, len(pair)
            else:
                r, p = stats.spearmanr(pair[dcol], pair[item])
                n = len(pair)
            bb_rows.append(
                {
                    "decode": dcol,
                    "behaviour": item,
                    "spearman_r": float(r) if np.isfinite(r) else np.nan,
                    "spearman_p": float(p) if np.isfinite(p) else np.nan,
                    "n": int(n),
                }
            )
    bb = pd.DataFrame(bb_rows)
    if len(bb):
        bb = bb.sort_values("spearman_p")
        bb.to_csv(out / "brain_behavior_correlations.csv", index=False)

    # Coverage summary
    cov = {
        "n_cohort": int(len(subj)),
        "n_with_behavior": int(subj["has_behavior"].sum()),
        "missing_behavior": missing.to_dict(orient="records"),
        "missing_by_group": missing["group"].value_counts().to_dict() if len(missing) else {},
        "no_iq_in_spreadsheet": True,
        "no_continuous_cap_in_spreadsheet": True,
        "noise_vs_dss_benefit": noise_dss,
        "cap_contrast_highlights": audit.loc[
            audit["contrast"] == "dyslexia_normal_vs_atypical_CAP",
            ["metric", "n_A", "n_B", "mean_A", "mean_B", "hedges_g", "welch_p", "perm_p"],
        ]
        .sort_values("perm_p")
        .head(20)
        .to_dict(orient="records"),
    }
    (out / "confound_audit_summary.json").write_text(json.dumps(cov, indent=2, ensure_ascii=False))

    # Plot age / keep_frac / residual noise by CAP
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, metric, title in zip(
        axes,
        ["age_years", "keep_frac", "mean_residual_noise"],
        ["Age (years)", "Keep fraction", "Mean residual noise"],
    ):
        if metric not in subj.columns:
            ax.set_visible(False)
            continue
        data, labels = [], []
        for gname in ["TD", "dyslexia_normal_CAP", "dyslexia_atypical_CAP"]:
            v = subj.loc[subj["group"] == gname, metric].dropna().to_numpy(float)
            data.append(v)
            labels.append(gname.replace("dyslexia_", "").replace("_CAP", ""))
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(out / "plots" / "confound_boxplots.png", dpi=140)
    plt.close(fig)

    print(f"Wrote {out / 'confound_audit.csv'} ({len(audit)} rows)")
    print(f"Behavior coverage: {cov['n_with_behavior']}/{cov['n_cohort']}")
    print(f"Missing: {cov['missing_by_group']}")
    if noise_dss:
        print(
            f"Noise vs DSS benefit: ρ={noise_dss['spearman_r']:.3f} p={noise_dss['spearman_p']:.3f} n={noise_dss['n']}"
        )
    # Print top CAP contrasts by effect size
    cap = audit.loc[audit["contrast"] == "dyslexia_normal_vs_atypical_CAP"].copy()
    cap = cap.reindex(cap["hedges_g"].abs().sort_values(ascending=False).index)
    print("\nCAP contrast (|g| ranked):")
    print(cap[["metric", "hedges_g", "welch_p", "perm_p", "n_A", "n_B"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
