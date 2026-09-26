#!/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
"""Join channel DVs, item predictors, group, and behaviour scores.

Does not fit a model. Drops the child's name. Writes lmm_input.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
V1 = POOL / "n400_storytime_v1" / "v1_strict"
WORD = V1 / "stimulus" / "word_table.csv"
TRIALS = V1 / "trials"
BEHAVIOR = POOL / "dyslexia_natualistics_listing" / "behavioral_data.xlsx"
GROUPS = (
    POOL
    / "dyslexia_natualistics_listing"
    / "b2b_decoding"
    / "joint_v4"
    / "cohort_groups.csv"
)
SUBJECTS = POOL / "dyslexia_natualistics_listing" / "b2b_decoding" / "subjects.txt"

MEASURES = {
    "识字量": "literacy",
    "语音意识-声调": "pa_tone",
    "语音意识-音位": "pa_phon",
    "语音意识-音节": "pa_syl",
    "快速命名-数字": "ran_digit",
    "快速命名-混合": "ran_mixed",
    "快速命名-物体": "ran_object",
    "快速命名-颜色": "ran_color",
    "汉字阅读流畅度": "fluency",
    "噪声间隙（识别门限）": "gap",
    "数字分听-自由回忆": "dichotic",
}


def parse_age(v) -> float:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return float("nan")
    s = str(v).strip().replace("；", ";").replace("：", ":")
    if s == "" or s.lower() == "nan":
        return float("nan")
    if ";" in s:
        a, b = s.split(";", 1)
        try:
            return float(a) + float(b) / 12.0
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def behaviour() -> pd.DataFrame:
    raw = pd.read_excel(BEHAVIOR, sheet_name="Sheet1")
    df = raw.rename(columns={c: str(c).strip() for c in raw.columns})
    age = pd.DataFrame(
        {
            "subject": df["编号"].astype(str).str.strip(),
            "age": [parse_age(v) for v in df["年龄"]],
        }
    )
    age = age.groupby("subject", as_index=False)["age"].first()
    measure = df["测试项目"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    long = pd.DataFrame(
        {
            "subject": df["编号"].astype(str).str.strip(),
            "measure": measure,
            "score": pd.to_numeric(df["测试结果"], errors="coerce"),
        }
    )
    long = long.loc[long["subject"].ne("") & long["subject"].ne("nan")]
    nunq = long.groupby(["subject", "measure"])["score"].nunique(dropna=False)
    if (nunq > 1).any():
        raise SystemExit(f"conflicting scores: {nunq[nunq > 1].index.tolist()[:8]}")
    wide = long.pivot_table(index="subject", columns="measure", values="score", aggfunc="mean")
    missing = [c for c in MEASURES if c not in wide.columns]
    if missing:
        raise SystemExit(f"missing behaviour measures: {missing}")
    slim = wide[list(MEASURES)].rename(columns=MEASURES).reset_index()
    out = age.merge(slim, on="subject", how="outer")
    return out


def main() -> None:
    subjects = [ln.strip() for ln in SUBJECTS.read_text().splitlines() if ln.strip()]
    frames = []
    missing = []
    for subject in subjects:
        path = TRIALS / subject / "dv.csv"
        if not path.is_file():
            missing.append(subject)
            continue
        dv = pd.read_csv(path)
        dv["subject"] = subject
        frames.append(dv)
    if missing:
        raise SystemExit(f"missing dv.csv for {len(missing)} subjects, e.g. {missing[:5]}")
    dv = pd.concat(frames, ignore_index=True)
    word = pd.read_csv(WORD)
    word = word.loc[word["in_model"].astype(bool)].copy()
    merged = dv.merge(word, on=["section", "word_index"], how="inner", validate="many_to_one")
    if len(merged) == 0:
        raise SystemExit("no rows after joining DVs to in_model items")
    groups = pd.read_csv(GROUPS)
    groups = groups.loc[groups["include_primary"].astype(int) == 1, ["participant", "group"]]
    groups = groups.rename(columns={"participant": "subject"})
    groups["subject"] = groups["subject"].astype(str)
    merged = merged.merge(groups, on="subject", how="left")
    if merged["group"].isna().any():
        raise SystemExit("trial subject missing a primary group")
    beh = behaviour()
    merged = merged.merge(beh, on="subject", how="left")
    gmap = {"TD": 0, "dyslexia_normal_CAP": 1, "dyslexia_atypical_CAP": 1}
    cmap = {"dyslexia_normal_CAP": 0, "dyslexia_atypical_CAP": 1}
    unknown = sorted(set(merged["group"]) - set(gmap))
    if unknown:
        raise SystemExit(f"unknown groups: {unknown}")
    merged["g01"] = merged["group"].map(gmap).astype(int)
    merged["cap01"] = merged["group"].map(cmap)
    merged["item"] = merged["item_id"].astype(str)
    z_cols = [
        "freq_z", "assoc_z", "cloze_z", "conc_z", "duration_z", "pos_sent_z", "sent_story_z",
        "freq_m1_z", "freq_p1_z", "assoc_m1_z", "assoc_p1_z", "cloze_m1_z", "cloze_p1_z",
    ]
    dv_cols = ["dv_cp", "dv_fz", "dv_cz", "dv_pz", "dv_oz"]
    if not np.isfinite(merged[z_cols + dv_cols].to_numpy(float)).all():
        raise SystemExit("non-finite predictor or DV in the model table")
    keep = [
        "subject", "item", "section", "word_index", "group", "g01", "cap01",
        *dv_cols, *z_cols,
        "age", "literacy", "pa_tone", "pa_phon", "pa_syl",
        "ran_digit", "ran_mixed", "ran_object", "ran_color",
        "fluency", "gap", "dichotic",
    ]
    out = TRIALS / "lmm_input.csv"
    merged[keep].to_csv(out, index=False)
    qc = {
        "n_trials": int(len(merged)),
        "n_subjects": int(merged["subject"].nunique()),
        "n_items": int(merged["item"].nunique()),
        "n_by_group": {k: int(v) for k, v in merged.groupby("group")["subject"].nunique().items()},
        "n_trials_by_group": {k: int(v) for k, v in merged["group"].value_counts().items()},
        "out": str(out),
    }
    (TRIALS / "assemble_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc))


if __name__ == "__main__":
    main()
