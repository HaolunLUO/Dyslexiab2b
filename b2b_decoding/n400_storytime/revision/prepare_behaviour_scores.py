#!/usr/bin/env python3
"""Subject-level Sheet1 scores for the revision behaviour models.

Writes raw scores only. Z-scoring happens in fit_behaviour.R on the
children who enter each model. No names are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
BEHAVIOR = POOL / "dyslexia_natualistics_listing" / "behavioral_data.xlsx"
OUT = POOL / "n400_storytime_v1" / "revision" / "behaviour" / "raw_scores.csv"

PA_COLS = ("语音意识-声调", "语音意识-音位", "语音意识-音节")
RAN_COLS = ("快速命名-数字", "快速命名-混合", "快速命名-物体", "快速命名-颜色")
KEEP = {
    "识字量": "literacy",
    "噪声间隙（识别门限）": "gap",
    "数字分听-自由回忆": "dichotic",
    "快速命名-物体": "ran_object",
    "语音意识-声调": "pa_tone",
    "语音意识-音位": "pa_phoneme",
    "语音意识-音节": "pa_syllable",
    "快速命名-数字": "ran_digits",
    "快速命名-混合": "ran_mixed",
    "快速命名-颜色": "ran_colors",
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


def main() -> None:
    raw = pd.read_excel(BEHAVIOR, sheet_name="Sheet1")
    df = raw.rename(columns={c: str(c).strip() for c in raw.columns})
    measure = (
        df["测试项目"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    )
    out = pd.DataFrame(
        {
            "participant": df["编号"].astype(str).str.strip(),
            "measure": measure,
            "score": pd.to_numeric(df["测试结果"], errors="coerce"),
        }
    )
    out = out.loc[out["participant"].ne("") & out["participant"].ne("nan")]
    nunq = out.groupby(["participant", "measure"])["score"].nunique(dropna=False)
    conflicts = nunq[nunq > 1]
    if len(conflicts):
        raise SystemExit(f"conflicting scores: {conflicts.index.tolist()}")
    wide = out.pivot_table(
        index="participant", columns="measure", values="score", aggfunc="mean"
    )
    keep = [c for c in KEEP if c in wide.columns]
    missing = [c for c in KEEP if c not in wide.columns]
    slim = wide[keep].rename(columns=KEEP).reset_index()
    slim = slim.rename(columns={"participant": "subject"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(OUT, index=False)
    meta = {
        "n_subjects": int(slim.shape[0]),
        "columns": list(slim.columns),
        "missing_measures": missing,
        "n_sheet_test_names": int(measure.nunique()),
        "ran_subtests": list(RAN_COLS),
        "pa_subtests": list(PA_COLS),
        "note": "Raw scores. Direction of RAN and 短文阅读 is unresolved.",
    }
    (OUT.parent / "raw_scores_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", OUT, slim.shape)


if __name__ == "__main__":
    main()
