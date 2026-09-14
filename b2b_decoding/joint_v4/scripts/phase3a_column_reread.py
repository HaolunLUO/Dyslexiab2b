#!/usr/bin/env python3
"""Phase 3a: free re-reads of existing H diagonals (no new features / fits).

Pitch → level vs dynamic; tone → one-hots vs tone_dev.
Per-column table normalised by family width (6-D vs 13-D trap).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "phase3a"

COL_NAMES = {
    "envelope": [
        "env_mean", "env_max", "env_std", "env_slope_50ms",
        "env_t00", "env_t01", "env_t02", "env_t03", "env_t04",
        "env_t05", "env_t06", "env_t07", "env_t08", "env_t09",
    ],
    "pitch": [
        "f0_mean", "f0_std", "f0_range",
        "f0_t00", "f0_t01", "f0_t02", "f0_t03", "f0_t04",
        "f0_t05", "f0_t06", "f0_t07", "f0_t08", "f0_t09",
    ],
    "tone": ["tone_1", "tone_2", "tone_3", "tone_4", "tone_dev", "tone_dev_valid"],
    "offset": ["offset_at_t0", "offset_max_50ms", "offset_mean_50ms", "isi"],
    "frequency": ["logfreq"],
    "surprisal": ["surprisal"],
}
# Sub-families for existence re-read (sum of column nullsub).
SUB = {
    "pitch": {"level": ["f0_mean"], "dynamic": [
        "f0_std", "f0_range",
        "f0_t00", "f0_t01", "f0_t02", "f0_t03", "f0_t04",
        "f0_t05", "f0_t06", "f0_t07", "f0_t08", "f0_t09",
    ]},
    "tone": {
        "onehot": ["tone_1", "tone_2", "tone_3", "tone_4"],
        "dev": ["tone_dev", "tone_dev_valid"],
    },
}


def _sign_flip_p(x: np.ndarray, n_perm: int = 10_000, seed: int = 42) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, x.size))
    null = np.abs((signs * x).mean(axis=1))
    return float((np.count_nonzero(null >= obs) + 1) / (n_perm + 1))


def _attach_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    names = []
    for _, r in out.iterrows():
        fam, col = r["family"], int(r["column"])
        if col == 0:
            names.append(f"{fam}_trace")
            continue
        lst = COL_NAMES.get(fam, [])
        names.append(lst[col - 1] if 0 < col <= len(lst) else r["feat_name"])
    out["col_name"] = names
    return out


def frame_tables(frame: str) -> None:
    scores = pd.read_csv(HERE / "existence" / f"scores_{frame}.csv")
    scores = _attach_names(scores)
    cols = scores[scores["column"] != 0].copy()
    cols["nullsub_per_col"] = cols["nullsub"]
    # Width-normalised: family-sum / K already in column==0 as mean_score analogue.
    # Here report per-column mean nullsub and z so 13-D pitch is not compared raw to 6-D tone.
    summary = (
        cols.groupby(["family", "col_name"], sort=False)
        .agg(
            n=("nullsub", "count"),
            mean_nullsub=("nullsub", "mean"),
            median_z=("z", "median"),
            pct_pos=("nullsub", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    summary["frame"] = frame
    summary.to_csv(OUT / f"column_scores_{frame}.csv", index=False)

    sub_rows = []
    for fam, mapping in SUB.items():
        fam_c = cols[cols["family"] == fam]
        parts = sorted(fam_c["participant"].unique())
        for sub, names in mapping.items():
            xs = []
            for p in parts:
                g = fam_c[(fam_c["participant"] == p) & (fam_c["col_name"].isin(names))]
                xs.append(float(g["nullsub"].sum()) if len(g) else float("nan"))
            xs = np.asarray(xs, float)
            pval = _sign_flip_p(xs)
            sub_rows.append(dict(
                frame=frame, family=fam, subfamily=sub, n_cols=len(names),
                n=int(np.isfinite(xs).sum()),
                mean_nullsub=float(np.nanmean(xs)),
                median_z=float(np.nanmedian(
                    fam_c[fam_c["col_name"].isin(names)].groupby("participant")["z"].mean()
                )),
                pct_pos=float(np.nanmean(xs > 0)),
                p_signflip=pval,
                exists=bool(pval < 0.05) if np.isfinite(pval) else False,
            ))
    return summary, pd.DataFrame(sub_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, subs = [], []
    for frame in ("onset", "offset"):
        sm, su = frame_tables(frame)
        summaries.append(sm)
        subs.append(su)
    all_sum = pd.concat(summaries, ignore_index=True)
    all_sub = pd.concat(subs, ignore_index=True)
    all_sum.to_csv(OUT / "column_scores.csv", index=False)
    all_sub.to_csv(OUT / "subfamily_existence.csv", index=False)

    lines = [
        "# Phase 3a — H-diagonal re-reads (no new features)",
        "",
        "Per-column `nullsub` from Phase 1. Family-sum existence is Gate 1;",
        "these rows ask which columns carry it. Do not compare raw sums across",
        "families of different width (13-D pitch vs 6-D tone).",
        "",
        "## Sub-family existence (sum of column nullsub, sign-flip, n = 63)",
        "",
        all_sub.to_string(index=False, float_format=lambda x: f"{x:.4g}"),
        "",
        "## Onset per-column mean nullsub (primary frame)",
        "",
    ]
    on = all_sum[all_sum["frame"] == "onset"]
    for fam, g in on.groupby("family", sort=False):
        lines.append(f"### {fam} (K = {len(g)})")
        lines.append(g[["col_name", "mean_nullsub", "median_z", "pct_pos"]]
                     .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        lines.append("")
    (OUT / "column_reread.md").write_text("\n".join(lines))
    print(all_sub.to_string(index=False))
    print("wrote", OUT / "column_reread.md")


if __name__ == "__main__":
    main()
