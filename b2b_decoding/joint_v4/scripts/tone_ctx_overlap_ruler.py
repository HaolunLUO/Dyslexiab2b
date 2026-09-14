#!/usr/bin/env python3
"""Event-overlap ruler for nucleus-offset B2B (no EEG, no window change).

t=0 is nucleus offset. Negative lags may still be inside the current nucleus.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "tone_ctx_v1" / "tone_ctx_tokens.parquet"
OUT = ROOT / "tone_ctx_v1"
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
LAGS = np.round(np.arange(-0.20, 0.01, 0.01), 2)


def main() -> None:
    df = pd.read_parquet(TOKENS).sort_values(["section", "nucleus_on"]).reset_index(drop=True)
    dur = (df["nucleus_off"] - df["nucleus_on"]).to_numpy(dtype=np.float64)
    onset_rel = df["nucleus_on"].to_numpy() - df["nucleus_off"].to_numpy()  # ≤ 0
    next_on = np.full(len(df), np.nan)
    for sid, g in df.groupby("section", sort=True):
        idx = g.index.to_numpy()
        off = g["nucleus_off"].to_numpy()
        nxt = g["nucleus_on"].to_numpy()
        next_on[idx[:-1]] = nxt[1:] - off[:-1]
    still = {f"{lag:.2f}": float(np.mean(onset_rel <= lag)) for lag in LAGS}
    isi = []
    for _, g in df.groupby("section", sort=True):
        t = np.sort(g["nucleus_on"].to_numpy(dtype=np.float64))
        isi.append(np.diff(t))
    isi = np.concatenate(isi)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    epoch_s = EPOCH_TMAX - EPOCH_TMIN
    med_isi = float(np.median(isi))
    p10_isi = float(np.percentile(isi, 10))
    min_k_window = int(np.ceil(epoch_s / max(p10_isi, 1e-6)))
    min_k = max(20, min_k_window)
    out = {
        "n_tokens": int(len(df)),
        "epoch_s": [EPOCH_TMIN, EPOCH_TMAX],
        "epoch_span_s": epoch_s,
        "nucleus_duration_s": {
            "p10": float(np.percentile(dur, 10)),
            "p50": float(np.percentile(dur, 50)),
            "p90": float(np.percentile(dur, 90)),
        },
        "nucleus_onset_rel_offset_s": {
            "p10": float(np.percentile(onset_rel, 10)),
            "p50": float(np.percentile(onset_rel, 50)),
            "p90": float(np.percentile(onset_rel, 90)),
        },
        "next_nucleus_onset_rel_offset_s": {
            "p10": float(np.nanpercentile(next_on, 10)),
            "p50": float(np.nanpercentile(next_on, 50)),
            "p90": float(np.nanpercentile(next_on, 90)),
        },
        "prop_still_inside_current_nucleus_at_lag": still,
        "onset_isi_s": {"p10": p10_isi, "p50": med_isi, "p90": float(np.percentile(isi, 90))},
        "min_k_from_window_p10_isi": min_k_window,
        "min_shift_syllables_frozen": min_k,
        "note": (
            "At t=−0.09 s many tokens are still inside the current nucleus. "
            "Describe M2 as temporally nonspecific around offset, not as demonstrated carryover. "
            "At +0.58 s many next nuclei have begun; call that a late overlap-sensitive peak."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlap_ruler.json").write_text(json.dumps(out, indent=2) + "\n")
    rows = [{"lag_s": float(lag), "prop_inside_current_nucleus": still[f"{lag:.2f}"]} for lag in LAGS]
    pd.DataFrame(rows).to_csv(OUT / "overlap_ruler_lags.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("no matplotlib; wrote json/csv only")
        print(json.dumps({k: out[k] for k in (
            "nucleus_duration_s", "nucleus_onset_rel_offset_s",
            "next_nucleus_onset_rel_offset_s", "min_shift_syllables_frozen",
        )}, indent=2))
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    lags = [r["lag_s"] for r in rows]
    props = [r["prop_inside_current_nucleus"] for r in rows]
    ax.plot(lags, props, color="0.15", lw=2)
    ax.axvline(-0.09, color="C1", ls="--", lw=1, label="descriptive M2 lag −0.09 s")
    for q, lab in (("p10", "onset p10"), ("p50", "onset p50"), ("p90", "onset p90")):
        ax.axvline(out["nucleus_onset_rel_offset_s"][q], color="C0", ls=":", lw=1, alpha=0.9)
    ax.set_xlabel("time relative to nucleus offset (s)")
    ax.set_ylabel("proportion still inside current nucleus")
    ax.set_xlim(-0.20, 0.0)
    ax.set_ylim(0, 1.02)
    ax.set_title("Event-overlap ruler (stimulus timing only)")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overlap_ruler.png", dpi=140)
    print(json.dumps({
        "duration_p10_p50_p90": out["nucleus_duration_s"],
        "onset_rel_p10_p50_p90": out["nucleus_onset_rel_offset_s"],
        "next_on_p10_p50_p90": out["next_nucleus_onset_rel_offset_s"],
        "prop_inside_at_-0.09": still["-0.09"],
        "min_shift_syllables": min_k,
    }, indent=2))
    print("wrote", OUT / "overlap_ruler.json")


if __name__ == "__main__":
    main()
