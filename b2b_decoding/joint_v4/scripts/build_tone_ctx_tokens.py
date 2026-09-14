#!/usr/bin/env python3
"""Build one-row-per-nucleus token table for tone_ctx_v1.

Writes joint_v4/tone_ctx_v1/tone_ctx_tokens.parquet and sandhi_audit.csv.
Does not touch frozen X_word_tone_v3.npy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tone_ctx_lib as L  # noqa: E402


def _load_word_extras(shared: Path, section: int) -> dict[int, dict]:
    timing = pd.read_csv(shared / f"section_{section:03d}" / "word_timing_relative.csv")
    surp_path = shared / f"section_{section:03d}" / "X_word_gpt2cn_surprisal.npy"
    surprisal = np.load(surp_path).reshape(-1) if surp_path.is_file() else np.zeros(len(timing))
    out = {}
    for i, row in timing.iterrows():
        out[int(i)] = {
            "word": str(row["word"]),
            "logfreq": float(row["logfreq"]) if np.isfinite(row["logfreq"]) else 0.0,
            "surprisal": float(surprisal[i]) if i < surprisal.size else 0.0,
            "word_on": float(row["onset_relative"]),
            "word_off": float(row["offset_relative"]),
        }
    return out


def _load_envelope(cache: Path, section: int) -> np.ndarray:
    path = cache / f"section_{section:03d}__fs{int(L.FS)}.npz"
    with np.load(path) as z:
        return np.asarray(z["envelope"], dtype=np.float64).ravel()


def collect_rows(shared: Path, cache: Path, tg: Path) -> list[dict]:
    v2 = L.load_v2()
    v3 = L.load_v3()
    v1 = v2._load_v1()
    mfa = v1._load_mfa_mod()
    rows: list[dict] = []
    for sid in L.SECTIONS:
        sec = v2.extract_section(v1, mfa, shared, cache, tg, sid)
        extras = _load_word_extras(shared, sid)
        env = _load_envelope(cache, sid)
        local_reg = L.running_median_logf0(sec["f0_hz"], sec["voiced"])
        voiced_log = np.where(
            sec["voiced"] & np.isfinite(sec["f0_hz"]) & (sec["f0_hz"] > 0),
            np.log(sec["f0_hz"]),
            np.nan,
        )
        speaker_reg = float(np.nanmedian(voiced_log)) if np.isfinite(voiced_log).any() else 0.0
        sylls = sec["all_sylls"]
        story_end = max(s["offset"] for s in sylls) if sylls else 1.0
        syl_i_in_word: dict[int, int] = {}
        for s in sylls:
            wi = int(s.get("word_i", 0))
            syl_i_in_word[wi] = syl_i_in_word.get(wi, -1) + 1
            s["_syl_i"] = syl_i_in_word[wi]

        for i, s in enumerate(sylls):
            lex = s.get("lexical_tone")
            surf = s.get("surface_tone")
            nxt_lex = sylls[i + 1].get("lexical_tone") if i + 1 < len(sylls) else None
            prev_surf = sylls[i - 1].get("surface_tone") if i > 0 else None
            next_surf = sylls[i + 1].get("surface_tone") if i + 1 < len(sylls) else None
            timed = s.get("phones_timed") or []
            rw = s.get("rime")
            if rw is None:
                continue
            nuc_on, nuc_off, nuc_phone = rw
            if i > 0 and nuc_on + 1e-9 < rows[-1]["nucleus_on"] and rows[-1]["section"] == sid:
                # keep later sort; do not drop
                pass
            init, final = L.split_initial_final(timed, v2)
            contour = s.get("contour")
            c3 = v3.legendre_coeffs(contour, n_coef=4) if contour is not None else None
            c2 = v3.legendre_coeffs(contour, n_coef=3) if contour is not None else None
            stats = L.contour_stats(contour)
            prev_stats = L.contour_stats(sylls[i - 1].get("contour") if i > 0 else None)
            meta = s.get("rime_meta") or {}
            env_mean, env_max, env_std, env_slope = L.slice_stats(env, nuc_on, nuc_off)
            a = max(0, int(round(nuc_on * L.FS)))
            loc = float(local_reg[min(a, local_reg.size - 1)]) if local_reg.size else speaker_reg
            if not np.isfinite(loc):
                loc = speaker_reg
            prev_off = float(sylls[i - 1]["offset"]) if i > 0 else float(s["onset"])
            gap = float(s["onset"] - prev_off)
            # speech rate: syllables in ±1 s / 2
            win = [
                t for t in sylls
                if abs(0.5 * (t["onset"] + t["offset"]) - 0.5 * (s["onset"] + s["offset"])) <= 1.0
            ]
            speech_rate = float(len(win) / 2.0)
            last_pause = 0.0
            for j in range(i, -1, -1):
                if j == 0 or (sylls[j]["onset"] - sylls[j - 1]["offset"]) >= L.PAUSE_GAP_S:
                    last_pause = float(sylls[j]["onset"])
                    break
            next_pause = story_end
            for j in range(i, len(sylls)):
                if j + 1 >= len(sylls) or (sylls[j + 1]["onset"] - sylls[j]["offset"]) >= L.PAUSE_GAP_S:
                    next_pause = float(sylls[j]["offset"])
                    break
            phrase_span = max(next_pause - last_pause, 1e-3)
            wextra = extras.get(int(s.get("word_i", -1)), {})
            rows.append({
                "section": sid,
                "word_i": int(s.get("word_i", -1)),
                "syllable_i": int(s.get("_syl_i", 0)),
                "syllable_id": f"s{sid}_{int(s.get('word_i', 0)):04d}_{int(s.get('_syl_i', 0))}",
                "word": wextra.get("word", s.get("word") or ""),
                "char": s.get("char") or "",
                "base_syllable": L.base_syllable(s.get("char") or "", s.get("phones") or []),
                "nucleus_on": float(nuc_on),
                "nucleus_off": float(nuc_off),
                "syl_on": float(s["onset"]),
                "syl_off": float(s["offset"]),
                "nucleus_phone": nuc_phone,
                "underlying_tone": lex or "",
                "surface_tone": surf or "",
                "sandhi_type": L.sandhi_type(s.get("char") or "", lex, surf, nxt_lex),
                "prev_surface_tone": prev_surf or "",
                "next_surface_tone": next_surf or "",
                "initial": init,
                "final": final,
                "duration": float(nuc_off - nuc_on),
                "F0_valid_fraction": float(meta.get("voiced_frac") or 0.0),
                "f0_valid": bool(contour is not None),
                "preceding_pause": max(gap, 0.0),
                "phrase_position": float((s["onset"] - last_pause) / phrase_span),
                "speech_rate": speech_rate,
                "speaker_register": speaker_reg,
                "local_register": loc,
                "c0": float(c2[0]) if c2 is not None else np.nan,
                "c1": float(c2[1]) if c2 is not None else np.nan,
                "c2": float(c2[2]) if c2 is not None else np.nan,
                "c3": float(c3[3]) if c3 is not None else np.nan,
                "rel_f0_mean": stats["rel_f0_mean"],
                "f0_change": stats["f0_change"],
                "f0_range": stats["f0_range"],
                "prev_f0_end": prev_stats["f0_end"],
                "prev_f0_slope": prev_stats["f0_slope"],
                "env_mean": env_mean,
                "env_max": env_max,
                "env_std": env_std,
                "env_slope": env_slope,
                "periodicity": float(meta.get("voiced_frac") or 0.0),
                "logfreq": float(wextra.get("logfreq", 0.0)),
                "surprisal": float(wextra.get("surprisal", 0.0)),
                "phones": " ".join(f"{p}{t or ''}" for p, t in (s.get("phones") or [])),
            })
    rows.sort(key=lambda r: (r["section"], r["nucleus_on"], r["word_i"], r["syllable_i"]))
    return rows


def write_outputs(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for sid, g in df.groupby("section"):
        ons = g["nucleus_on"].to_numpy()
        if ons.size and np.any(np.diff(ons) < -1e-9):
            raise SystemExit(f"section {sid}: nucleus_on not monotonic after sort")
    df.to_parquet(out_dir / "tone_ctx_tokens.parquet", index=False)
    sandhi = df[df["sandhi_type"] != "none"][
        ["section", "syllable_id", "word", "char", "underlying_tone",
         "surface_tone", "sandhi_type", "prev_surface_tone", "next_surface_tone"]
    ]
    sandhi.to_csv(out_dir / "sandhi_audit.csv", index=False)
    lengths = {int(s): int(n) for s, n in df.groupby("section").size().items()}
    meta = {
        "n_rows": int(len(df)),
        "n_per_section": lengths,
        "n_T5": int((df["surface_tone"] == "5").sum()),
        "n_T1T4": int(df["surface_tone"].isin(L.TONES).sum()),
        "n_f0_valid": int(df["f0_valid"].sum()),
        "n_sandhi": int(len(sandhi)),
        "sandhi_counts": sandhi["sandhi_type"].value_counts().to_dict() if len(sandhi) else {},
        "n_multi_syllable_words": int(
            df.groupby(["section", "word_i"]).size().gt(1).sum()
        ),
    }
    (out_dir / "token_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "sec_lengths.json").write_text(json.dumps(lengths) + "\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=L.DEFAULT_SHARED)
    ap.add_argument("--cache", type=Path, default=L.DEFAULT_CACHE)
    ap.add_argument("--textgrid-dir", type=Path, default=L.DEFAULT_TG)
    ap.add_argument("--out-dir", type=Path, default=L.CTX_DIR)
    args = ap.parse_args()
    print(f"shared={args.shared}")
    rows = collect_rows(args.shared, args.cache, args.textgrid_dir)
    df = write_outputs(rows, args.out_dir)
    print(
        f"wrote {args.out_dir / 'tone_ctx_tokens.parquet'}  "
        f"n={len(df)}  T1–T4={int(df.surface_tone.isin(L.TONES).sum())}  "
        f"T5={int((df.surface_tone == '5').sum())}  "
        f"sandhi={int((df.sandhi_type != 'none').sum())}"
    )
    print("section lengths:", df.groupby("section").size().to_dict())


if __name__ == "__main__":
    main()
