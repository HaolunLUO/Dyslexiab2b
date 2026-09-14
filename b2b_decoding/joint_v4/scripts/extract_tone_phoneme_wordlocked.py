#!/usr/bin/env python3
"""Word-locked tone family + 100 Hz phone/syllable onset impulses.

Replaces the raw 13-D pitch family for word-locked B2B:

  X_word_tone.npy   (n_words × 5)
      tone_1 … tone_4   one-hot surface tone of the word's first syllable
      tone_dev          RMSE of time-normalized log-F0 vs corpus template

Phone / syllable onsets are written as sparse 100 Hz impulse trains
(section_XXX__onsets_fs100.npz) for a later continuous / mTRF arm — not
as word-locked rows.

Tone labels come from MFA Chao / digit suffixes (dictionary / lexical).
A small Mandarin sandhi pass (3-3, 不, 一) approximates surface tone.
Templates are fit on the full story (T1–T4 only), never per group.

Usage:
  .venv_gpt2/bin/python scripts/extract_tone_phoneme_wordlocked.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
DEFAULT_CACHE = (
    POOL / "extracted_sections_wordlocked_shared_acoustic_residual" / "_acoustic_features_cache"
)
DEFAULT_TG = POOL / "mfa_lppcn_wordlocked" / "aligned"
FS = 100.0
NPTS = 10
SECTIONS = (1, 2)
MIN_VOICED = 3
TONE_COLS = ("tone_1", "tone_2", "tone_3", "tone_4", "tone_dev")
# MFA Mandarin nasal / rhotic finals after a toned nucleus.
CODAS = frozenset({"n", "ŋ", "ɴ", "N", "ɻ", "r"})


def _load_mfa_mod():
    spec = importlib.util.spec_from_file_location(
        "extract_mfa_v2", ROOT / "scripts" / "extract_mfa_phone_tone_envelope_wordlocked.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cjk_chars(word: str) -> list[str]:
    return [c for c in word if "\u4e00" <= c <= "\u9fff"]


def _syl_record(phones: list[tuple[float, float, str, str | None]]) -> dict:
    tones = [p[3] for p in phones if p[3] is not None]
    return {
        "onset": float(phones[0][0]),
        "offset": float(phones[-1][1]),
        "phones": [(p[2], p[3]) for p in phones],
        "phones_timed": [(float(p[0]), float(p[1]), p[2], p[3]) for p in phones],
        "lexical_tone": tones[-1] if tones else None,
    }


def group_syllables(
    phones: list[tuple[float, float, str, str | None]],
) -> list[dict]:
    """Split a word's phones at each toned nucleus; optional n/ŋ coda stays with it."""
    if not phones:
        return []
    nuc = [i for i, p in enumerate(phones) if p[3] is not None]
    if not nuc:
        return [_syl_record(phones)]

    n = len(phones)
    owner = [0] * n
    for k, ni in enumerate(nuc):
        owner[ni] = k
    for i in range(nuc[0]):
        owner[i] = 0
    for k in range(len(nuc) - 1):
        a, b = nuc[k], nuc[k + 1]
        mid = list(range(a + 1, b))
        if not mid:
            continue
        if phones[mid[0]][2] in CODAS:
            owner[mid[0]] = k
            for i in mid[1:]:
                owner[i] = k + 1
        else:
            for i in mid:
                owner[i] = k + 1
    for i in range(nuc[-1] + 1, n):
        owner[i] = len(nuc) - 1

    groups: list[list] = [[] for _ in nuc]
    for i, ph in enumerate(phones):
        groups[owner[i]].append(ph)
    return [_syl_record(g) for g in groups if g]


def apply_tone_sandhi(chars: list[str], lexical: list[str | None]) -> list[str | None]:
    """Lexical → approximate surface tone (不, 一, then 3-3). Neutral/None unchanged."""
    n = len(lexical)
    surface = list(lexical)
    if n == 0:
        return surface
    for i in range(n - 1):
        ch = chars[i] if i < len(chars) else ""
        nxt = surface[i + 1]
        if ch == "不" and nxt == "4":
            surface[i] = "2"
        elif ch == "一" and nxt in {"1", "2", "3", "4"}:
            surface[i] = "2" if nxt == "4" else "4"
    for i in range(n - 1):
        if surface[i] == "3" and surface[i + 1] == "3":
            surface[i] = "2"
    return surface


def voiced_mask(log_f0: np.ndarray) -> np.ndarray:
    """Unvoiced frames were filled with a constant mean in the acoustic cache."""
    vals, counts = np.unique(np.round(log_f0, 6), return_counts=True)
    fill = float(vals[int(np.argmax(counts))])
    return np.abs(log_f0 - fill) > 1e-4


def syllable_f0(
    log_f0: np.ndarray,
    voiced: np.ndarray,
    t_on: float,
    t_off: float,
    fs: float = FS,
    npts: int = NPTS,
) -> np.ndarray | None:
    a = max(0, int(round(t_on * fs)))
    b = min(log_f0.size, max(a + 1, int(round(t_off * fs))))
    seg = log_f0[a:b]
    v = voiced[a:b]
    idx = np.flatnonzero(v)
    if idx.size < MIN_VOICED:
        return None
    use = seg[idx]
    if float(np.std(use)) < 1e-4:
        return None
    # Keep voiced frames at their original relative times (do not compress gaps).
    denom = max(len(seg) - 1, 1)
    x = idx.astype(np.float64) / denom
    return np.interp(np.linspace(0.0, 1.0, npts), x, use)


def zscore_rows(rows: list[np.ndarray]) -> list[np.ndarray]:
    if not rows:
        return rows
    M = np.vstack(rows)
    mu = float(M.mean())
    sd = float(M.std())
    if sd < 1e-8:
        return [r - mu for r in rows]
    return [(r - mu) / sd for r in rows]


def extract_section(mod, shared: Path, cache: Path, tg_dir: Path, section: int) -> dict:
    words = mod._load_timing(shared / f"section_{section:03d}", section)
    phones = mod.load_mfa_phones(tg_dir, section)
    assigned = mod.assign_phones_to_words(phones, words)
    cache_path = cache / f"section_{section:03d}__fs{int(FS)}.npz"
    if not cache_path.is_file():
        raise SystemExit(f"Missing F0 cache {cache_path}")
    with np.load(cache_path) as z:
        log_f0 = np.asarray(z["log_f0"], dtype=np.float64).ravel()
    voiced = voiced_mask(log_f0)

    word_sylls: list[list[dict]] = []
    all_sylls: list[dict] = []
    for w, phs in zip(words, assigned):
        sylls = group_syllables(phs)
        chars = cjk_chars(w["word"])
        if len(chars) < len(sylls):
            chars = chars + [""] * (len(sylls) - len(chars))
        for s, ch in zip(sylls, chars):
            s["char"] = ch
            s["word"] = w["word"]
            s["word_i"] = w["i"]
            all_sylls.append(s)
        word_sylls.append(sylls)

    # Story-level sandhi (3-3 and 不/一 can cross MFA word boundaries).
    chars = [s.get("char") or "" for s in all_sylls]
    lex = [s["lexical_tone"] for s in all_sylls]
    surf = apply_tone_sandhi(chars, lex)
    for s, t in zip(all_sylls, surf):
        s["surface_tone"] = t
        s["contour"] = syllable_f0(log_f0, voiced, s["onset"], s["offset"])

    return {
        "words": words,
        "assigned": assigned,
        "word_sylls": word_sylls,
        "all_sylls": all_sylls,
        "log_f0": log_f0,
        "voiced": voiced,
        "n_phones": sum(1 for items in assigned for _ in items),
        "section": section,
    }


def fit_templates(sections: list[dict]) -> dict[str, np.ndarray]:
    by_tone: dict[str, list[np.ndarray]] = {t: [] for t in "1234"}
    for sec in sections:
        raw = [s["contour"] for s in sec["all_sylls"] if s["contour"] is not None]
        zrows = zscore_rows(raw)
        k = 0
        for s in sec["all_sylls"]:
            if s["contour"] is None:
                s["contour_z"] = None
                continue
            s["contour_z"] = zrows[k]
            t = s.get("surface_tone")
            lex = s.get("lexical_tone")
            dur = float(s["offset"] - s["onset"])
            # Canonical tokens only: no sandhi, T1–T4, long enough to have a contour.
            if t in by_tone and t == lex and dur >= 0.12:
                by_tone[t].append(zrows[k])
            k += 1
    templates = {}
    for t, rows in by_tone.items():
        templates[t] = np.mean(np.vstack(rows), axis=0) if rows else np.zeros(NPTS)
        templates[f"n_{t}"] = len(rows)
    return templates


def tone_deviation(
    contour_z: np.ndarray | None,
    tone: str | None,
    templates: dict[str, np.ndarray],
) -> float:
    if contour_z is None or tone is None or tone not in "1234" or tone not in templates:
        return 0.0
    d = contour_z - templates[tone]
    return float(np.sqrt(np.mean(d * d)))


def rasterize(times: list[float], n: int, fs: float = FS) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    for t in times:
        i = int(round(t * fs))
        if 0 <= i < n:
            x[i] = 1.0
    return x


def write_word_tone(section_dir: Path, X: np.ndarray, meta: dict) -> None:
    if X.shape[1] != 5:
        raise SystemExit(f"tone: expected 5 cols, got {X.shape}")
    if not np.isfinite(X).all():
        raise SystemExit(f"tone: {int(np.sum(~np.isfinite(X)))} non-finite values")
    np.save(section_dir / "X_word_tone.npy", X.astype(np.float64))
    (section_dir / "X_word_tone_feature_names.txt").write_text(
        "\n".join(TONE_COLS) + "\n", encoding="utf-8"
    )
    (section_dir / "X_word_tone_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def save_template_plot(templates: dict[str, np.ndarray], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        np.savez(path.with_suffix(".npz"), **{f"T{t}": templates[t] for t in "1234"})
        return
    x = np.linspace(0.0, 1.0, NPTS)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    styles = {
        "1": ("T1 high", "#c0392b", "-"),
        "2": ("T2 rise", "#2980b9", "-"),
        "3": ("T3 dip", "#27ae60", "-"),
        "4": ("T4 fall", "#8e44ad", "-"),
    }
    for t, (lab, color, ls) in styles.items():
        n = int(templates.get(f"n_{t}", 0))
        ax.plot(x, templates[t], ls, color=color, lw=2.2, label=f"{lab} (n={n})")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("Normalized syllable time")
    ax.set_ylabel("z-scored log F0")
    ax.set_title("Corpus tone templates (stimulus F0, T1–T4)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_syllable_table(sections: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "section", "word_i", "word", "char", "onset", "offset",
            "lexical_tone", "surface_tone", "phones", "has_contour", "tone_dev",
        ])
        for sec in sections:
            sid = sec["section"]
            for s in sec["all_sylls"]:
                w.writerow([
                    sid, s.get("word_i"), s.get("word"), s.get("char"),
                    f"{s['onset']:.4f}", f"{s['offset']:.4f}",
                    s.get("lexical_tone") or "",
                    s.get("surface_tone") or "",
                    " ".join(
                        f"{p}{t or ''}" for p, t in s["phones"]
                    ),
                    int(s.get("contour") is not None),
                    f"{s.get('tone_dev', 0.0):.4f}",
                ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--textgrid-dir", type=Path, default=DEFAULT_TG)
    ap.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "plots" / "tone_templates_corpus.png",
    )
    args = ap.parse_args()

    mod = _load_mfa_mod()
    print(f"shared={args.shared}")
    print(f"cache ={args.cache}")
    print(f"textgrid={args.textgrid_dir}")

    sections = []
    for sid in SECTIONS:
        sec = extract_section(mod, args.shared, args.cache, args.textgrid_dir, sid)
        sections.append(sec)
        n_syl = len(sec["all_sylls"])
        n_c = sum(s["contour"] is not None for s in sec["all_sylls"])
        print(
            f"section {sid}: words={len(sec['words'])}  syllables={n_syl}  "
            f"usable_contours={n_c}  phones={sec['n_phones']}"
        )

    templates = fit_templates(sections)
    print("tone templates (10-pt z-scored log-F0, first→last):")
    for t in "1234":
        row = templates[t]
        print(
            f"  T{t} n={templates[f'n_{t}']}: "
            f"{np.array2string(np.asarray(row), precision=3, suppress_small=True)}"
        )
    save_template_plot(templates, args.plot)
    print(f"  wrote {args.plot}")

    counts = {t: 0 for t in list("12345") + ["none"]}
    n_sandhi_total = 0
    for sec, sid in zip(sections, SECTIONS):
        n = len(sec["words"])
        X = np.zeros((n, 5), dtype=np.float64)
        n_sandhi = 0
        for i, sylls in enumerate(sec["word_sylls"]):
            if not sylls:
                continue
            first = sylls[0]
            t = first.get("surface_tone")
            lex = first.get("lexical_tone")
            if t is not None and t in "1234":
                X[i, int(t) - 1] = 1.0
                counts[t] += 1
            elif t == "5":
                counts["5"] += 1
            else:
                counts["none"] += 1
            if lex != t:
                n_sandhi += 1
            dev = tone_deviation(first.get("contour_z"), t, templates)
            first["tone_dev"] = dev
            for s in sylls[1:]:
                s["tone_dev"] = tone_deviation(s.get("contour_z"), s.get("surface_tone"), templates)
            X[i, 4] = dev

        phone_times = [p[0] for items in sec["assigned"] for p in items]
        syl_times = [s["onset"] for s in sec["all_sylls"]]
        n_t = sec["log_f0"].size
        phone_imp = rasterize(phone_times, n_t)
        syl_imp = rasterize(syl_times, n_t)
        onset_path = args.cache / f"section_{sid:03d}__onsets_fs100.npz"
        np.savez_compressed(
            onset_path,
            phone_onset=phone_imp,
            syllable_onset=syl_imp,
            fs=np.array(FS),
        )
        n_sandhi_total += n_sandhi
        meta = {
            "feature": "tone",
            "section_id": sid,
            "n_words": n,
            "n_dim": 5,
            "columns": list(TONE_COLS),
            "coding": "first-syllable surface tone one-hot (T1–T4) + contour RMSE",
            "sandhi": "不+T4→T2, 一 sandhi, then 3-3 (story-level)",
            "tone_source": "MFA Chao/digit on syllable nucleus",
            "n_syllables": len(sec["all_sylls"]),
            "n_phone_onsets": int(phone_imp.sum()),
            "n_syllable_onsets": int(syl_imp.sum()),
            "n_first_syl_sandhi": n_sandhi,
            "template_n_pts": NPTS,
            "template_n_by_tone": {t: int(templates[f"n_{t}"]) for t in "1234"},
            "onsets_npz": str(onset_path),
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0, ddof=1).tolist(),
        }
        out_dir = args.shared / f"section_{sid:03d}"
        write_word_tone(out_dir, X, meta)
        print(
            f"  wrote {out_dir / 'X_word_tone.npy'}  shape={X.shape}  "
            f"onehot_frac={X[:, :4].sum(axis=1).mean():.3f}  "
            f"dev_mean={X[:, 4].mean():.3f}  first_sandhi={n_sandhi}"
        )
        print(
            f"  wrote {onset_path}  phone_impulses={int(phone_imp.sum())}  "
            f"syl_impulses={int(syl_imp.sum())}"
        )

    table_path = ROOT / "plots" / "tone_syllables_spotcheck.csv"
    write_syllable_table(sections, table_path)
    print(f"  wrote {table_path}")
    print(f"first-syllable surface tone counts (words): {counts}")
    print(f"first-syllable sandhi changes (both sections): {n_sandhi_total}")
    print("Done.")


if __name__ == "__main__":
    main()
