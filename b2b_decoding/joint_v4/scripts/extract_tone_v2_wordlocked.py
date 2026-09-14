#!/usr/bin/env python3
"""Word-locked tone family v2: rime-sliced F0 + within-tone z-scored deviation.

Does not overwrite v1 (X_word_tone.npy). Writes:

  X_word_tone_v2.npy   (n_words × 6)
      tone_1 … tone_4     one-hot surface tone of the word's first syllable
      tone_dev            RMSE vs corpus template, z-scored within tone;
                          invalid tokens → 0 (mean imputation after z-score)
      tone_dev_valid      1 if the first-syllable rime contour was usable

F0 comes from section_XXX__f0_voicing_fs100_v2.npz (NaN = unvoiced).
Contours are sliced from the first vowel (nucleus) to syllable end.

Usage:
  .venv_gpt2/bin/python scripts/extract_tone_v2_wordlocked.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
MAX_GAP_S = 0.050
MIN_VOICED_FRAC = 0.50
MIN_FINAL_THIRD = 2
MIN_RMSE_PTS = 4
MIN_CANON_DUR = 0.12
TONE_COLS = ("tone_1", "tone_2", "tone_3", "tone_4", "tone_dev", "tone_dev_valid")

# MFA Mandarin nuclei that carry Chao / digit tone. Glides are onset, not rime.
VOWELS = frozenset({
    "a", "i", "o", "u", "e", "ə", "y",
    "aw", "aj", "ej", "ow",
    "ʐ̩", "z̩",
})
GLIDES = frozenset({"j", "w", "ɥ"})


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_v1():
    return _load("extract_tone_v1", ROOT / "scripts" / "extract_tone_phoneme_wordlocked.py")


def is_vowel(phone: str) -> bool:
    return phone in VOWELS


def rime_window(
    phones_timed: list[tuple[float, float, str, str | None]],
    syl_onset: float,
    syl_offset: float,
    *,
    assert_ok: bool = True,
) -> tuple[float, float, str] | None:
    """Nucleus onset = first vowel phone; rime = nucleus onset → syllable end."""
    vowels = [p for p in phones_timed if is_vowel(p[2])]
    if not vowels:
        return None
    nuc_on, _nuc_off, nuc_p, _nuc_t = vowels[0]
    if assert_ok:
        if nuc_on < syl_onset - 1e-6:
            raise AssertionError(
                f"rime starts before syllable: nuc_on={nuc_on:.4f} syl_on={syl_onset:.4f} "
                f"phones={phones_timed}"
            )
        if not is_vowel(nuc_p):
            raise AssertionError(f"nucleus is not a vowel: {nuc_p!r}")
        if nuc_p in GLIDES:
            raise AssertionError(f"nucleus is a glide: {nuc_p!r}")
    rime_on = max(float(nuc_on), float(syl_onset))
    rime_off = float(syl_offset)
    if rime_off <= rime_on + 1e-6:
        return None
    return rime_on, rime_off, nuc_p


def _runs_from_voiced_times(t_v: np.ndarray, max_gap_s: float = MAX_GAP_S) -> list[np.ndarray]:
    if t_v.size == 0:
        return []
    if t_v.size == 1:
        return [np.array([0], dtype=int)]
    cuts = np.where(np.diff(t_v) > max_gap_s)[0] + 1
    bounds = np.concatenate([[0], cuts, [t_v.size]])
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def time_normalize_rime(
    f0_hz: np.ndarray,
    t_on: float,
    t_off: float,
    *,
    fs: float = FS,
    npts: int = NPTS,
    max_gap_s: float = MAX_GAP_S,
) -> tuple[np.ndarray | None, dict]:
    """10-pt log-F0 on the rime, interpolating only inside voiced runs (gap ≤ 50 ms)."""
    a = max(0, int(round(t_on * fs)))
    b = min(f0_hz.size, max(a + 1, int(round(t_off * fs))))
    seg = f0_hz[a:b].astype(np.float64)
    n = seg.size
    meta = {
        "n_frames": n,
        "n_voiced": 0,
        "voiced_frac": 0.0,
        "n_final_third_voiced": 0,
        "valid": False,
        "reason": "",
    }
    if n < 2:
        meta["reason"] = "short_rime"
        return None, meta

    voiced = np.isfinite(seg) & (seg > 0)
    meta["n_voiced"] = int(voiced.sum())
    meta["voiced_frac"] = float(voiced.mean())
    third = max(1, n // 3)
    meta["n_final_third_voiced"] = int(voiced[-third:].sum())

    if meta["voiced_frac"] < MIN_VOICED_FRAC:
        meta["reason"] = "low_voiced_frac"
        return None, meta
    if meta["n_final_third_voiced"] < MIN_FINAL_THIRD:
        meta["reason"] = "sparse_final_third"
        return None, meta

    t_rel = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    y_v = np.log(seg[voiced])
    t_v_s = np.flatnonzero(voiced).astype(np.float64) / fs
    runs = _runs_from_voiced_times(t_v_s, max_gap_s=max_gap_s)

    contour = np.full(npts, np.nan, dtype=np.float64)
    q = np.linspace(0.0, 1.0, npts)
    for run in runs:
        if run.size == 0:
            continue
        tt = t_rel[voiced][run]
        yy = y_v[run]
        lo, hi = float(tt[0]), float(tt[-1])
        in_run = (q >= lo - 1e-12) & (q <= hi + 1e-12)
        if not in_run.any():
            continue
        if run.size == 1:
            contour[in_run] = yy[0]
        else:
            contour[in_run] = np.interp(q[in_run], tt, yy)

    if int(np.isfinite(contour).sum()) < MIN_RMSE_PTS:
        meta["reason"] = "too_few_grid_points"
        return None, meta
    meta["valid"] = True
    return contour, meta


def zscore_section(contours: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    pts = np.concatenate([c[np.isfinite(c)] for c in contours]) if contours else np.array([])
    if pts.size < 2:
        return list(contours), 0.0, 1.0
    mu = float(pts.mean())
    sd = float(pts.std())
    if sd < 1e-8:
        return [c - mu for c in contours], mu, 1.0
    return [(c - mu) / sd for c in contours], mu, sd


def rmse_voiced(contour: np.ndarray, template: np.ndarray, min_pts: int = MIN_RMSE_PTS) -> float | None:
    m = np.isfinite(contour) & np.isfinite(template)
    if int(m.sum()) < min_pts:
        return None
    d = contour[m] - template[m]
    return float(np.sqrt(np.mean(d * d)))


def load_f0_v2(cache: Path, section: int) -> tuple[np.ndarray, np.ndarray]:
    path = cache / f"section_{section:03d}__f0_voicing_fs100_v2.npz"
    if not path.is_file():
        raise SystemExit(f"Missing v2 F0 cache {path}. Run extract_f0_voicing_v2.py first.")
    with np.load(path) as z:
        f0_hz = np.asarray(z["f0_hz"], dtype=np.float64).ravel()
        voiced = np.asarray(z["voiced"], dtype=bool).ravel()
    if np.any(voiced & ~np.isfinite(f0_hz)):
        raise SystemExit(f"{path}: voiced frame with non-finite f0_hz")
    if np.any((~voiced) & np.isfinite(f0_hz)):
        # allow tracker to write 0; treat non-voiced as NaN
        f0_hz = np.where(voiced, f0_hz, np.nan)
    return f0_hz, voiced


def extract_section(v1, mfa, shared: Path, cache: Path, tg_dir: Path, section: int) -> dict:
    words = mfa._load_timing(shared / f"section_{section:03d}", section)
    phones = mfa.load_mfa_phones(tg_dir, section)
    assigned = mfa.assign_phones_to_words(phones, words)
    f0_hz, voiced = load_f0_v2(cache, section)

    word_sylls: list[list[dict]] = []
    all_sylls: list[dict] = []
    for w, phs in zip(words, assigned):
        sylls = v1.group_syllables(phs)
        chars = v1.cjk_chars(w["word"])
        if len(chars) < len(sylls):
            chars = chars + [""] * (len(sylls) - len(chars))
        for s, ch in zip(sylls, chars):
            s["char"] = ch
            s["word"] = w["word"]
            s["word_i"] = w["i"]
            timed = s.get("phones_timed") or [
                (s["onset"], s["offset"], p, t) for p, t in s["phones"]
            ]
            rw = rime_window(timed, s["onset"], s["offset"])
            s["rime"] = rw
            if rw is None:
                s["contour"] = None
                s["rime_meta"] = {"valid": False, "reason": "no_vowel_nucleus"}
            else:
                c, meta = time_normalize_rime(f0_hz, rw[0], rw[1])
                s["contour"] = c
                s["rime_meta"] = meta
            all_sylls.append(s)
        word_sylls.append(sylls)

    chars = [s.get("char") or "" for s in all_sylls]
    lex = [s["lexical_tone"] for s in all_sylls]
    surf = v1.apply_tone_sandhi(chars, lex)
    for s, t in zip(all_sylls, surf):
        s["surface_tone"] = t

    return {
        "words": words,
        "word_sylls": word_sylls,
        "all_sylls": all_sylls,
        "f0_hz": f0_hz,
        "voiced": voiced,
        "section": section,
    }


def fit_templates(sections: list[dict]) -> dict:
    by_tone: dict[str, list[np.ndarray]] = {t: [] for t in "1234"}
    z_stats = {}
    for sec in sections:
        valid = [s for s in sec["all_sylls"] if s.get("contour") is not None]
        zrows, mu, sd = zscore_section([s["contour"] for s in valid])
        z_stats[sec["section"]] = {"mu": mu, "sd": sd, "n": len(valid)}
        for s in sec["all_sylls"]:
            s["contour_z"] = None
        for s, z in zip(valid, zrows):
            s["contour_z"] = z
            t = s.get("surface_tone")
            lex = s.get("lexical_tone")
            dur = float(s["offset"] - s["onset"])
            if t in by_tone and t == lex and dur >= MIN_CANON_DUR:
                by_tone[t].append(z)
    templates = {"z_stats": z_stats}
    for t, rows in by_tone.items():
        if rows:
            templates[t] = np.nanmean(np.vstack(rows), axis=0)
        else:
            templates[t] = np.full(NPTS, np.nan)
        templates[f"n_{t}"] = len(rows)
    return templates


def is_prepausal(sylls: list[dict], i: int, gap_s: float = 0.20) -> bool:
    if i + 1 >= len(sylls):
        return True
    return (sylls[i + 1]["onset"] - sylls[i]["offset"]) >= gap_s


def template_pairwise_rmse(templates: dict) -> dict[str, float]:
    out = {}
    for a in "1234":
        for b in "1234":
            if a >= b:
                continue
            r = rmse_voiced(np.asarray(templates[a]), np.asarray(templates[b]), min_pts=6)
            out[f"T{a}_vs_T{b}"] = r if r is not None else 0.0
    return out


def nearest_template(contour_z: np.ndarray, templates: dict) -> str | None:
    best_t, best_r = None, np.inf
    for t in "1234":
        r = rmse_voiced(contour_z, np.asarray(templates[t]))
        if r is not None and r < best_r:
            best_t, best_r = t, r
    return best_t


def confusion_matrix(sections: list[dict], templates: dict) -> dict:
    """4×4 nearest-template assignment on valid T1–T4 tokens (resubstitution)."""
    labels = list("1234")
    idx = {t: i for i, t in enumerate(labels)}
    C = np.zeros((4, 4), dtype=int)
    n_skip = 0
    for sec in sections:
        for s in sec["all_sylls"]:
            t = s.get("surface_tone")
            cz = s.get("contour_z")
            if t not in idx or cz is None:
                continue
            pred = nearest_template(cz, templates)
            if pred is None:
                n_skip += 1
                continue
            C[idx[t], idx[pred]] += 1
    row_n = C.sum(axis=1)
    acc = {}
    modal = {}
    for i, t in enumerate(labels):
        n = int(row_n[i])
        acc[t] = float(C[i, i] / n) if n else 0.0
        modal[t] = labels[int(np.argmax(C[i]))] if n else None
    return {
        "labels": labels,
        "counts": C.tolist(),
        "n": int(C.sum()),
        "n_unassigned": n_skip,
        "hit_rate": acc,
        "modal": modal,
        "modal_is_self": {t: modal[t] == t for t in labels},
        "T4_self_beats_T1": bool(C[3, 3] > C[3, 0]),
        "overall_acc": float(np.trace(C) / C.sum()) if C.sum() else 0.0,
    }


def validate_templates(templates: dict, confusion: dict, pairwise: dict) -> dict[str, bool]:
    t1, t2, t3, t4 = (np.asarray(templates[t], dtype=np.float64) for t in "1234")
    t1_range = float(np.nanmax(t1) - np.nanmin(t1))
    t4_range = float(np.nanmax(t4) - np.nanmin(t4))
    # Discriminability gate (amended 2026-08-31). Citation T4-near-T3-floor
    # is informational only — connected-speech T4 is high→mid here.
    required = {
        "pairwise_separated": bool(pairwise and min(pairwise.values()) > 0.15),
        "confusion_modal_self": bool(all(confusion["modal_is_self"].values())),
        "confusion_T4_self_beats_T1": bool(confusion["T4_self_beats_T1"]),
    }
    info = {
        "T2_endpoint_gt_onset": bool(t2[-1] > t2[0]),
        "T4_endpoint_lt_onset": bool(t4[-1] < t4[0]),
        "T1_mean_high": bool(float(np.nanmean(t1)) > 0.0),
        "T1_flatter_than_T4": bool(t1_range < t4_range),
        "T3_low_falling": bool(t3[-1] < t3[0]),
        "T4_starts_highest": bool(t4[0] > t1[0] and t4[0] > t2[0] and t4[0] > t3[0]),
        "T4_ends_below_T1": bool(float(t4[-1]) < float(t1[-1])),
    }
    return {**required, **{f"info_{k}": v for k, v in info.items()}}


def gate_passed(checks: dict[str, bool]) -> bool:
    return all(v for k, v in checks.items() if not k.startswith("info_"))


def save_template_plot(
    templates: dict,
    extras: dict[str, np.ndarray],
    path: Path,
    checks: dict[str, bool],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1.0, NPTS)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    styles = {
        "1": ("T1 high", "#c0392b"),
        "2": ("T2 rise", "#2980b9"),
        "3": ("T3 dip", "#27ae60"),
        "4": ("T4 fall", "#8e44ad"),
    }
    for t, (lab, color) in styles.items():
        n = int(templates.get(f"n_{t}", 0))
        ax.plot(x, templates[t], color=color, lw=2.2, label=f"{lab} (n={n})")
    if "t3_prepausal" in extras:
        ax.plot(x, extras["t3_prepausal"], color="#27ae60", lw=1.3, ls="--",
                label="T3 pre-pausal")
    if "t4_prepausal" in extras:
        ax.plot(x, extras["t4_prepausal"], color="#8e44ad", lw=1.3, ls="--",
                label="T4 pre-pausal")
    if "t4_nonfinal" in extras:
        ax.plot(x, extras["t4_nonfinal"], color="#8e44ad", lw=1.1, ls=":",
                label="T4 non-final")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("Normalized rime time")
    ax.set_ylabel("z-scored log F0")
    ok = gate_passed(checks)
    ax.set_title("Tone templates v2 (rime, voiced-only)" + (" — PASS" if ok else " — FAIL"))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_f0_hist(sections: list[dict], path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {t: [] for t in "1234"}
    for sec in sections:
        f0 = sec["f0_hz"]
        for s in sec["all_sylls"]:
            t = s.get("surface_tone")
            rw = s.get("rime")
            if t not in by or rw is None or s.get("contour") is None:
                continue
            a = max(0, int(round(rw[0] * FS)))
            b = min(f0.size, int(round(rw[1] * FS)))
            hz = f0[a:b]
            hz = hz[np.isfinite(hz) & (hz > 0)]
            by[t].extend(hz.tolist())

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    summary = {}
    for ax, t in zip(axes.ravel(), "1234"):
        hz = np.asarray(by[t], dtype=np.float64)
        summary[t] = {
            "n": int(hz.size),
            "median": float(np.median(hz)) if hz.size else None,
            "p05": float(np.percentile(hz, 5)) if hz.size else None,
            "p95": float(np.percentile(hz, 95)) if hz.size else None,
        }
        if hz.size:
            ax.hist(hz, bins=40, color="0.35", alpha=0.85)
            # octave-error hint: secondary mode near 2× median
            med = float(np.median(hz))
            ax.axvline(med, color="#c0392b", lw=1)
            ax.axvline(2 * med, color="#2980b9", lw=1, ls="--")
        ax.set_title(f"T{t} rime F0 (Hz)")
    axes[1, 0].set_xlabel("F0 (Hz)")
    axes[1, 1].set_xlabel("F0 (Hz)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return summary


def write_word_tone(section_dir: Path, X: np.ndarray, meta: dict) -> None:
    if X.shape[1] != 6:
        raise SystemExit(f"tone_v2: expected 6 cols, got {X.shape}")
    if not np.isfinite(X).all():
        raise SystemExit(f"tone_v2: {int(np.sum(~np.isfinite(X)))} non-finite")
    np.save(section_dir / "X_word_tone_v2.npy", X.astype(np.float64))
    (section_dir / "X_word_tone_v2_feature_names.txt").write_text(
        "\n".join(TONE_COLS) + "\n", encoding="utf-8"
    )
    (section_dir / "X_word_tone_v2_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--textgrid-dir", type=Path, default=DEFAULT_TG)
    ap.add_argument("--plot", type=Path, default=ROOT / "plots" / "tone_templates_v2_rime.png")
    args = ap.parse_args()

    v1 = _load_v1()
    mfa = v1._load_mfa_mod()
    print(f"shared={args.shared}")
    print(f"cache ={args.cache}")

    sections = []
    for sid in SECTIONS:
        sec = extract_section(v1, mfa, args.shared, args.cache, args.textgrid_dir, sid)
        sections.append(sec)
        n_ok = sum(s.get("contour") is not None for s in sec["all_sylls"])
        print(
            f"section {sid}: words={len(sec['words'])}  sylls={len(sec['all_sylls'])}  "
            f"valid_rimes={n_ok}"
        )

    templates = fit_templates(sections)
    print("tone templates v2 (10-pt z log-F0, rime):")
    for t in "1234":
        row = np.asarray(templates[t])
        print(
            f"  T{t} n={templates[f'n_{t}']}: "
            f"{np.array2string(row, precision=3, suppress_small=True)}"
        )

    def _mean_by(tone: str, pred) -> tuple[np.ndarray | None, int]:
        rows = []
        for sec in sections:
            sylls = sec["all_sylls"]
            for i, s in enumerate(sylls):
                if s.get("surface_tone") != tone or s.get("contour_z") is None:
                    continue
                if pred(sylls, i):
                    rows.append(s["contour_z"])
        if not rows:
            return None, 0
        return np.nanmean(np.vstack(rows), axis=0), len(rows)

    extras = {}
    t3_prepausal, n_t3p = _mean_by("3", is_prepausal)
    t4_prepausal, n_t4p = _mean_by("4", is_prepausal)
    t4_nonfinal, n_t4n = _mean_by("4", lambda sylls, i: not is_prepausal(sylls, i))
    if t3_prepausal is not None:
        extras["t3_prepausal"] = t3_prepausal
        print(f"  T3 pre-pausal n={n_t3p}  end-start={float(t3_prepausal[-1] - t3_prepausal[0]):+.3f}")
    if t4_prepausal is not None:
        extras["t4_prepausal"] = t4_prepausal
        print(f"  T4 pre-pausal n={n_t4p}  end-start={float(t4_prepausal[-1] - t4_prepausal[0]):+.3f}")
    if t4_nonfinal is not None:
        extras["t4_nonfinal"] = t4_nonfinal
        print(f"  T4 non-final  n={n_t4n}  end-start={float(t4_nonfinal[-1] - t4_nonfinal[0]):+.3f}")

    pairwise = template_pairwise_rmse(templates)
    print("pairwise template RMSE:")
    for k, v in pairwise.items():
        print(f"  {k}: {v:.3f}")
    confusion = confusion_matrix(sections, templates)
    print(f"nearest-template confusion (n={confusion['n']}, overall_acc={confusion['overall_acc']:.3f}):")
    print("         pred T1   T2   T3   T4")
    for i, t in enumerate(confusion["labels"]):
        row = confusion["counts"][i]
        print(f"  true T{t}  {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}  "
              f"hit={confusion['hit_rate'][t]:.3f}  modal=T{confusion['modal'][t]}")

    checks = validate_templates(templates, confusion, pairwise)
    print("validation:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    save_template_plot(templates, extras, args.plot, checks)
    print(f"  wrote {args.plot}")
    hist_path = ROOT / "plots" / "tone_v2_rime_f0_hz_hist.png"
    hist = save_f0_hist(sections, hist_path)
    print(f"  wrote {hist_path}")
    print("  rime F0 Hz:", hist)

    # Per-token raw RMSE, then z-score within surface tone (pooled).
    raw_by_tone: dict[str, list[float]] = {t: [] for t in "1234"}
    for sec in sections:
        for s in sec["all_sylls"]:
            t = s.get("surface_tone")
            cz = s.get("contour_z")
            if t is None or t not in "1234" or cz is None:
                s["dev_raw"] = None
                continue
            raw = rmse_voiced(cz, templates[t])
            s["dev_raw"] = raw
            if raw is not None:
                raw_by_tone[t].append(raw)
    tone_mu = {}
    tone_sd = {}
    for t in "1234":
        arr = np.asarray(raw_by_tone[t], dtype=np.float64)
        tone_mu[t] = float(arr.mean()) if arr.size else 0.0
        sd = float(arr.std()) if arr.size > 1 else 1.0
        tone_sd[t] = sd if sd > 1e-8 else 1.0

    counts = {t: 0 for t in list("12345") + ["none"]}
    n_valid_first = 0
    for sec, sid in zip(sections, SECTIONS):
        n = len(sec["words"])
        X = np.zeros((n, 6), dtype=np.float64)
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
            raw = first.get("dev_raw")
            if raw is not None and t in tone_mu:
                X[i, 4] = (raw - tone_mu[t]) / tone_sd[t]
                X[i, 5] = 1.0
                n_valid_first += 1
            # else already 0, 0  — mean imputation after within-tone z-score
        meta = {
            "feature": "tone_v2",
            "section_id": sid,
            "n_words": n,
            "n_dim": 6,
            "columns": list(TONE_COLS),
            "coding": (
                "first-syllable surface tone one-hot + rime RMSE "
                "z-scored within tone; invalid → 0 after z-score"
            ),
            "sandhi": "不+T4→T2, 一 sandhi, then 3-3 (story-level)",
            "tone_source": "MFA Chao/digit on syllable nucleus",
            "rime": "first vowel → syllable end; voiced-only 10-pt; gap>50ms = NaN",
            "n_syllables": len(sec["all_sylls"]),
            "n_valid_rimes": sum(s.get("contour") is not None for s in sec["all_sylls"]),
            "n_first_syl_sandhi": n_sandhi,
            "n_first_valid_dev": int(X[:, 5].sum()),
            "template_n_by_tone": {t: int(templates[f"n_{t}"]) for t in "1234"},
            "dev_raw_mean_by_tone": tone_mu,
            "dev_raw_sd_by_tone": tone_sd,
            "validation": checks,
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0, ddof=1).tolist(),
        }
        write_word_tone(args.shared / f"section_{sid:03d}", X, meta)
        print(
            f"  wrote X_word_tone_v2.npy section {sid}  shape={X.shape}  "
            f"onehot_frac={X[:, :4].sum(1).mean():.3f}  "
            f"valid_frac={X[:, 5].mean():.3f}  "
            f"dev_std={X[:, 4].std():.3f}"
        )

    gate = {
        "passed": gate_passed(checks),
        "gate": "discriminability_2026-08-31",
        "checks": checks,
        "pairwise_rmse": pairwise,
        "confusion": confusion,
        "templates": {t: np.asarray(templates[t]).tolist() for t in "1234"},
        "template_n": {t: int(templates[f"n_{t}"]) for t in "1234"},
        "t3_prepausal_end_minus_start": (
            float(t3_prepausal[-1] - t3_prepausal[0]) if t3_prepausal is not None else None
        ),
        "t4_prepausal_end_minus_start": (
            float(t4_prepausal[-1] - t4_prepausal[0]) if t4_prepausal is not None else None
        ),
        "t4_nonfinal_end_minus_start": (
            float(t4_nonfinal[-1] - t4_nonfinal[0]) if t4_nonfinal is not None else None
        ),
        "n_t4_prepausal": n_t4p,
        "n_t4_nonfinal": n_t4n,
        "f0_hz_by_tone": hist,
        "first_syllable_tone_counts": counts,
        "n_valid_first_dev": n_valid_first,
    }
    gate_path = ROOT / "plots" / "tone_v2_validation.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    print(f"  wrote {gate_path}")
    print("GATE", "PASS" if gate["passed"] else "FAIL")
    print("Done.")


if __name__ == "__main__":
    main()
