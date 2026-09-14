#!/usr/bin/env python3
"""Shared helpers for the exploratory tone_ctx_v1 branch.

Does not read or write frozen tone_v3 word-locked matrices.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
DEFAULT_CACHE = (
    POOL / "extracted_sections_wordlocked_shared_acoustic_residual" / "_acoustic_features_cache"
)
DEFAULT_TG = POOL / "mfa_lppcn_wordlocked" / "aligned"
DEFAULT_NUCLEUS_SHARED = POOL / "extracted_sections_nucleuslocked_shared" / "_shared_wordlocked_features"
CTX_DIR = ROOT / "tone_ctx_v1"
FS = 100.0
SECTIONS = (1, 2)
TONES = ("1", "2", "3", "4")
PAUSE_GAP_S = 0.20

INITIALS = (
    "", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w", "other",
)
FINALS = (
    "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "i", "ia", "ian",
    "iang", "iao", "ie", "in", "ing", "iong", "iu", "o", "ong", "ou", "u",
    "ua", "uai", "uan", "uang", "ue", "ui", "un", "uo", "v", "van", "ve",
    "vn", "other",
)
MFA_FINAL_MAP = {
    "ə": "e", "y": "v", "aw": "ao", "aj": "ai", "ej": "ei", "ow": "ou",
    "ʐ̩": "i", "z̩": "i",
}

CONTROL_ACOUSTIC = (
    "env_mean", "env_max", "env_std", "env_slope", "periodicity", "nucleus_onset",
)
CONTROL_STRUCT = (
    "duration", "phrase_position", "preceding_pause", "speech_rate",
    "logfreq", "surprisal",
)
CONTROL_PREV = (
    "prev_t1", "prev_t2", "prev_t3", "prev_t4", "prev_f0_end", "prev_f0_slope",
)
CONTROL_INIT = tuple(f"init_{x or 'none'}" for x in INITIALS)
CONTROL_FINAL = tuple(f"fin_{x}" for x in FINALS)
CONTROL_NAMES = CONTROL_ACOUSTIC + CONTROL_STRUCT + CONTROL_PREV + CONTROL_INIT + CONTROL_FINAL
# rel_f0_mean omitted: r≈0.999 with c0_register, unidentified under OLS-H.
PITCH_NAMES = ("c0_register", "c1_slope", "c2_curvature", "f0_change", "f0_range")
EVIDENCE_NAMES = ("tone_evidence_1", "tone_evidence_2", "tone_evidence_3")
RESID_NAMES = ("tone_resid_1", "tone_resid_2", "tone_resid_3")

N_CTX_CONTROLS = len(CONTROL_NAMES)
N_CTX_PITCH = len(PITCH_NAMES)
N_CTX_EVIDENCE = 3
N_CTX_RESID = 3

assert N_CTX_CONTROLS == 6 + 6 + 6 + 25 + 36


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_v2():
    return load_module("extract_tone_v2", ROOT / "scripts" / "extract_tone_v2_wordlocked.py")


def load_v3():
    return load_module("extract_tone_v3", ROOT / "scripts" / "extract_tone_v3_wordlocked.py")


def orthonormal_helmert(k: int = 4) -> np.ndarray:
    """k × (k-1) orthonormal Helmert contrasts. Columns sum to 0."""
    if k < 2:
        raise ValueError("helmert needs k >= 2")
    H = np.zeros((k, k - 1), dtype=np.float64)
    for j in range(k - 1):
        H[: j + 1, j] = 1.0 / (j + 1)
        H[j + 1, j] = -1.0
        nrm = np.linalg.norm(H[:, j])
        H[:, j] /= nrm
    return H


def one_hot_tone(tone: str | None, k: int = 4) -> np.ndarray:
    y = np.zeros(k, dtype=np.float64)
    if tone in TONES:
        y[int(tone) - 1] = 1.0
    return y


def sandhi_type(char: str, lexical: str | None, surface: str | None, next_lex: str | None) -> str:
    if lexical is None or surface is None or lexical == surface:
        return "none"
    if char == "不" and next_lex == "4" and surface == "2":
        return "bu_t4_to_t2"
    if char == "一" and next_lex in TONES:
        if next_lex == "4" and surface == "2":
            return "yi_to_2"
        if next_lex != "4" and surface == "4":
            return "yi_to_4"
    if lexical == "3" and surface == "2":
        return "t3t3_to_t2"
    return "other"


def base_syllable(char: str, phones: list[tuple[str, str | None]]) -> str:
    if char:
        try:
            from pypinyin import Style, pinyin
            tok = pinyin(char, style=Style.NORMAL, strict=False, errors="ignore")
            if tok and tok[0] and tok[0][0]:
                return str(tok[0][0]).lower()
        except Exception:
            pass
    parts = [p for p, _ in phones if p]
    return "".join(parts) or "unk"


def split_initial_final(phones_timed: list[tuple[float, float, str, str | None]], v2) -> tuple[str, str]:
    vowels = [p for p in phones_timed if v2.is_vowel(p[2])]
    if not vowels:
        segs = [p[2] for p in phones_timed]
        return (_canon_init("".join(segs)), "other")
    first_v = phones_timed.index(vowels[0])
    init_raw = "".join(p[2] for p in phones_timed[:first_v])
    fin_raw = "".join(MFA_FINAL_MAP.get(p[2], p[2]) for p in phones_timed[first_v:])
    return _canon_init(init_raw), _canon_final(fin_raw)


def _canon_init(raw: str) -> str:
    s = (raw or "").lower()
    if s in INITIALS:
        return s
    for init in ("zh", "ch", "sh"):
        if s.startswith(init):
            return init
    if s[:1] in INITIALS:
        return s[:1]
    return "other" if s else ""


def _canon_final(raw: str) -> str:
    s = (raw or "").lower()
    if s in FINALS:
        return s
    for fin in sorted(FINALS, key=len, reverse=True):
        if fin != "other" and s.startswith(fin):
            return fin
    return "other"


def one_hot_label(value: str, inventory: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(inventory), dtype=np.float64)
    if value in inventory:
        out[inventory.index(value)] = 1.0
    else:
        out[inventory.index("other")] = 1.0
    return out


def contour_stats(contour: np.ndarray | None) -> dict[str, float]:
    if contour is None:
        return {"rel_f0_mean": 0.0, "f0_change": 0.0, "f0_range": 0.0,
                "f0_end": 0.0, "f0_slope": 0.0}
    y = np.asarray(contour, dtype=np.float64)
    m = np.isfinite(y)
    if int(m.sum()) < 2:
        return {"rel_f0_mean": 0.0, "f0_change": 0.0, "f0_range": 0.0,
                "f0_end": 0.0, "f0_slope": 0.0}
    yy = y[m]
    return {
        "rel_f0_mean": float(yy.mean()),
        "f0_change": float(yy[-1] - yy[0]),
        "f0_range": float(yy.max() - yy.min()),
        "f0_end": float(yy[-1]),
        "f0_slope": float(yy[-1] - yy[0]),
    }


def slice_stats(x: np.ndarray, t_on: float, t_off: float, fs: float = FS) -> tuple[float, float, float, float]:
    a = max(0, int(round(t_on * fs)))
    b = min(x.size, max(a + 1, int(round(t_off * fs))))
    seg = np.asarray(x[a:b], dtype=np.float64)
    if seg.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    denv = np.diff(seg, prepend=seg[:1]) * fs
    return float(seg.mean()), float(seg.max()), float(seg.std()), float(denv[: max(1, int(round(0.05 * fs)))].mean())


def running_median_logf0(f0_hz: np.ndarray, voiced: np.ndarray, win_s: float = 2.0) -> np.ndarray:
    v3 = load_v3()
    return v3.running_median(
        np.where(voiced & np.isfinite(f0_hz) & (f0_hz > 0), np.log(f0_hz), np.nan),
        int(round(win_s * FS)),
    )
