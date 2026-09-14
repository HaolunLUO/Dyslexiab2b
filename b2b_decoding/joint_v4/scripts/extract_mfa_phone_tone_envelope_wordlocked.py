#!/usr/bin/env python3
"""Word-locked MFA Mandarin phones + tone + envelope on existing lPPCN onsets.

Keeps the shared word_timing_relative.csv events (1753 + 1800 words). MFA
(or a pypinyin G2P fallback) only supplies phone/tone *labels and durations*
inside those intervals. Envelope is taken from the section WAV at the same
onsets.

v2 outputs (do not overwrite legacy keys):
  X_word_envelope_v2.npy
  X_word_mfa_phones_v2.npy      (or X_word_g2p_phones_v2.npy)
  X_word_mfa_tone_v2.npy        (or X_word_g2p_tone_v2.npy)
plus feature-name, meta, TextGrid, and QC sidecars.

Examples:
  python scripts/extract_mfa_phone_tone_envelope_wordlocked.py --aligner g2p
  python scripts/extract_mfa_phone_tone_envelope_wordlocked.py --aligner mfa \\
      --mfa-sif /path/to/mfa.sif --mfa-root /orcd/pool/005/haolun52/mfa_root
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import hilbert

POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
EXPECTED = {1: 1753, 2: 1800}
ENV_FS = 100.0
N_ENV_FRAMES = 10          # 200 ms @ 50 Hz, same span as Whisper encoder windows
ENV_FRAME_HZ = 50.0
SLOPE_WIN_S = 0.050
FEATURE_TAG = "v2"
SILENCE = frozenset({"", "sil", "sp", "spn", "pau", "<unk>", "unk"})
# Longest-suffix Chao mapping for MFA Mandarin (v3 phone set).
# Order matters: longer contours must precede single-level marks.
CHAO_TO_TONE = (
    ("˨˩˦", "3"),  # dipping (Mandarin tone 3)
    ("˧˥", "2"),   # rising (tone 2)
    ("˥˩", "4"),   # falling (tone 4)
    ("˥", "1"),    # high (tone 1)
    ("˧", "5"),    # mid / neutral
    ("˩", "3"),    # low; treated as tone-3 alternate in MFA inventories
)
TONE_CHARS = frozenset("˥˧˩˨˦")
PINYIN_INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)
STD_EPS = 1e-12


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_timing(section_dir: Path, section_id: int) -> list[dict]:
    csv_path = section_dir / "word_timing_relative.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing timing csv: {csv_path}")
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n != EXPECTED[section_id]:
        raise SystemExit(f"section {section_id}: {n} words != {EXPECTED[section_id]}")
    out = []
    for i, r in enumerate(rows):
        on = float(r["onset_relative"])
        off = float(r["offset_relative"])
        if off < on:
            raise SystemExit(f"section {section_id} word {i}: offset < onset")
        out.append({
            "i": i,
            "word": str(r["word"]),
            "onset": on,
            "offset": off if off > on else on + 1e-3,
        })
    return out


def _resolve_wav(section_dir: Path, section_id: int) -> Path:
    candidates = [
        section_dir / f"task-lppCN_section_{section_id}.wav",
        POOL / f"task-lppCN_section_{section_id}.wav",
        section_dir / f"task-lppCN_section_{section_id:03d}.wav",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise SystemExit("No wav for section "
                     f"{section_id}. Tried:\n  " + "\n  ".join(str(c) for c in candidates))


def _write_feature(section_dir: Path, key: str, X: np.ndarray, names: list[str], meta: dict) -> None:
    if X.ndim != 2:
        raise SystemExit(f"{key}: expected 2d array, got {X.shape}")
    if X.shape[1] != len(names):
        raise SystemExit(f"{key}: {X.shape[1]} cols != {len(names)} names")
    if not np.isfinite(X).all():
        n_bad = int(np.sum(~np.isfinite(X)))
        raise SystemExit(f"{key}: {n_bad} non-finite values")
    np.save(section_dir / f"X_word_{key}.npy", X.astype(np.float64))
    (section_dir / f"X_word_{key}_feature_names.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    (section_dir / f"X_word_{key}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  wrote X_word_{key}.npy  shape={tuple(X.shape)}", flush=True)


def drop_constant_columns(
    X: np.ndarray, names: list[str], *, eps: float = STD_EPS
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop columns with near-zero std across the concatenated story."""
    keep = []
    dropped = []
    for j, name in enumerate(names):
        if float(np.std(X[:, j], ddof=0)) < eps:
            dropped.append(name)
        else:
            keep.append(j)
    if not keep:
        raise SystemExit(f"All columns constant among {names}")
    return X[:, keep], [names[j] for j in keep], dropped


def drop_half_fragile_columns(
    X: np.ndarray,
    names: list[str],
    sec_lengths: list[int],
    *,
    n_blocks: int = 8,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop columns that are all-zero on either canonical 4/8 blocked half.

    Matches B2B section-wise blocked partitions enough to keep OLS-H identifiable
    on held-out halves without ridge.
    """
    offsets = np.cumsum([0] + list(sec_lengths))

    def half_idx(half_blocks: tuple[int, ...]) -> np.ndarray:
        parts = []
        for i, n_sec in enumerate(sec_lengths):
            edges = np.round(np.linspace(0, n_sec, n_blocks + 1)).astype(int)
            local = []
            for b in half_blocks:
                local.extend(range(int(edges[b]), int(edges[b + 1])))
            parts.append(np.asarray(local, dtype=int) + int(offsets[i]))
        return np.concatenate(parts)

    h1 = half_idx((0, 1, 2, 3))
    h2 = half_idx((4, 5, 6, 7))
    keep, dropped = [], []
    for j, name in enumerate(names):
        if not (np.any(np.abs(X[h1, j]) > 0) and np.any(np.abs(X[h2, j]) > 0)):
            dropped.append(name)
        else:
            keep.append(j)
    if not keep:
        raise SystemExit("All columns fragile across blocked halves")
    return X[:, keep], [names[j] for j in keep], dropped


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def _resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    try:
        import librosa
        return librosa.resample(
            np.asarray(x, dtype=np.float64),
            orig_sr=int(round(fs_in)),
            target_sr=int(round(fs_out)),
        )
    except Exception:
        g = np.gcd(int(round(fs_in)), int(round(fs_out)))
        up = int(round(fs_out)) // g
        down = int(round(fs_in)) // g
        return signal.resample_poly(np.asarray(x, dtype=np.float64), up, down)


def envelope_hilbert(audio: np.ndarray, fs_audio: float, fs_target: float = ENV_FS) -> np.ndarray:
    """Broadband Hilbert envelope, log1p-compressed (same as regress_out_acoustic_mne).

    Hilbert is computed at 16 kHz, not native 44.1 kHz — equivalent for
    a 100 Hz envelope and much faster on ~10 min section WAVs.
    """
    fs_hilbert = 16000.0
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if abs(fs_audio - fs_hilbert) > 1.0:
        x = _resample_1d(x, fs_audio, fs_hilbert)
    else:
        fs_hilbert = fs_audio
    env = np.abs(hilbert(x))
    env = _resample_1d(env, fs_hilbert, fs_target)
    return np.log1p(np.maximum(env, 0.0))


def word_envelope_features(env: np.ndarray, words: list[dict], fs: float = ENV_FS) -> tuple[np.ndarray, list[str]]:
    """Onset-locked envelope summary. env_t00 is the onset sample (no env_onset dup)."""
    n = len(words)
    names = [
        "env_mean", "env_max", "env_std", "env_slope_50ms",
    ] + [f"env_t{k:02d}" for k in range(N_ENV_FRAMES)]
    X = np.zeros((n, len(names)), dtype=np.float64)
    n_env = env.size
    slope_n = max(1, int(round(SLOPE_WIN_S * fs)))

    for i, w in enumerate(words):
        a = max(0, min(n_env, int(round(w["onset"] * fs))))
        b = max(a + 1, min(n_env, int(round(w["offset"] * fs))))
        seg = env[a:b]
        X[i, 0] = float(seg.mean())
        X[i, 1] = float(seg.max())
        X[i, 2] = float(seg.std(ddof=0))
        a2 = min(n_env - 1, a + slope_n)
        X[i, 3] = float((env[a2] - env[a]) / max(a2 - a, 1) * fs)
        for k in range(N_ENV_FRAMES):
            t = w["onset"] + (k / ENV_FRAME_HZ)
            idx = int(round(t * fs))
            idx = max(0, min(n_env - 1, idx))
            X[i, 4 + k] = float(env[idx])
    return X, names


# ---------------------------------------------------------------------------
# Phone / tone parsing
# ---------------------------------------------------------------------------

def split_phone_tone(label: str) -> tuple[str, str | None]:
    """Split an MFA / digit-tagged phone into (base_phone, tone).

    Tone is stripped only as a *suffix* (longest Chao contour first, else [1-5]).
    Residual Chao marks in the base phone are a hard error for callers to catch.
    """
    s = (label or "").strip()
    if s.lower() in SILENCE or s == "":
        return "sil", None
    m = re.search(r"([1-5])$", s)
    if m:
        return s[: m.start()], m.group(1)
    for mark, tone in CHAO_TO_TONE:
        if s.endswith(mark):
            return s[: -len(mark)], tone
    return s, None


def phone_has_tone_contour(phone: str) -> bool:
    return any(c in phone for c in TONE_CHARS)


def parse_textgrid_intervals(path: Path, tier_name: str | None = None) -> list[tuple[float, float, str]]:
    """Parse Praat long or short TextGrid interval tiers."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Long format
    items = re.split(r"item\s*\[\s*\d+\s*\]\s*:", text)
    intervals: list[tuple[float, float, str]] = []
    if len(items) > 1:
        for item in items[1:]:
            cls = re.search(r'class\s*=\s*"([^"]+)"', item)
            name = re.search(r'name\s*=\s*"([^"]+)"', item)
            if cls and "IntervalTier" not in cls.group(1):
                continue
            if tier_name is not None and name and name.group(1).lower() != tier_name.lower():
                continue
            if tier_name is not None and name is None:
                continue
            for m in re.finditer(
                r"intervals\s*\[\s*\d+\s*\]\s*:\s*"
                r"xmin\s*=\s*([^\n]+)\s*"
                r"xmax\s*=\s*([^\n]+)\s*"
                r'text\s*=\s*"([^"]*)"',
                item,
            ):
                intervals.append((float(m.group(1)), float(m.group(2)), m.group(3)))
            if intervals:
                return intervals
    # Short format: "IntervalTier" \n "phones" \n xmin xmax n  then triples
    if '"IntervalTier"' in text or "IntervalTier" in text:
        parts = re.split(r'"IntervalTier"', text)
        for part in parts[1:]:
            toks = _short_textgrid_tokens(part)
            if len(toks) < 5:
                continue
            name = toks[0].strip('"')
            if tier_name is not None and name.lower() != tier_name.lower():
                continue
            try:
                n_int = int(float(toks[3]))
            except ValueError:
                continue
            vals = toks[4:]
            got = []
            for j in range(n_int):
                if 3 * j + 2 >= len(vals):
                    break
                xmin = float(vals[3 * j])
                xmax = float(vals[3 * j + 1])
                lab = vals[3 * j + 2].strip('"')
                got.append((xmin, xmax, lab))
            if got:
                return got
    raise SystemExit(f"Could not parse IntervalTier {tier_name!r} from {path}")


def _short_textgrid_tokens(part: str) -> list[str]:
    return re.findall(r'"[^"]*"|[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', part)


def assign_phones_to_words(
    phones: list[tuple[float, float, str]],
    words: list[dict],
    min_overlap: float = 0.01,
) -> list[list[tuple[float, float, str, str | None]]]:
    """Exclusive assignment: each non-silence phone → word with max overlap.

    Durations are *clipped* to the overlap with the chosen word so boundary-
    straddling phones do not inflate occupancy beyond the word interval.
    """
    assigned: list[list[tuple[float, float, str, str | None]]] = [[] for _ in words]
    onsets = np.array([w["onset"] for w in words], dtype=np.float64)
    offsets = np.array([w["offset"] for w in words], dtype=np.float64)
    for xmin, xmax, raw in phones:
        phone, tone = split_phone_tone(raw)
        if phone == "sil":
            continue
        if phone_has_tone_contour(phone):
            raise SystemExit(
                f"Tone contour left in phone base after split: raw={raw!r} base={phone!r}"
            )
        ov = np.minimum(offsets, xmax) - np.maximum(onsets, xmin)
        j = int(np.argmax(ov))
        if ov[j] < min_overlap:
            continue
        clip_on = max(float(xmin), float(onsets[j]))
        clip_off = min(float(xmax), float(offsets[j]))
        if clip_off <= clip_on:
            continue
        assigned[j].append((clip_on, clip_off, phone, tone))
    return assigned


# ---------------------------------------------------------------------------
# G2P fallback (pypinyin) — same word intervals, dictionary phones/tones
# ---------------------------------------------------------------------------

def _split_initial_final(syl: str) -> tuple[str, str]:
    s = syl.replace("ü", "v").replace("u:", "v")
    for init in PINYIN_INITIALS:
        if s.startswith(init) and len(s) > len(init):
            return init, s[len(init):]
    return "", s


def g2p_word_phones(word: str) -> list[tuple[str, str | None]]:
    try:
        from pypinyin import Style, pinyin
    except ImportError as e:
        raise SystemExit("pypinyin is required for --aligner g2p. "
                         "Install: pip install pypinyin") from e
    tones = pinyin(word, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
    out: list[tuple[str, str | None]] = []
    for tok in tones:
        syl = (tok[0] if tok else "").strip().lower()
        if not syl:
            continue
        m = re.match(r"^([a-züv:]+)([1-5])?$", syl.replace("ü", "v"))
        if not m:
            out.append((re.sub(r"[^a-z]", "", syl) or "unk", None))
            continue
        body, tone = m.group(1), m.group(2)
        init, final = _split_initial_final(body)
        if init:
            out.append((init, None))
        if final:
            out.append((final, tone))
        elif tone:
            out.append(("spn", tone))
    return out or [("unk", None)]


def g2p_assign(words: list[dict]) -> list[list[tuple[float, float, str, str | None]]]:
    assigned = []
    for w in words:
        seq = g2p_word_phones(w["word"])
        dur = max(w["offset"] - w["onset"], 1e-3)
        step = dur / len(seq)
        items = []
        for k, (ph, tone) in enumerate(seq):
            xmin = w["onset"] + k * step
            xmax = w["onset"] + (k + 1) * step
            items.append((xmin, xmax, ph, tone))
        assigned.append(items)
    return assigned


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

# Rare phones make blocked half-partitions singular under OLS H. Keep phones /
# first-phone one-hots that appear in enough words; collapse the rest to other.
DEFAULT_MIN_PHONE_WORDS = 100
DEFAULT_MIN_PH1_WORDS = 400
DEFAULT_MIN_TONE_WORDS = 100


def _phone_word_counts(
    assigned_by_section: dict[int, list[list[tuple[float, float, str, str | None]]]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (any-position word counts, first-phone word counts)."""
    counts: dict[str, int] = {}
    first_counts: dict[str, int] = {}
    for assigned in assigned_by_section.values():
        for items in assigned:
            seen_word: set[str] = set()
            first_ph = None
            for _, _, ph, _ in items:
                if ph == "sil" or phone_has_tone_contour(ph):
                    continue
                if first_ph is None:
                    first_ph = ph
                if ph in seen_word:
                    continue
                seen_word.add(ph)
                counts[ph] = counts.get(ph, 0) + 1
            if first_ph is not None:
                first_counts[first_ph] = first_counts.get(first_ph, 0) + 1
    return counts, first_counts


def build_phone_tone_matrices(
    assigned_by_section: dict[int, list[list[tuple[float, float, str, str | None]]]],
    words_by_section: dict[int, list[dict]],
    *,
    min_phone_words: int = DEFAULT_MIN_PHONE_WORDS,
    min_ph1_words: int = DEFAULT_MIN_PH1_WORDS,
    min_tone_words: int = DEFAULT_MIN_TONE_WORDS,
) -> tuple[dict[int, np.ndarray], list[str], dict[int, np.ndarray], list[str], list[str], dict]:
    raw_counts, first_counts = _phone_word_counts(assigned_by_section)
    for ph in raw_counts:
        if phone_has_tone_contour(ph):
            raise SystemExit(f"Phone inventory retains tone mark: {ph!r}")
    if not raw_counts:
        raise SystemExit("No phones found (alignment empty?)")

    kept_phones = sorted(p for p, c in raw_counts.items() if c >= min_phone_words)
    rare_phones = sorted(p for p, c in raw_counts.items() if c < min_phone_words)
    # Gate first-phone one-hots on *first-phone* frequency, not any-position count.
    ph1_phones = sorted(
        p for p in kept_phones if first_counts.get(p, 0) >= min_ph1_words
    )
    use_other = bool(rare_phones)
    phones = list(kept_phones) + (["other"] if use_other else [])
    if not phones:
        raise SystemExit(
            f"No phones meet min_phone_words={min_phone_words}; "
            f"top counts={sorted(raw_counts.items(), key=lambda kv: -kv[1])[:10]}"
        )

    phone_index = {p: i for i, p in enumerate(phones)}
    ph1_index = {p: i for i, p in enumerate(ph1_phones)}
    n_ph = len(phones)
    n_ph1 = len(ph1_phones)
    # No dependent "coverage" column; keep n_phones + mean_phone_dur.
    phone_names = (
        [f"ph_{p}" for p in phones]
        + [f"ph1_{p}" for p in ph1_phones]
        + ["n_phones", "mean_phone_dur"]
    )
    tone_names = (
        [f"tone_{t}" for t in "12345"]
        + [f"tone1_{t}" for t in "12345"]
        + ["n_syllables", "tone_change"]
    )

    X_phone: dict[int, np.ndarray] = {}
    X_tone: dict[int, np.ndarray] = {}
    qc = {
        "n_phones_assigned": 0,
        "n_words_no_phone": 0,
        "occupancy_sum_max": 0.0,
        "occupancy_sum_mean": 0.0,
        "tone_counts": {t: 0 for t in "12345"},
        "min_phone_words": min_phone_words,
        "min_ph1_words": min_ph1_words,
        "min_tone_words": min_tone_words,
        "kept_phones": kept_phones,
        "rare_phones": rare_phones,
        "ph1_phones": ph1_phones,
        "raw_phone_word_counts": raw_counts,
        "first_phone_word_counts": first_counts,
    }
    occ_sums_all: list[float] = []

    for sid, assigned in assigned_by_section.items():
        words = words_by_section[sid]
        n = len(words)
        xp = np.zeros((n, len(phone_names)), dtype=np.float64)
        xt = np.zeros((n, len(tone_names)), dtype=np.float64)
        for i, items in enumerate(assigned):
            wdur = max(words[i]["offset"] - words[i]["onset"], 1e-6)
            durs = []
            tones_here: list[str] = []
            first_ph = None
            first_tone = None
            for xmin, xmax, ph, tone in items:
                d = max(xmax - xmin, 0.0)
                if d <= 0:
                    continue
                durs.append(d)
                qc["n_phones_assigned"] += 1
                key = ph if ph in phone_index else ("other" if use_other else None)
                if key is not None:
                    j = phone_index[key]
                    xp[i, j] += d / wdur
                    if first_ph is None:
                        first_ph = key
                        if key in ph1_index:
                            xp[i, n_ph + ph1_index[key]] = 1.0
                if tone is not None and tone in "12345":
                    tones_here.append(tone)
                    qc["tone_counts"][tone] += 1
                    xt[i, int(tone) - 1] += d / wdur
                    if first_tone is None:
                        first_tone = tone
                        xt[i, 5 + int(tone) - 1] = 1.0
            occ = float(xp[i, :n_ph].sum())
            occ_sums_all.append(occ)
            xp[i, n_ph + n_ph1 + 0] = float(len(items))
            xp[i, n_ph + n_ph1 + 1] = float(np.mean(durs)) if durs else 0.0
            xt[i, 10] = float(len(tones_here))
            xt[i, 11] = float(len(set(tones_here)) > 1)
            if first_ph is None:
                qc["n_words_no_phone"] += 1
        X_phone[sid] = xp
        X_tone[sid] = xt

    if occ_sums_all:
        qc["occupancy_sum_max"] = float(max(occ_sums_all))
        qc["occupancy_sum_mean"] = float(np.mean(occ_sums_all))

    # Drop sparse tone columns (e.g. rare neutral) that break blocked halves.
    tone_stack = np.vstack([X_tone[sid] for sid in sorted(X_tone)])
    tone_nnz = (np.abs(tone_stack) > 0).sum(axis=0)
    tone_keep = [j for j, _n in enumerate(tone_names) if int(tone_nnz[j]) >= min_tone_words]
    tone_dropped_sparse = [tone_names[j] for j in range(len(tone_names)) if j not in tone_keep]
    if not tone_keep:
        raise SystemExit("All tone columns below min_tone_words")
    for sid in list(X_tone):
        X_tone[sid] = X_tone[sid][:, tone_keep]
    tone_names = [tone_names[j] for j in tone_keep]
    qc["dropped_sparse_tone_columns"] = tone_dropped_sparse
    return X_phone, phone_names, X_tone, tone_names, phones, qc


# ---------------------------------------------------------------------------
# MFA
# ---------------------------------------------------------------------------

def _mfa_cmd(args: argparse.Namespace, extra: list[str]) -> list[str]:
    if args.mfa_sif:
        sif = str(Path(args.mfa_sif).resolve())
        cmd = [
            "apptainer", "exec",
            "--bind", f"{POOL}:{POOL}",
            "--pwd", str(Path(args.work_dir).resolve()),
        ]
        if args.mfa_root:
            cmd += ["--env", f"MFA_ROOT_DIR={args.mfa_root}"]
        cmd += [sif, "mfa"]
        return cmd + extra
    mfa = args.mfa_bin or shutil.which("mfa")
    if not mfa:
        raise SystemExit("mfa not on PATH; pass --mfa-sif or --mfa-bin")
    return [mfa] + extra


def prepare_mfa_corpus(work_dir: Path, shared: Path, sections: Iterable[int]) -> Path:
    corpus = work_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for sid in sections:
        sec = shared / f"section_{sid:03d}"
        words = _load_timing(sec, sid)
        wav = _resolve_wav(sec, sid)
        dst_wav = corpus / f"section_{sid:03d}.wav"
        if dst_wav.is_symlink() or dst_wav.exists():
            dst_wav.unlink()
        dst_wav.symlink_to(wav.resolve())
        lab = " ".join(w["word"] for w in words)
        (corpus / f"section_{sid:03d}.lab").write_text(lab + "\n", encoding="utf-8")
    return corpus


def run_mfa_align(args: argparse.Namespace, corpus: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.mfa_root:
        env["MFA_ROOT_DIR"] = args.mfa_root
        Path(args.mfa_root).mkdir(parents=True, exist_ok=True)

    def run(extra: list[str]) -> None:
        cmd = _mfa_cmd(args, extra)
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)

    run(["model", "download", "acoustic", args.acoustic_model])
    run(["model", "download", "dictionary", args.dictionary])
    run(["model", "download", "g2p", args.g2p_model])
    align = [
        "align", str(corpus), args.dictionary, args.acoustic_model, str(out_dir),
        "--clean", "--overwrite",
        "--single_speaker",
        "--num_jobs", str(args.num_jobs),
        "--beam", str(args.beam),
        "--g2p_model_path", args.g2p_model,
        # MFA 3.4 auto-detects Chinese and requires spacy-pkuseg inside the
        # container. Our .lab files are already space-tokenized words.
        "--no_tokenization",
    ]
    run(align)


def load_mfa_phones(textgrid_dir: Path, section_id: int) -> list[tuple[float, float, str]]:
    candidates = [
        textgrid_dir / f"section_{section_id:03d}.TextGrid",
        textgrid_dir / f"section_{section_id:03d}.textgrid",
    ]
    # MFA sometimes nests by speaker
    if not any(p.is_file() for p in candidates):
        found = list(textgrid_dir.rglob(f"section_{section_id:03d}.TextGrid"))
        found += list(textgrid_dir.rglob(f"section_{section_id:03d}.textgrid"))
        if found:
            candidates = found
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise SystemExit(f"No TextGrid for section {section_id} under {textgrid_dir}")
    for tier in ("phones", "phone", "Phones"):
        try:
            return parse_textgrid_intervals(path, tier)
        except SystemExit:
            continue
    # Last resort: first interval tier
    return parse_textgrid_intervals(path, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-dir", type=str, default=str(DEFAULT_SHARED))
    ap.add_argument("--sections", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--aligner", choices=["mfa", "g2p", "auto"], default="auto")
    ap.add_argument("--work-dir", type=str,
                    default=str(POOL / "mfa_lppcn_wordlocked"))
    ap.add_argument("--mfa-sif", type=str, default="")
    ap.add_argument("--mfa-bin", type=str, default="")
    ap.add_argument("--mfa-root", type=str, default=str(POOL / "mfa_root"))
    ap.add_argument("--textgrid-dir", type=str, default="")
    ap.add_argument("--acoustic-model", type=str, default="mandarin_mfa")
    ap.add_argument("--dictionary", type=str, default="mandarin_china_mfa")
    ap.add_argument("--g2p-model", type=str, default="mandarin_china_mfa")
    ap.add_argument("--num-jobs", type=int, default=4)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--skip-align", action="store_true",
                    help="Reuse existing TextGrids in --textgrid-dir / work-dir/aligned")
    ap.add_argument("--feature-tag", type=str, default=FEATURE_TAG,
                    help="Suffix on written keys (default: v2)")
    ap.add_argument("--min-phone-words", type=int, default=DEFAULT_MIN_PHONE_WORDS,
                    help="Collapse phones below this story-wide word count into ph_other")
    ap.add_argument("--min-ph1-words", type=int, default=DEFAULT_MIN_PH1_WORDS,
                    help="Keep first-phone one-hots only for phones at/above this count")
    ap.add_argument("--min-tone-words", type=int, default=DEFAULT_MIN_TONE_WORDS,
                    help="Drop tone columns with fewer non-zero words")
    args = ap.parse_args()
    tag = args.feature_tag.strip()
    if not tag:
        raise SystemExit("--feature-tag must be non-empty (use v2)")

    shared = Path(args.shared_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    tg_dir = Path(args.textgrid_dir) if args.textgrid_dir else work / "aligned"

    words_by_section = {}
    wav_by_section = {}
    for sid in args.sections:
        sec = shared / f"section_{sid:03d}"
        words_by_section[sid] = _load_timing(sec, sid)
        wav_by_section[sid] = _resolve_wav(sec, sid)

    have_tg = all(
        list(tg_dir.rglob(f"section_{sid:03d}.TextGrid"))
        or list(tg_dir.rglob(f"section_{sid:03d}.textgrid"))
        for sid in args.sections
    ) if tg_dir.is_dir() else False

    aligner = args.aligner
    if aligner == "auto":
        if have_tg or args.mfa_sif or args.mfa_bin or shutil.which("mfa"):
            aligner = "mfa"
        else:
            aligner = "g2p"
            print("WARNING: MFA not available; using pypinyin G2P phones/tones "
                  "inside existing word intervals (not acoustic alignment).",
                  flush=True)

    assigned: dict[int, list] = {}
    if aligner == "mfa":
        if not (have_tg and args.skip_align):
            corpus = prepare_mfa_corpus(work, shared, args.sections)
            run_mfa_align(args, corpus, tg_dir)
        for sid in args.sections:
            phones = load_mfa_phones(tg_dir, sid)
            assigned[sid] = assign_phones_to_words(phones, words_by_section[sid])
            n_ph = sum(len(x) for x in assigned[sid])
            n_empty = sum(1 for x in assigned[sid] if not x)
            print(f"  MFA section {sid}: {len(phones)} phone intervals, "
                  f"{n_ph} assigned (clipped), {n_empty} words with no phone", flush=True)
        phone_key, tone_key = f"mfa_phones_{tag}", f"mfa_tone_{tag}"
    else:
        for sid in args.sections:
            assigned[sid] = g2p_assign(words_by_section[sid])
        phone_key, tone_key = f"g2p_phones_{tag}", f"g2p_tone_{tag}"
    env_key = f"envelope_{tag}"

    X_phone, phone_names, X_tone, tone_names, phone_inv, build_qc = build_phone_tone_matrices(
        assigned, words_by_section,
        min_phone_words=args.min_phone_words,
        min_ph1_words=args.min_ph1_words,
        min_tone_words=args.min_tone_words,
    )

    # Drop globally constant columns across the concatenated story (story-level).
    phone_stack = np.vstack([X_phone[sid] for sid in args.sections])
    tone_stack = np.vstack([X_tone[sid] for sid in args.sections])
    phone_stack, phone_names, phone_dropped = drop_constant_columns(phone_stack, phone_names)
    tone_stack, tone_names, tone_dropped = drop_constant_columns(tone_stack, tone_names)
    sec_lengths = [len(words_by_section[sid]) for sid in args.sections]
    phone_stack, phone_names, phone_fragile = drop_half_fragile_columns(
        phone_stack, phone_names, sec_lengths
    )
    tone_stack, tone_names, tone_fragile = drop_half_fragile_columns(
        tone_stack, tone_names, sec_lengths
    )
    phone_dropped = phone_dropped + phone_fragile
    tone_dropped = tone_dropped + tone_fragile
    # Split back
    offsets = np.cumsum([0] + sec_lengths)
    for i, sid in enumerate(args.sections):
        X_phone[sid] = phone_stack[offsets[i]:offsets[i + 1]]
        X_tone[sid] = tone_stack[offsets[i]:offsets[i + 1]]

    if any(phone_has_tone_contour(p) for p in phone_inv):
        raise SystemExit("Phone inventory still contains Chao tone marks")
    if not any(build_qc["tone_counts"][t] > 0 for t in "1234"):
        raise SystemExit("No lexical tones (1–4) found after Chao parsing — check mapping")
    if build_qc["occupancy_sum_max"] > 1.0 + 1e-6:
        raise SystemExit(
            f"Clipped occupancy sum exceeds 1.0 (max={build_qc['occupancy_sum_max']:.4f})"
        )

    for sid in args.sections:
        sec = shared / f"section_{sid:03d}"
        wav_path = wav_by_section[sid]
        print(f"  envelope section {sid} from {wav_path.name} …", flush=True)
        audio, fs = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        env = envelope_hilbert(audio, float(fs), ENV_FS)
        X_env, env_names = word_envelope_features(env, words_by_section[sid], ENV_FS)

        n_empty = int(sum(1 for x in assigned[sid] if not x))
        meta_common = {
            "section_id": sid,
            "n_words": len(words_by_section[sid]),
            "aligner": aligner,
            "feature_tag": tag,
            "wav": str(wav_path),
            "timing_csv": str(sec / "word_timing_relative.csv"),
            "same_onsets": True,
            "n_words_no_phone": n_empty,
            "phone_duration_clipped_to_word": True,
            "occupancy_sum_mean": build_qc["occupancy_sum_mean"],
            "occupancy_sum_max": build_qc["occupancy_sum_max"],
            "tone_counts": build_qc["tone_counts"],
        }
        _write_feature(sec, env_key, X_env, env_names, {
            **meta_common,
            "family": "envelope",
            "method": "hilbert_log1p",
            "env_fs": ENV_FS,
            "n_onset_frames": N_ENV_FRAMES,
            "onset_frame_hz": ENV_FRAME_HZ,
            "note": "env_t00 is onset sample; no separate env_onset column",
        })
        _write_feature(sec, phone_key, X_phone[sid], phone_names, {
            **meta_common,
            "family": "phones",
            "phone_inventory": phone_inv,
            "coding": "clipped_duration_weighted_occupancy + first_phone_onehot "
                      "(rare phones collapsed to other)",
            "kept_phones": build_qc.get("kept_phones"),
            "rare_phones": build_qc.get("rare_phones"),
            "ph1_phones": build_qc.get("ph1_phones"),
            "min_phone_words": build_qc.get("min_phone_words"),
            "min_ph1_words": build_qc.get("min_ph1_words"),
            "dropped_constant_columns": phone_dropped,
        })
        _write_feature(sec, tone_key, X_tone[sid], tone_names, {
            **meta_common,
            "family": "tone",
            "coding": "clipped Chao / digit tone occupancy + first_tone_onehot "
                      "(1-4 lexical, 5 neutral; sparse tones dropped)",
            "chao_map": [{"suffix": a, "tone": b} for a, b in CHAO_TO_TONE],
            "dropped_sparse_tone_columns": build_qc.get("dropped_sparse_tone_columns"),
            "min_tone_words": build_qc.get("min_tone_words"),
            "dropped_constant_columns": tone_dropped,
        })

        long_path = work / f"section_{sid:03d}_phones_{tag}_long.csv"
        with long_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["word_index", "word", "word_onset", "word_offset",
                         "phone_onset", "phone_offset", "phone", "tone"])
            for i, items in enumerate(assigned[sid]):
                w = words_by_section[sid][i]
                for xmin, xmax, ph, tone in items:
                    wr.writerow([i, w["word"], w["onset"], w["offset"],
                                 xmin, xmax, ph, tone or ""])

        qc = {
            **meta_common,
            "phone_inventory": phone_inv,
            "n_phone_types": len(phone_inv),
            "dropped_phone_columns": phone_dropped,
            "dropped_tone_columns": tone_dropped,
            "long_csv": str(long_path),
        }
        (sec / f"X_word_{phone_key}_qc.json").write_text(
            json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"section {sid}: empty_words={n_empty}  n_phone_types={len(phone_inv)}  "
              f"tone_counts={build_qc['tone_counts']}  "
              f"occ_max={build_qc['occupancy_sum_max']:.3f}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
