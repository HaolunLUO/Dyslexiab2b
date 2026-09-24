"""Shared paths and montage for the Storytime N400 analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
SHARED = POOL / "extracted_sections_wordlocked_shared"
WORD_ROOT = SHARED / "_shared_wordlocked_features"
SCRIPT_DIR = Path(__file__).resolve().parent
SUBJECTS_TXT = POOL / "dyslexia_natualistics_listing" / "b2b_decoding" / "subjects.txt"
GROUPS_CSV = (
    POOL
    / "dyslexia_natualistics_listing"
    / "b2b_decoding"
    / "joint_v4"
    / "cohort_groups.csv"
)
BEHAVIOR_XLSX = POOL / "dyslexia_natualistics_listing" / "behavioral_data.xlsx"
OUT_ROOT = POOL / "n400_storytime_v1"
STIM_DIR = OUT_ROOT / "stimulus"
EMBED_VEC = STIM_DIR / "embeddings" / "wiki.zh.vec"

SECTIONS = (1, 2)
SFREQ = 100.0
TMIN = -0.2
TMAX = 1.0
BASELINE = (-0.2, 0.0)
N400_WIN = (0.35, 0.55)
CP_CHANNELS = ("Pz", "Cz", "Oz")
FZ_CHANNEL = "Fz"

# Row order used by joint_v4/eeg_band.jl when metadata has no channel_names.
WORDLOCKED_CHAN_65 = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT7", "FT8", "FT9", "FT10",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "TP9", "TP10",
    "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]

UD_CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# Jieba flags counted as N / V / ADJ / ADV, plus close content classes
# (idiom, set phrase, distinguish-word, state-word, time word, place word).
JIEBA_CONTENT_EXACT = {"i", "l", "b", "z", "t", "s"}


def epoch_times(sfreq: float = SFREQ) -> np.ndarray:
    n = int(round((TMAX - TMIN) * sfreq)) + 1
    return np.round(TMIN + np.arange(n) / sfreq, 5)


def load_subjects() -> list[str]:
    return [ln.strip() for ln in SUBJECTS_TXT.read_text().splitlines() if ln.strip()]


def load_groups() -> pd.DataFrame:
    g = pd.read_csv(GROUPS_CSV)
    g["participant"] = g["participant"].astype(str).str.strip()
    g = g.loc[g["include_primary"].astype(int) == 1, ["participant", "group"]].copy()
    return g.reset_index(drop=True)


def parse_age_value(v) -> float:
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


def load_ages(path: Path = BEHAVIOR_XLSX) -> pd.DataFrame:
    """Age in years from Sheet1 column 年龄, values stored as 年;月."""
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "编号" not in df.columns or "年龄" not in df.columns:
        raise SystemExit(f"behavioral workbook missing 编号/年龄: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "participant": df["编号"].astype(str).str.strip(),
            "age": [parse_age_value(v) for v in df["年龄"]],
        }
    )
    out = out.dropna(subset=["participant"])
    out = out.groupby("participant", as_index=False)["age"].first()
    return out


def jieba_is_content(flag: str) -> bool:
    f = (flag or "").strip().lower()
    if not f or f == "x" or f == "split":
        return False
    if f in JIEBA_CONTENT_EXACT:
        return True
    return f[0] in {"n", "v", "a", "d"}


def jieba_tag_token(word: str) -> tuple[str, bool]:
    """Tag one existing token without resegmenting the story.

    Returns (flag, atomic). atomic is False when jieba splits the token,
    which would break 1:1 alignment with GPT2-CN surprisal.
    """
    import jieba.posseg as pseg

    pairs = list(pseg.cut(word, HMM=False))
    if len(pairs) != 1:
        return "+".join(p.flag for p in pairs), False
    return pairs[0].flag, True


def ud_tags_for_section(section: int) -> list[str]:
    sec = f"section_{section:03d}"
    names = (WORD_ROOT / sec / "X_word_wordinfo_feature_names.txt").read_text().split()
    wi = np.load(WORD_ROOT / sec / "X_word_wordinfo.npy")
    pos_idx = [i for i, n in enumerate(names) if n.startswith("pos_")]
    tags = []
    for row in wi:
        hit = [names[j][4:] for j in pos_idx if row[j] > 0.5]
        tags.append(hit[0] if hit else "X")
    return tags


def read_word_timing(section: int) -> list[dict]:
    path = WORD_ROOT / f"section_{section:03d}" / "word_timing_relative.csv"
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def confirm_montage(n_ch: int) -> dict:
    if n_ch != len(WORDLOCKED_CHAN_65):
        raise SystemExit(
            f"EEG has {n_ch} rows; WORDLOCKED_CHAN_65 has {len(WORDLOCKED_CHAN_65)}. "
            "Refusing to guess a montage."
        )
    missing = [ch for ch in (*CP_CHANNELS, FZ_CHANNEL) if ch not in WORDLOCKED_CHAN_65]
    if missing:
        raise SystemExit(f"Required midline channels absent: {missing}")
    idx = {ch: WORDLOCKED_CHAN_65.index(ch) for ch in (*CP_CHANNELS, FZ_CHANNEL)}
    return {"n_ch": n_ch, "names": list(WORDLOCKED_CHAN_65), "index": idx}
