#!/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
"""Stimulus table for v1-strict. Does not read EEG and does not fit a model.

Sentence boundaries, concreteness, cloze probability, and item-level z-scores.
The spec is docs/n400-v1-strict-spec.md and is not revised from this script.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
SRC = POOL / "n400_storytime_v1" / "stimulus" / "word_table.csv"
OUT = POOL / "n400_storytime_v1" / "v1_strict" / "stimulus"
PUNCT = OUT / "punct_xiaowangzi.txt"
NORM = OUT / "xu_li_2020_concreteness.xlsx"
ANCHOR = "当我还只有六岁"


def _population_z(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    m = float(np.mean(x))
    s = float(np.sqrt(np.mean((x - m) ** 2)))
    if not np.isfinite(s) or s == 0:
        raise SystemExit("zero sd while z-scoring")
    return (x - m) / s, m, s


def _sentence_fields(words: list[str], text: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map 。！？ in the punctuated text onto existing tokens. No resegmentation."""
    body: list[str] = []
    end_after: list[bool] = []
    buf = False
    for ch in text:
        if ch in "。！？":
            buf = True
        elif "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            if body and buf:
                end_after[-1] = True
            body.append(ch)
            end_after.append(False)
            buf = False
    body_s = "".join(body)
    k = body_s.find(ANCHOR)
    if k < 0:
        raise SystemExit(f"anchor {ANCHOR!r} not in punctuated text")
    tok = "".join(words)
    sm = SequenceMatcher(a=tok, b=body_s[k:], autojunk=False)
    char_end = [False] * len(tok)
    matched = [False] * len(tok)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for a, b in zip(range(i1, i2), range(j1, j2)):
                matched[a] = True
                if end_after[k + b]:
                    char_end[a] = True
        elif tag == "replace" and i2 > i1:
            for a in range(i1, i2):
                matched[a] = True
            if any(end_after[k + b] for b in range(j1, j2)):
                char_end[i2 - 1] = True
    spans = []
    c = 0
    for w in words:
        spans.append((c, c + len(w)))
        c += len(w)
    if c != len(tok):
        raise SystemExit("token character cursor drifted")
    sent = 1
    pos = 0
    sent_id = np.empty(len(words), dtype=np.int32)
    pos_id = np.empty(len(words), dtype=np.int32)
    for i, ((a, b), _w) in enumerate(zip(spans, words)):
        pos += 1
        sent_id[i] = sent
        pos_id[i] = pos
        if any(char_end[a:b]):
            sent += 1
            pos = 0
    unmatched_tokens = 0
    for a, b in spans:
        if not all(matched[a:b]):
            unmatched_tokens += 1
    first = "".join(words[i] for i in range(len(words)) if sent_id[i] == 1)
    if "野兽" not in first:
        raise SystemExit(f"first sentence did not reach 野兽: {first[:40]}")
    qc = {
        "n_sentences": int(sent_id.max()),
        "n_tokens": len(words),
        "unmatched_tokens": int(unmatched_tokens),
        "matched_chars": int(sum(matched)),
        "token_chars": len(tok),
        "first_sentence": first,
        "anchor": ANCHOR,
        "punct_file": str(PUNCT),
    }
    return sent_id, pos_id, qc


def _concreteness(words: pd.Series) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(NORM, sheet_name="Concreteness Ratings")
    if list(raw.columns)[:3] != ["Word ID", "Word", "Mean of Valid Ratings"]:
        raise SystemExit(f"unexpected concreteness columns: {list(raw.columns)}")
    norm = (
        raw.dropna(subset=["Word", "Mean of Valid Ratings"])
        .assign(Word=lambda d: d["Word"].astype(str).str.strip())
        .drop_duplicates("Word")
    )
    word_map = dict(zip(norm["Word"], norm["Mean of Valid Ratings"].astype(float)))
    buckets: dict[str, list[float]] = {}
    for w, m in word_map.items():
        if len(w) != 2:
            continue
        for ch in w:
            buckets.setdefault(ch, []).append(float(m))
    char_map = {ch: float(np.mean(v)) for ch, v in buckets.items()}

    scores = []
    sources = []
    for w in words.astype(str):
        if w in word_map:
            scores.append(word_map[w])
            sources.append("word")
            continue
        chars = [char_map[ch] for ch in w if ch in char_map]
        if chars and len(chars) == len(w):
            scores.append(float(np.mean(chars)))
            sources.append("char_mean")
        else:
            scores.append(np.nan)
            sources.append("missing")
    out = pd.DataFrame({"concreteness": scores, "conc_source": sources})
    return out, {
        "n_norm_words": int(len(word_map)),
        "n_char_scores": int(len(char_map)),
        "rating_column": "Mean of Valid Ratings",
        "citation": "Xu & Li 2020 PLOS ONE e0232133 S1",
    }


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")
    if not PUNCT.is_file():
        raise SystemExit(f"missing punctuated text {PUNCT}")
    if not NORM.is_file():
        raise SystemExit(f"missing concreteness workbook {NORM}")
    df = pd.read_csv(SRC)
    need = [
        "section", "word_index", "item_id", "word", "onset_sample", "duration", "content",
        "analysis_complete", "surprisal", "log_frequency", "association",
        "freq_m1", "freq_p1", "assoc_m1", "assoc_p1", "sur_m1", "sur_p1",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"word table missing {missing}")
    df = df.sort_values(["section", "word_index"]).reset_index(drop=True)
    if not df["section"].isin([1, 2]).all():
        raise SystemExit("unexpected section ids")
    # Sections are already in order 1 then 2; sentence index is continuous.
    if df["section"].iloc[0] != 1 or df["section"].iloc[-1] != 2:
        raise SystemExit("word table is not section 1 then section 2")

    words = df["word"].astype(str).tolist()
    sent_id, pos_id, sent_qc = _sentence_fields(words, PUNCT.read_text(encoding="utf-8"))
    df["pos_in_sent"] = pos_id
    df["sent_in_story"] = sent_id
    conc, conc_qc = _concreteness(df["word"])
    df = pd.concat([df, conc], axis=1)

    df["cloze_p"] = np.exp(-df["surprisal"].astype(float))
    df["cloze_m1"] = np.exp(-df["sur_m1"].astype(float))
    df["cloze_p1"] = np.exp(-df["sur_p1"].astype(float))

    content = df["content"].astype(bool) & df["analysis_complete"].astype(bool)
    finite_pos = np.isfinite(df["pos_in_sent"]) & np.isfinite(df["sent_in_story"])
    finite_cloze = (
        np.isfinite(df["cloze_p"]) & np.isfinite(df["cloze_m1"]) & np.isfinite(df["cloze_p1"])
    )
    finite_conc = np.isfinite(df["concreteness"])
    df["in_model"] = content & finite_pos & finite_cloze & finite_conc

    z_cols = {
        "freq_z": "log_frequency",
        "assoc_z": "association",
        "cloze_z": "cloze_p",
        "conc_z": "concreteness",
        "duration_z": "duration",
        "pos_sent_z": "pos_in_sent",
        "sent_story_z": "sent_in_story",
        "freq_m1_z": "freq_m1",
        "freq_p1_z": "freq_p1",
        "assoc_m1_z": "assoc_m1",
        "assoc_p1_z": "assoc_p1",
        "cloze_m1_z": "cloze_m1",
        "cloze_p1_z": "cloze_p1",
    }
    scaler = {}
    sub = df.loc[df["in_model"]]
    for zname, raw in z_cols.items():
        z, m, s = _population_z(sub[raw].to_numpy(float))
        df[zname] = np.nan
        df.loc[df["in_model"], zname] = z
        scaler[zname] = {"source": raw, "mean": m, "population_sd": s, "n_items": int(len(sub))}

    for raw, tert in (
        ("log_frequency", "tert_freq"),
        ("association", "tert_assoc"),
        ("cloze_p", "tert_cloze"),
    ):
        ranks = sub[raw].rank(method="average")
        cuts = pd.qcut(ranks, 3, labels=[0, 1, 2])
        df[tert] = np.nan
        df.loc[df["in_model"], tert] = cuts.astype(int)

    base_items = df.loc[content]
    conc_on_base = base_items["conc_source"].value_counts().to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "section", "word_index", "item_id", "word", "onset_sample",
        "content", "analysis_complete",
        "duration", "surprisal", "log_frequency", "association",
        "freq_m1", "freq_p1", "assoc_m1", "assoc_p1", "sur_m1", "sur_p1",
        "cloze_p", "cloze_m1", "cloze_p1",
        "pos_in_sent", "sent_in_story", "concreteness", "conc_source", "in_model",
        *z_cols.keys(),
        "tert_freq", "tert_assoc", "tert_cloze",
    ]
    out_csv = OUT / "word_table.csv"
    df[keep_cols].to_csv(out_csv, index=False)

    corr_cols = ["cloze_p", "log_frequency", "association", "concreteness", "duration",
                 "pos_in_sent", "sent_in_story"]
    corr = sub[corr_cols].corr(method="pearson")
    corr.to_csv(OUT / "predictor_correlations.csv")

    qc = {
        "n_tokens": int(len(df)),
        "n_content_analysis_complete": int(content.sum()),
        "n_in_model": int(df["in_model"].sum()),
        "dropped_missing_concreteness_among_complete_content": int((content & ~finite_conc).sum()),
        "dropped_missing_sentence_among_complete_content": int((content & ~finite_pos).sum()),
        "concreteness_source_among_complete_content": {k: int(v) for k, v in conc_on_base.items()},
        "concreteness_source_in_model": {
            k: int(v) for k, v in sub["conc_source"].value_counts().to_dict().items()
        },
        "sentences": sent_qc,
        "concreteness_norm": conc_qc,
        "scaler": scaler,
        "z_rule": "population sd on in_model items, one row per item",
        "cloze": "exp(-surprisal_nats)",
    }
    (OUT / "stimulus_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_in_model": qc["n_in_model"],
        "n_content_complete": qc["n_content_analysis_complete"],
        "conc": qc["concreteness_source_among_complete_content"],
        "n_sentences": sent_qc["n_sentences"],
        "unmatched_tokens": sent_qc["unmatched_tokens"],
        "out": str(out_csv),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
