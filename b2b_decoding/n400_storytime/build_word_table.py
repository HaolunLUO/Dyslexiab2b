#!/usr/bin/env python3
"""Stimulus-only word table for the Storytime N400 analysis.

Does not read EEG. Writes word_table.csv, correlation check, and QC json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import n400_common as C


def _load_fasttext(vocab: set[str], path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"fastText vectors missing: {path}")
    found: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split()
        if len(header) != 2 or not header[0].isdigit():
            raise SystemExit(f"unexpected fastText header: {header[:5]}")
        n_rows, dim = int(header[0]), int(header[1])
        for i, line in enumerate(f, start=1):
            if i % 100000 == 0:
                print(f"  scanned {i} vectors, kept {len(found)}", flush=True)
            line = line.strip()
            if not line:
                continue
            w, rest = line.split(" ", 1)
            if w not in vocab or w in found:
                continue
            vec = np.fromstring(rest, sep=" ", dtype=np.float32, count=dim)
            if vec.shape[0] != dim or not np.isfinite(vec).all():
                continue
            found[w] = vec
            if len(found) == len(vocab):
                break
    print(f"fastText scan done rows~{n_rows} dim={dim} kept={len(found)}", flush=True)
    return found


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _section_frame(section: int, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    timing = C.read_word_timing(section)
    sec = f"section_{section:03d}"
    sur = np.load(C.WORD_ROOT / sec / "X_word_fs100_gpt2cn_surprisal.npy").ravel()
    sur_ref = np.load(C.WORD_ROOT / sec / "X_word_gpt2cn_surprisal.npy").ravel()
    if sur.shape != sur_ref.shape or not np.allclose(sur, sur_ref, equal_nan=True):
        raise SystemExit(f"{sec}: fs100 surprisal != X_word_gpt2cn_surprisal")
    freq = np.load(C.WORD_ROOT / sec / "X_word_fs100_lexical_frequency.npy").ravel()
    ud = C.ud_tags_for_section(section)
    if not (len(timing) == len(sur) == len(freq) == len(ud)):
        raise SystemExit(
            f"{sec} length mismatch timing={len(timing)} sur={len(sur)} "
            f"freq={len(freq)} ud={len(ud)}"
        )

    rows = []
    for i, rec in enumerate(timing):
        word = str(rec["word"]).strip()
        onset = float(rec["onset_relative"])
        offset = float(rec["offset_relative"])
        flag, atomic = C.jieba_tag_token(word)
        if atomic:
            content = C.jieba_is_content(flag)
            pos_source = "jieba"
        else:
            content = ud[i] in C.UD_CONTENT
            pos_source = "ud_fallback_jieba_split"
        rows.append(
            {
                "section": section,
                "word_index": i,
                "item_id": f"s{section}_w{i:04d}",
                "word": word,
                "onset": onset,
                "offset": offset,
                "duration": offset - onset,
                "position": i,
                "onset_sample": int(float(rec["onset_sample"])),
                "surprisal": float(sur[i]),
                "log_frequency": float(freq[i]),
                "pos_jieba": flag,
                "pos_ud": ud[i],
                "pos_source": pos_source,
                "content": bool(content),
                "content_ud": ud[i] in C.UD_CONTENT,
                "has_vector": word in vectors,
            }
        )
    df = pd.DataFrame(rows)

    assoc = np.full(len(df), np.nan)
    n_ctx = np.zeros(len(df), dtype=int)
    # Previous content-word vectors in order (None if that content word is OOV).
    prev: list[np.ndarray | None] = []
    for i, rec in enumerate(df.itertuples(index=False)):
        vec = vectors.get(rec.word)
        if prev and vec is not None:
            take = prev[-3:]
            sims = [_cosine(vec, v) for v in take if v is not None]
            n_ctx[i] = len(sims)
            if sims:
                assoc[i] = float(np.mean(sims))
        if rec.content:
            usable = vec if vec is not None and float(np.linalg.norm(vec)) > 1e-8 else None
            prev.append(usable)
    df["association"] = assoc
    df["n_assoc_ctx"] = n_ctx

    for col, src in (
        ("sur_m1", "surprisal"),
        ("sur_p1", "surprisal"),
        ("freq_m1", "log_frequency"),
        ("freq_p1", "log_frequency"),
        ("assoc_m1", "association"),
        ("assoc_p1", "association"),
    ):
        s = df[src]
        df[col] = s.shift(1) if col.endswith("m1") else s.shift(-1)
    return df


def _complete_mask(df: pd.DataFrame) -> pd.Series:
    cols = [
        "duration",
        "position",
        "surprisal",
        "log_frequency",
        "association",
        "sur_m1",
        "sur_p1",
        "freq_m1",
        "freq_p1",
        "assoc_m1",
        "assoc_p1",
    ]
    ok = df["content"] & (df["duration"] > 0)
    for c in cols:
        ok = ok & np.isfinite(df[c].to_numpy(float))
    return ok


def _corr_block(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = ["surprisal", "log_frequency", "association", "duration"]
    sub = df.loc[:, cols]
    rows = []
    for method in ("pearson", "spearman"):
        mat = sub.corr(method=method)
        for a in cols:
            for b in cols:
                if a >= b:
                    continue
                rows.append(
                    {
                        "set": label,
                        "method": method,
                        "a": a,
                        "b": b,
                        "r": float(mat.loc[a, b]),
                        "n": int(len(sub)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    import logging

    import jieba

    jieba.setLogLevel(logging.ERROR)
    C.STIM_DIR.mkdir(parents=True, exist_ok=True)
    # vocab from both sections before the scan
    vocab: set[str] = set()
    for section in C.SECTIONS:
        for rec in C.read_word_timing(section):
            vocab.add(str(rec["word"]).strip())
    print(f"unique word types {len(vocab)}", flush=True)
    vectors = _load_fasttext(vocab, C.EMBED_VEC)

    frames = [_section_frame(s, vectors) for s in C.SECTIONS]
    df = pd.concat(frames, ignore_index=True)
    df["analysis_complete"] = _complete_mask(df)

    oov_types = sorted(w for w in vocab if w not in vectors)
    n_tok = len(df)
    n_oov_tok = int((~df["has_vector"]).sum())
    content = df.loc[df["content"]]
    complete = df.loc[df["analysis_complete"]]

    corr = pd.concat(
        [
            _corr_block(
                content.loc[
                    np.isfinite(content["surprisal"])
                    & np.isfinite(content["log_frequency"])
                    & np.isfinite(content["association"])
                    & np.isfinite(content["duration"])
                ],
                "content_finite_assoc",
            ),
            _corr_block(complete, "lmm_complete_case"),
        ],
        ignore_index=True,
    )

    word_path = C.STIM_DIR / "word_table.csv"
    corr_path = C.STIM_DIR / "predictor_correlations.csv"
    qc_path = C.STIM_DIR / "stimulus_qc.json"
    df.to_csv(word_path, index=False)
    corr.to_csv(corr_path, index=False)

    both = df.loc[df["pos_source"] == "jieba"]
    agree = float((both["content"] == both["content_ud"]).mean()) if len(both) else float("nan")
    qc = {
        "embedding_source": "Facebook fastText wiki.zh.vec (Chinese Wikipedia, 300-d)",
        "embedding_url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec",
        "embedding_path": str(C.EMBED_VEC),
        "embedding_header": "332647 300",
        "n_word_tokens": n_tok,
        "n_word_types": len(vocab),
        "n_oov_types": len(oov_types),
        "oov_rate_types": len(oov_types) / len(vocab),
        "n_oov_tokens": n_oov_tok,
        "oov_rate_tokens": n_oov_tok / n_tok,
        "oov_examples": oov_types[:40],
        "pos_primary": "jieba 0.42.1 HMM=False on existing GPT2-CN tokens",
        "pos_fallback": "Universal POS in X_word_wordinfo when jieba splits a token",
        "n_jieba_atomic": int((df["pos_source"] == "jieba").sum()),
        "n_ud_fallback": int((df["pos_source"] != "jieba").sum()),
        "content_agreement_jieba_vs_ud_on_atomic": agree,
        "n_content": int(df["content"].sum()),
        "n_content_ud": int(df["content_ud"].sum()),
        "n_analysis_complete": int(df["analysis_complete"].sum()),
        "surprisal_key": "X_word_fs100_gpt2cn_surprisal (identical to X_word_gpt2cn_surprisal)",
        "surprisal_units": "nats (sum of token -log p), uer_gpt2_chinese",
        "frequency_key": "X_word_fs100_lexical_frequency",
        "content_rule": "jieba flag initial n/v/a/d or exact i/l/b/z/t/s; UD NOUN/PROPN/VERB/ADJ/ADV on split fallback",
    }
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2))
    print(json.dumps({k: qc[k] for k in qc if k != "oov_examples"}, indent=2))
    print("oov examples", oov_types[:25])
    print(corr.to_string(index=False))
    print("wrote", word_path)


if __name__ == "__main__":
    main()
