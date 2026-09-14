#!/usr/bin/env python3
"""Derive Gwilliams-style syntactic operation/state from lppCN_tree.txt.

Writes:
  X_word_hdc_syntactic_operation.npy  (n_open, n_close, sentence_end)
  X_word_hdc_syntactic_state.npy      (n_open_nodes, depth, depth_pm1, order, order_pm1)

Terminals are aligned to word_timing_relative.csv by exact string match with
a greedy leftover-token skip if a tree word is missing from timing (or vice versa).

Usage:
  python3 scripts/prepare_hdc_syntax.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACTOR = Path(
    os.environ.get(
        "B2B_EXTRACTOR_DIR",
        "/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared",
    )
)
TREE_PATH = EXTRACTOR / "lppCN_tree.txt"
SHARED = EXTRACTOR / "_shared_wordlocked_features"
SECTIONS = (1, 2)
EXPECTED = (1753, 1800)
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def parse_tree_line(line: str):
    """Return per-terminal dicts: n_opening, n_closing, depth, n_open_nodes, is_end."""
    tokens = TOKEN_RE.findall(line.strip())
    depth = 0
    pending_open = 0
    rows = []
    i = 0
    ntok = len(tokens)
    while i < ntok:
        tok = tokens[i]
        if tok == "(":
            depth += 1
            pending_open += 1
            i += 1
            # consume the node label (NP, VV, P, ...); not a terminal
            if i < ntok and tokens[i] not in ("(", ")"):
                i += 1
            continue
        if tok == ")":
            depth = max(depth - 1, 0)
            i += 1
            continue
        n_opening = pending_open
        pending_open = 0
        word = tok
        n_closing = 0
        k = i + 1
        while k < ntok and tokens[k] == ")":
            n_closing += 1
            depth = max(depth - 1, 0)
            k += 1
        i = k
        rows.append(
            {
                "word": word,
                "n_opening": n_opening,
                "n_closing": n_closing,
                "depth": depth + n_closing,
                "n_open_nodes": depth,
            }
        )
    if rows:
        rows[-1]["sentence_end"] = 1.0
        for r in rows[:-1]:
            r["sentence_end"] = 0.0
        for idx, r in enumerate(rows):
            r["linear_order"] = idx / max(len(rows) - 1, 1)
    return rows


def find_span(tree_words: list[str], target: list[str], min_hit: float = 0.7) -> int:
    """Return start index in tree_words that best matches target sequence."""
    n_t, n_w = len(tree_words), len(target)
    if n_t < n_w:
        return 0
    # exact prefix search on first 12 words
    needle = target[:12]
    for i in range(0, n_t - len(needle) + 1):
        if tree_words[i : i + len(needle)] == needle:
            return i
    best_i, best = 0, -1.0
    step = max(1, n_w // 20)
    for i in range(0, n_t - n_w + 1, step):
        hit = sum(a == b for a, b in zip(tree_words[i : i + n_w], target)) / n_w
        if hit > best:
            best, best_i = hit, i
            if hit >= min_hit:
                break
    return best_i


def load_all_tree_rows(path: Path):
    all_rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        all_rows.extend(parse_tree_line(line))
    return all_rows


def align_to_timing(tree_rows, words: list[str]):
    """Greedy align tree terminals to timing words."""
    n_t, n_w = len(tree_rows), len(words)
    out = [None] * n_w
    ti = 0
    for wi, w in enumerate(words):
        # search forward a small window
        found = None
        for k in range(ti, min(ti + 8, n_t)):
            if tree_rows[k]["word"] == w:
                found = k
                break
        if found is None:
            # unmatched timing word: copy previous or zeros
            prev = out[wi - 1] if wi else {
                "n_opening": 0, "n_closing": 0, "depth": 0,
                "n_open_nodes": 0, "sentence_end": 0.0, "linear_order": 0.0,
            }
            out[wi] = dict(prev)
            out[wi]["unmatched"] = True
        else:
            out[wi] = dict(tree_rows[found])
            out[wi]["unmatched"] = False
            ti = found + 1
    return out


def add_pm1(vals: np.ndarray) -> np.ndarray:
    lag = np.empty((vals.shape[0], 3), dtype=np.float64)
    lag[:, 1] = vals
    lag[0, 0] = vals[0]
    lag[1:, 0] = vals[:-1]
    lag[-1, 2] = vals[-1]
    lag[:-1, 2] = vals[1:]
    return lag


def main() -> None:
    if not TREE_PATH.is_file():
        raise SystemExit(f"missing {TREE_PATH}")
    tree_rows = load_all_tree_rows(TREE_PATH)
    tree_words = [r["word"] for r in tree_rows]
    print(f"Parsed {len(tree_rows)} tree terminals from {TREE_PATH}")

    membership = {"feature_keys": ["hdc_syntactic_operation", "hdc_syntactic_state"]}
    for i, sid in enumerate(SECTIONS):
        d = SHARED / f"section_{sid:03d}"
        timing = pd.read_csv(d / "word_timing_relative.csv")
        words = [str(w) for w in timing["word"].tolist()]
        n = len(words)
        if n != EXPECTED[i]:
            raise SystemExit(f"section {sid}: {n} != {EXPECTED[i]}")
        start = find_span(tree_words, words)
        chunk = tree_rows[start : start + n + 50]
        aligned = align_to_timing(chunk, words)
        n_un = sum(1 for r in aligned if r.get("unmatched"))
        print(f"  section {sid}: tree_start={start} unmatched={n_un}/{n}")

        n_open = np.array([r["n_opening"] for r in aligned], dtype=np.float64)
        n_close = np.array([r["n_closing"] for r in aligned], dtype=np.float64)
        sent_end = np.array([r["sentence_end"] for r in aligned], dtype=np.float64)
        depth = np.array([r["depth"] for r in aligned], dtype=np.float64)
        n_open_nodes = np.array([r["n_open_nodes"] for r in aligned], dtype=np.float64)
        order = np.array([r["linear_order"] for r in aligned], dtype=np.float64)

        op = np.column_stack([n_open, n_close, sent_end]).astype(np.float32)
        st = np.column_stack(
            [
                n_open_nodes,
                add_pm1(depth),
                add_pm1(order),
            ]
        ).astype(np.float32)
        op_names = ["n_opening", "n_closing", "sentence_end"]
        st_names = [
            "n_open_nodes",
            "depth_m1", "depth", "depth_p1",
            "order_m1", "linear_order", "order_p1",
        ]
        np.save(d / "X_word_hdc_syntactic_operation.npy", op)
        np.save(d / "X_word_hdc_syntactic_state.npy", st)
        (d / "X_word_hdc_syntactic_operation_feature_names.txt").write_text(
            "\n".join(op_names) + "\n"
        )
        (d / "X_word_hdc_syntactic_state_feature_names.txt").write_text(
            "\n".join(st_names) + "\n"
        )
        print(f"  wrote operation {op.shape}  state {st.shape}")

    membership["n_operation"] = 3
    membership["n_state"] = 7
    (SHARED / "hdc_syntax_membership.json").write_text(
        json.dumps(membership, indent=2) + "\n"
    )
    print("HDC syntax features ready.")


if __name__ == "__main__":
    main()
