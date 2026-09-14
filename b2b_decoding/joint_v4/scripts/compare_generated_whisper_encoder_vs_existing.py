#!/usr/bin/env python3
"""
Compare generated Whisper encoder embeddings vs existing reference matrices.

Designed for quick verification runs on a subset of words.

Usage example:
  python3 compare_generated_whisper_encoder_vs_existing.py \
    --section-id 1 \
    --stem whispertiny \
    --generated-dir /path/to/out \
    --existing-root /orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/_shared_wordlocked_features \
    --max-words 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_SHARED = Path(
    os.environ.get(
        "B2B_SHARED_FEATURES_DIR",
        "/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/_shared_wordlocked_features",
    )
)


def _cosine_similarity_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A,B: (N,D)
    An = np.linalg.norm(A, axis=1, keepdims=True)
    Bn = np.linalg.norm(B, axis=1, keepdims=True)
    denom = An * Bn
    denom[denom == 0] = np.nan
    return (A * B).sum(axis=1) / denom[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section-id", type=int, default=1)
    ap.add_argument("--stem", type=str, required=True, help="whispertiny or whisperlargev2")
    ap.add_argument("--generated-dir", type=str, required=True)
    ap.add_argument("--existing-root", type=str, default=str(DEFAULT_SHARED))
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument(
        "--start-word-idx",
        type=int,
        default=0,
        help="Row offset into the existing matrix (must match extraction --start-word-idx).",
    )
    args = ap.parse_args()

    sec = f"section_{args.section_id:03d}"
    gen_dir = Path(args.generated_dir)
    ex_dir = Path(args.existing_root) / sec

    variants = ["acoustic", "speech"]
    for v in variants:
        gen = gen_dir / f"X_word_{args.stem}_{v}.npy"
        ex = ex_dir / f"X_word_{args.stem}_{v}.npy"
        if not gen.is_file():
            raise SystemExit(f"Missing generated: {gen}")
        if not ex.is_file():
            raise SystemExit(f"Missing existing: {ex}")

        A = np.load(ex, mmap_mode="r")
        B = np.load(gen, mmap_mode="r")

        i0 = args.start_word_idx
        n = min(args.max_words, B.shape[0], max(0, A.shape[0] - i0))
        if B.shape[1] != A.shape[1]:
            print(f"[{v}] DIM MISMATCH: existing {A.shape} vs gen {B.shape}")
            continue
        if n <= 0:
            print(f"[{v}] nothing to compare (start={i0}, existing={A.shape}, gen={B.shape})")
            continue

        A_sub = np.asarray(A[i0 : i0 + n, :])
        B_sub = np.asarray(B[:n, :])
        print(f"[{v}] comparing gen[:{n}] vs existing[{i0}:{i0 + n}] (full existing {A.shape}, gen {B.shape})")
        cos = _cosine_similarity_rows(A_sub, B_sub)
        finite = np.isfinite(cos)
        if finite.any():
            print(
                f"[{v}] cosine similarity (first {n} words): "
                f"mean={np.nanmean(cos):.4f} std={np.nanstd(cos):.4f} "
                f"min={np.nanmin(cos):.4f} max={np.nanmax(cos):.4f}"
            )
        else:
            print(f"[{v}] cosine similarity produced no finite values.")


if __name__ == "__main__":
    main()

