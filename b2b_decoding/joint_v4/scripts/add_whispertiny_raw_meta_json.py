#!/usr/bin/env python3
"""
Add missing raw meta json files for whispertiny_* in:
  extracted_sections_wordlocked_shared/_shared_wordlocked_features/section_00{1,2}

Observed in this workspace:
  - whispertiny raw *_meta.json files are missing
  - wpca *_meta.json files already exist
  - large-v2 raw *_meta.json files exist

This script writes minimal meta JSON matching the large-v2 raw schema:
  keys: [word_rows, shape, variant, tag]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_SHARED = Path(
    os.environ.get(
        "B2B_SHARED_FEATURES_DIR",
        "/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/_shared_wordlocked_features",
    )
)


VARIANT_MAP = {
    "whispertiny_acoustic": ("acoustic", "WhisperTiny"),
    "whispertiny_speech": ("speech", "WhisperTiny"),
    "whispertiny_language_audio_fused": ("language_audio_fused", "WhisperTiny"),
    "whispertiny_language_text_only": ("language_text_only", "WhisperTiny"),
}


def _load_npy_shape(path: Path) -> tuple[int, int]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {path}, got {arr.shape}")
    return int(arr.shape[0]), int(arr.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-dir", type=str, default=str(DEFAULT_SHARED))
    ap.add_argument("--sections", default="1,2", help="Comma list: 1,2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shared = Path(args.shared_dir)
    section_ids = [s.strip() for s in args.sections.split(",") if s.strip()]

    files_written = 0

    for sid in section_ids:
        sec_dir = shared / f"section_{int(sid):03d}"
        if not sec_dir.is_dir():
            raise SystemExit(f"Missing section dir: {sec_dir}")

        for key, (variant, tag) in VARIANT_MAP.items():
            npy = sec_dir / f"X_word_{key}.npy"
            meta = sec_dir / f"X_word_{key}_meta.json"
            if not npy.is_file():
                raise SystemExit(f"Missing npy: {npy}")
            if meta.is_file():
                continue

            n_words, dim = _load_npy_shape(npy)
            word_rows = list(range(n_words))
            out = {
                "word_rows": word_rows,
                "shape": [n_words, dim],
                "variant": variant,
                "tag": tag,
            }

            if not args.dry_run:
                meta.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
                files_written += 1
                print(f"[wrote] {meta.relative_to(shared.parent)}")
            else:
                print(f"[would write] {meta}")

    if args.dry_run:
        print("Dry-run complete. No files written.")
    else:
        print(f"Done. Wrote {files_written} meta json files.")


if __name__ == "__main__":
    main()

