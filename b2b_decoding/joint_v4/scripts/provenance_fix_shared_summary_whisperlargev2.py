#!/usr/bin/env python3
"""
Fix provenance bookkeeping in:
  extracted_sections_wordlocked_shared/_shared_wordlocked_features/shared_summary.json

Problem observed in local workspace:
  - shared_summary.json lists whispertiny_* but not whisperlargev2_*.

This script scans the on-disk _shared_wordlocked_features/section_00{1,2}
matrices for whisperlargev2 variants and updates shared_summary.json's
sections[<id>]["X_word"] mapping accordingly.
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


WHISPERLARGE2_KEYS = [
    "whisperlargev2_acoustic",
    "whisperlargev2_speech",
    "whisperlargev2_language_audio_fused",
    "whisperlargev2_language_text_only",
]


def _load_npy_shape_dtype(path: Path) -> tuple[list[int], str]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D npy, got shape {arr.shape} for {path}")
    return [int(arr.shape[0]), int(arr.shape[1])], str(arr.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-dir", type=str, default=str(DEFAULT_SHARED))
    ap.add_argument("--dry-run", action="store_true", help="Don't write changes.")
    args = ap.parse_args()

    shared = Path(args.shared_dir)
    summary_path = shared / "shared_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing shared_summary.json: {summary_path}")
    if not shared.is_dir():
        raise SystemExit(f"Missing shared dir: {shared}")

    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    sections = summary.get("sections")
    if not isinstance(sections, dict):
        raise SystemExit("Unexpected shared_summary.json schema: sections must be a dict")

    changed = False

    for sec_id, sec_obj in sections.items():
        sec_dir = shared / f"section_{int(sec_id):03d}"
        if not sec_dir.is_dir():
            raise SystemExit(f"Section dir missing: {sec_dir}")
        x_word = sec_obj.get("X_word")
        if not isinstance(x_word, dict):
            raise SystemExit(f"Unexpected schema: sections[{sec_id}].X_word must be a dict")

        for key in WHISPERLARGE2_KEYS:
            npy = sec_dir / f"X_word_{key}.npy"
            names = sec_dir / f"X_word_{key}_feature_names.txt"
            if not npy.is_file():
                raise SystemExit(f"Missing npy for {sec_dir.name} {key}: {npy}")
            if not names.is_file():
                raise SystemExit(f"Missing names file for {sec_dir.name} {key}: {names}")

            shape, dtype = _load_npy_shape_dtype(npy)
            entry = x_word.get(key)
            new_entry = {
                "npy": f"X_word_{key}.npy",
                "feature_names_txt": f"X_word_{key}_feature_names.txt",
                "shape": shape,
                "dtype": dtype,
            }

            if entry != new_entry:
                x_word[key] = new_entry
                changed = True

    if not changed:
        print("No changes needed: shared_summary already contains whisperlargev2 entries.")
        return

    if args.dry_run:
        print("Dry-run: would update shared_summary.json whisperlargev2 entries.")
        return

    tmp = summary_path.with_suffix(".tmp.json")
    json.dump(summary, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    tmp.replace(summary_path)
    print(f"Updated {summary_path} with whisperlargev2 entries.")


if __name__ == "__main__":
    main()

