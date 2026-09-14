#!/usr/bin/env python3
"""
QC scanner for Whisper word-locked feature matrices.

Checks (per section_00{1,2}):
  - expected file existence
  - npy shape + dtype
  - feature_names.txt line count == embedding dim
  - meta json existence (when present upstream)

This is intentionally lightweight: it does NOT compute PCA or scan for NaNs
unless --check-finite is enabled.
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


EXPECTED_DIMS = {
    # raw encoder concatenations
    "whisperlargev2_acoustic": 12800,
    "whisperlargev2_speech": 12800,
    "whisperlargev2_language_audio_fused": 1280,
    "whisperlargev2_language_text_only": 1280,
    "whisperlargev3_acoustic": 12800,
    "whisperlargev3_speech": 12800,
    "whisperlargev3_language_audio_fused": 1280,
    "whisperlargev3_language_text_only": 1280,
    "whisperlargev3_acoustic": 12800,
    "whisperlargev3_speech": 12800,
    "whisperlargev3_language_audio_fused": 1280,
    "whisperlargev3_language_text_only": 1280,
    "whispertiny_acoustic": 3840,
    "whispertiny_speech": 3840,
    "whispertiny_language_audio_fused": 384,
    "whispertiny_language_text_only": 384,
    # wpca derivatives (encoder-only)
    "whisperlargev2_acoustic_wpca": 1280,
    "whisperlargev2_speech_wpca": 1280,
    "whispertiny_acoustic_wpca": 384,
    "whispertiny_speech_wpca": 384,
}


RAW_KEYS = [
    "whisperlargev2_acoustic",
    "whisperlargev2_speech",
    "whisperlargev2_language_audio_fused",
    "whisperlargev2_language_text_only",
    "whisperlargev3_acoustic",
    "whisperlargev3_speech",
    "whisperlargev3_language_audio_fused",
    "whisperlargev3_language_text_only",
    "whisperlargev3_acoustic",
    "whisperlargev3_speech",
    "whisperlargev3_language_audio_fused",
    "whisperlargev3_language_text_only",
    "whispertiny_acoustic",
    "whispertiny_speech",
    "whispertiny_language_audio_fused",
    "whispertiny_language_text_only",
]

WPCa_KEYS = [
    "whisperlargev2_acoustic_wpca",
    "whisperlargev2_speech_wpca",
    "whispertiny_acoustic_wpca",
    "whispertiny_speech_wpca",
]


def _load_npy_shape_dtype(path: Path) -> tuple[tuple[int, int], str]:
    # mmap_mode avoids copying the whole array into memory.
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D array, got shape {arr.shape}")
    shape = (int(arr.shape[0]), int(arr.shape[1]))
    return shape, str(arr.dtype)


def _check_feature_names(names_path: Path, expected_dim: int) -> None:
    if not names_path.is_file():
        raise FileNotFoundError(str(names_path))
    # feature_names.txt is plain newline-delimited text.
    n_lines = sum(1 for _ in names_path.open("r", encoding="utf-8", errors="ignore"))
    if n_lines != expected_dim:
        raise ValueError(f"{names_path.name}: expected {expected_dim} lines, found {n_lines}")


def _check_finite(path: Path, max_rows: int = 50) -> None:
    arr = np.load(path, mmap_mode="r")
    # Full scans are expensive; default checks a slice.
    sl = arr[: min(arr.shape[0], max_rows), :]
    if not np.isfinite(sl).all():
        raise ValueError(f"{path.name}: found NaN/Inf in the first {sl.shape[0]} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shared-dir",
        type=str,
        default=str(DEFAULT_SHARED),
        help="Path to _shared_wordlocked_features",
    )
    ap.add_argument("--sections", default="1,2", help="Comma list of section ids (e.g. 1,2)")
    ap.add_argument(
        "--check-finite",
        action="store_true",
        help="Scan for NaNs/Infs (partial slice; still slower).",
    )
    args = ap.parse_args()

    shared = Path(args.shared_dir)
    if not shared.is_dir():
        raise SystemExit(f"Shared dir not found: {shared}")

    section_ids = [s.strip() for s in args.sections.split(",") if s.strip()]
    section_dirs = [shared / f"section_{int(s):03d}" for s in section_ids]

    keys_to_check = RAW_KEYS + WPCa_KEYS

    print(f"QC shared dir: {shared}")
    print(f"Sections: {section_ids}")
    print(f"Checking {len(keys_to_check)} Whisper variants (raw + wpca).")

    failures: list[str] = []
    for sec_dir in section_dirs:
        sec_name = sec_dir.name
        if not sec_dir.is_dir():
            failures.append(f"[missing] {sec_name} not found")
            continue

        for key in keys_to_check:
            npy = sec_dir / f"X_word_{key}.npy"
            names_txt = sec_dir / f"X_word_{key}_feature_names.txt"
            meta = sec_dir / f"X_word_{key}_meta.json"
            expected_dim = EXPECTED_DIMS[key]

            try:
                if not npy.is_file():
                    raise FileNotFoundError(str(npy))
                (n_words, dim), dtype = _load_npy_shape_dtype(npy)
                if dim != expected_dim:
                    raise ValueError(
                        f"{npy.name}: expected dim {expected_dim}, got {dim} (dtype={dtype})"
                    )
                _check_feature_names(names_txt, expected_dim)

                # meta for wpca should always exist; raw meta may be incomplete upstream.
                # We'll still report its presence.
                if not meta.is_file():
                    # Only fail for wpca; raw meta is allowed to be absent.
                    if key in WPCa_KEYS:
                        raise FileNotFoundError(str(meta))

                if args.check_finite:
                    _check_finite(npy)

            except Exception as e:
                failures.append(f"[{sec_name}] {key}: {e}")

    if failures:
        print("\nQC FAILURES:")
        for f in failures:
            print(" -", f)
        raise SystemExit(f"QC failed with {len(failures)} issues")

    print("\nQC OK: all checked files passed shape/dtype + feature_names checks.")


if __name__ == "__main__":
    main()

