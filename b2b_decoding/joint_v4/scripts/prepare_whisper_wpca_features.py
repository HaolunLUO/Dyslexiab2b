#!/usr/bin/env python3
"""Within-word PCA on precomputed Whisper encoder features (247-encoding style).

Reads concatenated multi-window embeddings (e.g. 1280 × 10 → 12800 for
Whisper Large V2 speech/acoustic), applies per-word PCA with one whitened
component across temporal windows, and writes B2B-ready matrices:

  X_word_<variant>_wpca.npy
  X_word_<variant>_wpca_feature_names.txt
  X_word_<variant>_wpca_meta.json

This matches hassonlab/247-encoding ``--window-num 1-10-pca`` on encoder
variants. Language streams (single 1280-d vectors) are unchanged.

Usage:
  python3 scripts/prepare_whisper_wpca_features.py
  B2B_EXTRACTOR_DIR=/path/to/extracted_sections_wordlocked_shared \\
    python3 scripts/prepare_whisper_wpca_features.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

EXTRACTOR = Path(
    os.environ.get(
        "B2B_EXTRACTOR_DIR",
        "/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared",
    )
)
SHARED = EXTRACTOR / "_shared_wordlocked_features"
SECTIONS = tuple(int(x) for x in os.environ.get("WPCa_SECTIONS", "1,2").split(","))
EXPECTED_ROWS = {1: 1753, 2: 1800}

# Source feature stem → (hidden dim, number of temporal windows concatenated).
ENCODER_VARIANTS: dict[str, tuple[int, int]] = {
    "whisperlargev2_acoustic": (1280, 10),
    "whisperlargev2_speech": (1280, 10),
    "whispertiny_acoustic": (384, 10),
    "whispertiny_speech": (384, 10),
}

OUT_SUFFIX = "_wpca"
WINDOW_START = int(os.environ.get("WPCa_WINDOW_START", "1"))  # 1-indexed, inclusive
WINDOW_END = int(os.environ.get("WPCa_WINDOW_END", "10"))  # 1-indexed, inclusive


def _read_meta(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def word_pca_row(row: np.ndarray, emb_dim: int, n_windows: int) -> np.ndarray:
    """Collapse concatenated windows for one word → emb_dim vector."""
    win = row.reshape(n_windows, emb_dim)
    # 247-encoding uses windows [start:end] on the concatenated vector.
    w0 = WINDOW_START - 1
    w1 = WINDOW_END
    win = win[w0:w1]
    if win.shape[0] == 0:
        raise ValueError(f"empty window slice [{WINDOW_START}, {WINDOW_END}]")
    # (n_sel_windows, emb_dim).T → (emb_dim, n_sel_windows)
    mat = win.T
    pca = PCA(n_components=1, svd_solver="auto", whiten=True)
    out = pca.fit_transform(mat).squeeze()
    if out.shape != (emb_dim,):
        out = np.atleast_1d(out)
        if out.size != emb_dim:
            raise ValueError(f"unexpected word_pca shape {out.shape} for emb_dim={emb_dim}")
    return out.astype(np.float64)


def process_variant(src_stem: str, emb_dim: int, n_windows: int) -> None:
    out_stem = f"{src_stem}{OUT_SUFFIX}"
    print(f"\n=== {src_stem} → {out_stem}  ({emb_dim}×{n_windows} → {emb_dim}) ===")

    for sid in SECTIONS:
        sec_dir = SHARED / f"section_{sid:03d}"
        src_path = sec_dir / f"X_word_{src_stem}.npy"
        if not src_path.is_file():
            print(f"  [skip] section {sid}: missing {src_path.name}")
            continue

        X = np.load(src_path).astype(np.float64)
        n_words, p = X.shape
        exp_n = EXPECTED_ROWS.get(sid)
        if exp_n is not None and n_words != exp_n:
            raise SystemExit(
                f"section {sid} {src_stem}: rows {n_words} != expected {exp_n}"
            )
        if p != emb_dim * n_windows:
            raise SystemExit(
                f"section {sid} {src_stem}: cols {p} != {emb_dim}×{n_windows}={emb_dim * n_windows}"
            )
        if not np.isfinite(X).all():
            raise SystemExit(f"section {sid} {src_stem}: non-finite values in source")

        Y = np.vstack([word_pca_row(X[i], emb_dim, n_windows) for i in range(n_words)])
        if not np.isfinite(Y).all():
            raise SystemExit(f"section {sid} {out_stem}: non-finite after word_pca")

        out_npy = sec_dir / f"X_word_{out_stem}.npy"
        np.save(out_npy, Y)

        names_path = sec_dir / f"X_word_{out_stem}_feature_names.txt"
        names = [f"{out_stem}_{j}" for j in range(emb_dim)]
        names_path.write_text("\n".join(names) + "\n")

        src_meta = _read_meta(sec_dir / f"X_word_{src_stem}_meta.json")
        meta = {
            "tag": src_meta.get("tag", "Whisper"),
            "variant": f"{src_meta.get('variant', src_stem)}{OUT_SUFFIX}",
            "source_variant": src_stem,
            "shape": [int(n_words), int(emb_dim)],
            "word_rows": src_meta.get("word_rows", list(range(n_words))),
            "preprocessing": {
                "method": "within_word_pca",
                "reference": "hassonlab/247-encoding whisper-paper-1 (window-num 1-10-pca)",
                "emb_dim": emb_dim,
                "n_windows_total": n_windows,
                "window_range_1indexed": [WINDOW_START, WINDOW_END],
                "pca_n_components": 1,
                "pca_whiten": True,
            },
        }
        meta_path = sec_dir / f"X_word_{out_stem}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        norms = np.linalg.norm(Y, axis=1)
        print(
            f"  section {sid}: {src_path.name} {X.shape} → {out_npy.name} {Y.shape}  "
            f"‖x‖ mean={norms.mean():.2f} std={norms.std():.2f}"
        )


def main() -> None:
    if not SHARED.is_dir():
        raise SystemExit(f"Shared feature dir not found: {SHARED}")

    print(f"Extractor : {EXTRACTOR}")
    print(f"Sections  : {SECTIONS}")
    print(f"Windows   : {WINDOW_START}-{WINDOW_END} (1-indexed, inclusive)")

    for src_stem, (emb_dim, n_windows) in ENCODER_VARIANTS.items():
        process_variant(src_stem, emb_dim, n_windows)

    print("\nDone. Point B2B at e.g.:")
    print("  B2B_FEAT_ACOUSTIC=whisperlargev2_acoustic_wpca")
    print("  B2B_FEAT_SPEECH=whisperlargev2_speech_wpca")
    print("  B2B_FEAT_LANGUAGE=whisperlargev2_language_audio_fused")


if __name__ == "__main__":
    main()
