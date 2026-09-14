#!/usr/bin/env python3
"""Build Gwilliams-style HDC pilot features from existing wordinfo + GloVe.

Writes per-family matrices into the shared wordlocked feature dirs:

  X_word_hdc_phonetic.npy
  X_word_hdc_word_form.npy
  X_word_hdc_lexical_syntactic.npy
  X_word_hdc_syntax_proxy.npy
  X_word_hdc_semantic.npy

plus feature-name files, a family membership JSON, and GloVe PCA loadings.

Near-constant columns (std < 1e-8, or binary with < MIN_POS positives in
the concatenated story) are dropped.

Usage:
  python3 scripts/prepare_hdc_features.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

EXTRACTOR = Path(
    os.environ.get(
        "B2B_EXTRACTOR_DIR",
        "/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared",
    )
)
SHARED = EXTRACTOR / "_shared_wordlocked_features"
SECTIONS = (1, 2)
EXPECTED = (1753, 1800)
N_GLOVE_PCS = 10
MIN_POS = int(os.environ.get("HDC_MIN_POS", "5"))
STD_EPS = 1e-8
# Optional suffix on written feature keys, e.g. HDC_FEATURE_TAG=mp50 → hdc_phonetic_mp50
FEATURE_TAG = os.environ.get("HDC_FEATURE_TAG", "").strip()
# Always drop: exact duplicate on Mandarin (syllables≡morphemes); near-constant
# manner fraction that is linearly dependent (manner cols sum to 1).
FORCE_DROP = frozenset({"wf_n_morphemes", "ph_vowel"})

FAMILY_SPEC = {
    "hdc_phonetic": {
        "family": "phonetic",
        "prefixes": ("ph_",),
        "exact": (),
    },
    "hdc_word_form": {
        "family": "word_form",
        "prefixes": ("wf_",),
        "exact": ("logfreq",),
    },
    "hdc_lexical_syntactic": {
        "family": "lexical_syntactic",
        "prefixes": ("pos_",),
        "exact": (),
    },
    "hdc_syntax_proxy": {
        "family": "syntax_proxy",
        "prefixes": (),
        "exact": ("top_down", "bottom_up", "left_corner"),
    },
}


def _read_names(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _select_columns(names: list[str], spec: dict) -> list[int]:
    idx = []
    for i, n in enumerate(names):
        if n in spec["exact"] or any(n.startswith(p) for p in spec["prefixes"]):
            idx.append(i)
    return idx


def _is_binary(col: np.ndarray) -> bool:
    u = np.unique(col[np.isfinite(col)])
    return u.size <= 2 and set(np.round(u, 8)).issubset({0.0, 1.0})


def _keep_mask(X: np.ndarray, names: list[str] | None = None) -> np.ndarray:
    """True for columns with usable variance across the concatenated story."""
    keep = np.ones(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        if names is not None and names[j] in FORCE_DROP:
            keep[j] = False
            continue
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            keep[j] = False
            continue
        if float(np.std(finite, ddof=1) if finite.size > 1 else 0.0) < STD_EPS:
            keep[j] = False
            continue
        if _is_binary(col) and int(np.sum(np.abs(finite) > 0.5)) < MIN_POS:
            keep[j] = False
    return keep


def _pca_fit(X: np.ndarray, n_comp: int):
    mu = X.mean(axis=0)
    xc = X - mu
    # SVD on centered design (n × p). Loadings are V.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(n_comp, vt.shape[0])
    loadings = vt[:k].T  # p × k
    ev = (s[:k] ** 2) / max(X.shape[0] - 1, 1)
    scores = xc @ loadings
    return mu, loadings, ev, scores


def main() -> None:
    wordinfo = []
    glove = []
    names = None
    for i, sid in enumerate(SECTIONS):
        d = SHARED / f"section_{sid:03d}"
        wi = np.load(d / "X_word_wordinfo.npy").astype(np.float64)
        gl = np.load(d / "X_word_glove.npy").astype(np.float64)
        nms = _read_names(d / "X_word_wordinfo_feature_names.txt")
        if wi.shape[0] != EXPECTED[i]:
            raise SystemExit(f"section {sid}: wordinfo rows {wi.shape[0]} != {EXPECTED[i]}")
        if gl.shape[0] != EXPECTED[i]:
            raise SystemExit(f"section {sid}: glove rows {gl.shape[0]} != {EXPECTED[i]}")
        if names is None:
            names = nms
        elif nms != names:
            raise SystemExit("wordinfo feature names differ across sections")
        wordinfo.append(wi)
        glove.append(gl)

    Xwi = np.vstack(wordinfo)
    Xgl = np.vstack(glove)
    if not np.isfinite(Xwi).all():
        n_bad = int(np.sum(~np.isfinite(Xwi)))
        # Impute non-finite with column min of finites (same as lexical logfreq).
        for j in range(Xwi.shape[1]):
            col = Xwi[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                raise SystemExit(f"wordinfo col {names[j]} has no finite values")
            col[~np.isfinite(col)] = finite.min()
            Xwi[:, j] = col
        print(f"Imputed {n_bad} non-finite wordinfo entries")
    if not np.isfinite(Xgl).all():
        raise SystemExit("GloVe contains non-finite values")

    dropped: dict[str, list[str]] = {}
    kept_by_key: dict[str, list[str]] = {}
    matrices: dict[str, np.ndarray] = {}

    for key, spec in FAMILY_SPEC.items():
        cols = _select_columns(names, spec)
        if not cols:
            raise SystemExit(f"No columns matched family {key}")
        fam_names = [names[j] for j in cols]
        X = Xwi[:, cols]
        keep = _keep_mask(X, fam_names)
        drop = [fam_names[j] for j, k in enumerate(keep) if not k]
        keep_names = [fam_names[j] for j, k in enumerate(keep) if k]
        if not keep_names:
            raise SystemExit(f"All columns dropped for {key}")
        dropped[key] = drop
        kept_by_key[key] = keep_names
        matrices[key] = X[:, keep]
        print(f"{key}: kept {len(keep_names)}/{len(fam_names)}  dropped={drop}")

    mu, loadings, ev, scores = _pca_fit(Xgl, N_GLOVE_PCS)
    sem_names = [f"glove_pc{j+1:02d}" for j in range(loadings.shape[1])]
    matrices["hdc_semantic"] = scores
    kept_by_key["hdc_semantic"] = sem_names
    dropped["hdc_semantic"] = []
    evr = ev / max(float(ev.sum()), 1e-30)
    print(
        f"hdc_semantic: GloVe PCA-{loadings.shape[1]}  "
        f"evr={evr.sum():.3f}  evr1={evr[0]:.3f}"
    )

    offsets = np.cumsum([0] + [m.shape[0] for m in wordinfo])
    family_order = list(FAMILY_SPEC.keys()) + ["hdc_semantic"]
    membership = {
        "pipeline": "hdc_pilot_v1",
        "n_glove_pcs": int(loadings.shape[1]),
        "glove_explained_variance_ratio": evr.tolist(),
        "dropped_columns": dropped,
        "families": [],
        "min_pos": MIN_POS,
        "std_eps": STD_EPS,
        "force_drop": sorted(FORCE_DROP),
    }
    for key in family_order:
        spec_family = (
            FAMILY_SPEC[key]["family"] if key in FAMILY_SPEC else "semantic"
        )
        membership["families"].append(
            {
                "feature_key": key,
                "family": spec_family,
                "n_cols": int(matrices[key].shape[1]),
                "columns": kept_by_key[key],
            }
        )

    pca_payload = {
        "mean": mu,
        "loadings": loadings,
        "eigenvalues": ev,
        "n_words": int(Xgl.shape[0]),
        "n_raw_dims": int(Xgl.shape[1]),
        "n_components": int(loadings.shape[1]),
    }

    for i, sid in enumerate(SECTIONS):
        d = SHARED / f"section_{sid:03d}"
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        for key in family_order:
            out_key = f"{key}_{FEATURE_TAG}" if FEATURE_TAG else key
            X = np.asarray(matrices[key][sl], dtype=np.float32)
            if X.shape[0] != EXPECTED[i]:
                raise SystemExit(f"{key} section {sid} rows {X.shape[0]} != {EXPECTED[i]}")
            np.save(d / f"X_word_{out_key}.npy", X)
            (d / f"X_word_{out_key}_feature_names.txt").write_text(
                "\n".join(kept_by_key[key]) + "\n"
            )
            meta = {
                "feature": out_key,
                "base_feature": key,
                "feature_tag": FEATURE_TAG or None,
                "min_pos": MIN_POS,
                "family": next(
                    f["family"] for f in membership["families"] if f["feature_key"] == key
                ),
                "section_id": sid,
                "n_words": int(X.shape[0]),
                "n_cols": int(X.shape[1]),
                "columns": kept_by_key[key],
            }
            (d / f"X_word_{out_key}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"  wrote {d / f'X_word_{out_key}.npy'}  shape={X.shape}")

        (d / "hdc_pilot_membership.json").write_text(
            json.dumps(membership, indent=2) + "\n"
        )

    np.savez(
        SHARED / "hdc_glove_pca.npz",
        mean=mu.astype(np.float64),
        loadings=loadings.astype(np.float64),
        eigenvalues=ev.astype(np.float64),
    )
    (SHARED / "hdc_pilot_membership.json").write_text(
        json.dumps(membership, indent=2) + "\n"
    )
    (SHARED / "hdc_glove_pca_meta.json").write_text(
        json.dumps(
            {
                k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in pca_payload.items()
                if k != "mean" and k != "loadings"
            }
            | {"explained_variance_ratio": evr.tolist()},
            indent=2,
        )
        + "\n"
    )
    print("HDC pilot feature files ready.")


if __name__ == "__main__":
    main()
