#!/usr/bin/env python3
"""Preflight QC for MFA v2 phone/tone/envelope encoding features.

Refuses analysis unless:
  - no phone name contains a Chao tone contour
  - expected Mandarin tones 1–4 are represented; no unintended constant cols remain
  - clipped occupancy is bounded (≤ 1) and interval overcount is zero
  - standardized joint design is full rank with acceptable condition number

Also verifies feature-name / column alignment and records a QC JSON sidecar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TONE_CHARS = frozenset("˥˧˩˨˦")
DEFAULT_SHARED = Path(
    "/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/_shared_wordlocked_features"
)
DEFAULT_KEYS = ("envelope_v2", "mfa_phones_v2", "mfa_tone_v2")
# Soft upper bound for OLS-H designs (matches Julia KAPPA_INVALID spirit, but
# gate earlier at prepare time with a stricter practical limit).
COND_MAX = 1e8
RANK_TOL_SCALE = 1e-10
HALF_COND_MAX = 85.0  # match Julia KAPPA_INVALID for OLS-H halves


def _blocked_half_indices(n: int, n_blocks: int = 8, half_blocks=(0, 1, 2, 3)) -> np.ndarray:
    edges = np.round(np.linspace(0, n, n_blocks + 1)).astype(int)
    idx = []
    for b in half_blocks:
        idx.extend(range(edges[b], edges[b + 1]))
    return np.asarray(idx, dtype=int)


def _half_rank_cond(X_full: np.ndarray, idx: np.ndarray) -> tuple[int, float, int]:
    mu = X_full.mean(axis=0)
    sd = X_full.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X_full[idx] - mu) / sd
    s = np.linalg.svd(Z, compute_uv=False)
    tol = max(Z.shape) * np.finfo(float).eps * float(s[0])
    rank = int(np.sum(s > max(tol, RANK_TOL_SCALE * float(s[0]))))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    n_zero = int(np.sum(Z.std(axis=0, ddof=0) < 1e-12))
    return rank, cond, n_zero



def _load_pair(section_dir: Path, key: str) -> tuple[np.ndarray, list[str], dict]:
    X = np.load(section_dir / f"X_word_{key}.npy")
    names_path = section_dir / f"X_word_{key}_feature_names.txt"
    meta_path = section_dir / f"X_word_{key}_meta.json"
    if not names_path.is_file():
        raise SystemExit(f"Missing feature names: {names_path}")
    names = [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if X.ndim != 2 or X.shape[1] != len(names):
        raise SystemExit(f"{key}: shape {X.shape} != names {len(names)}")
    if not np.isfinite(X).all():
        raise SystemExit(f"{key}: non-finite values")
    return X, names, meta


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    μ = X.mean(axis=0)
    σ = X.std(axis=0, ddof=0)
    σ = np.where(σ > 1e-12, σ, 1.0)
    return (X - μ) / σ


def run_preflight(
    shared: Path,
    keys: tuple[str, ...],
    sections: tuple[int, ...] = (1, 2),
    cond_max: float = COND_MAX,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    stacks: list[np.ndarray] = []
    all_names: list[str] = []
    report: dict = {"sections": {}, "keys": list(keys)}

    for key in keys:
        blocks = []
        names_ref = None
        for sid in sections:
            sec = shared / f"section_{sid:03d}"
            path = sec / f"X_word_{key}.npy"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            X, names, meta = _load_pair(sec, key)
            if names_ref is None:
                names_ref = names
            elif names != names_ref:
                errors.append(f"{key}: feature names differ across sections")
            blocks.append(X)
            report.setdefault("meta", {})[f"section_{sid:03d}/{key}"] = {
                "shape": list(X.shape),
                "n_names": len(names),
                "occupancy_sum_max": meta.get("occupancy_sum_max"),
                "tone_counts": meta.get("tone_counts"),
                "dropped_constant_columns": meta.get("dropped_constant_columns"),
            }
            if key.endswith("phones_v2") or "phones" in key:
                bad = [n for n in names if any(c in n for c in TONE_CHARS)]
                if bad:
                    errors.append(f"{key} s{sid}: phone names retain tone marks: {bad[:8]}")
                inv = meta.get("phone_inventory") or []
                bad_inv = [p for p in inv if any(c in p for c in TONE_CHARS)]
                if bad_inv:
                    errors.append(f"{key} s{sid}: inventory retains tone: {bad_inv[:8]}")
                occ = meta.get("occupancy_sum_max")
                if occ is not None and float(occ) > 1.0 + 1e-6:
                    errors.append(f"{key} s{sid}: occupancy_sum_max={occ} > 1 (clipping failed)")
                if "coverage" in names:
                    errors.append(f"{key}: dependent coverage column still present")
            if key.endswith("tone_v2") or "tone" in key:
                tc = meta.get("tone_counts") or {}
                for t in ("1", "2", "3", "4"):
                    if int(tc.get(t, 0)) <= 0:
                        errors.append(f"{key} s{sid}: tone {t} count is 0")
            if key.startswith("envelope") and "env_onset" in names:
                errors.append(f"{key}: env_onset duplicate still present (use env_t00 only)")
        if not blocks:
            continue
        Xcat = np.vstack(blocks)
        # Residual constant columns after extraction are a hard fail.
        std = Xcat.std(axis=0, ddof=0)
        const_idx = np.where(std < 1e-12)[0]
        if const_idx.size:
            const_names = [names_ref[i] for i in const_idx]
            errors.append(f"{key}: constant columns remain: {const_names}")
        stacks.append(_zscore_cols(Xcat))
        all_names.extend(f"{key}::{n}" for n in names_ref)

    if errors:
        report["ok"] = False
        report["errors"] = errors
        return report

    X_joint = np.hstack(stacks)
    n, p = X_joint.shape
    # Rank / condition on standardized joint design
    try:
        s = np.linalg.svd(X_joint, compute_uv=False)
    except np.linalg.LinAlgError as e:
        errors.append(f"SVD failed on joint design: {e}")
        report["ok"] = False
        report["errors"] = errors
        return report
    tol = max(n, p) * np.finfo(float).eps * float(s[0])
    rank = int(np.sum(s > max(tol, RANK_TOL_SCALE * float(s[0]))))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    report["design"] = {
        "n": n,
        "p": p,
        "rank": rank,
        "cond": cond,
        "singular_min": float(s[-1]),
        "singular_max": float(s[0]),
        "feature_names": all_names,
    }
    if rank < p:
        errors.append(f"joint design rank-deficient: rank={rank} < p={p}")
    if not np.isfinite(cond) or cond > cond_max:
        errors.append(f"joint condition number unacceptable: cond={cond:.3e} (max {cond_max:.3e})")
    if n < p:
        errors.append(f"n_words={n} < n_features={p}; OLS H unidentified")

    # Section-aware blocked halves (4/8 blocks per section), matching B2B partitions.
    row_counts = []
    for sid in sections:
        sec = shared / f"section_{sid:03d}"
        X0 = np.load(sec / f"X_word_{keys[0]}.npy")
        row_counts.append(X0.shape[0])
    offsets = np.cumsum([0] + row_counts)
    half_parts = []
    comp_parts = []
    for i, n_sec in enumerate(row_counts):
        local = np.arange(n_sec)
        h = _blocked_half_indices(n_sec, n_blocks=8, half_blocks=(0, 1, 2, 3))
        c = _blocked_half_indices(n_sec, n_blocks=8, half_blocks=(4, 5, 6, 7))
        half_parts.append(local[h] + offsets[i])
        comp_parts.append(local[c] + offsets[i])
    half_idx = np.concatenate(half_parts)
    comp_idx = np.concatenate(comp_parts)
    half_reports = []
    for lab, idx in (("half1", half_idx), ("half2", comp_idx)):
        r, c, z = _half_rank_cond(X_joint, idx)
        half_reports.append({"half": lab, "n": int(len(idx)), "rank": r, "cond": c, "zero_var_cols": z})
        if r < p or (not np.isfinite(c)) or c > HALF_COND_MAX:
            errors.append(
                f"blocked {lab} design unidentified: rank={r}/{p} cond={c:.3e} "
                f"zero_var_cols={z} (raise min_phone_words / drop sparse features)"
            )
    report["design"]["blocked_halves"] = half_reports

    report["ok"] = len(errors) == 0
    report["errors"] = errors
    report["warnings"] = warnings
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--keys", type=str, default=",".join(DEFAULT_KEYS))
    ap.add_argument("--sections", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cond-max", type=float, default=COND_MAX)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    keys = tuple(k.strip() for k in args.keys.replace(";", ",").split(",") if k.strip())
    report = run_preflight(args.shared_dir, keys, tuple(args.sections), args.cond_max)
    out = args.out_json
    if out is None:
        out = args.shared_dir / "mfa_v2_preflight_qc.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "errors", "warnings", "design") if k in report},
                     indent=2)[:4000])
    print(f"Wrote {out}")
    if not report["ok"]:
        print("PREFLIGHT FAILED", file=sys.stderr)
        for e in report["errors"]:
            print(" -", e, file=sys.stderr)
        raise SystemExit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
