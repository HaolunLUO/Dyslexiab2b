#!/usr/bin/env python3
"""Write nucleus-locked X families into a NEW shared tree.

Never writes into extracted_sections_wordlocked_shared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tone_ctx_lib as L  # noqa: E402


def _fill(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(x, nan=default, posinf=default, neginf=default)


def build_controls(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    X = np.zeros((n, L.N_CTX_CONTROLS), dtype=np.float64)
    names = list(L.CONTROL_NAMES)
    idx = {n: i for i, n in enumerate(names)}
    X[:, idx["env_mean"]] = _fill(df, "env_mean")
    X[:, idx["env_max"]] = _fill(df, "env_max")
    X[:, idx["env_std"]] = _fill(df, "env_std")
    X[:, idx["env_slope"]] = _fill(df, "env_slope")
    X[:, idx["periodicity"]] = _fill(df, "periodicity")
    # nucleus_onset is constant 1 and makes passthrough OLS-H unidentified
    X[:, idx["nucleus_onset"]] = 0.0
    X[:, idx["duration"]] = _fill(df, "duration")
    X[:, idx["phrase_position"]] = _fill(df, "phrase_position")
    X[:, idx["preceding_pause"]] = _fill(df, "preceding_pause")
    X[:, idx["speech_rate"]] = _fill(df, "speech_rate")
    X[:, idx["logfreq"]] = _fill(df, "logfreq")
    X[:, idx["surprisal"]] = _fill(df, "surprisal")
    for t, col in zip(L.TONES, ("prev_t1", "prev_t2", "prev_t3", "prev_t4")):
        X[:, idx[col]] = (df["prev_surface_tone"].astype(str) == t).to_numpy(dtype=np.float64)
    X[:, idx["prev_f0_end"]] = _fill(df, "prev_f0_end")
    X[:, idx["prev_f0_slope"]] = _fill(df, "prev_f0_slope")
    for i, lab in enumerate(L.INITIALS):
        X[:, idx[f"init_{lab or 'none'}"]] = (df["initial"].fillna("").astype(str) == lab).to_numpy(dtype=np.float64)
    for i, lab in enumerate(L.FINALS):
        X[:, idx[f"fin_{lab}"]] = (df["final"].fillna("other").astype(str) == lab).to_numpy(dtype=np.float64)
    # leftover initials/finals → other
    init_block = slice(idx["init_none"], idx["init_other"] + 1)
    fin_block = slice(idx["fin_a"], idx["fin_other"] + 1)
    miss_i = X[:, init_block].sum(1) < 0.5
    X[miss_i, idx["init_other"]] = 1.0
    miss_f = X[:, fin_block].sum(1) < 0.5
    X[miss_f, idx["fin_other"]] = 1.0
    return X


def build_pitch(df: pd.DataFrame) -> np.ndarray:
    # Level (c0) vs dynamic (c1, c2, f0_change, f0_range). rel_f0_mean dropped:
    # it is r≈0.999 with c0 and makes split-half κ ~ 1e3.
    return np.column_stack([
        _fill(df, "c0"),
        _fill(df, "c1"),
        _fill(df, "c2"),
        _fill(df, "f0_change"),
        _fill(df, "f0_range"),
    ])


def build_evidence(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_fill(df, c) for c in ("e1", "e2", "e3")])


def build_resid(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_fill(df, c) for c in ("r1", "r2", "r3")])


def write_timing(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame({
        "word": df["word"].astype(str),
        "lemma": df["char"].astype(str),
        "logfreq": _fill(df, "logfreq"),
        "onset_relative": df["nucleus_on"].to_numpy(dtype=np.float64),
        "offset_relative": df["nucleus_off"].to_numpy(dtype=np.float64),
    })
    out["onset_sample"] = np.round(out["onset_relative"] * L.FS).astype(int)
    out["offset_sample"] = np.round(out["offset_relative"] * L.FS).astype(int)
    if np.any(np.diff(out["onset_sample"].to_numpy()) < 0):
        raise SystemExit(f"{path}: onset_sample not monotonic")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


# Split-halves of ~n/2 must stay identified (KAPPA_INVALID=85). Rare
# exclusive dummies (fin_un n=1, fin_ie n=3) vanish in a half and make κ~1e16.
MIN_DUMMY_COUNT = 80
KAPPA_INVALID = 85.0


def _merge_rare_dummies(
    X: np.ndarray, names: list[str], prefix: str, min_count: int
) -> tuple[np.ndarray, list[str]]:
    names = list(names)
    X = np.asarray(X, dtype=np.float64).copy()
    other_name = f"{prefix}other"
    cols = [i for i, n in enumerate(names) if n.startswith(prefix) and n != other_name]
    if other_name in names:
        other_i = names.index(other_name)
    else:
        X = np.column_stack([X, np.zeros(X.shape[0], dtype=np.float64)])
        names.append(other_name)
        other_i = len(names) - 1
    drop: list[int] = []
    for i in cols:
        if float(X[:, i].sum()) < min_count:
            X[:, other_i] += X[:, i]
            drop.append(i)
    X[:, other_i] = np.clip(X[:, other_i], 0.0, 1.0)
    keep = [i for i in range(X.shape[1]) if i not in set(drop)]
    return X[:, keep], [names[i] for i in keep]


def prune_fullrank(
    X: np.ndarray,
    names: list[str],
    dummy_prefixes: tuple[str, ...] = (),
    min_count: int = MIN_DUMMY_COUNT,
) -> tuple[np.ndarray, list[str]]:
    """Merge rare dummies, drop constants, drop one dummy per exclusive block."""
    names = list(names)
    X = np.asarray(X, dtype=np.float64).copy()
    for pref in dummy_prefixes:
        X, names = _merge_rare_dummies(X, names, pref, min_count)
    keep = []
    for j, name in enumerate(names):
        if name == "nucleus_onset":
            continue
        if float(np.std(X[:, j])) < 1e-8:
            continue
        keep.append(j)
    X = X[:, keep]
    names = [names[j] for j in keep]
    drop = []
    for pref in dummy_prefixes:
        cols = [i for i, n in enumerate(names) if n.startswith(pref)]
        if len(cols) < 2:
            continue
        freq = X[:, cols].sum(0)
        drop.append(cols[int(np.argmax(freq))])
    keep = [i for i in range(len(names)) if i not in set(drop)]
    return X[:, keep], [names[i] for i in keep]


def split_half_max_cond(X: np.ndarray, n_parts: int = 40, seed: int = 42) -> float:
    """Global z-score, then cond of random n/2 row halves (B2B partition analog)."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Z = (X - X.mean(0)) / sd
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    worst = 0.0
    for _ in range(n_parts):
        idx = rng.permutation(n)
        for half in (idx[: n // 2], idx[n // 2 :]):
            s = np.linalg.svd(Z[half], compute_uv=False)
            if s.size == 0 or s[-1] <= 0 or not np.isfinite(s[-1]):
                return float("inf")
            worst = max(worst, float(s[0] / s[-1]))
    return worst


def write_family(sec_dir: Path, key: str, X: np.ndarray, names: tuple[str, ...] | list[str]) -> None:
    if X.shape[1] != len(names):
        raise SystemExit(f"{key}: dim {X.shape[1]} != {len(names)}")
    if not np.isfinite(X).all():
        raise SystemExit(f"{key}: non-finite")
    np.save(sec_dir / f"X_word_{key}.npy", X.astype(np.float64))
    (sec_dir / f"X_word_{key}_feature_names.txt").write_text("\n".join(names) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=L.CTX_DIR / "tone_ctx_tokens_scored.parquet")
    ap.add_argument("--out-shared", type=Path, default=L.DEFAULT_NUCLEUS_SHARED)
    args = ap.parse_args()
    frozen = Path("/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared")
    if Path(args.out_shared).resolve() == (frozen / "_shared_wordlocked_features").resolve():
        raise SystemExit("Refusing to write into the frozen word-locked shared tree")
    df = pd.read_parquet(args.scored)
    n_noninit = int((df["syllable_i"] > 0).sum())
    print(f"parent-word surprisal reused on {n_noninit} non-initial syllables (known limitation)")
    groups = []
    for sid, g in df.groupby("section", sort=True):
        groups.append((int(sid), g.sort_values(["nucleus_on", "word_i", "syllable_i"]).reset_index(drop=True)))
    Xc_all = np.vstack([build_controls(g) for _, g in groups])
    Xc_all, ctrl_names = prune_fullrank(Xc_all, list(L.CONTROL_NAMES), ("init_", "fin_"))
    print(f"controls after prune: {Xc_all.shape[1]}  {ctrl_names}")
    families = {
        "ctx_controls": Xc_all,
        "ctx_pitch": np.vstack([build_pitch(g) for _, g in groups]),
        "ctx_evidence": np.vstack([build_evidence(g) for _, g in groups]),
        "ctx_resid": np.vstack([build_resid(g) for _, g in groups]),
    }
    joint = np.hstack([families[k] for k in ("ctx_controls", "ctx_pitch", "ctx_evidence", "ctx_resid")])
    nested = {
        "M0": families["ctx_controls"],
        "M1": np.hstack([families["ctx_controls"], families["ctx_pitch"]]),
        "M2": np.hstack([families["ctx_controls"], families["ctx_pitch"], families["ctx_evidence"]]),
        "M3": joint,
    }
    kappas = {k: split_half_max_cond(X) for k, X in nested.items()}
    print("split-half max κ", "  ".join(f"{k}={v:.3g}" for k, v in kappas.items()))
    bad = {k: v for k, v in kappas.items() if v > KAPPA_INVALID}
    if bad:
        raise SystemExit(
            f"split-half design unidentified: {bad} (limit {KAPPA_INVALID}). "
            "Fix feature redundancy before EEG."
        )
    offset = 0
    lengths = {}
    for sid, g in groups:
        n = len(g)
        sec = args.out_shared / f"section_{sid:03d}"
        write_timing(g, sec / "word_timing_relative.csv")
        write_family(sec, "ctx_controls", families["ctx_controls"][offset:offset + n], ctrl_names)
        write_family(sec, "ctx_pitch", families["ctx_pitch"][offset:offset + n], L.PITCH_NAMES)
        write_family(sec, "ctx_evidence", families["ctx_evidence"][offset:offset + n], L.EVIDENCE_NAMES)
        write_family(sec, "ctx_resid", families["ctx_resid"][offset:offset + n], L.RESID_NAMES)
        lengths[sid] = n
        offset += n
        print(f"section {sid}: n={n} → {sec}")
    meta = {
        "n_per_section": lengths,
        "n_total": int(sum(lengths.values())),
        "dims": {
            "ctx_controls": int(Xc_all.shape[1]),
            "ctx_pitch": L.N_CTX_PITCH,
            "ctx_evidence": L.N_CTX_EVIDENCE,
            "ctx_resid": L.N_CTX_RESID,
        },
        "control_names": ctrl_names,
        "min_dummy_count": MIN_DUMMY_COUNT,
        "split_half_max_kappa": kappas,
        "n_noninitial_surprisal_inherited": n_noninit,
    }
    args.out_shared.parent.mkdir(parents=True, exist_ok=True)
    (args.out_shared / "tone_ctx_v1_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (L.CTX_DIR / "sec_lengths.json").write_text(json.dumps(lengths) + "\n")
    print("wrote", args.out_shared)


if __name__ == "__main__":
    main()
