#!/usr/bin/env python3
"""Tone family v3: deviation in standardized Legendre coefficient space.

Reuses v2 rime contours (voiced-only 10-pt). Distance is Euclidean in
z-scored (c0, c1, c2) = (register, slope, curvature), so register variance
cannot drown slope. Escalates A → B → C until the discriminability gate
passes (or all fail).

  A  coefficient space on section-z-scored 10-pt contours (v2 contours)
  B  same, after subtracting a 2 s running-median local F0 baseline
  C  B + templates split by pre-pausal vs non-final position

Writes X_word_tone_v3.npy only if the gate passes. Never overwrites v1/v2.

Usage:
  .venv_gpt2/bin/python scripts/extract_tone_v3_wordlocked.py
  .venv_gpt2/bin/python scripts/extract_tone_v3_wordlocked.py --step A
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
DEFAULT_CACHE = (
    POOL / "extracted_sections_wordlocked_shared_acoustic_residual" / "_acoustic_features_cache"
)
DEFAULT_TG = POOL / "mfa_lppcn_wordlocked" / "aligned"
FS = 100.0
NPTS = 10
N_COEF = 3
SECTIONS = (1, 2)
MIN_CANON_DUR = 0.12
MIN_FIT_PTS = 5
WIN_S = 2.0
TONE_COLS = ("tone_1", "tone_2", "tone_3", "tone_4", "tone_dev", "tone_dev_valid")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_v2():
    return _load("extract_tone_v2", ROOT / "scripts" / "extract_tone_v2_wordlocked.py")


def legendre_coeffs(contour: np.ndarray, n_coef: int = N_COEF) -> np.ndarray | None:
    """Least-squares Legendre P0..P{n-1} on finite grid points (x in [-1, 1])."""
    y = np.asarray(contour, dtype=np.float64)
    m = np.isfinite(y)
    if int(m.sum()) < max(MIN_FIT_PTS, n_coef + 1):
        return None
    x = np.linspace(-1.0, 1.0, y.size)
    P = np.polynomial.legendre.legvander(x[m], n_coef - 1)
    c, *_ = np.linalg.lstsq(P, y[m], rcond=None)
    if not np.isfinite(c).all():
        return None
    return c.astype(np.float64)


def zscore_cols(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (M - mu) / sd, mu, sd


def running_median(x: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling median; NaNs ignored. win should be odd."""
    import pandas as pd
    if win % 2 == 0:
        win += 1
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=max(5, win // 10))
        .median()
        .to_numpy(dtype=np.float64)
    )


def local_logf0_residual(f0_hz: np.ndarray, voiced: np.ndarray, win_s: float = WIN_S,
                         fs: float = FS) -> np.ndarray:
    log_f0 = np.where(voiced & np.isfinite(f0_hz) & (f0_hz > 0), np.log(f0_hz), np.nan)
    base = running_median(log_f0, int(round(win_s * fs)))
    resid = log_f0 - base
    resid[~np.isfinite(log_f0)] = np.nan
    return resid


def time_normalize_log(
    log_f0: np.ndarray,
    t_on: float,
    t_off: float,
    v2,
    *,
    fs: float = FS,
    npts: int = NPTS,
) -> tuple[np.ndarray | None, dict]:
    """Same validity / gap rules as v2, but the series is already log-F0 (NaN unvoiced)."""
    a = max(0, int(round(t_on * fs)))
    b = min(log_f0.size, max(a + 1, int(round(t_off * fs))))
    seg = np.asarray(log_f0[a:b], dtype=np.float64)
    n = seg.size
    meta = {
        "n_frames": n, "n_voiced": 0, "voiced_frac": 0.0,
        "n_final_third_voiced": 0, "valid": False, "reason": "",
    }
    if n < 2:
        meta["reason"] = "short_rime"
        return None, meta
    voiced = np.isfinite(seg)
    meta["n_voiced"] = int(voiced.sum())
    meta["voiced_frac"] = float(voiced.mean())
    third = max(1, n // 3)
    meta["n_final_third_voiced"] = int(voiced[-third:].sum())
    if meta["voiced_frac"] < v2.MIN_VOICED_FRAC:
        meta["reason"] = "low_voiced_frac"
        return None, meta
    if meta["n_final_third_voiced"] < v2.MIN_FINAL_THIRD:
        meta["reason"] = "sparse_final_third"
        return None, meta
    t_rel = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    y_v = seg[voiced]
    t_v_s = np.flatnonzero(voiced).astype(np.float64) / fs
    runs = v2._runs_from_voiced_times(t_v_s, max_gap_s=v2.MAX_GAP_S)
    contour = np.full(npts, np.nan, dtype=np.float64)
    q = np.linspace(0.0, 1.0, npts)
    for run in runs:
        if run.size == 0:
            continue
        tt = t_rel[voiced][run]
        yy = y_v[run]
        lo, hi = float(tt[0]), float(tt[-1])
        in_run = (q >= lo - 1e-12) & (q <= hi + 1e-12)
        if not in_run.any():
            continue
        if run.size == 1:
            contour[in_run] = yy[0]
        else:
            contour[in_run] = np.interp(q[in_run], tt, yy)
    if int(np.isfinite(contour).sum()) < v2.MIN_RMSE_PTS:
        meta["reason"] = "too_few_grid_points"
        return None, meta
    meta["valid"] = True
    return contour, meta


def token_context(sylls: list[dict], i: int, v2) -> str:
    return "prepausal" if v2.is_prepausal(sylls, i) else "nonfinal"


def attach_contours_from_log(sections: list[dict], log_by_sec: dict[int, np.ndarray], v2) -> None:
    for sec in sections:
        log_f0 = log_by_sec[sec["section"]]
        for s in sec["all_sylls"]:
            rw = s.get("rime")
            if rw is None:
                s["contour"] = None
                s["rime_meta"] = {"valid": False, "reason": "no_vowel_nucleus"}
                continue
            c, meta = time_normalize_log(log_f0, rw[0], rw[1], v2)
            s["contour"] = c
            s["rime_meta"] = meta


def collect_valid(sections: list[dict], v2) -> list[dict]:
    out = []
    for sec in sections:
        sylls = sec["all_sylls"]
        for i, s in enumerate(sylls):
            if s.get("contour") is None:
                continue
            t = s.get("surface_tone")
            if t is None or t not in "1234":
                continue
            s["_ctx"] = token_context(sylls, i, v2)
            s["_section"] = sec["section"]
            out.append(s)
    return out


def fit_coeff_space(valid: list[dict], *, context_split: bool) -> dict:
    coeffs = []
    keep = []
    for s in valid:
        c = legendre_coeffs(s["contour"])
        s["coeff_raw"] = c
        if c is None:
            continue
        coeffs.append(c)
        keep.append(s)
    if not keep:
        raise SystemExit("No tokens with fittable Legendre coefficients")
    M = np.vstack(coeffs)
    Z, mu, sd = zscore_cols(M)
    for s, z in zip(keep, Z):
        s["coeff_z"] = z
    dropped = len(valid) - len(keep)

    templates: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    def _key(s: dict) -> str:
        if context_split:
            return f"{s['surface_tone']}_{s['_ctx']}"
        return s["surface_tone"]

    buckets: dict[str, list[np.ndarray]] = {}
    for s in keep:
        t = s.get("surface_tone")
        lex = s.get("lexical_tone")
        dur = float(s["offset"] - s["onset"])
        if t not in "1234" or t != lex or dur < MIN_CANON_DUR:
            continue
        buckets.setdefault(_key(s), []).append(s["coeff_z"])
    keys = [f"{t}_{ctx}" for t in "1234" for ctx in ("nonfinal", "prepausal")] if context_split else list("1234")
    for k in keys:
        rows = buckets.get(k, [])
        templates[k] = np.mean(np.vstack(rows), axis=0) if rows else np.zeros(N_COEF)
        counts[k] = len(rows)
    return {
        "templates": templates,
        "counts": counts,
        "z_mu": mu,
        "z_sd": sd,
        "n_fit": len(keep),
        "n_dropped": dropped,
        "context_split": context_split,
        "keep": keep,
    }


def template_key(s: dict, context_split: bool) -> str:
    if context_split:
        return f"{s['surface_tone']}_{s['_ctx']}"
    return s["surface_tone"]


def nearest_tone(z: np.ndarray, templates: dict, *, ctx: str | None) -> str | None:
    best_t, best_d = None, np.inf
    for t in "1234":
        key = f"{t}_{ctx}" if ctx is not None else t
        if key not in templates:
            continue
        d = float(np.linalg.norm(z - templates[key]))
        if d < best_d:
            best_t, best_d = t, d
    return best_t


def confusion_coeff(keep: list[dict], templates: dict, context_split: bool) -> dict:
    labels = list("1234")
    idx = {t: i for i, t in enumerate(labels)}
    C = np.zeros((4, 4), dtype=int)
    for s in keep:
        t = s.get("surface_tone")
        z = s.get("coeff_z")
        if t not in idx or z is None:
            continue
        ctx = s["_ctx"] if context_split else None
        pred = nearest_tone(z, templates, ctx=ctx)
        if pred is None:
            continue
        C[idx[t], idx[pred]] += 1
    row_n = C.sum(axis=1)
    acc, modal = {}, {}
    for i, t in enumerate(labels):
        n = int(row_n[i])
        acc[t] = float(C[i, i] / n) if n else 0.0
        modal[t] = labels[int(np.argmax(C[i]))] if n else None
    return {
        "labels": labels,
        "counts": C.tolist(),
        "n": int(C.sum()),
        "hit_rate": acc,
        "modal": modal,
        "modal_is_self": {t: modal[t] == t for t in labels},
        "T4_self_beats_T1": bool(C[3, 3] > C[3, 0]),
        "overall_acc": float(np.trace(C) / C.sum()) if C.sum() else 0.0,
    }


def gate_pass(confusion: dict) -> bool:
    return bool(all(confusion["modal_is_self"].values()) and confusion["T4_self_beats_T1"])


def print_confusion(tag: str, confusion: dict) -> None:
    print(f"[{tag}] nearest-template confusion  n={confusion['n']}  "
          f"acc={confusion['overall_acc']:.3f}")
    print("         pred T1   T2   T3   T4")
    for i, t in enumerate(confusion["labels"]):
        row = confusion["counts"][i]
        print(f"  true T{t}  {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}  "
              f"hit={confusion['hit_rate'][t]:.3f}  modal=T{confusion['modal'][t]}")
    print(f"  T4→T4 beats T4→T1: {confusion['T4_self_beats_T1']}")
    print(f"  GATE {'PASS' if gate_pass(confusion) else 'FAIL'}")


def save_confusion_plot(confusion: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = np.asarray(confusion["counts"], dtype=float)
    row = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(row, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(C[i, j])}\n{row[i, j]:.2f}", ha="center", va="center",
                    color="white" if row[i, j] > 0.45 else "black", fontsize=9)
    labs = ["T1", "T2", "T3", "T4"]
    ax.set_xticks(range(4), labs)
    ax.set_yticks(range(4), labs)
    ax.set_xlabel("Nearest template")
    ax.set_ylabel("Surface tone")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row fraction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_coeff_plot(keep: list[dict], templates: dict, path: Path, context_split: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    names = ("c0 register", "c1 slope", "c2 curvature")
    colors = {"1": "#c0392b", "2": "#2980b9", "3": "#27ae60", "4": "#8e44ad"}
    for j, ax in enumerate(axes):
        for t in "1234":
            vals = [s["coeff_z"][j] for s in keep if s.get("surface_tone") == t]
            if not vals:
                continue
            ax.scatter(np.full(len(vals), int(t)) + np.random.default_rng(0).normal(0, 0.06, len(vals)),
                       vals, s=4, alpha=0.15, c=colors[t], linewidths=0)
            key = t if not context_split else f"{t}_nonfinal"
            if key in templates:
                ax.plot(int(t), templates[key][j], "D", color=colors[t], ms=7)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xticks([1, 2, 3, 4], ["T1", "T2", "T3", "T4"])
        ax.set_title(names[j])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_word_tone(shared: Path, sections: list[dict], fit: dict) -> dict:
    templates = fit["templates"]
    context_split = fit["context_split"]
    raw_by_tone: dict[str, list[float]] = {t: [] for t in "1234"}
    keep_ids = {id(s) for s in fit["keep"]}
    for s in fit["keep"]:
        t = s["surface_tone"]
        key = template_key(s, context_split)
        d = float(np.linalg.norm(s["coeff_z"] - templates[key]))
        s["dev_raw"] = d
        raw_by_tone[t].append(d)
    tone_mu, tone_sd = {}, {}
    for t in "1234":
        arr = np.asarray(raw_by_tone[t], dtype=np.float64)
        tone_mu[t] = float(arr.mean()) if arr.size else 0.0
        sd = float(arr.std()) if arr.size > 1 else 1.0
        tone_sd[t] = sd if sd > 1e-8 else 1.0

    counts = {t: 0 for t in list("12345") + ["none"]}
    n_valid = 0
    for sec in sections:
        sid = sec["section"]
        n = len(sec["words"])
        X = np.zeros((n, 6), dtype=np.float64)
        for i, sylls in enumerate(sec["word_sylls"]):
            if not sylls:
                continue
            first = sylls[0]
            t = first.get("surface_tone")
            if t is not None and t in "1234":
                X[i, int(t) - 1] = 1.0
                counts[t] += 1
            elif t == "5":
                counts["5"] += 1
            else:
                counts["none"] += 1
            if id(first) in keep_ids and first.get("dev_raw") is not None and t in tone_mu:
                X[i, 4] = (first["dev_raw"] - tone_mu[t]) / tone_sd[t]
                X[i, 5] = 1.0
                n_valid += 1
        if not np.isfinite(X).all():
            raise SystemExit("tone_v3 non-finite")
        out = shared / f"section_{sid:03d}"
        np.save(out / "X_word_tone_v3.npy", X.astype(np.float64))
        (out / "X_word_tone_v3_feature_names.txt").write_text(
            "\n".join(TONE_COLS) + "\n", encoding="utf-8"
        )
        meta = {
            "feature": "tone_v3",
            "section_id": sid,
            "n_words": n,
            "n_dim": 6,
            "columns": list(TONE_COLS),
            "coding": (
                "first-syllable surface one-hot + Euclidean distance in "
                "z-scored Legendre (c0,c1,c2), z-scored within tone; invalid→0"
            ),
            "space": "legendre3_zscored",
            "context_split": context_split,
            "n_first_valid_dev": int(X[:, 5].sum()),
            "dev_raw_mean_by_tone": tone_mu,
            "dev_raw_sd_by_tone": tone_sd,
            "mean": X.mean(0).tolist(),
            "std": X.std(0, ddof=1).tolist(),
        }
        (out / "X_word_tone_v3_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  wrote X_word_tone_v3.npy section {sid}  shape={X.shape}  "
              f"valid_frac={X[:, 5].mean():.3f}")
    return {"counts": counts, "n_valid_first": n_valid, "tone_mu": tone_mu, "tone_sd": tone_sd}


def run_step(step: str, sections: list[dict], v2, v1, shared, cache, tg) -> dict:
    if step == "A":
        # v2 extract_section already attached section-z-scored contour_z; use raw 10-pt
        # contour (pre section-z) so c0 is absolute register, then we z-score coeffs.
        valid = collect_valid(sections, v2)
        print(f"[A] valid T1–T4 rimes={len(valid)}")
        fit = fit_coeff_space(valid, context_split=False)
    elif step == "B":
        log_by = {}
        for sec in sections:
            resid = local_logf0_residual(sec["f0_hz"], sec["voiced"])
            log_by[sec["section"]] = resid
        attach_contours_from_log(sections, log_by, v2)
        valid = collect_valid(sections, v2)
        print(f"[B] valid T1–T4 rimes after local baseline={len(valid)}")
        fit = fit_coeff_space(valid, context_split=False)
    elif step == "C":
        # assume B contours already on sections; if not, apply B first
        if not any(s.get("_ctx") for sec in sections for s in sec["all_sylls"]):
            log_by = {}
            for sec in sections:
                log_by[sec["section"]] = local_logf0_residual(sec["f0_hz"], sec["voiced"])
            attach_contours_from_log(sections, log_by, v2)
        valid = collect_valid(sections, v2)
        print(f"[C] valid T1–T4 rimes={len(valid)}  context-split templates")
        fit = fit_coeff_space(valid, context_split=True)
    else:
        raise SystemExit(f"unknown step {step}")
    confusion = confusion_coeff(fit["keep"], fit["templates"], fit["context_split"])
    print_confusion(step, confusion)
    print("  template n:", fit["counts"])
    print("  coeff z-score mu", np.round(fit["z_mu"], 3), "sd", np.round(fit["z_sd"], 3))
    return {"step": step, "fit": fit, "confusion": confusion, "pass": gate_pass(confusion)}


def load_v2_sections(v2, shared, cache, tg):
    v1 = v2._load_v1()
    mfa = v1._load_mfa_mod()
    sections = []
    for sid in SECTIONS:
        sec = v2.extract_section(v1, mfa, shared, cache, tg, sid)
        sections.append(sec)
        n_ok = sum(s.get("contour") is not None for s in sec["all_sylls"])
        print(f"section {sid}: words={len(sec['words'])} sylls={len(sec['all_sylls'])} valid_rimes={n_ok}")
    return sections, v1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--textgrid-dir", type=Path, default=DEFAULT_TG)
    ap.add_argument("--step", choices=["A", "B", "C", "auto"], default="auto")
    args = ap.parse_args()

    v2 = _load_v2()
    print(f"shared={args.shared}")
    sections, v1 = load_v2_sections(v2, args.shared, args.cache, args.textgrid_dir)

    order = ["A", "B", "C"] if args.step == "auto" else [args.step]
    results = []
    winner = None
    for step in order:
        r = run_step(step, sections, v2, v1, args.shared, args.cache, args.textgrid_dir)
        results.append({
            "step": step,
            "pass": r["pass"],
            "confusion": r["confusion"],
            "template_n": r["fit"]["counts"],
            "n_fit": r["fit"]["n_fit"],
        })
        save_confusion_plot(
            r["confusion"],
            ROOT / "plots" / f"tone_v3_confusion_step{step}.png",
            f"Tone v3 step {step}  n={r['confusion']['n']}  acc={r['confusion']['overall_acc']:.2f}",
        )
        save_coeff_plot(
            r["fit"]["keep"], r["fit"]["templates"],
            ROOT / "plots" / f"tone_v3_coeffs_step{step}.png",
            r["fit"]["context_split"],
        )
        if r["pass"]:
            winner = r
            break

    gate = {
        "passed": winner is not None,
        "winning_step": winner["step"] if winner else None,
        "steps": results,
        "gate": "discriminability_coeff_space_2026-08-31",
    }
    (ROOT / "plots" / "tone_v3_validation.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    print(f"wrote plots/tone_v3_validation.json  winner={gate['winning_step']}")

    if winner is None:
        print("GATE FAIL on A/B/C — not writing X_word_tone_v3.npy")
        return

    extra = write_word_tone(args.shared, sections, winner["fit"])
    gate["write"] = extra
    gate["winning_confusion"] = winner["confusion"]
    (ROOT / "plots" / "tone_v3_validation.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    print("GATE PASS — wrote X_word_tone_v3.npy")


if __name__ == "__main__":
    main()
