#!/usr/bin/env python3
"""Recoverability of nested M0–M3 unique traces on the real design matrices."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tone_ctx_lib as L  # noqa: E402


def load_joint(shared: Path) -> dict[str, np.ndarray]:
    fams = ("ctx_controls", "ctx_pitch", "ctx_evidence", "ctx_resid")
    blocks = {f: [] for f in fams}
    for sid in L.SECTIONS:
        sec = shared / f"section_{sid:03d}"
        for f in fams:
            blocks[f].append(np.load(sec / f"X_word_{f}.npy"))
    return {f: np.vstack(blocks[f]) for f in fams}


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (X - mu) / sd


def ridge_G(Y: np.ndarray, X: np.ndarray, lam: float) -> np.ndarray:
    # G: (n_y x n_x) mapping Y -> X   Xhat = Y @ G
    YtY = Y.T @ Y
    ev, U = np.linalg.eigh(YtY)
    ev = np.maximum(ev, 0.0)
    inv = U @ np.diag(1.0 / (ev + lam)) @ U.T
    return inv @ (Y.T @ X)


def ols_H(Xhat: np.ndarray, X: np.ndarray) -> np.ndarray:
    H, *_ = np.linalg.lstsq(Xhat, X, rcond=None)
    return H


def family_traces(H: np.ndarray, slices: dict[str, slice]) -> dict[str, float]:
    d = np.diag(H)
    return {k: float(d[sl].sum()) for k, sl in slices.items()}


def b2b_once(Y: np.ndarray, X: np.ndarray, slices: dict[str, slice], lam: float, rng) -> dict[str, float]:
    n = Y.shape[0]
    idx = rng.permutation(n)
    mid = n // 2
    tr, te = idx[:mid], idx[mid:]
    G = ridge_G(Y[tr], X[tr], lam)
    Xhat = Y[te] @ G
    H = ols_H(Xhat, X[te])
    return family_traces(H, slices)


def make_slices(dims: dict[str, int]) -> dict[str, slice]:
    out = {}
    start = 0
    for k, p in dims.items():
        out[k] = slice(start, start + p)
        start += p
    return out


def colored_noise(n: int, d: int, rng, rho: float = 0.7) -> np.ndarray:
    e = rng.normal(size=(n, d))
    for t in range(1, n):
        e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    return e


def simulate(X_by: dict[str, np.ndarray], condition: str, rng, n_ch: int = 8, snr: float = 0.6):
    Xc = zscore(X_by["ctx_controls"])
    Xp = zscore(X_by["ctx_pitch"])
    Xe = zscore(X_by["ctx_evidence"])
    Xr = zscore(X_by["ctx_resid"])
    n = Xc.shape[0]
    W = {
        "controls": rng.normal(scale=0.3, size=(Xc.shape[1], n_ch)),
        "pitch": rng.normal(scale=0.5, size=(Xp.shape[1], n_ch)),
        "evidence": rng.normal(scale=0.5, size=(Xe.shape[1], n_ch)),
        "resid": rng.normal(scale=0.5, size=(Xr.shape[1], n_ch)),
    }
    signal = np.zeros((n, n_ch))
    if condition in ("controls", "all"):
        signal += Xc @ W["controls"]
    if condition in ("pitch", "all"):
        signal += Xp @ W["pitch"]
    if condition in ("evidence", "all"):
        signal += Xe @ W["evidence"]
    if condition in ("resid", "all"):
        signal += Xr @ W["resid"]
    noise = colored_noise(n, n_ch, rng)
    sig_std = signal.std() + 1e-8
    Y = signal + (sig_std / max(snr, 1e-3)) * (noise / (noise.std() + 1e-8))
    X = np.hstack([Xc, Xp, Xe, Xr])
    return Y, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=Path, default=L.DEFAULT_NUCLEUS_SHARED)
    ap.add_argument("--out", type=Path, default=L.CTX_DIR / "recoverability.json")
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()
    X_by = load_joint(args.shared)
    dims = {k: X_by[k].shape[1] for k in ("ctx_controls", "ctx_pitch", "ctx_evidence", "ctx_resid")}
    slices = make_slices(dims)
    rng = np.random.default_rng(7)
    lams = (1.0, 10.0, 100.0)
    summary = {}
    for cond in ("controls", "pitch", "evidence", "resid", "all"):
        traces = {k: [] for k in slices}
        for lam in lams:
            for _ in range(args.repeats):
                Y, X = simulate(X_by, cond, rng)
                tr = b2b_once(Y, X, slices, lam, rng)
                for k, v in tr.items():
                    traces[k].append(v)
        summary[cond] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in traces.items()}
        print(cond, {k: round(v["mean"], 3) for k, v in summary[cond].items()})

    m3_false = max(
        summary["controls"]["ctx_resid"]["mean"],
        summary["pitch"]["ctx_resid"]["mean"],
    )
    m3_true = summary["resid"]["ctx_resid"]["mean"]
    m2_true = summary["evidence"]["ctx_evidence"]["mean"]
    m2_false = summary["pitch"]["ctx_evidence"]["mean"]
    gates = {
        "no_false_M3": bool(m3_true > m3_false + 0.02),
        "M3_recoverable": bool(m3_true > 0.05),
        "M2_recoverable": bool(m2_true > m2_false + 0.02),
        "sign_stable_across_lambda": True,
    }
    # lambda stability: resid condition should top-rank ctx_resid at each lam
    for lam in lams:
        Y, X = simulate(X_by, "resid", np.random.default_rng(11))
        tr = b2b_once(Y, X, slices, lam, np.random.default_rng(12))
        if tr["ctx_resid"] < max(tr["ctx_pitch"], tr["ctx_evidence"]) - 0.05:
            gates["sign_stable_across_lambda"] = False
    out = {"dims": dims, "summary": summary, "gates": gates, "launch": bool(all(gates.values()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(gates, indent=2))
    if not out["launch"]:
        print("RECOVERABILITY FAILED — do not launch the 63-subject array.")
        sys.exit(2)
    print("RECOVERABILITY PASSED")


if __name__ == "__main__":
    main()
