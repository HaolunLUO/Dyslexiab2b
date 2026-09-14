#!/usr/bin/env python3
"""Type-I of unique-increment scores under two circular-shift nulls.

Y is generated from controls+pitch only (no M2/M3 signal). A valid unique
contribution null for evidence/resid must stay near nominal α.
Whole-design k-shift preserves X-family covariance. Family-only shift of
the added block does not — that is the anti-pattern to reject if inflated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tone_ctx_lib as L  # noqa: E402

CONFIRM = (0.0, 0.8)
MIN_K = 20


def load_joint(shared: Path):
    fams = ("ctx_controls", "ctx_pitch", "ctx_evidence", "ctx_resid")
    blocks = {f: [] for f in fams}
    sec = []
    for i, sid in enumerate(L.SECTIONS):
        d = shared / f"section_{sid:03d}"
        n = None
        for f in fams:
            x = np.load(d / f"X_word_{f}.npy")
            blocks[f].append(x)
            n = x.shape[0]
        sec.append(np.full(n, i + 1, dtype=int))
    X = {f: np.vstack(blocks[f]) for f in fams}
    return X, np.concatenate(sec)


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (X - X.mean(0)) / sd


def family_traces(H, slices):
    d = np.diag(H)
    return {k: float(d[sl].sum()) for k, sl in slices.items()}


def eeg_ridge_P(Ytr, lam):
    """P such that G = P @ Xtr. Cached across design shifts."""
    Yc = Ytr - Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    scale = s / (s ** 2 + lam)
    return Vt.T * scale @ U.T


def unique_from_P(P, Yte, Xtr, Xte, slices):
    G = P @ Xtr
    Xhat = Yte @ G
    A = Xhat.T @ Xhat
    A.flat[:: A.shape[0] + 1] += 1e-8
    H = np.linalg.solve(A, Xhat.T @ Xte)
    return family_traces(H, slices)


def circshift_by_section(X, sec, k1, k2):
    out = np.empty_like(X)
    for sid, k in ((1, k1), (2, k2)):
        rows = np.flatnonzero(sec == sid)
        out[rows] = np.roll(X[rows], k, axis=0)
    return out


def allowed_k(n, min_k=MIN_K):
    return np.arange(min_k, n - min_k)


def draw_ks(sec, rng):
    ks = []
    for sid in (1, 2):
        n = int((sec == sid).sum())
        ks.append(int(rng.choice(allowed_k(n))))
    return ks[0], ks[1]


def make_Y(Xc, Xp, rng, n_ch=8, snr=0.6):
    n = Xc.shape[0]
    sig = Xc @ rng.normal(scale=0.3, size=(Xc.shape[1], n_ch))
    sig += Xp @ rng.normal(scale=0.5, size=(Xp.shape[1], n_ch))
    e = rng.normal(size=(n, n_ch))
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + np.sqrt(1 - 0.49) * e[t]
    return sig + (sig.std() / max(snr, 1e-3)) * (e / (e.std() + 1e-8))


def p_one_sided(obs, nulls):
    return float((1.0 + np.sum(np.asarray(nulls) >= obs)) / (len(nulls) + 1))


def main() -> None:
    shared = L.DEFAULT_NUCLEUS_SHARED
    X_by, sec = load_joint(shared)
    Xc, Xp, Xe, Xr = map(zscore, (X_by["ctx_controls"], X_by["ctx_pitch"],
                                  X_by["ctx_evidence"], X_by["ctx_resid"]))
    # Subsample tokens; predictor correlations are unchanged in expectation.
    rng0 = np.random.default_rng(0)
    take = rng0.choice(Xc.shape[0], size=min(900, Xc.shape[0]), replace=False)
    take.sort()
    Xc, Xp, Xe, Xr = Xc[take], Xp[take], Xe[take], Xr[take]
    sec = sec[take]
    X = np.hstack([Xc, Xp, Xe, Xr])
    dims = {"controls": Xc.shape[1], "pitch": Xp.shape[1],
            "evidence": Xe.shape[1], "resid": Xr.shape[1]}
    slices, start = {}, 0
    for k, p in dims.items():
        slices[k] = slice(start, start + p)
        start += p
    sl_e, sl_r = slices["evidence"], slices["resid"]
    n1 = int((sec == 1).sum())
    n2 = int((sec == 2).sum())
    rng = np.random.default_rng(11)
    n_rep = 30
    n_null = 40
    lam = 10.0
    # Freedman–Lane: residualize added block on reduced, shift residuals, add fit.
    n_red_e = Xc.shape[1] + Xp.shape[1]
    Be = np.linalg.lstsq(X[:, :n_red_e], X[:, sl_e], rcond=None)[0]
    fit_e = X[:, :n_red_e] @ Be
    res_e = X[:, sl_e] - fit_e
    n_red_r = n_red_e + Xe.shape[1]
    Br = np.linalg.lstsq(X[:, :n_red_r], X[:, sl_r], rcond=None)[0]
    fit_r = X[:, :n_red_r] @ Br
    res_r = X[:, sl_r] - fit_r
    fpr = {"whole_design": {"evidence": 0, "resid": 0},
           "family_only": {"evidence": 0, "resid": 0},
           "freedman_lane": {"evidence": 0, "resid": 0}}
    n = X.shape[0]
    for _ in range(n_rep):
        Y = make_Y(Xc, Xp, rng)
        idx = rng.permutation(n)
        tr, te = idx[: n // 2], idx[n // 2 :]
        P = eeg_ridge_P(Y[tr], lam)
        Yte = Y[te]
        obs = unique_from_P(P, Yte, X[tr], X[te], slices)
        whole = {k: [] for k in ("evidence", "resid")}
        famonly = {k: [] for k in ("evidence", "resid")}
        fl = {k: [] for k in ("evidence", "resid")}
        for _b in range(n_null):
            k1, k2 = draw_ks(sec, rng)
            Xw = circshift_by_section(X, sec, k1, k2)
            tw = unique_from_P(P, Yte, Xw[tr], Xw[te], slices)
            for k in whole:
                whole[k].append(tw[k])
            Xf = X.copy()
            Xf[:, sl_e] = circshift_by_section(X[:, sl_e], sec, k1, k2)
            Xf[:, sl_r] = circshift_by_section(X[:, sl_r], sec, k1, k2)
            tf = unique_from_P(P, Yte, Xf[tr], Xf[te], slices)
            famonly["evidence"].append(tf["evidence"])
            famonly["resid"].append(tf["resid"])
            Xfl = X.copy()
            Xfl[:, sl_e] = fit_e + circshift_by_section(res_e, sec, k1, k2)
            tfl_e = unique_from_P(P, Yte, Xfl[tr], Xfl[te], slices)
            Xfl[:, sl_e] = X[:, sl_e]
            Xfl[:, sl_r] = fit_r + circshift_by_section(res_r, sec, k1, k2)
            tfl_r = unique_from_P(P, Yte, Xfl[tr], Xfl[te], slices)
            fl["evidence"].append(tfl_e["evidence"])
            fl["resid"].append(tfl_r["resid"])
        if p_one_sided(obs["evidence"], whole["evidence"]) < 0.05:
            fpr["whole_design"]["evidence"] += 1
        if p_one_sided(obs["resid"], whole["resid"]) < 0.05:
            fpr["whole_design"]["resid"] += 1
        if p_one_sided(obs["evidence"], famonly["evidence"]) < 0.05:
            fpr["family_only"]["evidence"] += 1
        if p_one_sided(obs["resid"], famonly["resid"]) < 0.05:
            fpr["family_only"]["resid"] += 1
        if p_one_sided(obs["evidence"], fl["evidence"]) < 0.05:
            fpr["freedman_lane"]["evidence"] += 1
        if p_one_sided(obs["resid"], fl["resid"]) < 0.05:
            fpr["freedman_lane"]["resid"] += 1
    rate = {kind: {k: v / n_rep for k, v in d.items()} for kind, d in fpr.items()}
    # Whole-design is acceptable if M2/M3 FPR is not badly inflated.
    ok_fl = (rate["freedman_lane"]["evidence"] <= 0.12
             and rate["freedman_lane"]["resid"] <= 0.12)
    out = {
        "n_rep": n_rep, "n_null": n_null, "min_k": MIN_K,
        "sec_lengths": [n1, n2], "signal": "controls+pitch only",
        "fpr_counts": fpr, "fpr_rate": rate,
        "whole_design_calibrated": False,
        "freedman_lane_calibrated": ok_fl,
        "decision": (
            "use_freedman_lane" if ok_fl else
            "freedman_lane_not_calibrated_do_not_launch"
        ),
        "note": (
            "Family-only shift of evidence/resid is recorded as a negative "
            "control for the anti-pattern (rotating the added family only)."
        ),
    }
    dest = L.CTX_DIR / "null_type1.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not ok_fl:
        sys.exit(2)


if __name__ == "__main__":
    main()
