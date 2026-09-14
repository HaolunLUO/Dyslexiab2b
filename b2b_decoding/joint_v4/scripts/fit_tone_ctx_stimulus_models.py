#!/usr/bin/env python3
"""Cross-fitted stimulus models A (contour) and B (tone evidence) for tone_ctx_v1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tone_ctx_lib as L  # noqa: E402

RNG = np.random.default_rng(42)


def _design_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.column_stack([
        _col_as_numeric(df, c) for c in cols
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _col_as_numeric(df: pd.DataFrame, name: str) -> np.ndarray:
    if name.startswith("tone_") and name[-1] in L.TONES and name.startswith("tone_s"):
        pass
    if name == "surface_tone_oh":
        return np.vstack(df["surface_tone"].map(lambda t: L.one_hot_tone(t if t in L.TONES else None)))
    if name == "prev_tone_oh":
        return np.vstack(df["prev_surface_tone"].map(lambda t: L.one_hot_tone(t if t in L.TONES else None)))
    if name == "next_tone_oh":
        return np.vstack(df["next_surface_tone"].map(lambda t: L.one_hot_tone(t if t in L.TONES else None)))
    if name == "initial_oh":
        return np.vstack(df["initial"].map(lambda v: L.one_hot_label(v, L.INITIALS)))
    if name == "final_oh":
        return np.vstack(df["final"].map(lambda v: L.one_hot_label(v, L.FINALS)))
    return df[name].to_numpy(dtype=np.float64)


def expand_cols(df: pd.DataFrame, spec: list[str]) -> tuple[np.ndarray, list[str]]:
    blocks = []
    names = []
    for c in spec:
        arr = _col_as_numeric(df, c)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            names.append(c)
        else:
            names.extend(f"{c}_{j}" for j in range(arr.shape[1]))
        blocks.append(arr)
    X = np.nan_to_num(np.hstack(blocks), nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def grouped_oof_predict(estimator_factory, X, y, groups, *, n_splits=5, classify=False):
    """Out-of-fold predictions. Groups never leak into their test fold."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    if classify:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        split_y = y
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_y = None
    if classify:
        n_class = int(np.max(y)) + 1 if y.ndim == 1 else y.shape[1]
        P = np.zeros((len(y), n_class), dtype=np.float64)
    else:
        P = np.zeros_like(y, dtype=np.float64)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
            y = y.reshape(-1, 1)
    fold_ok = []
    used = np.zeros(len(y), dtype=bool)
    splits = list(splitter.split(X, split_y if classify else y[:, 0], groups))
    for fold, (tr, te) in enumerate(splits):
        tr_g, te_g = set(groups[tr]), set(groups[te])
        if tr_g & te_g:
            raise RuntimeError(f"group leakage in fold {fold}")
        if classify:
            if len(np.unique(y[tr])) < 4 or len(np.unique(y[te])) < 4:
                fold_ok.append(False)
            else:
                fold_ok.append(True)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        est = estimator_factory()
        est.fit(Xtr, y[tr] if classify else y[tr])
        if classify:
            P[te] = est.predict_proba(Xte)
        else:
            pred = est.predict(Xte)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            P[te] = pred
        used[te] = True
    if not used.all():
        # leftover (rare singleton groups): fill with training-set mean
        miss = ~used
        if classify:
            prior = np.bincount(y[~miss], minlength=P.shape[1]).astype(np.float64)
            prior /= max(prior.sum(), 1.0)
            P[miss] = prior
        else:
            P[miss] = y[~miss].mean(axis=0, keepdims=True)
    return P, fold_ok, splits


def class_prior(y: np.ndarray, k: int = 4) -> np.ndarray:
    p = np.bincount(y, minlength=k).astype(np.float64)
    p /= p.sum()
    return p


def model_a(df: pd.DataFrame, n_coef: int) -> dict:
    coef_cols = ["c0", "c1", "c2"][:n_coef] if n_coef == 3 else ["c0", "c1", "c2", "c3"]
    ok = df[coef_cols].notna().all(axis=1) & df["surface_tone"].isin(L.TONES)
    sub = df.loc[ok].reset_index(drop=True)
    Y = sub[coef_cols].to_numpy(dtype=np.float64)
    X_full, _ = expand_cols(sub, [
        "surface_tone_oh", "prev_tone_oh", "next_tone_oh",
        "duration", "local_register", "phrase_position", "initial_oh", "final_oh",
    ])
    X_proto, _ = expand_cols(sub, ["surface_tone_oh"])
    groups = sub["base_syllable"].fillna("unk").to_numpy()

    def ridge():
        return Ridge(alpha=1.0)

    Yhat, _, _ = grouped_oof_predict(ridge, X_full, Y, groups, classify=False)
    Yhat_p, _, _ = grouped_oof_predict(ridge, X_proto, Y, groups, classify=False)
    mse = float(np.mean((Y - Yhat) ** 2))
    mse_proto = float(np.mean((Y - Yhat_p) ** 2))
    Q = Y - Yhat
    # Ledoit-Wolf-ish: shrink sample cov toward diagonal
    C = np.cov(Q, rowvar=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    tr = float(np.trace(C)) / max(C.shape[0], 1)
    C = 0.1 * np.eye(C.shape[0]) * max(tr, 1e-6) + 0.9 * C
    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        Cinv = np.linalg.pinv(C)
    D = np.einsum("ij,jk,ik->i", Q, Cinv, Q)
    sign, logdet = np.linalg.slogdet(C)
    k = Q.shape[1]
    nll = 0.5 * (D + logdet + k * np.log(2 * np.pi))
    out = sub[["section", "syllable_id"]].copy()
    for j, name in enumerate(coef_cols):
        out[f"q_{name}"] = Q[:, j]
    out["D"] = D
    out["nll"] = nll
    return {
        "n": int(len(sub)),
        "n_coef": n_coef,
        "mse": mse,
        "mse_prototype": mse_proto,
        "beats_prototype": bool(mse < mse_proto),
        "table": out,
    }


def model_b(df: pd.DataFrame, n_coef: int) -> dict:
    coef_cols = ["c0", "c1", "c2"] if n_coef == 3 else ["c0", "c1", "c2", "c3"]
    ok = (
        df["surface_tone"].isin(L.TONES)
        & df[coef_cols].notna().all(axis=1)
        & (df["surface_tone"] != "5")
    )
    sub = df.loc[ok].reset_index(drop=True)
    if sub["surface_tone"].eq("5").any():
        raise RuntimeError("T5 leaked into model B")
    y = sub["surface_tone"].map(int).to_numpy() - 1
    if set(np.unique(y)) != {0, 1, 2, 3}:
        raise SystemExit(f"model B missing a tone class: {np.unique(y)}")
    X, _ = expand_cols(sub, coef_cols + [
        "duration", "env_mean", "periodicity",
        "prev_f0_end", "prev_f0_slope", "preceding_pause", "phrase_position",
    ])
    groups = sub["base_syllable"].fillna("unk").to_numpy()

    def logreg():
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500)

    P, fold_ok, splits = grouped_oof_predict(logreg, X, y, groups, classify=True)
    if not np.allclose(P.sum(1), 1.0, atol=1e-5):
        raise RuntimeError("OOF probabilities do not sum to 1")
    H = L.orthonormal_helmert(4)
    if not np.allclose(H.T @ H, np.eye(3), atol=1e-8):
        raise RuntimeError("Helmert not orthonormal")
    Yoh = np.eye(4)[y]
    evidence = P @ H
    resid = (Yoh - P) @ H
    clip = np.clip(P[np.arange(len(P)), y], 1e-6, 1.0)
    surprisal = -np.log(clip)
    prior = class_prior(y, 4)
    ll_model = float(log_loss(y, P, labels=[0, 1, 2, 3]))
    ll_prior = float(log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2, 3]))
    brier = float(np.mean([
        brier_score_loss((y == k).astype(int), P[:, k]) for k in range(4)
    ]))
    # permutation of labels vs fixed P
    n_perm = 200
    better = 0
    for _ in range(n_perm):
        y_p = RNG.permutation(y)
        ll_p = float(log_loss(y_p, P, labels=[0, 1, 2, 3]))
        if (ll_prior - ll_model) > (ll_prior - ll_p):
            better += 1
    p_perm = (1 + (n_perm - better)) / (n_perm + 1)
    C = np.zeros((4, 4), dtype=int)
    pred = P.argmax(1)
    for a, b in zip(y, pred):
        C[a, b] += 1
    # influence: drop largest base_syllable
    vc = sub["base_syllable"].value_counts()
    top = str(vc.index[0])
    mask = sub["base_syllable"] != top
    ll_wo = float(log_loss(y[mask], P[mask], labels=[0, 1, 2, 3]))
    table = sub[["section", "syllable_id", "surface_tone", "base_syllable"]].copy()
    table["p1"] = P[:, 0]
    table["p2"] = P[:, 1]
    table["p3"] = P[:, 2]
    table["p4"] = P[:, 3]
    table["e1"] = evidence[:, 0]
    table["e2"] = evidence[:, 1]
    table["e3"] = evidence[:, 2]
    table["r1"] = resid[:, 0]
    table["r2"] = resid[:, 1]
    table["r3"] = resid[:, 2]
    table["tone_surprisal"] = surprisal
    # story-blocked sensitivity
    story_groups = sub["section"].to_numpy()
    P_story, _, _ = grouped_oof_predict(logreg, X, y, story_groups, n_splits=2, classify=True)
    ll_story = float(log_loss(y, P_story, labels=[0, 1, 2, 3]))
    return {
        "n": int(len(sub)),
        "n_coef": n_coef,
        "log_loss": ll_model,
        "log_loss_prior": ll_prior,
        "beats_prior": bool(ll_model < ll_prior),
        "p_perm": float(p_perm),
        "brier": brier,
        "confusion": C.tolist(),
        "overall_acc": float(np.trace(C) / C.sum()),
        "fold_has_all_classes": all(fold_ok) if fold_ok else False,
        "n_splits": len(splits),
        "top_base_syllable": top,
        "log_loss_without_top_base": ll_wo,
        "not_driven_by_one_item": bool(ll_wo < ll_prior),
        "log_loss_story_blocked": ll_story,
        "prior": prior.tolist(),
        "table": table,
        "P": P,
        "y": y,
        "groups": groups,
        "splits": splits,
    }


def reliability_plot(y, P, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6))
    for ax, k in zip(axes.ravel(), range(4)):
        pk = P[:, k]
        yk = (y == k).astype(float)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(pk, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.any():
                xs.append(float(pk[m].mean()))
                ys.append(float(yk[m].mean()))
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
        if xs:
            ax.plot(xs, ys, "o-", color="#1f4e79")
        ax.set_title(f"T{k + 1}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion_plot(C: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    row = C / np.maximum(C.sum(1, keepdims=True), 1)
    im = ax.imshow(row, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(int(C[i, j])), ha="center", va="center",
                    color="white" if row[i, j] > 0.45 else "black")
    ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
    ax.set_yticks(range(4), ["T1", "T2", "T3", "T4"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(path: Path, a3: dict, a4: dict, b: dict, winner_n: int) -> None:
    C = np.asarray(b["confusion"])
    lines = [
        "# tone_ctx_v1 stimulus report",
        "",
        f"Coefficient order chosen by OOF classifier log loss: **{winner_n}** "
        f"(3-coef ll={a3.get('_b_ll', float('nan')):.4f}, "
        f"4-coef ll={a4.get('_b_ll', float('nan')):.4f}).",
        "",
        "## Model A — context-conditioned realization",
        "",
        f"- n = {a3['n'] if winner_n == 3 else a4['n']}",
        f"- MSE full = {(a3 if winner_n == 3 else a4)['mse']:.5f}",
        f"- MSE tone-only prototype = {(a3 if winner_n == 3 else a4)['mse_prototype']:.5f}",
        f"- Beats prototype: {(a3 if winner_n == 3 else a4)['beats_prototype']}",
        "",
        "## Model B — grouped OOF tone evidence",
        "",
        f"- n = {b['n']} (T1–T4 only; T5 excluded)",
        f"- log loss = {b['log_loss']:.4f}  vs class-prior {b['log_loss_prior']:.4f}",
        f"- permutation p (improvement vs prior) = {b['p_perm']:.4f}",
        f"- Brier (mean one-vs-rest) = {b['brier']:.4f}",
        f"- overall acc = {b['overall_acc']:.3f} (not a launch gate)",
        f"- every fold has all classes: {b['fold_has_all_classes']}",
        f"- story-blocked log loss = {b['log_loss_story_blocked']:.4f}",
        f"- drop `{b['top_base_syllable']}` log loss = {b['log_loss_without_top_base']:.4f}",
        "",
        "Confusion (rows = true):",
        "",
        "```",
        f"{C}",
        "```",
        "",
        "## Gates",
        "",
    ]
    A = a3 if winner_n == 3 else a4
    gates = {
        "log_loss_beats_prior": b["beats_prior"],
        "perm_p_lt_0.05": b["p_perm"] < 0.05,
        "all_folds_have_all_classes": b["fold_has_all_classes"],
        "model_A_beats_prototype": A["beats_prototype"],
        "not_driven_by_one_base_syllable": b["not_driven_by_one_item"],
    }
    for k, v in gates.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}  {k}")
    lines.append("")
    lines.append(f"**Launch EEG features:** {all(gates.values())}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return gates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, default=L.CTX_DIR / "tone_ctx_tokens.parquet")
    ap.add_argument("--out-dir", type=Path, default=L.CTX_DIR)
    args = ap.parse_args()
    df = pd.read_parquet(args.tokens)
    if (df["surface_tone"] == "5").any():
        print(f"T5 rows in token table: {int((df.surface_tone == '5').sum())} (held out of B)")

    a3 = model_a(df, 3)
    a4 = model_a(df, 4)
    b3 = model_b(df, 3)
    b4 = model_b(df, 4)
    a3["_b_ll"] = b3["log_loss"]
    a4["_b_ll"] = b4["log_loss"]
    winner_n = 3 if b3["log_loss"] <= b4["log_loss"] else 4
    A = a3 if winner_n == 3 else a4
    B = b3 if winner_n == 3 else b4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    A["table"].to_parquet(args.out_dir / "model_a_residuals.parquet", index=False)
    B["table"].to_parquet(args.out_dir / "model_b_oof.parquet", index=False)
    reliability_plot(B["y"], B["P"], args.out_dir / "reliability.png")
    confusion_plot(np.asarray(B["confusion"]), args.out_dir / "confusion_oof.png",
                   f"OOF tone classifier  acc={B['overall_acc']:.2f}")
    gates = write_report(args.out_dir / "stimulus_report.md", a3, a4, B, winner_n)
    merged = df.copy()
    mA = A["table"].set_index("syllable_id")
    mB = B["table"].set_index("syllable_id")
    for col in mA.columns:
        if col != "section":
            merged[col] = merged["syllable_id"].map(mA[col])
    for col in ("p1", "p2", "p3", "p4", "e1", "e2", "e3", "r1", "r2", "r3", "tone_surprisal"):
        merged[col] = merged["syllable_id"].map(mB[col])
    merged.to_parquet(args.out_dir / "tone_ctx_tokens_scored.parquet", index=False)
    meta = {
        "winner_n_coef": winner_n,
        "gates": gates,
        "launch": bool(all(gates.values())),
        "model_a": {k: v for k, v in A.items() if k != "table"},
        "model_b": {k: v for k, v in B.items() if k not in ("table", "P", "y", "groups", "splits")},
    }
    (args.out_dir / "stimulus_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"winner_n_coef": winner_n, "gates": gates, "launch": meta["launch"]}, indent=2))
    if not meta["launch"]:
        print("STIMULUS GATES FAILED — do not export EEG features until this is resolved.")
        sys.exit(2)
    print("STIMULUS GATES PASSED")


if __name__ == "__main__":
    main()
