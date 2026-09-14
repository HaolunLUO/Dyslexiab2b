#!/usr/bin/env python3
"""Unit tests for tone_ctx_v1 helpers (no EEG, no cluster)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tone_ctx_lib as L  # noqa: E402
import fit_tone_ctx_stimulus_models as M  # noqa: E402


def test_helmert_orthonormal_and_zero_sum():
    H = L.orthonormal_helmert(4)
    assert H.shape == (4, 3)
    assert np.allclose(H.T @ H, np.eye(3), atol=1e-8)
    assert np.allclose(H.sum(0), 0.0, atol=1e-8)


def test_sandhi_types():
    assert L.sandhi_type("不", "4", "2", "4") == "bu_t4_to_t2"
    assert L.sandhi_type("一", "1", "2", "4") == "yi_to_2"
    assert L.sandhi_type("一", "1", "4", "3") == "yi_to_4"
    assert L.sandhi_type("很", "3", "2", "3") == "t3t3_to_t2"
    assert L.sandhi_type("好", "3", "3", "1") == "none"


def test_t5_never_in_onehot():
    assert L.one_hot_tone("5").sum() == 0
    assert L.one_hot_tone(None).sum() == 0
    assert L.one_hot_tone("2")[1] == 1


def test_grouped_oof_no_group_leak():
    rng = np.random.default_rng(0)
    n = 80
    groups = np.array([f"g{i // 4}" for i in range(n)])
    y = np.array([i % 4 for i in range(n)])
    X = rng.normal(size=(n, 6))
    X[np.arange(n), y] += 2.0

    def logreg():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=200)

    P, _, splits = M.grouped_oof_predict(logreg, X, y, groups, n_splits=4, classify=True)
    assert np.allclose(P.sum(1), 1.0, atol=1e-5)
    for tr, te in splits:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_control_dim_matches_lib():
    assert L.N_CTX_CONTROLS == len(L.CONTROL_NAMES) == 79
    assert L.N_CTX_PITCH == 5
    assert L.N_CTX_EVIDENCE == 3


def test_prune_merges_rare_dummies():
    import extract_tone_ctx_v1_nucleuslocked as E

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 5, size=n)
    y[:4] = 5
    names = ["x"] + [f"init_{k}" for k in range(5)] + ["init_other"]
    X = np.zeros((n, 7), dtype=np.float64)
    X[:, 0] = rng.normal(size=n)
    for k in range(6):
        X[:, 1 + k] = (y == k).astype(np.float64)
    Xp, npn = E.prune_fullrank(X, names, ("init_",), min_count=20)
    assert "init_other" in npn
    other = Xp[:, npn.index("init_other")]
    assert other.sum() >= 4
    assert E.split_half_max_cond(Xp, n_parts=10, seed=1) < E.KAPPA_INVALID


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
