#!/usr/bin/env python3
"""Tests for tone v3 Legendre coefficient space."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_tone_v3",
    ROOT / "scripts" / "extract_tone_v3_wordlocked.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_legendre_flat_vs_slope():
    c = mod.legendre_coeffs(np.full(10, 2.0))
    assert c is not None
    assert abs(c[0] - 2.0) < 0.02
    assert abs(c[1]) < 0.02
    assert abs(c[2]) < 0.02
    x = np.linspace(-1.0, 1.0, 10)
    c = mod.legendre_coeffs(x)
    assert abs(c[0]) < 0.05
    assert abs(c[1] - 1.0) < 0.05
    assert abs(c[2]) < 0.05


def test_legendre_nan_uses_finite_points():
    y = np.linspace(-1.0, 1.0, 10)
    y[4] = np.nan
    c = mod.legendre_coeffs(y)
    assert c is not None
    assert abs(c[1] - 1.0) < 0.15


def test_zscore_equalizes_column_variance():
    rng = np.random.default_rng(0)
    M = np.column_stack([rng.normal(0, 5, 200), rng.normal(0, 0.2, 200), rng.normal(1, 1, 200)])
    Z, mu, sd = mod.zscore_cols(M)
    assert np.allclose(Z.std(0), 1.0, atol=1e-6)
    # After z-score, a slope-only difference is not drowned by register
    a = np.array([5.0, 0.0, 0.0])  # high register, flat
    b = np.array([5.0, -2.0, 0.0])  # high register, falling
    az = (a - mu) / sd
    bz = (b - mu) / sd
    # raw Euclidean dominated by unused; in z-space c1 difference is large
    assert abs(az[1] - bz[1]) > abs(az[0] - bz[0])


def test_gate_requires_self_modal_and_t4_over_t1():
    ok = {
        "modal_is_self": {"1": True, "2": True, "3": True, "4": True},
        "T4_self_beats_T1": True,
    }
    assert mod.gate_pass(ok)
    bad = dict(ok)
    bad["T4_self_beats_T1"] = False
    assert not mod.gate_pass(bad)
    bad2 = {
        "modal_is_self": {"1": True, "2": True, "3": True, "4": False},
        "T4_self_beats_T1": True,
    }
    assert not mod.gate_pass(bad2)
