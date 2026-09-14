#!/usr/bin/env python3
"""Unit tests for tone v2 rime slicing and deviation coding."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_tone_v2",
    ROOT / "scripts" / "extract_tone_v2_wordlocked.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_rime_starts_at_vowel_not_glide():
    phones = [
        (0.00, 0.04, "j", None),
        (0.04, 0.18, "ow", "3"),
        (0.18, 0.22, "n", None),
    ]
    rw = mod.rime_window(phones, 0.00, 0.22)
    assert rw is not None
    assert abs(rw[0] - 0.04) < 1e-9
    assert abs(rw[1] - 0.22) < 1e-9
    assert rw[2] == "ow"


def test_rime_rejects_pre_syllable_nucleus():
    phones = [(0.00, 0.10, "a", "1")]
    try:
        mod.rime_window(phones, 0.05, 0.20)
    except AssertionError:
        return
    raise AssertionError("expected nucleus-before-syllable to raise")


def test_rime_requires_vowel():
    phones = [(0.0, 0.05, "t", None), (0.05, 0.10, "s", None)]
    assert mod.rime_window(phones, 0.0, 0.10) is None
    # glide-only is not a nucleus
    phones = [(0.0, 0.08, "j", None)]
    assert mod.rime_window(phones, 0.0, 0.08) is None


def test_time_normalize_skips_long_gaps():
    fs = 100.0
    f0 = np.full(40, np.nan)
    # voiced 0–80 ms, then 90 ms gap, then 180–250 ms
    f0[0:8] = 200.0
    f0[18:26] = 180.0
    c, meta = mod.time_normalize_rime(f0, 0.0, 0.40, fs=fs)
    # 16/40 = 0.4 voiced < 0.5 → invalid
    assert c is None
    assert meta["reason"] == "low_voiced_frac"

    f0 = np.full(40, np.nan)
    f0[0:12] = np.linspace(220, 200, 12)
    f0[28:40] = np.linspace(160, 120, 12)  # 160 ms gap
    c, meta = mod.time_normalize_rime(f0, 0.0, 0.40, fs=fs)
    assert meta["valid"]
    assert c is not None
    # middle of the 10-pt grid falls in the gap → NaN
    assert not np.isfinite(c[4]) or not np.isfinite(c[5])
    assert np.isfinite(c[0]) and np.isfinite(c[-1])


def test_validity_final_third():
    fs = 100.0
    f0 = np.full(30, 180.0)
    f0[20:] = np.nan  # last third unvoiced
    c, meta = mod.time_normalize_rime(f0, 0.0, 0.30, fs=fs)
    assert c is None
    assert meta["reason"] == "sparse_final_third"


def test_rising_rime_preserves_shape():
    fs = 100.0
    f0 = np.linspace(120.0, 220.0, 40)
    c, meta = mod.time_normalize_rime(f0, 0.0, 0.40, fs=fs)
    assert meta["valid"]
    assert c[0] < c[-1]
    assert abs(c[0] - np.log(120)) < 0.05
    assert abs(c[-1] - np.log(220)) < 0.05


def test_rmse_and_within_tone_imputation():
    tmpl = np.zeros(10)
    c = np.ones(10)
    assert abs(mod.rmse_voiced(c, tmpl) - 1.0) < 1e-9
    c2 = np.array([0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0])
    assert mod.rmse_voiced(c2, tmpl) is None  # 3 comparable points < 4
    # invalid → 0 after z-score is a coding choice tested in assembly:
    # a z-scored valid set has mean 0, so 0 is mean imputation
    raw = np.array([1.0, 2.0, 3.0])
    z = (raw - raw.mean()) / raw.std()
    assert abs(z.mean()) < 1e-12
    imputed = 0.0
    assert abs(imputed - z.mean()) < 1e-12


def test_nearest_template_prefers_own_shape():
    templates = {
        "1": np.full(10, 0.4),
        "2": np.linspace(-0.5, 0.0, 10),
        "3": np.linspace(0.0, -1.0, 10),
        "4": np.linspace(0.5, 0.1, 10),
    }
    assert mod.nearest_template(np.full(10, 0.4), templates) == "1"
    assert mod.nearest_template(np.linspace(0.5, 0.1, 10), templates) == "4"
    # T4 token closer to T4 than T1
    t4 = np.linspace(0.48, 0.12, 10)
    assert mod.nearest_template(t4, templates) == "4"
    pw = mod.template_pairwise_rmse(templates)
    assert min(pw.values()) > 0.15
