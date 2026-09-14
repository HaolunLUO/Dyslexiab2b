#!/usr/bin/env python3
"""Unit tests for compare_b2b_groups.py (joint B2B v4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_b2b_groups", ROOT / "compare_b2b_groups.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["compare_b2b_groups"] = mod
spec.loader.exec_module(mod)


def test_cohort_counts_reject():
    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.csv"
        rows = [{"participant": f"P{i}", "group": "TD", "include_primary": 1} for i in range(24)]
        rows += [
            {"participant": f"N{i}", "group": "dyslexia_normal_CAP", "include_primary": 1}
            for i in range(18)  # wrong: should be 19
        ]
        rows += [
            {"participant": f"A{i}", "group": "dyslexia_atypical_CAP", "include_primary": 1}
            for i in range(20)
        ]
        pd.DataFrame(rows).to_csv(p, index=False)
        try:
            mod.load_groups(str(p))
            raise AssertionError("expected SystemExit")
        except SystemExit:
            pass


def test_max_cluster_across_families():
    t = {
        "acoustic": np.array([0.0, 0.0, 3.0, 3.0, 0.0]),
        "speech": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "language": np.array([0.0, 4.0, 4.0, 4.0, 0.0]),
    }
    thr = 2.0
    m = mod.max_cluster_across_families(t, thr)
    # language cluster mass = 12, acoustic = 6
    assert m == 12.0


def test_label_perm_multifamily_runs():
    rng = np.random.default_rng(0)
    n_t = 21
    times = np.linspace(0.0, 0.8, n_t)
    mats_A = {f: rng.normal(0.2, 1.0, size=(12, n_t)) for f in mod.FAMILIES}
    mats_B = {f: rng.normal(0.0, 1.0, size=(15, n_t)) for f in mod.FAMILIES}
    res = mod.label_perm_contrast_multifamily(
        mats_A, mats_B, times, n_perm=50, seed=1, win=(0.0, 0.8)
    )
    assert 0.0 < res["p_cluster_joint"] <= 1.0
    assert set(res["per_family"]) == set(mod.FAMILIES)
    assert res["null_method"] == "subject_label_permutation"


def test_stimulus_shift_existence_runs():
    rng = np.random.default_rng(1)
    n_subj, n_null, n_t = 10, 30, 21
    times = np.linspace(0.0, 0.8, n_t)
    mats = {f: rng.normal(0.3, 1.0, size=(n_subj, n_t)) for f in mod.FAMILIES}
    null_cubes = {
        f: rng.normal(0.0, 1.0, size=(n_subj, n_null, n_t)) for f in mod.FAMILIES
    }
    res = mod.stimulus_shift_existence(mats, null_cubes, times, win=(0.0, 0.8))
    assert res["null_method"] == "participant_stimulus_shift"
    assert res["n_null"] == n_null
    assert 0.0 < res["p_cluster_corrected"] <= 1.0


def test_parse_exclude_list():
    assert mod.parse_exclude_list("D034d, D035d,RN105") == ["D034d", "D035d", "RN105"]
    assert mod.parse_exclude_list("") == []
    assert mod.parse_exclude_list("RN105 RN105,RN106") == ["RN105", "RN106"]


if __name__ == "__main__":
    test_cohort_counts_reject()
    test_max_cluster_across_families()
    test_label_perm_multifamily_runs()
    test_stimulus_shift_existence_runs()
    test_parse_exclude_list()
    print("All compare_b2b_groups v4 tests passed.")
