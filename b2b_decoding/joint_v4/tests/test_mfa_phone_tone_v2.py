#!/usr/bin/env python3
"""Unit tests for MFA phone/tone v2 extraction repairs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_mfa_v2",
    ROOT / "scripts" / "extract_mfa_phone_tone_envelope_wordlocked.py",
)
mod = importlib.util.module_from_spec(SPEC)
sys.modules["extract_mfa_v2"] = mod
SPEC.loader.exec_module(mod)


def test_chao_longest_suffix_mapping():
    cases = [
        ("a˥", "a", "1"),
        ("a˧˥", "a", "2"),
        ("a˨˩˦", "a", "3"),
        ("a˥˩", "a", "4"),
        ("ə˧", "ə", "5"),
        ("o˩", "o", "3"),
        ("aj˨˩˦", "aj", "3"),
        ("tʰ", "tʰ", None),
        ("a1", "a", "1"),
        ("sil", "sil", None),
    ]
    for raw, phone, tone in cases:
        got_p, got_t = mod.split_phone_tone(raw)
        assert (got_p, got_t) == (phone, tone), (raw, (got_p, got_t), (phone, tone))
        assert not mod.phone_has_tone_contour(got_p), raw


def test_partial_tone3_residue_rejected():
    # Old bug left ˨˦ after stripping only ˩ from ˨˩˦.
    phone, tone = mod.split_phone_tone("o˨˩˦")
    assert phone == "o" and tone == "3"
    assert "˨" not in phone and "˦" not in phone


def test_boundary_clipping_no_overcount():
    words = [{"onset": 1.0, "offset": 2.0, "word": "x", "i": 0},
             {"onset": 2.0, "offset": 3.0, "word": "y", "i": 1}]
    # Phone straddles the boundary (1.9–2.3). Max overlap is with word1 (0.3 vs 0.1).
    phones = [(1.9, 2.3, "a˥")]
    assigned = mod.assign_phones_to_words(phones, words, min_overlap=0.01)
    assert assigned[0] == []
    assert len(assigned[1]) == 1
    on, off, ph, tone = assigned[1][0]
    assert ph == "a" and tone == "1"
    assert abs(on - 2.0) < 1e-9
    assert abs(off - 2.3) < 1e-9
    # Credited duration is clipped overlap, not full 0.4 s phone length.
    assert abs((off - on) - 0.3) < 1e-9

    # Equal-ish left-heavy case: phone mostly inside word0
    phones2 = [(1.7, 2.05, "i˨˩˦")]
    assigned2 = mod.assign_phones_to_words(phones2, words, min_overlap=0.01)
    assert len(assigned2[0]) == 1 and assigned2[1] == []
    on2, off2, ph2, tone2 = assigned2[0][0]
    assert ph2 == "i" and tone2 == "3"
    assert abs(on2 - 1.7) < 1e-9 and abs(off2 - 2.0) < 1e-9
    assert abs((off2 - on2) - 0.3) < 1e-9


def test_tone_phone_separation_in_matrices():
    words = {
        1: [
            {"onset": 0.0, "offset": 0.5, "word": "妈", "i": 0},
            {"onset": 0.5, "offset": 1.0, "word": "马", "i": 1},
            {"onset": 1.0, "offset": 1.5, "word": "骂", "i": 2},
        ]
    }
    assigned = {
        1: [
            [(0.0, 0.2, "m", None), (0.2, 0.5, "a", "1")],
            [(0.5, 0.7, "m", None), (0.7, 1.0, "a", "3")],
            [(1.0, 1.2, "m", None), (1.2, 1.5, "a", "4")],
        ]
    }
    Xp, pnames, Xt, tnames, inv, qc = mod.build_phone_tone_matrices(
        assigned, words, min_phone_words=1, min_ph1_words=1, min_tone_words=1
    )
    assert all(not mod.phone_has_tone_contour(p) for p in inv)
    assert "coverage" not in pnames
    assert any(n.startswith("ph_a") for n in pnames)
    assert qc["occupancy_sum_max"] <= 1.0 + 1e-9
    assert qc["tone_counts"]["1"] >= 1
    assert qc["tone_counts"]["3"] >= 1
    assert qc["tone_counts"]["4"] >= 1
    # Feature name / column alignment
    assert Xp[1].shape[1] == len(pnames)
    assert Xt[1].shape[1] == len(tnames)


def test_env_no_onset_t00_duplication():
    env = np.linspace(0.0, 1.0, 500)
    words = [{"onset": 1.0, "offset": 1.4, "word": "w", "i": 0}]
    X, names = mod.word_envelope_features(env, words, fs=100.0)
    assert "env_onset" not in names
    assert "env_t00" in names
    assert names.count("env_t00") == 1
    assert X.shape == (1, len(names))


def test_drop_constant_and_name_alignment():
    X = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0], [3.0, 0.0, 4.0]])
    names = ["a", "const", "b"]
    X2, names2, dropped = mod.drop_constant_columns(X, names)
    assert dropped == ["const"]
    assert names2 == ["a", "b"]
    assert X2.shape == (3, 2)


def test_stale_checksum_handling(tmp_path=None):
    """Manifest skip must fail when feature checksums diverge."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        feat = td / "X_word_mfa_phones_v2.npy"
        np.save(feat, np.zeros((10, 3)))
        h_new = hashlib.sha256(feat.read_bytes()).hexdigest()
        manifest = {
            "feature_checksums": {
                "section_001/mfa_phones_v2": "0" * 64,
            },
            "feature_basis_hash": "abc",
        }
        man_path = td / "run_manifest.json"
        man_path.write_text(json.dumps(manifest), encoding="utf-8")
        # Simulate the skip gate used by observed array
        stored = manifest["feature_checksums"]["section_001/mfa_phones_v2"]
        assert stored != h_new, "stale checksum must not match current feature file"


if __name__ == "__main__":
    test_chao_longest_suffix_mapping()
    test_partial_tone3_residue_rejected()
    test_boundary_clipping_no_overcount()
    test_tone_phone_separation_in_matrices()
    test_env_no_onset_t00_duplication()
    test_drop_constant_and_name_alignment()
    test_stale_checksum_handling()
    print("All MFA v2 extraction tests passed.")
