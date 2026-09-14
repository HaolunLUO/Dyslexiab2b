#!/usr/bin/env python3
"""Unit tests for tone / phoneme-onset extraction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_tone",
    ROOT / "scripts" / "extract_tone_phoneme_wordlocked.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _ph(seq):
    t = 0.0
    out = []
    for phone, tone in seq:
        out.append((t, t + 0.05, phone, tone))
        t += 0.05
    return out


def test_group_typical_bisyllables():
    # 看到 kan4 dao4
    g = mod.group_syllables(_ph([
        ("kʰ", None), ("a", "4"), ("n", None), ("t", None), ("aw", "4"),
    ]))
    assert [s["lexical_tone"] for s in g] == ["4", "4"]
    assert [p[0] for p in g[0]["phones"]] == ["kʰ", "a", "n"]
    assert [p[0] for p in g[1]["phones"]] == ["t", "aw"]

    # 森林 sen1 lin2
    g = mod.group_syllables(_ph([
        ("s", None), ("ə", "1"), ("n", None), ("ʎ", None), ("i", "2"), ("n", None),
    ]))
    assert [s["lexical_tone"] for s in g] == ["1", "2"]
    assert [p[0] for p in g[0]["phones"]] == ["s", "ə", "n"]
    assert [p[0] for p in g[1]["phones"]] == ["ʎ", "i", "n"]

    # 插画 cha1 hua4 — no nasal coda
    g = mod.group_syllables(_ph([
        ("ʈʂʰ", None), ("a", "1"), ("xʷ", None), ("a", "4"),
    ]))
    assert [s["lexical_tone"] for s in g] == ["1", "4"]
    assert [p[0] for p in g[0]["phones"]] == ["ʈʂʰ", "a"]


def test_group_onsetless_and_empty():
    g = mod.group_syllables(_ph([("a", "1"), ("n", None)]))
    assert len(g) == 1 and g[0]["lexical_tone"] == "1"
    assert mod.group_syllables([]) == []
    g = mod.group_syllables(_ph([("t", None), ("s", None)]))
    assert len(g) == 1 and g[0]["lexical_tone"] is None


def test_sandhi_bu_yi_and_t3():
    # 你好 ni3 hao3 → 2-3
    assert mod.apply_tone_sandhi(["你", "好"], ["3", "3"]) == ["2", "3"]
    # 很好 (across words, same stream)
    assert mod.apply_tone_sandhi(["很", "好"], ["3", "3"]) == ["2", "3"]
    # 不是 bu2 shi4
    assert mod.apply_tone_sandhi(["不", "是"], ["4", "4"]) == ["2", "4"]
    # 不去 would be 不+T4
    assert mod.apply_tone_sandhi(["不", "去"], ["4", "4"]) == ["2", "4"]
    # 一起 yi4 qi3 (一 before non-T4 → 4)
    assert mod.apply_tone_sandhi(["一", "起"], ["1", "3"]) == ["4", "3"]
    # 一个 yi2 ge4
    assert mod.apply_tone_sandhi(["一", "个"], ["1", "4"]) == ["2", "4"]
    # 3-3-3 → 2-2-3
    assert mod.apply_tone_sandhi(["a", "b", "c"], ["3", "3", "3"]) == ["2", "2", "3"]
    # Neutral / missing stay put
    assert mod.apply_tone_sandhi(["的"], ["5"]) == ["5"]
    assert mod.apply_tone_sandhi(["x"], [None]) == [None]


def test_tone_deviation_rmse():
    tmpl = {"1": np.zeros(10), "2": np.ones(10)}
    assert mod.tone_deviation(np.zeros(10), "1", tmpl) == 0.0
    assert abs(mod.tone_deviation(np.ones(10), "1", tmpl) - 1.0) < 1e-9
    assert mod.tone_deviation(None, "1", tmpl) == 0.0
    assert mod.tone_deviation(np.ones(10), "5", tmpl) == 0.0


def test_rasterize_and_voiced_mask():
    x = mod.rasterize([0.00, 0.10, 0.105], n=20, fs=100.0)
    assert x[0] == 1.0 and x[10] == 1.0
    assert x.sum() == 2.0  # 0.105 rounds to the same bin as 0.10
    f0 = np.concatenate([np.full(50, 5.5), np.linspace(5.0, 6.0, 20), np.full(30, 5.5)])
    v = mod.voiced_mask(f0)
    assert v[:50].sum() == 0
    assert v[50:70].sum() == 20


def test_syllable_f0_requires_voiced():
    log_f0 = np.full(100, 5.5)
    voiced = np.zeros(100, dtype=bool)
    assert mod.syllable_f0(log_f0, voiced, 0.1, 0.4) is None
    log_f0[10:40] = np.linspace(5.0, 6.0, 30)
    voiced[10:40] = True
    c = mod.syllable_f0(log_f0, voiced, 0.10, 0.40)
    assert c is not None and c.shape == (10,)
    assert c[0] < c[-1]
