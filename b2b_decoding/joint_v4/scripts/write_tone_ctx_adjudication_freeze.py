#!/usr/bin/env python3
"""Immutable adjudication freeze for tone_ctx_v1. Do not edit after write."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/orcd/pool/005/haolun52")
SHARED = POOL / "extracted_sections_nucleuslocked_shared" / "_shared_wordlocked_features"
DEST = ROOT / "TONECTX_V1_ADJUDICATION_FREEZE.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def main() -> None:
    feats = {}
    for sid in (1, 2):
        sec = SHARED / f"section_{sid:03d}"
        feats[f"section_{sid:03d}"] = {
            p.name: sha256(p)
            for p in sorted(sec.glob("X_word_ctx_*.npy"))
        }
        feats[f"section_{sid:03d}"]["word_timing_relative.csv"] = sha256(
            sec / "word_timing_relative.csv"
        )
    code = {}
    for rel in (
        "config.jl", "estimator.jl", "existence_nulls.jl", "split_half.jl",
        "pipeline.jl", "feature_basis.jl", "b2b_joint_v4_pipeline.jl",
        "scripts/tone_ctx_v1_env.sh", "scripts/submit_tone_ctx_v1.sh",
        "scripts/extract_tone_ctx_v1_nucleuslocked.py",
        "scripts/tone_ctx_lib.py",
    ):
        p = ROOT / rel
        if p.is_file():
            code[rel] = sha256(p)
    stimulus = {}
    for name in (
        "tone_ctx_tokens.parquet", "tone_ctx_tokens_scored.parquet",
        "stimulus_report.md", "recoverability.json", "overlap_ruler.json",
        "null_type1.json", "sec_lengths.json",
    ):
        p = ROOT / "tone_ctx_v1" / name
        if p.is_file():
            stimulus[name] = sha256(p)
    overlap = {}
    op = ROOT / "tone_ctx_v1" / "overlap_ruler.json"
    if op.is_file():
        overlap = json.loads(op.read_text())
    type1 = {}
    tp = ROOT / "tone_ctx_v1" / "null_type1.json"
    if tp.is_file():
        type1 = json.loads(tp.read_text())
    min_k = int(overlap.get("min_shift_syllables_frozen", 20))
    spec = {
        "tag": "tonectx-v1-adjudication",
        "date": str(date.today()),
        "stage": "adjudication",
        "claim": (
            "Frozen after n=63 observed identification. Existence and "
            "split-half only. No TD/DD or brain-behavior until Holm "
            "existence and SB>=0.30 on that increment."
        ),
        "git_commit": git_head(),
        "mean_trace": (
            "For family F with columns C_F, mean_trace(t) = sum_{k in C_F} H_kk(t). "
            "Subject score T = mean of mean_trace over confirm window. "
            "Increment scores are the unique family block in that arm: "
            "pitch in M1, evidence in M2, resid in M3."
        ),
        "channel_aggregation": "none (all 65 ICA-cleaned channels; B2B_SPATIAL_DENOISE=none)",
        "confirm_win_s": [0.0, 0.8],
        "negctrl_win_s": [-0.2, 0.0],
        "epoch": {"anchor": "offset", "tmin_s": -0.2, "tmax_s": 0.8},
        "do_not_redefine_window_from_peaks": True,
        "folds": {
            "n_outer_partitions": 20,
            "n_inner_folds": 4,
            "n_blocks_per_section": 8,
            "guard_words": 5,
            "seed": 42,
        },
        "estimator": {
            "passthrough": True,
            "pca_k": 1,
            "h_ridge_kappa": 0,
            "ols_H": True,
            "lambda": "normalized ridge G; alpha tuned on observed partitions and reused in nulls",
            "alpha_grid": "10^{−8:0.5:8}",
        },
        "exclusions": {
            "node1617": True,
            "T5_held_out_of_M2_M3": True,
            "min_valid_partitions": 16,
            "kappa_invalid": 85.0,
        },
        "null": {
            "algorithm": "freedman_lane_residual_shift_of_added_family",
            "preserves": [
                "reduced nested design X_red unpermuted (nuisance structure fixed)",
                "added-family columns jointly via OLS residuals (circshift resid, add back X_red B)",
                "spatial covariance (EEG unpermuted; all channels together)",
                "story boundaries (independent k per section)",
            ],
            "min_shift_syllables": min_k,
            "min_shift_also_exceeds_epoch_span": True,
            "epoch_span_s": 1.0,
            "n_draws": 5000,
            "seed_offset": 100000,
            "reuse_observed_alpha": True,
            "test": "one_sided_positive",
            "p": "(1 + #{T* >= T}) / (B + 1)",
            "T": "mean across subjects of confirm-window unique mean_trace",
            "T_star_b": "mean across subjects of shift-b unique mean_trace",
            "holm_family": ["M2_evidence", "M3_resid"],
            "m1_is_positive_control_not_in_holm": True,
            "do_not_require_M2_before_M3": True,
            "type1_file": "tone_ctx_v1/null_type1.json",
            "type1_decision": type1.get("decision", "pending"),
            "rejected_anti_pattern": "circular-shift only the added family while leaving M0/M1 fixed",
        },
        "reliability": {
            "primary_split": "interleaved_60s_odd_even_blocks_per_story",
            "not": "first_vs_second_half",
            "score": "confirm-window unique mean_trace recomputed independently in each half",
            "across_subject_r": "Pearson(odd, even) over n=63",
            "spearman_brown": "2r/(1+r)",
            "sb_gate": 0.30,
            "sensitivity": "repeated 60s phase offsets 0/15/30/45 s; report median and 95% interval if launched",
            "also_report": [
                "SB on raw observed increment",
                "SB on increment minus subject/half-specific null floor when half-nulls exist",
            ],
            "pitch_evidence_r_is_not_reliability": True,
        },
        "decision_rules": {
            "M1_fails_proper_null": "pause; audit timing, leakage, null calibration",
            "M2_or_M3_fail_holm": "stop inferential work for that increment",
            "existence_pass_SB_lt_0.30": "group-level existence only; no TD/DD or behavior",
            "existence_pass_SB_ge_0.30": "prespecified group model allowed later",
            "M2_pass_but_no_postoffset_specificity": "tone-evidence around the nucleus, not a post-offset response",
            "M3_pass": "category-residual increment; conditional-label sensitivity before stronger claims",
        },
        "observed_outdirs": {
            "M0": str(POOL / "encoding_results_b2b_tone_ctx_v1_M0_nucleus_offset"),
            "M1": str(POOL / "encoding_results_b2b_tone_ctx_v1_M1_nucleus_offset"),
            "M2": str(POOL / "encoding_results_b2b_tone_ctx_v1_M2_nucleus_offset"),
            "M3": str(POOL / "encoding_results_b2b_tone_ctx_v1_M3_nucleus_offset"),
        },
        "existence_outdirs": {
            "M1": str(POOL / "encoding_results_b2b_tone_ctx_v1_M1_nucleus_offset_existence_nulls"),
            "M2": str(POOL / "encoding_results_b2b_tone_ctx_v1_M2_nucleus_offset_existence_nulls"),
            "M3": str(POOL / "encoding_results_b2b_tone_ctx_v1_M3_nucleus_offset_existence_nulls"),
        },
        "splithalf_outdirs": {
            "M1": str(POOL / "encoding_results_b2b_tone_ctx_v1_M1_nucleus_offset_splithalf"),
            "M2": str(POOL / "encoding_results_b2b_tone_ctx_v1_M2_nucleus_offset_splithalf"),
            "M3": str(POOL / "encoding_results_b2b_tone_ctx_v1_M3_nucleus_offset_splithalf"),
        },
        "n_primary": 63,
        "dims": {"ctx_controls": 39, "ctx_pitch": 5, "ctx_evidence": 3, "ctx_resid": 3},
        "feature_sha256": feats,
        "code_sha256": code,
        "stimulus_sha256": stimulus,
        "overlap_ruler": {
            "nucleus_onset_rel_offset_s": overlap.get("nucleus_onset_rel_offset_s"),
            "next_nucleus_onset_rel_offset_s": overlap.get("next_nucleus_onset_rel_offset_s"),
            "prop_inside_at_minus_0.09": (
                overlap.get("prop_still_inside_current_nucleus_at_lag", {}).get("-0.09")
            ),
        },
        "interpretation_corrections": {
            "M2_negative_lag": "temporally nonspecific around nucleus offset; not demonstrated carryover",
            "M3_plus_0.58": "late overlap-sensitive decodability peak; not an implausible evoked latency",
        },
        "hold": ["TD_DD", "brain_behavior"],
    }
    DEST.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", DEST)


if __name__ == "__main__":
    main()
