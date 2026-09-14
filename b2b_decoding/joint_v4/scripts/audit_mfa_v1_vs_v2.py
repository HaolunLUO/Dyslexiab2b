#!/usr/bin/env python3
"""Short old-vs-v2 MFA encoding audit.

Confirms subjects, feature checksums, basis hashes, and inference inputs
are consistent for the v2 outdir, and contrasts them with superseded v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

POOL = Path("/orcd/pool/005/haolun52")
DEFAULT_V1 = POOL / "encoding_results_b2b_mfa_phone_tone_env_onset_passthrough"
DEFAULT_V2 = POOL / "encoding_results_b2b_mfa_v2_phone_tone_env_onset_passthrough"
SHARED = POOL / "extracted_sections_wordlocked_shared" / "_shared_wordlocked_features"
COHORT = POOL / "dyslexia_natualistics_listing" / "b2b_decoding" / "joint_v4" / "cohort_groups.csv"
V1_KEYS = ("envelope", "mfa_phones", "mfa_tone")
V2_KEYS = ("envelope_v2", "mfa_phones_v2", "mfa_tone_v2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_subjects(cohort: Path) -> list[str]:
    df = pd.read_csv(cohort)
    return df.loc[df["include_primary"].astype(str).isin(["1", "True", "true"]), "participant"].tolist()


def feature_checksums(keys: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for sid in (1, 2):
        sec = SHARED / f"section_{sid:03d}"
        for k in keys:
            p = sec / f"X_word_{k}.npy"
            if p.is_file():
                out[f"section_{sid:03d}/{k}"] = sha(p)
    return out


def audit_outdir(outdir: Path, subjects: list[str], expected_keys: tuple[str, ...]) -> dict:
    basis_meta = outdir / "_basis" / "feature_basis_meta.json"
    report: dict = {
        "outdir": str(outdir),
        "exists": outdir.is_dir(),
        "superseded_note": (outdir / "SUPERSEDED.txt").is_file(),
        "n_subjects_expected": len(subjects),
        "n_family_agg": 0,
        "missing_agg": [],
        "manifest_mismatches": [],
        "basis_meta_present": basis_meta.is_file(),
    }
    if basis_meta.is_file():
        report["basis_meta_sha256"] = sha(basis_meta)
        meta = json.loads(basis_meta.read_text(encoding="utf-8"))
        report["basis_raw_feature_keys"] = meta.get("raw_feature_keys")
        report["basis_passthrough"] = meta.get("passthrough")
        report["basis_gate_b_cond"] = (meta.get("gate_b") or {}).get("cond_joint")
        report["basis_feature_checksums"] = meta.get("feature_checksums")
    current_feats = feature_checksums(expected_keys)
    report["current_feature_checksums"] = current_feats

    for p in subjects:
        agg = outdir / p / f"{p}_b2b_family_agg.csv"
        man = outdir / p / f"{p}_run_manifest.json"
        if agg.is_file():
            report["n_family_agg"] += 1
        else:
            report["missing_agg"].append(p)
            continue
        if not man.is_file():
            report["manifest_mismatches"].append({"participant": p, "error": "missing_manifest"})
            continue
        man_d = json.loads(man.read_text(encoding="utf-8"))
        issues = []
        if report.get("basis_meta_sha256") and man_d.get("feature_basis_hash") != report["basis_meta_sha256"]:
            issues.append("basis_hash_mismatch")
        stored = man_d.get("feature_checksums") or {}
        for k, h in current_feats.items():
            got = stored.get(k)
            if got is None:
                issues.append(f"missing_checksum:{k}")
            elif got != h:
                issues.append(f"stale_checksum:{k}")
        h_kappa = man_d.get("h_ridge_kappa")
        if h_kappa is not None and float(h_kappa) != 0.0 and "v2" in str(outdir):
            issues.append(f"h_ridge_kappa={h_kappa}")
        if issues:
            report["manifest_mismatches"].append({"participant": p, "issues": issues})
    report["ok"] = (
        report["exists"]
        and report["n_family_agg"] == len(subjects)
        and not report["manifest_mismatches"]
        and report["basis_meta_present"]
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-outdir", type=Path, default=DEFAULT_V1)
    ap.add_argument("--v2-outdir", type=Path, default=DEFAULT_V2)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    subjects = primary_subjects(args.cohort)
    report = {
        "n_primary_subjects": len(subjects),
        "v1": audit_outdir(args.v1_outdir, subjects, V1_KEYS),
        "v2": audit_outdir(args.v2_outdir, subjects, V2_KEYS),
        "feature_key_change": {"v1": list(V1_KEYS), "v2": list(V2_KEYS)},
        "v1_labeled_superseded": (args.v1_outdir / "SUPERSEDED.txt").is_file(),
    }
    # Cross-check: v1 and v2 feature files must differ
    v1c = feature_checksums(V1_KEYS)
    v2c = feature_checksums(V2_KEYS)
    report["v1_v2_features_distinct"] = bool(v1c) and bool(v2c) and set(v1c.values()).isdisjoint(v2c.values())
    out = args.out_json or (args.v2_outdir / "old_vs_v2_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "v1_n_agg": report["v1"]["n_family_agg"],
        "v1_superseded": report["v1_labeled_superseded"],
        "v2_n_agg": report["v2"]["n_family_agg"],
        "v2_ok": report["v2"]["ok"],
        "v2_mismatches": len(report["v2"]["manifest_mismatches"]),
        "features_distinct": report["v1_v2_features_distinct"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
