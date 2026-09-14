# HPC Execution Guide — B2B joint_v4 (Large V2, onset-locked)

All commands from `b2b_decoding/joint_v4/` unless noted. Do **not** write into
`encoding_results_b2b_joint_v2` or `encoding_results_b2b_joint_v3`.

## 0. Sync

Transfer **only** shared Large V2 arrays under
`extracted_sections_wordlocked_shared/_shared_wordlocked_features/section_00{1,2}/`:

- `X_word_whisperlargev2_{acoustic,speech,language_audio_fused,language_text_only}.npy`
- matching `*_feature_names.txt` / `*_meta.json`
- `word_timing_relative.csv`

Exclude `._*` AppleDouble files and participant-level feature copies.
Verify shapes: sec1 1753×{12800,12800,1280,1280}; sec2 1800×same; dtype float32.

```bash
# Example checksum snapshot after sync
sha256sum \
  ../../extracted_sections_wordlocked_shared/_shared_wordlocked_features/section_00*/X_word_whisperlargev2_*.npy \
  | grep -v '/\._'
```

## 1. Integrity tests (no EEG)

```bash
module load julia/1.12.6
export JULIA_DEPOT_PATH=/home/haolun52/orcd/pool/.julia
julia --project=.. tests/test_b2b_joint_v4.jl
python3 tests/test_compare_b2b_groups.py
```

## 2. Basis preparation

```bash
mkdir -p logs
sbatch scripts/run_prepare_basis.sh
# Resources: 8 CPUs, 48 GB, 4 h
```

Inspect:

- `_basis/feature_basis_meta.json` — feature keys, checksums, drift retained var,
  gate_a / gate_b, cohort hash, git commit, Slurm job ID
- `_basis/basis_diagnostics.json` — cond(joint), canonical correlations

Independent mode: cross-family canon must **not** be forced to ~0.

## 3. Pilot observed fits

```bash
bash scripts/run_pilot.sh   # RN109, D007d, D011d
```

Record for each: `n_valid_partitions`, `max_kappa`, λ-boundary flags, H rank.

## 4. Null chunk benchmark

```bash
# After pilots pass stop gates:
sbatch --array=1 scripts/run_null_array.sh
```

Confirm finite traces and approx-centered null max statistics.

## 5. Full observed array (63)

```bash
sbatch scripts/run_observed_array.sh
```

## 6. Full null array (315 = 63 × 5)

**Only after all observed gates pass.**

```bash
sbatch scripts/run_null_array.sh
```

## 7. Aggregate + group inference

```bash
julia --project=.. aggregate_v4.jl \
  /home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k8

# Review _qc/null_completeness_report.json and keep_frac_by_group.csv

python3 compare_b2b_groups.py \
  --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k8 \
  --groups cohort_groups.csv \
  --out-dir group_comparison_b2b_largev2_onset_joint_k8
```

## 8. Observed-only sensitivities

Prep a basis in each outdir first (`B2B_OUTDIR=… B2B_BASIS_DIR=$B2B_OUTDIR/_basis`).

| Arm | Key env |
|-----|---------|
| Text-only language | `B2B_FEAT_LANGUAGE=whisperlargev2_language_text_only` → `…_textonly_k8` |
| Ordered basis | `B2B_BASIS_MODE=ordered` → `…_ordered_k8` |
| K=4 / K=16 | `B2B_PCA_K=4|16` → `…_joint_k4|k16` |
| Offset locking | `B2B_EPOCH_ANCHOR=offset` → `…_offset_joint_k8` |

```bash
# Example: offset-locked primary twin
export B2B_OUTDIR=/home/haolun52/orcd/pool/encoding_results_b2b_largev2_offset_joint_k8
export B2B_BASIS_DIR=$B2B_OUTDIR/_basis
export B2B_EPOCH_ANCHOR=offset
sbatch scripts/run_prepare_basis.sh
# then observed array with the same exports

# Or the bundled sensitivity array (K4 / K16 / textonly / ordered):
sbatch scripts/run_sensitivity_array.sh
```

Optional nuisance block (observed sensitivity):

```bash
export B2B_INCLUDE_NUISANCE=1
# new outdir required (manifest lock)
```

## Stop gates

Do **not** launch the full null array unless:

1. Every pilot has ≥16 valid partitions (prefer all 20)
2. No H rank failures
3. Held-out design κ < 85; investigate values > 32
4. Ridge selection does not repeatedly hit grid boundaries
5. Null traces are finite and approximately centered
6. Onset-locked word retention is acceptable and not systematically different by group

If independent K=8 exceeds conditioning limits → test K=4 in a **separate**
outdir. Do not silently switch to ordered residualization or regularize H.

## Deliverables checklist

- [ ] Code commit (`joint_v4` sources)
- [ ] This execution README
- [ ] Slurm scripts under `scripts/`
- [ ] Basis diagnostics (`_basis/basis_diagnostics.json`)
- [ ] Pilot report (logs + QC)
- [ ] Cohort/QC summary (`_qc/`)
- [ ] Null completeness report
- [ ] Group-analysis outputs (`group_comparison_b2b_largev2_onset_joint_k8/`)
