# Joint-family B2B decoding V4 — Whisper Large V2, onset-locked

Primary B2B analysis with Whisper Large V2 embeddings and word-onset-locked
scalp EEG. V2/V3 outputs under `encoding_results_b2b_joint_v{2,3}` are left
untouched.

## Estimand

Conditional linear alignment between multichannel EEG and retained acoustic,
speech, and language embedding subspaces. Primary language feature is
**audio-fused** (`whisperlargev2_language_audio_fused`). Text-only language is
a separate sensitivity outdir, not a fourth primary family.

## Locked primary configuration

| Knob | Value |
|------|-------|
| Epoch anchor | `onset` (`onset_sample`) |
| Epoch window | −0.2 … 1.0 s |
| Confirmatory interval | 0 … 0.8 s |
| Sampling rate | 100 Hz |
| Families | acoustic, speech, language (audio-fused) |
| PCA | K=8 independently per family |
| Basis | standardize, whiten, 40-knot per-section drift removal |
| Residualization | **none** (`B2B_BASIS_MODE=independent`) |
| G | CV ridge; H | unregularized OLS |
| Score | `family_trace = Σ diag(H_f)`; primary `family_score = trace/K` |
| Partitions | 20 blocked + word guards |
| Null | 200 section-wise circular stimulus shifts (common across families) |

## Output paths

| Analysis | `B2B_OUTDIR` |
|----------|--------------|
| Primary | `encoding_results_b2b_largev2_onset_joint_k8` |
| Text-only language | `encoding_results_b2b_largev2_onset_textonly_k8` |
| Ordered basis | `encoding_results_b2b_largev2_onset_ordered_k8` |
| Offset locking | `encoding_results_b2b_largev2_offset_joint_k8` |
| K=4 / K=16 | `…_joint_k4` / `…_joint_k16` |

## Environment knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `B2B_FEAT_ACOUSTIC` | `whisperlargev2_acoustic` | Acoustic feature key |
| `B2B_FEAT_SPEECH` | `whisperlargev2_speech` | Speech feature key |
| `B2B_FEAT_LANGUAGE` | `whisperlargev2_language_audio_fused` | Language feature key |
| `B2B_EPOCH_ANCHOR` | `onset` | `onset` \| `offset` |
| `B2B_BASIS_MODE` | `independent` | `independent` \| `ordered` |
| `B2B_INCLUDE_NUISANCE` | `0` | Append duration/logfreq/position/section/char-length |
| `B2B_PCA_K` | `8` | PCs per family |
| `B2B_DRIFT_KNOTS` | `40` | RBF drift knots per section |
| `B2B_OUTDIR` / `B2B_BASIS_DIR` | see above | Isolated result roots |

Appending to an outdir whose `_analysis_manifest.json` differs from the current
config is refused.

## HPC execution order

See [EXECUTION.md](EXECUTION.md) for the full sequence, stop gates, and
sensitivity recipes. Short form:

```bash
cd joint_v4
module load julia/1.12.6
export JULIA_DEPOT_PATH=/home/haolun52/orcd/pool/.julia

# 1. Integrity tests (no EEG)
julia --project=.. tests/test_b2b_joint_v4.jl
python3 tests/test_compare_b2b_groups.py

# 2. Basis (8 CPU / 48 GB / 4 h)
mkdir -p logs && sbatch scripts/run_prepare_basis.sh

# 3. Inspect _basis/basis_diagnostics.json (retained var, canon, cond)

# 4. Pilot
bash scripts/run_pilot.sh

# 5–9. Observed → null benchmark → full null → aggregate → group inference
#     (only after stop gates; see EXECUTION.md)
```

## Cohort

`cohort_groups.csv`: TD 24, dyslexia-normal-CAP 19, dyslexia-atypical-CAP 20
(pooled dyslexia 39). Any other counts are rejected.

## Group inference

- Representation existence: participant **stimulus-shift** nulls
- Group differences: **subject-label** permutations
- Statistic: one **max-cluster mass across time × three families**
- Primary contrast: TD 24 vs pooled dyslexia 39
- Secondary: normal-CAP 19 vs atypical-CAP 20

## Stop gates (do not launch full nulls unless)

- Every pilot ≥16 valid partitions (prefer 20)
- No H rank failures
- Held-out κ < 85; investigate > 32
- Ridge selection not repeatedly at grid boundaries
- Null traces finite and approximately centered
- Onset-locked retention acceptable and not systematically group-biased

If independent K=8 exceeds conditioning limits, test **K=4 in a separate
outdir**. Do not silently switch to ordered residualization or regularize H.
