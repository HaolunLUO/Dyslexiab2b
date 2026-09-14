#!/usr/bin/env bash
# Aggregate null chunks for the HDC dss40_bin20 arm, then rerun group compares
# so existence / unique-family tests un-skip.
#
# Usage (from joint_v4/):
#   bash scripts/aggregate_hdc_snr_nulls_and_compare.sh

set -euo pipefail
cd "$(dirname "$0")/.."
OUTDIR="${B2B_OUTDIR:-/home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_OUTDIR="$OUTDIR"
export B2B_BASIS_DIR="${B2B_BASIS_DIR:-${OUTDIR}/_basis}"
export B2B_EXTRACTOR_DIR="${B2B_EXTRACTOR_DIR:-/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared_acoustic_residual}"
export B2B_EPOCH_ANCHOR=offset
export B2B_PASSTHROUGH=1
export B2B_PCA_K=1
export B2B_SPATIAL_DENOISE=dss
export B2B_SPATIAL_N_COMP=40
export B2B_TIME_BIN_MS=20
export B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,hdc_lexical_syntactic,hdc_syntactic_operation,hdc_syntactic_state,hdc_semantic
export B2B_FAMILY_NAMES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic

module load julia/1.12.6
julia --project=.. aggregate_v4.jl --nulls-only

PY="${HDC_PYTHON:-}"
if [[ -z "$PY" && -x .venv_gpt2/bin/python3 ]]; then PY=.venv_gpt2/bin/python3; fi
if [[ -z "$PY" ]]; then PY=python3; fi

FAMILIES=phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
for SCORE in mean_score mean_z; do
  OUT="group_comparison_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20"
  [[ "$SCORE" == "mean_z" ]] && OUT="${OUT}_meanz"
  echo "=== compare score=$SCORE → $OUT ==="
  "$PY" scripts/compare_hdc_groups.py \
    --results-dir "$OUTDIR" \
    --groups cohort_groups.csv \
    --out-dir "$OUT" \
    --families "$FAMILIES" \
    --score-col "$SCORE"
done

echo "Done. Check unique_family_*.json and representation_existence in group_comparison JSON."
