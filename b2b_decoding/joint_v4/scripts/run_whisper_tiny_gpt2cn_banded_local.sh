#!/usr/bin/env bash
# Local Tiny speech + GPT2-CN B2B on one banded EEG arm.
#
# Usage:
#   bash scripts/run_whisper_tiny_gpt2cn_banded_local.sh <band> <onset|offset> [concurrency]
set -euo pipefail
cd "$(dirname "$0")/.."

BAND="${1:?band required (delta|theta|alpha|beta|broadband)}"
ANCHOR="${2:?anchor required (onset|offset)}"
NJOBS="${3:-2}"

case "$BAND" in delta|theta|alpha|beta|broadband) ;; *)
  echo "Unknown band: $BAND"; exit 1 ;;
esac
case "$ANCHOR" in onset|offset) ;; *)
  echo "Unknown anchor: $ANCHOR"; exit 1 ;;
esac

POOL="${POOL:-/orcd/pool/005/haolun52}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/home/haolun52/orcd/pool/.julia}"
export B2B_BLAS_THREADS="${B2B_BLAS_THREADS:-2}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$B2B_BLAS_THREADS}"
export B2B_MODE=observed
export B2B_PCA_K=8
export B2B_BASIS_MODE=independent
export B2B_EPOCH_ANCHOR="$ANCHOR"
export B2B_FEATURE_SETS=whispertiny_speech,gpt2_l24
export B2B_FAMILY_NAMES=speech,gpt2cn
export B2B_ALLOW_ZERO_FEATURE_ROWS=1
export B2B_NO_PLOT=1
export B2B_DO_TG=0
export B2B_EXTRACTOR_DIR="${POOL}/extracted_sections_wordlocked_banded_b2b/${BAND}"
export B2B_OUTDIR="${POOL}/encoding_results_b2b_tinyspeech_gpt2cn_${ANCHOR}_${BAND}_zqc_k8_reseg"
export B2B_BASIS_DIR="${POOL}/encoding_results_b2b_tinyspeech_gpt2cn_${ANCHOR}_k8/_basis"

test -d "$B2B_EXTRACTOR_DIR" || {
  echo "Missing extractor overlay: $B2B_EXTRACTOR_DIR"
  echo "Run: bash scripts/wire_banded_b2b_extractors.sh"
  exit 1
}
test -f "${B2B_BASIS_DIR}/feature_basis.npz" || {
  echo "Missing basis: ${B2B_BASIS_DIR}/feature_basis.npz"; exit 1;
}

module load julia/1.12.6
mkdir -p logs "logs/banded_tiny_gpt2cn_${BAND}_${ANCHOR}"

echo "=== Tiny speech + GPT2-CN banded local ==="
echo "  band=${BAND}  anchor=${ANCHOR}  jobs=${NJOBS}  blas=${B2B_BLAS_THREADS}"
echo "  extractor=${B2B_EXTRACTOR_DIR}"
echo "  out=${B2B_OUTDIR}"
echo "  basis=${B2B_BASIS_DIR}"

run_one() {
  local idx="$1"
  local part
  part=$(python3 - <<PY
import csv
rows=[r for r in csv.DictReader(open("cohort_groups.csv")) if r["include_primary"] in ("1","True","true")]
print(rows[${idx}-1]["participant"])
PY
)
  local out_csv="${B2B_OUTDIR}/${part}/${part}_b2b_family_agg.csv"
  if [[ -f "$out_csv" && "${B2B_FORCE:-0}" != "1" ]]; then
    echo "[skip] $idx $part"
    return 0
  fi
  echo "[start] $idx $part $(date -Is)"
  if julia --project=/orcd/pool/005/haolun52/dyslexia_natualistics_listing/b2b_decoding \
      b2b_joint_v4_pipeline.jl "$idx" \
      > "logs/banded_tiny_gpt2cn_${BAND}_${ANCHOR}/${idx}_${part}.out" 2>&1; then
    echo "[done]  $idx $part $(date -Is)"
  else
    echo "[FAIL]  $idx $part" | tee -a "logs/banded_tiny_gpt2cn_${BAND}_${ANCHOR}/failures.txt"
    return 0
  fi
}
export -f run_one
export B2B_OUTDIR B2B_BASIS_DIR B2B_EXTRACTOR_DIR JULIA_DEPOT_PATH B2B_BLAS_THREADS
export JULIA_NUM_THREADS OPENBLAS_NUM_THREADS
export B2B_MODE B2B_EPOCH_ANCHOR B2B_BASIS_MODE B2B_PCA_K B2B_FEATURE_SETS B2B_FAMILY_NAMES
export B2B_ALLOW_ZERO_FEATURE_ROWS B2B_NO_PLOT B2B_DO_TG B2B_FORCE BAND ANCHOR

set +e
seq 1 63 | xargs -P "$NJOBS" -I{} bash -c 'run_one "$@"' _ {}
rc=$?
set -e

n_done=$(find "$B2B_OUTDIR" -name '*_b2b_family_agg.csv' 2>/dev/null | wc -l)
echo "Arm finished: ${BAND}/${ANCHOR}  done=${n_done}/63  xargs_rc=${rc}"
exit 0
