#!/bin/bash
# mit_quicktest allows 8 submitted jobs per user, so the 63-subject
# array is submitted in waves of 8. Each wave blocks until it finishes.
set -euo pipefail
cd "$(dirname "$0")"
: > logs/wave_job_ids.txt
for start in 1 9 17 25 33 41 49 57; do
  end=$((start + 7))
  if (( end > 63 )); then end=63; fi
  jid=$(sbatch --parsable -p mit_quicktest -t 00:15:00 -a "${start}-${end}" run_epoch_array.sh)
  echo "wave ${start}-${end} job ${jid}"
  echo "${jid} ${start}-${end}" >> logs/wave_job_ids.txt
  while squeue -u "$USER" -n n400_epoch -h | grep -q .; do
    sleep 8
  done
  echo "wave ${start}-${end} cleared"
done
echo WAVES_DONE
n=$(ls -d /orcd/pool/005/haolun52/n400_storytime_v1/*/trials.csv 2>/dev/null | wc -l)
echo "trial_dirs ${n}"
