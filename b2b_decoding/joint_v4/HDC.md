# Hierarchical dynamic coding (HDC) arm — Gwilliams et al. PNAS 2025
#
# Isolated from Whisper Large-V2 and lexical duration/freq/surprisal outdirs.
#
# Pilot features (Stage 1): wordinfo + GloVe PCA-10, offset-locked, no PCA,
# no drift RBF. Family score = mean diag(H) within the family.
# H is ridge-shrunk to κ≤32 (collinear linguistic columns); Whisper stays OLS.
# Collinear drops: wf_n_morphemes (≡ n_syllables), ph_vowel (manner sum-to-1).
#
# ```bash
# cd joint_v4
# bash scripts/run_hdc_pilot.sh
# .venv_gpt2/bin/python3 scripts/analyze_hdc_pilot.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot
# ```
#
# If the diagonal stop gate passes:
#
# ```bash
# B2B_DO_TG=1 bash scripts/run_hdc_pilot.sh
# .venv_gpt2/bin/python3 scripts/analyze_hdc_tg.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot
# # Prefer Slurm (login-node user cgroup is ~8–10 GB):
# sbatch --export=NONE -p mit_preemptable --requeue scripts/run_hdc_observed_array.sh
# .venv_gpt2/bin/python3 scripts/compare_hdc_groups.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot \
#     --groups cohort_groups.csv \
#     --out-dir group_comparison_b2b_hdc_offset_pilot
# ```
#
# Tree syntax (Stage 2c) replaces `syntax_proxy` with `syntactic_operation`
# and `syntactic_state` from `lppCN_tree.txt` (separate outdir + basis):
#
# ```bash
# bash scripts/run_hdc_syntax_pilot.sh
# sbatch --export=NONE -p mit_preemptable --requeue scripts/run_hdc_syntax_observed_array.sh
# .venv_gpt2/bin/python3 scripts/compare_hdc_groups.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \
#     --groups cohort_groups.csv \
#     --out-dir group_comparison_b2b_hdc_offset_syntax \
#     --families phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
# ```
#
# Acoustic residualization control (Gwilliams SI §1.6): regress envelope +
# log-F0 out of continuous EEG with MNE ReceptiveField (laplacian ridge,
# lags −200…0 ms), then re-run the syntax HDC arm on residual EEG.
# Families whose cluster duration drops >50% vs raw EEG are flagged as
# acoustically confounded (expect syntactic_operation / possibly phonetic).
#
# ```bash
# # 1) Residualize EEG (Slurm array, or local smoke):
# .venv_gpt2/bin/python3 scripts/regress_out_acoustic_mne.py --participants RN109
# RID=$(sbatch --export=NONE --parsable scripts/run_acoustic_residual_array.sh)
# # 2) HDC syntax on residual EEG (new outdir + basis; wait for residual array):
# bash scripts/run_hdc_syntax_acoures_pilot.sh
# sbatch --export=NONE --dependency=afterok:${RID} scripts/run_hdc_syntax_acoures_observed_array.sh
# # 3) Compare raw vs residual (after both arms finish):
# .venv_gpt2/bin/python3 scripts/compare_hdc_acoustic_residual.py \
#     --raw-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \
#     --res-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures \
#     --groups cohort_groups.csv \
#     --out-dir group_comparison_b2b_hdc_offset_syntax_acoures
# ```
# Note: continuous EEG envelope/pitch TRF R² is often ≪1% here (vs MEG in the
# paper); residualization still removes the fit acoustic component. The residual
# extractor symlinks `_shared_wordlocked_features` and `lppCN_tree.txt`.
#
# SNR arm (child ~20-min EEG): DSS spatial denoise (40 comps) + 20 ms temporal
# bins before B2B. Isolated outdir; ENV defaults leave legacy arms unchanged.
#
# ```bash
# bash scripts/run_hdc_snr_dss_bin_pilot.sh
# BID=$(sbatch --export=NONE --parsable scripts/run_hdc_snr_dss_bin_prepare_basis.sh)
# sbatch --export=NONE --dependency=afterok:${BID} scripts/run_hdc_snr_dss_bin_observed_array.sh
# .venv_gpt2/bin/python3 scripts/compare_hdc_groups.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20 \
#     --groups cohort_groups.csv \
#     --out-dir group_comparison_b2b_hdc_offset_syntax_acoures_zqc_dss40_bin20 \
#     --families phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
# ```
# Knobs: `B2B_SPATIAL_DENOISE=none|pca|dss`, `B2B_SPATIAL_N_COMP=40`,
# `B2B_TIME_BIN_MS=20` (0 = off). Fingerprinted in `_analysis_manifest.json`.
#
# Gwilliams-style metrics on existing TG + family_agg (no Julia re-run):
# subject threshold = z * SE from split_sd; group cluster duration; hierarchy
# with semantic excluded from the primary sustain set; TD vs dyslexia on the
# new metrics. Writes ``…/gwilliams_metrics/``.
#
# ```bash
# .venv_gpt2/bin/python3 scripts/recompute_hdc_gwilliams_metrics.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_pilot \
#     --groups cohort_groups.csv \
#     --out-dir group_comparison_b2b_hdc_offset_pilot/gwilliams_metrics
# .venv_gpt2/bin/python3 scripts/recompute_hdc_gwilliams_metrics.py \
#     --results-dir /home/haolun52/orcd/pool/encoding_results_b2b_hdc_offset_syntax \
#     --groups cohort_groups.csv \
#     --out-dir group_comparison_b2b_hdc_offset_syntax/gwilliams_metrics \
#     --families phonetic,word_form,lexical_syntactic,syntactic_operation,syntactic_state,semantic
# ```
#