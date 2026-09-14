# Lexical King-style B2B (onset-locked)
#
# Predictors (joint in one H, K=1 each):
#   1. duration   — word acoustic duration (onset → offset)
#   2. frequency  — log word frequency
#   3. surprisal  — GPT2CN next-token surprisal (nats, summed over word tokens)
#
# Epochs locked to word onset (−0.2 … 1.0 s), same B2B estimator as joint_v4.
#
# ## Setup
#
# 1. Compute GPT2CN surprisal (once):
#
# ```bash
# module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64
# cd joint_v4
# python3 -m venv .venv_gpt2 && . .venv_gpt2/bin/activate
# pip install --upgrade pip
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install transformers numpy pandas
# python scripts/compute_gpt2cn_surprisal.py \
#   --timing-root /home/haolun52/orcd/pool/extracted_sections_wordlocked_shared/_shared_wordlocked_features \
#   --out-root    /home/haolun52/orcd/pool/extracted_sections_wordlocked_shared/_shared_wordlocked_features \
#   --sections 1 2 \
#   --model uer/gpt2-chinese-cluecorpussmall
# ```
#
# 2. Pilot (builds lexical duration/frequency features + basis + 3 subjects):
#
# ```bash
# bash scripts/run_lexical_pilot.sh
# # or: bash scripts/run_lexical_pilot.sh RN109
# ```
#
# Outdir default:
#   `/home/haolun52/orcd/pool/encoding_results_b2b_lexical_onset_dur_freq_surp_k1`
#
# Group compare (after full observed). Uses `.venv_gpt2` (Python 3.10 + numpy/pandas);
# scipy is installed on first run if missing.
#
# ```bash
# bash scripts/run_lexical_group_comparison.sh
# # plots only (skips 10k permutations if summary already exists):
# PLOT_ONLY=1 bash scripts/run_lexical_group_comparison.sh
# ```
#
# Plots: `group_comparison_b2b_lexical_onset_dur_freq_surp_k1/plots/group_means_by_family.png`
#
# Note: "word onset" here means (i) epochs locked to acoustic word onset and
# (ii) duration measured from onset→offset (speech analog of word length in
# King et al. 2020). Surprisal is from Chinese GPT-2, not the Qwen SAE surprisal
# under analysisEV/.
