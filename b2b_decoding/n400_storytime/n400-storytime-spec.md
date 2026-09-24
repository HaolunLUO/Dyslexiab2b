# Storytime N400 — analysis spec

Adaptation of Levari & Snedeker (2024) single-trial N400 to the 63-child Mandarin story EEG. Written before epoching. EEG is used as stored: no fit-time bandpass and no re-reference.

## Cohort

- Subjects: `dyslexia_natualistics_listing/b2b_decoding/subjects.txt` (63).
- Groups from `joint_v4/cohort_groups.csv` (`include_primary=1`): TD 24, dyslexia_normal_CAP (nCAP) 19, dyslexia_atypical_CAP (aCAP) 20.
- Age: `behavioral_data.xlsx` sheet Sheet1, column `年龄`, values `年;月` converted as years + months/12. Same parser as `run_compensation_cap_primary_tests.py`. Moderation models are complete-case on age.

## EEG and channels

- Input: `extracted_sections_wordlocked_shared/<ID>/section_00N/eeg_data.npy` plus `train_keep_mask.npy` and `metadata.json`.
- Confirmed on RN102 section_001 before epoching: shape `(65, 56498)`, `sfreq` 100 Hz, `metadata.json` has no `channel_names`.
- Montage is `WORDLOCKED_CHAN_65` from `joint_v4/eeg_band.jl` (the wordlocked encoder order). Required names are present, so nothing was renamed:
  - Fz index 7, Cz index 27, Pz index 46, Oz index 62.
- Centro-parietal ROI: mean of Pz, Cz, Oz. Exploratory anterior site: Fz.
- Each subject script re-checks `eeg.shape[0]==65` and that those four names exist before any epoch is saved.

## Word table (stimulus only)

One row per token in `word_timing_relative.csv` for section 001 (1753) and section 002 (1800). Token order is the GPT2-CN alignment. Surprisal is `X_word_fs100_gpt2cn_surprisal`, checked identical to `X_word_gpt2cn_surprisal`. Units are nats (sum of token −log p) from `uer_gpt2_chinese`. Log frequency is `X_word_fs100_lexical_frequency` (stored log SUBTLEX-CH; non-finite csv `logfreq` cells were already filled in that array). Duration is offset − onset (seconds). Position is the 0-based token index within section.

### Content words

jieba 0.42.1 was installed for this analysis (LTP was not on ORCD). Each existing token is tagged with `posseg.cut(word, HMM=False)` so the story is not resegmented. A token is content when the flag starts with n/v/a/d, or is one of i, l, b, z, t, s (idiom, set phrase, distinguish-word, state-word, time word, place word). That is the N/V/ADJ/ADV set plus those close content classes.

22 tokens that jieba splits (for example 大人们, 孩子们, 旷无人烟) would break 1:1 alignment. Those use the stored Universal POS in `X_word_wordinfo` (content = NOUN, PROPN, VERB, ADJ, ADV). On atomic tokens, jieba vs UD content labels agree on 89.9%. Analysis N: 1857 content tokens; 1605 have complete predictors and both spillover neighbors.

### Semantic association

Facebook fastText Chinese Wikipedia vectors, `wiki.zh.vec`, 332,647 types × 300 dimensions (`https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec`), saved at `n400_storytime_v1/stimulus/embeddings/wiki.zh.vec`. No fastText or Tencent vectors were already on the pool.

Association is the mean cosine to the previous up to 3 content words. OOV context words are skipped in the average; the target itself must have a vector. OOV: 74/894 types (8.3%), 149/3553 tokens (4.2%). Examples: 小王子, 大人们, B612, 喜笑颜开, 一会儿.

### Spillover and analysis set

N−1 and N+1 surprisal, log frequency, and association, taken from the adjacent tokens in the full word sequence (not content-only neighbors). The LMM sample is content words with positive duration and finite duration, position, surprisal, frequency, association, and all six spillover columns. 1605 items (section 001: 809; section 002: 796).

### Stimulus correlations (Pearson, complete-case items, N=1605)

|  | surprisal | log frequency | association |
|---|---:|---:|---:|
| log frequency | −0.486 |  |  |
| association | −0.383 | 0.465 |  |
| duration | 0.390 | −0.546 | −0.563 |

Predictors are z-scored on these 1605 stimulus rows (population sd) and the same mean/sd is applied to every trial. The EEG amplitude is not scaled.

## Epochs and the dependent variable

Slurm array, one task per subject. `mit_quicktest` is used in waves of 8 because that QOS allows 8 submitted jobs per user (a 63-wide array is rejected) and `mit_preemptable` was priority-pending behind about 2,000 jobs. First preemptable submission was 23611506 (cancelled before it started). Script root `b2b_decoding/n400_storytime/`. Outdir `n400_storytime_v1/<ID>/`. Nothing is written into existing B2B or TRF outdirs.

- Epoch −0.2 to 1.0 s inclusive at the content-word onset sample (121 samples at 100 Hz).
- Baseline −0.2 to 0 s (20 samples, not including the onset sample), subtracted from the ROI mean.
- Drop epochs that fall outside the section, overlap `bad_intervals`, or contain any sample with `train_keep_mask` false. In this extract every section has an empty `bad_intervals` list and an all-true mask; the checks still run.
- Artifact rule, because physical units are unknown: among remaining epochs, within subject, reject if the peak-to-peak of the baseline-corrected CP-ROI mean exceeds median + 5×MAD. MAD is the raw median absolute deviation (no 1.4826 scale). If MAD is 0, amplitude rejection is skipped and logged.
- DV: mean amplitude 350–550 ms inclusive (21 samples) on the CP-ROI mean, and the same window at Fz.
- Saved per subject: `trials.csv`, `timecourse_kept.npz` (baseline-corrected CP and Fz time courses of kept epochs), `reject_log.json`.

No latency window is chosen from group results. 350–550 ms is fixed.

## Models

Tooling: R 4.5.2, `lme4` + `lmerTest` (Satterthwaite). Crossed random intercepts for subject and item (`item_id` = section + word index, shared across children). By-subject random slopes for surprisal, frequency, and association, plus a by-item surprisal slope, are attempted on the full model. If that fit errors, exceeds 480 s of CPU time, or is singular, item slopes are dropped first, then subject slopes, leaving intercepts only. The same random-effect structure is used for every model in a nesting.

Fixed-effect z predictors:

- Base: duration, position, section, and N−1/N+1 surprisal, frequency, and association.
- Then forward likelihood-ratio tests (ML): base → +frequency → +association → +surprisal.
- Backwards from the full fixed model (REML p>0.05, highest p first). Section is kept. This follows the paper’s best-fit step. The prespecified surprisal test is the forward LRT and the full-model coefficient, not the backwards path.
- Same nesting at Fz.
- Expectation, locked before seeing EEG: surprisal slope at CP is negative; the frequency effect should weaken once surprisal is in the model.

### Group moderation (after the pooled models)

On the best CP fixed structure, with frequency and surprisal forced back in if backwards removed them:

1. All children with age: group = TD (0) vs pooled dyslexia (1), age centered on the subjects in that sample, plus surprisal×group and frequency×group.
2. Dyslexia only, with age: group = nCAP (0) vs aCAP (1), same interactions.

Holm family (four tests, prespecified): surprisal×group for both contrasts (primary), frequency×group for both contrasts (secondary). A two-test Holm on the surprisal interactions alone is saved as a sensitivity, not as a replacement. Per-group slopes and Wald CIs are reported whether or not an interaction is significant. Descriptive full models are also fit inside TD, pooled dyslexia, nCAP, and aCAP.

## Outputs

- Stimulus: `n400_storytime_v1/stimulus/word_table.csv`, `predictor_correlations.csv`, `stimulus_qc.json`.
- Trials and LMMs under `n400_storytime_v1/`.
- Results narrative: `docs/n400-storytime-results.md`.
- Plots: `media/n400-storytime/`.
