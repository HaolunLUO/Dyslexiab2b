# Analysis log — tone_v2 / combined F0 arm

Pre-registered before any combined-arm B2B compute. Do not treat later
frame-shopping as confirmatory.

## Feature versions

- `X_word_tone.npy` (v1) — syllable-span log-F0, voicing reverse-engineered
  from the filled-mean cache. Kept for a v1-vs-v2 supplement. Not overwritten.
- `X_word_tone_v2.npy` — rime-sliced F0 from `section_XXX__f0_voicing_fs100_v2.npz`
  (Praat; NaN = unvoiced). 6-D: T1–T4 one-hot + within-tone z-scored RMSE +
  `tone_dev_valid`. Invalid tokens coded 0 after z-score (mean imputation).
- Phone/syllable onset impulses stay out of word-locked B2B.

## Template gate (must pass before combined B2B)

See `plots/tone_v2_validation.json` and `plots/tone_templates_v2_rime.png`.

### Amendment — 2026-08-31 (before combined-arm launch)

Citation-form criterion (“T4 endpoint near T3 floor”) is inappropriate for
connected speech. Non-final T4 is routinely reduced to ~53 / high→mid; this
narrator’s empirical T4 (last voiced ~270 Hz vs T3 ~213 Hz) is that reduced
shape, not fill leakage. Artifact causes were excluded independently:
voicing runs to the rime end (`last_frac` ≈ 1.0), no +1 octave mode,
syllabification asserts in place.

`tone_dev` must use the speaker’s empirical template. Scoring against an
idealized 51 fall would mislabel normally-reduced T4 as deviant.

**Replaced with a token-level discriminability gate:**

1. Pairwise template RMSE — all four mutually separated in 10-pt space.
2. Confusion: assign each valid token to the nearest template by the same
   RMSE used for `tone_dev`. Pass = every tone’s modal assignment is itself,
   and T4→T4 beats T4→T1.

Shape checks (T2 rises, T4 falls, T1 high, T3 low-falling, T4 starts
highest) stay informational. Optional T4 pre-pausal vs non-final split is
logged as methods justification, not a launch gate.

### Discriminability result — 2026-08-31 (launch withheld)

Confusion on all valid tokens (n=3221, overall acc=0.40):

| true\pred | T1 | T2 | T3 | T4 | modal |
|---|---|---|---|---|---|
| T1 | 355 | 106 | 90 | 72 | T1 |
| T2 | 247 | 307 | 253 | 58 | T2 |
| T3 | 52 | 33 | 337 | 85 | T3 |
| T4 | **521** | 109 | 299 | 297 | **T1** |

T4→T1 (521) beats T4→T4 (297). T1–T4 pairwise RMSE=0.141.
T4 pre-pausal fall is deeper (end−start −0.65) than non-final (−0.36),
so the shallow pooled T4 is contextual reduction — but that reduction
collapses T4 vs T1 at the token level. Combined B2B not launched.

## tone_v3 — coefficient space (2026-08-31, before combined launch)

Raw 10-pt RMSE was ~90% register. v3 fits Legendre (c0 register, c1 slope,
c2 curvature) on the same v2 rime grid, z-scores each coefficient across
tokens, and uses Euclidean distance in that 3-D space.

Step A passed. B/C not needed.

| space | acc | T1 | T2 | T3 | T4 modal | T4→T4 vs T4→T1 |
|---|---|---|---|---|---|---|
| v2 10-pt RMSE | 0.40 | self 0.57 | self 0.36 | self 0.67 | **T1** | 297 vs **521** |
| v3 Legendre z (A) | 0.53 | self 0.62 | self 0.53 | self 0.66 | **T4** | **533** vs 328 |

`X_word_tone_v3.npy`: same 6 columns; `tone_dev` is within-tone z-scored
distance in (c0,c1,c2). Combined arm uses `tone_v3`, not v2.

## Combined arm (Step 5 / 6) — launch only after the gate

Features (same fold structure as existing ICA env/pitch arms):

```
envelope_v2 : pitch : tone_v2 : offset : lexical_frequency : gpt2cn_surprisal
```

### Natural frames (primary)

| Family | Primary frame | Window | Confirm |
|---|---|---|---|
| tone, pitch | **onset-locked** | −0.3 … 1.0 s | 0–0.6 s |
| surprisal | **offset-locked** | −0.5 … 1.0 s | 0–0.8 s |
| envelope, frequency, offset | report both; no preferred frame | same as the run | same as the run |

The other frame is supplementary only. Predicted double dissociation:
tone/pitch stronger onset-locked; surprisal stronger offset-locked.

### Readout (combined model)

| Result | Interpretation |
|---|---|
| Tone survives with pitch present | Categorical tone beyond acoustics |
| Tone collapses to ~0 | v1 tone was acoustic leakage (reportable null) |
| Pitch drops when tone is added | Some “F0 tracking” was category-level |

### Duration / next-word leakage

Stimulus word durations are shared across subjects, so they cannot masquerade
as a group effect. Late-window estimates can still reflect the following word
(onset-locked 1 s window; offset-locked pitch 0.3–0.6 s). Word duration is a
candidate nuisance, not a default in the first combined run.

## Onset impulses

`phone_onset` / `syllable_onset` at 100 Hz remain parked for a continuous
mTRF arm. Do not collapse to per-word counts/ISI.

---

# Phase 0 — Freeze and pre-register — 2026-09-02

Written **before** Phase 1 existence-null compute. Git tag: `tonev3-observed`.
No feature engineering after this date. Do not build tone v4.

## Frozen design (combined arm)

```
envelope_v2 : pitch : tone_v3 : offset : lexical_frequency : gpt2cn_surprisal
```

Passthrough K=1, independent families, OLS-H. Observed 63/63 both frames
(as of 2026-09-02). Features live in the shared word-locked tree (not this
git repo). SHA256 at freeze (`TONEV3_FREEZE.json`):

| file | section 001 | section 002 |
|---|---|---|
| `X_word_envelope_v2.npy` | `be315d5ee2ab7efd…` | `7ae95bb1396aac6a…` |
| `X_word_pitch.npy` | `9eecb2c7ea8059ff…` | `8c7ab119e7724fd0…` |
| `X_word_tone_v3.npy` | `661a90d12014e23a…` | `ac0fc0c24c14ebe5…` |
| `X_word_offset.npy` | `bf0279a392aa1642…` | `f72722d068a04c50…` |
| `X_word_lexical_frequency.npy` | `14034452f8bb4695…` | `21405a4dfe29d317…` |
| `X_word_gpt2cn_surprisal.npy` | `1e9025f856bdee63…` | `c716c93f43fa2b5b…` |

Confirm window locked to the pipeline constant `CONFIRM_WIN_S = (0.0, 0.8)`.
Score = `mean_trace` (Σ diag(H) over the family block) averaged in that window.

Offset outdir may contain **standard** pipeline nulls (min-shift 80, λ
re-tuned). Those are a different procedure and are **not** Gate 1.

Outdirs (do not overwrite):

- onset −0.3…1.0 s:
  `encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_passthrough`
- offset −0.5…1.0 s:
  `encoding_results_b2b_mne_ica_v1_envpitch_tonev3_woffset_tmin05_passthrough`

## Existence rule (Gate 1)

A family **exists** in a frame if the group-level **sign-flip permutation
test** on per-subject **null-subtracted** confirm-window scores gives
**p < .05** (pooled, n = 63).

- Null: circular-shift the **entire design matrix** (all families jointly)
  by *k* words **within each story**, wrap within story, exclude `|k| < 20`.
  200 shifts. Reuse observed CV folds and ridge α; do not re-tune per shift.
- `nullsub = observed − null_median` (per subject, per family/column).
- **Primary frame = onset.** Offset is supplementary.
- Standard pipeline nulls (min-shift 80, λ re-tuned) are a different
  procedure and are **not** Gate 1.

If group null floors differ (TD vs DD Mann–Whitney *p* < .1), carry the
per-subject null median as a covariate into every Phase 4 model.

## Reliability rule (Gate 2)

Split-half ICC(2,1) on null-subtracted confirm-window scores, Spearman–Brown
corrected. Also ICC of peak latency and fractional-area latency.

| ICC | Use |
|---|---|
| ≥ .5 | eligible for individual-differences and classifier analyses |
| .3–.5 | group-mean tests only |
| < .3 | excluded |

**No group test is reported for a family that failed Gate 1.**

## Confirmatory set (Holm, α = .05, 4 tests)

Everything not listed here is **exploratory**.

| ID | Hypothesis | Direction |
|---|---|---|
| C1 | delta-band (0.5–4 Hz) envelope unique variance | TD > DD, one-tailed |
| C2 | dynamic-pitch sub-family unique variance | TD > DD, one-tailed |
| C3 | transfer asymmetry, delta envelope | TD→DD < TD→TD-held-out |
| C4 | tone awareness → dynamic-pitch unique variance, all 63 | β > 0; age and NVIQ covaried |

C1–C3 use Phase 3b band-limited scores. C2/C4 use the pitch **dynamic**
sub-family (slope / Δf0 / f0_std / range / derivative), not the full 13-D
family. One-hots vs `tone_dev` are exploratory re-reads (Phase 3a).

## Decision (Phase 5)

- Any confirmatory survives Holm, **or** omnibus AUC permutation *p* < .05
  → wake `onsets_fs100` continuous delta/theta TRF.
- Nothing survives, omnibus at chance → the group statement is that
  word-locked B2B on this montage, n = 63, shows no group signature in
  envelope, F0, or tone at this design’s sensitivity. Tone-category null
  to supplement as a bounded negative. **Do not build v4.**

---

# Phase 1 submit — 2026-09-02

Existence nulls (min |k| = 20, 200 shifts, reuse observed α, cached YᵀY).
New outdirs; observed outdirs not written.

| frame | job | outdir |
|---|---|---|
| onset (primary) | 21790859 | `encoding_results_b2b_mne_ica_v1_envpitch_tonev3_onset_tmin03_existence_nulls` |
| offset | 21790860 | `encoding_results_b2b_mne_ica_v1_envpitch_tonev3_woffset_tmin05_existence_nulls` |

Standard offset nulls (min-shift 80, λ retuned) remain ignored for Gate 1.
After 63/63: `python scripts/aggregate_existence_nulls.py --frame onset`

## Phase 3a stimulus-side (2026-09-02, no EEG)

Lag-1 tone transitions (n = 2742 valid pairs): mean self-transition 0.21;
T3→T2 = 0.26 (sandhi-consistent, not T3 self-loop). OLS of tone(n) one-hots
on pitch(n−1) 13-D: mean R² = 0.013. Tone-category null wording: not a
missing lag-1 F0 story; unique variance after contemporaneous pitch is
already in the joint model. Tables in `phase3a/`.

---

# Gate 1 — 2026-09-02

Sign-flip of per-subject `nullsub = observed − null_median` confirm-window
`mean_trace` (pooled n = 63, 10 k flips). Primary frame = onset. All six
families **exist** in both frames (p = .0001 floor). Stronger than the
pre-registered guess (tone was expected to fail).

| frame | family | mean nullsub | median z | % subjects > 0 | p | exist |
|---|---|---|---|---|---|---|
| onset | envelope | 0.00129 | 2.66 | 90 | .0001 | yes |
| onset | pitch | 0.00108 | 1.56 | 87 | .0001 | yes |
| onset | tone | 0.00076 | 2.15 | 95 | .0001 | yes |
| onset | offset | 0.00324 | 11.23 | 100 | .0001 | yes |
| onset | frequency | 0.00014 | 0.70 | 68 | .0001 | yes |
| onset | surprisal | 0.00133 | 9.44 | 98 | .0001 | yes |
| offset | envelope | 0.00103 | 2.41 | 94 | .0001 | yes |
| offset | pitch | 0.00164 | 3.17 | 90 | .0001 | yes |
| offset | tone | 0.00080 | 2.46 | 92 | .0001 | yes |
| offset | offset | 0.00172 | 6.79 | 98 | .0001 | yes |
| offset | frequency | 0.00074 | 4.83 | 97 | .0001 | yes |
| offset | surprisal | 0.00067 | 5.03 | 98 | .0001 | yes |

Tone existence is the **family-sum** (one-hots + `tone_dev`). Phase 3a
sub-family split is still required before wording a categorical-tone claim.
Onset family-level `nullsub` intercorrelations are high (r ≈ .84–.94 among
envelope/pitch/tone/offset/surprisal) — shared SNR, not six independent
encodings. Gate 2 decides which of these are stable enough for group tests.

**Floors (TD vs DD Mann–Whitney on per-subject null median).** Onset: all
p > .20. Offset envelope p = .039 (< .10). Pre-registered rule: carry
per-subject null median as a covariate into **every Phase 4 model**.

Outputs: `existence/gate1.csv`, `scores_{frame}.csv`, `floors_by_group.csv`,
`nulls_{frame}.parquet`, `null_traces_{frame}.npz`. D030d onset nulls
resubmitted (job 21797845) after `node1617` lacked Julia.

---

# Phase 2 submit — 2026-09-02

Interleaved ~60 s odd/even split-half (not first/second half). New outdirs.

| frame | job | outdir |
|---|---|---|
| onset | 21800122 | `…_onset_tmin03_splithalf` |
| offset | 21800123 | `…_woffset_tmin05_splithalf` |

After 63/63: `python scripts/aggregate_split_half.py --frame onset` then `--frame offset`.

## Phase 3a H-diagonal re-reads — 2026-09-02

From Phase 1 per-column `nullsub` (no new fits). Sub-family sign-flip (n = 63):

| frame | family | sub | mean nullsub | median z (cols) | % > 0 | exist |
|---|---|---|---|---|---|---|
| onset | pitch | level (`f0_mean`) | 0.00076 | 2.23 | 90 | yes |
| onset | pitch | dynamic (12 cols) | 0.00046 | 0.01 | 67 | yes (sum) |
| onset | tone | one-hot | 0.00065 | 0.83 | 98 | yes |
| onset | tone | `tone_dev` + valid | 0.00016 | 0.44 | 81 | yes |
| offset | pitch | level | 0.00132 | 4.42 | 98 | yes |
| offset | pitch | dynamic | 0.00046 | 0.01 | 63 | yes (sum) |
| offset | tone | one-hot | 0.00052 | 0.70 | 92 | yes |
| offset | tone | `tone_dev` + valid | 0.00033 | 1.25 | 92 | yes |

Pitch family-sum existence is mostly **level** (`f0_mean`). Dynamic columns
exist as a sum but the typical column is at chance (median z ≈ 0). C2 is
about this weaker dynamic block — Gate 2 must pass before any group test.
Tone one-hots exist as a sum; that is still not a category-selective claim
(6-D vs 13-D trap: report per-column / per-K, not raw family sums).
Tables: `phase3a/column_reread.md`, `subfamily_existence.csv`.

---

# Gate 2 — 2026-09-02

Split-half 63/63 both frames (jobs 21800122 / 21800123). ICC(2,1) and
Spearman–Brown on Phase-1-null-subtracted confirm-window scores. Gate
uses SB-corrected reliability (each half is half the data).

| frame | family / sub | ICC(2,1) | SB | gate |
|---|---|---|---|---|
| onset | envelope | .49 | .66 | id + classifier |
| onset | pitch | .66 | .80 | id + classifier |
| onset | pitch **level** | .62 | .76 | id + classifier |
| onset | pitch **dynamic** | .08 | .16 | **excluded** |
| onset | tone | .64 | .78 | id + classifier |
| onset | tone one-hot | .64 | .78 | id + classifier |
| onset | tone `tone_dev` | .06 | .12 | **excluded** |
| onset | offset | .91 | .95 | id + classifier |
| onset | frequency | .37 | .54 | id + classifier |
| onset | surprisal | .79 | .88 | id + classifier |
| offset | envelope | .31 | .47 | group-mean only |
| offset | pitch | .65 | .79 | id + classifier |
| offset | pitch level | .51 | .68 | id + classifier |
| offset | pitch dynamic | .60 | .75 | id + classifier |
| offset | tone | .64 | .78 | id + classifier |
| offset | tone one-hot | .62 | .77 | id + classifier |
| offset | tone `tone_dev` | .37 | .54 | id + classifier |
| offset | offset | .72 | .84 | id + classifier |
| offset | frequency | .79 | .88 | id + classifier |
| offset | surprisal | .67 | .80 | id + classifier |

**Hold the line.** C2 and C4 are pre-registered on **dynamic-pitch** unique
variance. On the primary frame (onset) that block is excluded (SB = .16).
No confirmatory group test on onset dynamic pitch. Offset dynamic pitch
passes Gate 2 but is not the primary frame.

Peak latency is excluded everywhere. Fractional-area latency is
group-mean only for a few family sums (onset envelope / offset / pitch /
surprisal); do not use latency in the omnibus except where ICC allows.

`reliability/icc_splithalf.csv`. Next: Phase 3b band-limited B2B (C1 lives
on delta envelope, not this broadband Gate 2).

---

# Phase 3b/3c submit — 2026-09-02

Band-limit **continuous** EEG (Butterworth-4 `filtfilt`) before epoching,
then observed + 200 Phase-1-style existence nulls. Reuse frozen tone_v3
`_basis`. New outdirs only; never write into `*passthrough*`.

| arm | band | frame | channels | outdir suffix |
|---|---|---|---|---|
| C1 primary | delta 0.5–4 | onset | all 65 | `…_onset_tmin03_band_delta` |
| supp | delta | offset | all | `…_woffset_tmin05_band_delta` |
| supp | theta 4–8 | onset | all | `…_onset_tmin03_band_theta` |
| supp | theta | offset | all | `…_woffset_tmin05_band_theta` |
| 3c LI | delta | onset | T7/TP7/P7/C5 | `…_onset_tmin03_band_delta_left_temporal` |
| 3c LI | delta | onset | T8/TP8/P8/C6 | `…_onset_tmin03_band_delta_right_temporal` |

C1 is **all-channel delta envelope** nullsub, not laterality. LI =
(R−L)/(R+L) on delta-envelope nullsub after both temporal arms finish.

Submit: `bash scripts/submit_band_b2b.sh delta onset all` first (this
entry). Remaining arms: `bash scripts/submit_phase3bc_remaining.sh` after
the first array is healthy. Always `--exclude=node1617`.
`B2B_OBSERVED_OUTDIR=$B2B_OUTDIR` and `B2B_ALLOW_INPLACE_NULLS=1` are
required at julia start.

**2026-09-02 04:15 EDT:** slurmctld on `slurm001` was DOWN (`scontrol ping`).
Code and tests are ready; the delta-onset array is not yet in the queue.
Re-run the submit line above when the controller is restored. Job id goes
in `logs/band_b2b_job_ids.txt`.

After 63/63, reuse `scripts/aggregate_existence_nulls.py` with
`--exist-dir/--obs-dir` into `joint_v4/band_delta/` (and `band_theta/`).
Laterality: `scripts/aggregate_laterality.py --frame onset`.

---

# tone_ctx_v1 — exploratory nucleus residual B2B — 2026-09-03

New branch. **Does not amend Phase 0 C1–C4.** Frozen tone_v3 features and
outdirs are not written. Tag state remains `tonev3-observed`.

## Question

Does EEG contain three nested representations: physical F0 (M1), nonlinear
acoustic tone evidence (M2), and surface-tone identity left after that
evidence (M3)? M3 is the operational “tone beyond modeled acoustics.”

## Stimulus (no EEG)

- Token table: `tone_ctx_v1/tone_ctx_tokens.parquet`
  n = **4962** nuclei (sec1=2372, sec2=2590). T1–T4=4517, T5=56 held out of
  the classifier. Sandhi=372 (`t3t3_to_t2` 248, `yi_to_4` 64, `yi_to_2` 32,
  `bu_t4_to_t2` 28). Multi-syllable words=1286. Parent-word surprisal is
  reused on 1429 non-initial syllables (known limitation).
- Stimulus-only CV chose **3** Legendre coefficients (OOF ll 1.148 vs 1.150).
- Model A beats a tone-only prototype (MSE 0.0138 vs 0.0177).
- Model B grouped-by-`base_syllable` OOF: log loss 1.148 vs prior 1.329
  (perm p=0.005). Acc=0.546 (not a gate). All stimulus gates **PASS**.
  See `tone_ctx_v1/stimulus_report.md`.

## Recoverability

On the real nested design, injected M2/M3 recover; controls/pitch-only do
not create a false M3 increment. `tone_ctx_v1/recoverability.json`.
Re-run after the identification prune (dims 39/5/3/3): still **PASS**.

## Identification (before any array)

First RN102 M0 smoke had 0/20 valid partitions (max κ ~ 1e16). Two
causes, both stimulus-side:

1. Rare exclusive dummies (`fin_un` n=1, `fin_ie` n=3) vanish in a
   split-half. Initials/finals with count < 80 are now merged into
   `*_other`; one reference dummy per block is still dropped.
2. `rel_f0_mean` is r≈0.999 with `c0_register`. Dropped. Level vs
   dynamic stay separate (`c0` vs `c1`/`c2`/`f0_change`/`f0_range`).

Controls are 39-D; pitch is 5-D. Stimulus split-half max κ:
M0=10.3, M1=11.4, M2=12, M3=12. Overlay wires only `eeg_data.npy`,
`metadata.json`, and `train_keep_mask.npy` — not `word_events.csv`
(that file is word-locked and would override nucleus timing).

## EEG path (pilot)

Nucleus-offset B2B, epoch −0.2…0.8 s, confirm 0–0.8 s.

| Arm | families |
|---|---|
| M0 | ctx_controls |
| M1 | controls + pitch |
| M2 | + evidence `(p)H` |
| M3 | + category residual `(y−p)H` |

Overlay: `extracted_sections_nucleuslocked_mne_ica_v1_b2b`
Outdirs: `encoding_results_b2b_tone_ctx_v1_{M0,M1,M2,M3}_nucleus_offset`

**Pilot subjects (no group contrast):** RN102, RN103, RN105, D001d, D011d.

RN102 M0 login-node smoke **2026-09-03**: 20/20 valid, max κ=10.3,
n_usable=4360. `slurmctld` on slurm001 is **DOWN** (same outage as
Phase 3b). Cannot `sbatch` the 5-subject arrays, n=63, existence
nulls, or split-half. Local fallback completed the pre-registered
5 × 4 smoke instead:

```
bash scripts/run_tone_ctx_v1_local_pilot.sh
```

All **20/20** subject×arm cells valid (max κ 10.2–12.1). Parameters
frozen from this smoke: epoch −0.2…0.8 s, offset anchor, passthrough
K=1, OLS-H, independent families, dims 39/5/3/3. Descriptive unique
`mean_trace` (confirm 0–0.8 s) is in `tone_ctx_v1/pilot_smoke.json`;
**do not** treat n=5 as a group or existence result.

When the controller is back:

```
MODEL=M0 bash scripts/submit_tone_ctx_v1.sh
MODEL=M1 bash scripts/submit_tone_ctx_v1.sh
MODEL=M2 bash scripts/submit_tone_ctx_v1.sh
MODEL=M3 bash scripts/submit_tone_ctx_v1.sh
```

(drop `--pilot`; the five smoke IDs already have observed outputs and
will be overwritten by the n=63 array). Then `--existence` and
`--splithalf` on M2 and M3. Holm over M2−M1 and M3−M2. No group test
until those gates pass. Use this arm’s null median, not tone_v3 floors.

Always `--exclude=node1617`.

## n=63 launch — 2026-09-03 (slurmctld back up)

First arrays **21874591–21874594** all 63×4 failed in ~11 s:
`B2B_EXPECTED_SEC_LENGTHS must have 2 integers; got [2372]`. Cause: comma
in `sbatch --export` (`2372,2590` truncated to `2372`). Env now uses
`2372:2590`. Resubmitted:

| Arm | jobid |
|---|---|
| M0 | 21878002 |
| M1 | 21878003 |
| M2 | 21878004 |
| M3 | 21878005 |

**Observed n=63 complete (2026-09-03 evening).** All 63×4 tasks COMPLETED, 0 failures.
QC: 63/63 valid 20/20 partitions on every arm (M0 κ 10.2–10.5; M1 11.4–12;
M2/M3 11.9–12.5).

## Adjudication freeze — 2026-09-03

Identification PASS. Existence and reliability PENDING. TD/DD and
brain–behavior HOLD. Windows stay 0–0.8 s; do not move them to the
descriptive +0.05 or +0.58 s peaks.

**Descriptive corrections (no inference):**
- M2 at −0.09 s is **temporally nonspecific around offset**, not demonstrated
  carryover. Nucleus duration p50=70 ms; 36% of tokens are still inside the
  current nucleus at −0.09 s. Next-nucleus onset p50=80 ms after offset
  (`tone_ctx_v1/overlap_ruler.json`).
- M3 at +0.58 s is a **late, overlap-sensitive decodability peak**, not an
  implausible evoked latency. Next-nucleus p90=0.47 s, so many later syllables
  have begun.

**Null calibration:** whole-design k-shift is **not** a unique-contribution
null (Type-I for evidence ≈ 0.93 when Y has only controls+pitch). Raw
added-family shift is the same anti-pattern (FPR=1). Freedman–Lane
residual shift of the last family, after OLS on the reduced nested design,
had FPR 0/30 for both evidence and resid (`tone_ctx_v1/null_type1.json`).
Existence uses that method. |k|≥20 (already exceeds the 1.0 s epoch at the
p10 ISI). B=5000. One-sided p=(1+#{T*≥T})/(B+1). Holm only on M2 and M3.
M1 is the positive-control null, not in Holm. Do not require M2 before M3.

Immutable spec: `TONECTX_V1_ADJUDICATION_FREEZE.json`.

| Job | id |
|---|---|
| existence M1 | 21884409 |
| existence M2 | 21884410 |
| existence M3 | 21884411 |
| split-half M1 | 21884412 |
| split-half M2 | 21884413 |
| split-half M3 | 21884414 |

After 63/63: `python scripts/aggregate_tone_ctx_adjudication.py`.
SB gate 0.30. No group or behavior test before that.



## Adjudication recovery — 2026-09-08

No active jobs at recovery. Existence-confirm file coverage: M1 63/63,
M2 61/63, M3 26/63; split-half 0/63 for all three models.
Split-half failed during subfamily reporting: legacy 13-D pitch indices
were applied to 5-D ctx_pitch (RN102 BoundsError, columns 41:52 in a
44-column matrix). Corrected reporting to select ctx_pitch level 1 and
dynamic columns 2:5 by feature key, with dimension assertions. The
family-sum estimand, fitting, windows, null method, and frozen features
are unchanged. This is a bug fix to the frozen implementation; its
split_half.jl hash now differs from the original freeze. All 12 tests
in tests/test_split_half.jl passed with the shared Julia depot.

Submitted via scripts/submit_tone_ctx_recovery.sh:

| Work | Job | Tasks |
|---|---|---|
| M1 split-half | 22320967 | 63 |
| M2 existence | 22320969 | missing indices 7,33 |
| M2 split-half | 22320970 | 63 |
| M3 existence | 22320971 | 37 missing participants |
| M3 split-half | 22320972 | 63 |

Existence retains 5000 Freedman–Lane draws, min shift 20; completed
existence outputs are preserved. No group or behavioral tests launched.
Submission is not completion; validate all participant outputs before
running scripts/aggregate_tone_ctx_adjudication.py.
