# Storytime N400 — results

Single-trial adaptation of Levari & Snedeker (2024) to the 63-child Mandarin stories. EEG was used as stored (no fit-time bandpass, no re-reference). Spec: [n400-storytime-spec.md](n400-storytime-spec.md).

## Headline

At the centro-parietal ROI (Pz, Cz, Oz), GPT2-CN surprisal predicts a more negative 350–550 ms amplitude. In the full model the slope is **−0.630 amplitude units per 1 SD** (95% CI [−0.737, −0.522], p = 2.6×10⁻²⁹), which is **−0.192 per nat**. The likelihood-ratio test against the model without surprisal is χ²(1) = 127.3, p = 1.6×10⁻²⁹.

The classic positive frequency effect does not survive once surprisal is included. Before surprisal, log frequency has a positive slope (β = +0.119 per SD, p = 0.043). In the full model the partial frequency slope is −0.119 (95% CI [−0.239, 0.0004], p = 0.051). Association does not improve fit at this ROI.

Surprisal × group (TD vs pooled dyslexia) remains after Holm correction across the four prespecified moderation tests (p_Holm = 0.018): the TD slope is steeper (more negative) than the dyslexia slope. The nCAP vs aCAP surprisal interaction and both frequency interactions do not survive that Holm family.

## Sample and jobs

Epoch array: 63/63 tasks `COMPLETED` exit 0 on `mit_quicktest`, waves of 8 (QOS submit limit). Job IDs: 23611674, 23611693, 23611704, 23611729, 23611773, 23611786, 23611802, 23611858. An earlier `mit_preemptable` array, 23611506, was cancelled while still priority-pending.

| Stage | N |
|---|---|
| Subjects epoched | 63 (TD 24, nCAP 19, aCAP 20) |
| Content-word epochs kept | 113,077 |
| LMM trials (complete predictors) | 97,967 trials, 63 subjects, 1,603 items |
| Age-complete moderation | 57 subjects, 88,620 trials (TD 23, nCAP 18, aCAP 16) |
| Dyslexia-only moderation | 34 subjects, 53,144 trials |

Age missing, dropped from moderation only: D043d (TD), D047d (nCAP), D041d, D042d, D048d, D049d (aCAP). `年龄` was read as `年;月` from Sheet1.

Artifact rejection (peak-to-peak of the CP mean > median + 5×MAD) removed about 3.7% of candidate epochs in TD, 3.4% in aCAP, and 1.6% in nCAP. Boundary cuts removed a few onset-edge words. `bad_intervals` were empty and `train_keep_mask` was all true, so those filters removed no epochs.

## Channels, cloze proxy, embeddings

Fz (index 7), Cz (27), Pz (46), and Oz (62) are in `WORDLOCKED_CHAN_65`. No channel was substituted. ROI = mean of Pz, Cz, Oz.

Surprisal is `X_word_fs100_gpt2cn_surprisal` (identical to `X_word_gpt2cn_surprisal`), nats from `uer_gpt2_chinese`. Log frequency is `X_word_fs100_lexical_frequency`.

Association uses Facebook fastText wiki Chinese vectors (`wiki.zh.vec`, 332,647 × 300). OOV: 74/894 types (8.3%) and 149/3,553 tokens (4.2%). Mean cosine to the previous content words is 0.93 (SD 0.026), so the raw association scale is tight; models use the z-scored values. Content words: jieba 0.42.1 on the existing GPT2-CN tokens (22 split tokens fell back to stored Universal POS). 1,605 stimulus items had complete predictors; 1,603 of those had at least one kept epoch.

## Pooled models (crossed intercepts)

By-subject slopes and a by-item surprisal slope exceeded 480 s of CPU time on both the maximal structure and the subject-slope structure. Reported models use random intercepts for subject and item only.

Forward LRTs at CP (ML), then the full-model REML coefficients:

| Step | χ²(1) | p |
|---|---:|---:|
| + frequency | 4.13 | 0.042 |
| + association | 1.85 | 0.174 |
| + surprisal | 127.3 | 1.6×10⁻²⁹ |

Full CP fixed effects of interest: surprisal −0.630 [−0.737, −0.522]; frequency −0.119 [−0.239, 0.0004]; association −0.006 [−0.151, 0.138], p = 0.93. Duration is positive (β = 0.446, p = 4.9×10⁻¹²). Backwards pruning kept duration, section, current surprisal, current frequency (then p = 0.044, still negative), and the N+1 surprisal and association spillovers. N+1 surprisal is itself negative (β = −0.255, p = 1.1×10⁻⁵); that was not a prespecified test.

At Fz the surprisal LRT is smaller but in the same direction: χ²(1) = 5.28, p = 0.022; β = −0.203 [−0.377, −0.029]. Frequency does not enter (LRT p = 0.51; full-model p = 0.76) and backwards drops it. Association does not enter (LRT p = 0.39). This is not the anterior frequency effect in the English storytime paper.

## Group moderation

Prespecified Holm family of four tests on the age-complete CP models (age centered within the subjects in each contrast). Frequency and surprisal were in the model; association stayed because backwards had not removed every control, but the tested interactions are only surprisal×group and frequency×group.

| Test | Role | p | p_Holm (4 tests) |
|---|---|---:|---:|
| surprisal × TD vs dyslexia | primary | 0.0046 | **0.018** |
| surprisal × nCAP vs aCAP | primary | 0.034 | 0.102 |
| frequency × TD vs dyslexia | secondary | 0.072 | 0.102 |
| frequency × nCAP vs aCAP | secondary | 0.041 | 0.102 |

A two-test Holm on the surprisal interactions alone (sensitivity, not the prespecified family) gives 0.009 and 0.034.

Simple slopes from those interaction models (amplitude per 1 SD surprisal, Wald CI):

| Contrast | Group coded 0 | Group coded 1 |
|---|---|---|
| TD vs dyslexia | TD −0.769 [−0.914, −0.624] | dyslexia −0.541 [−0.667, −0.414] |
| nCAP vs aCAP | nCAP −0.446 [−0.610, −0.282] | aCAP −0.669 [−0.842, −0.496] |

Within-group refits on all kept trials, not only age-complete rows (same full fixed structure, intercepts):

| Group | Subjects | Trials | Surprisal β [95% CI] |
|---|---:|---:|---|
| TD | 24 | 37,054 | −0.725 [−0.861, −0.589] |
| Dyslexia pooled | 39 | 60,913 | −0.572 [−0.701, −0.443] |
| nCAP | 19 | 29,964 | −0.465 [−0.626, −0.304] |
| aCAP | 20 | 30,949 | −0.676 [−0.844, −0.507] |

Every group slope is negative. The reliable group difference is TD steeper than pooled dyslexia. nCAP is numerically shallower than aCAP; that interaction does not survive the four-test Holm.

## Plots

- `media/n400-storytime/erp_surprisal_tertile.png` — subject-equal grand average by surprisal tertile, CP and Fz, shaded 350–550 ms. High-surprisal tertiles sit lower (more negative) in the window.
- `media/n400-storytime/coefficients_cp_fz.png` — full-model surprisal, frequency, and association coefficients.
- `media/n400-storytime/subject_surprisal_slopes.png` — within-subject partial surprisal slopes. Medians: TD −0.68, nCAP −0.33, aCAP −0.55.

## Limits

Surprisal is model surprisal in nats, not human cloze, so the slope is not on the paper’s cloze scale. Absolute amplitude is in as-stored units, not confirmed microvolts. Random slopes did not finish inside the CPU limit, so the tests use intercepts only; standard errors do not include by-subject slope variance. FastText association is a local cosine, not LSA, and the cosines are compressed near 0.93. Six children lack age and are out of the moderation models only.
