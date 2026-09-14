# tone_ctx_v1 stimulus report

Coefficient order chosen by OOF classifier log loss: **3** (3-coef ll=1.1476, 4-coef ll=1.1498).

## Model A — context-conditioned realization

- n = 3219
- MSE full = 0.01380
- MSE tone-only prototype = 0.01774
- Beats prototype: True

## Model B — grouped OOF tone evidence

- n = 3219 (T1–T4 only; T5 excluded)
- log loss = 1.1476  vs class-prior 1.3294
- permutation p (improvement vs prior) = 0.0050
- Brier (mean one-vs-rest) = 0.1512
- overall acc = 0.546 (not a launch gate)
- every fold has all classes: True
- story-blocked log loss = 1.1333
- drop `yi` log loss = 1.1394

Confusion (rows = true):

```
[[ 67 150  12 394]
 [ 37 517  58 252]
 [  2  77 215 212]
 [ 43 154  71 958]]
```

## Gates

- PASS  log_loss_beats_prior
- PASS  perm_p_lt_0.05
- PASS  all_folds_have_all_classes
- PASS  model_A_beats_prototype
- PASS  not_driven_by_one_base_syllable

**Launch EEG features:** True

