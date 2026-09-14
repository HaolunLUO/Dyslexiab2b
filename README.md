# b2b_joint_v4

Backward-to-backward (B2B) EEG decoding pipeline for naturalistic Mandarin
listening (`b2b_decoding/joint_v4`).

This repository is a **source-only** split from
[`dyslexia_natualistics_listing`](https://github.com/HaolunLUO/dyslexia_natualistics_listing)
(branch `add-joint-v4-source`). It excludes Python virtualenvs, EEG feature
arrays, and encoding result trees.

## Layout

```
b2b_decoding/
  Project.toml          # Julia project deps
  joint_v4/             # pipeline, scripts, tests, freeze specs
```

See `b2b_decoding/joint_v4/README.md` and `EXECUTION.md` for locked configs
and how to run observed / existence / split-half jobs.
