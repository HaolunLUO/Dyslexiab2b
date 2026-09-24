#!/usr/bin/env python3
"""Phase 0 audits for the Storytime N400 revision.

Does not fit the revision LMMs. Writes tables under
/orcd/pool/005/haolun52/n400_storytime_v1/revision/audit/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

POOL = Path("/orcd/pool/005/haolun52")
ROOT = POOL / "n400_storytime_v1"
STIM = ROOT / "stimulus"
SHARED = POOL / "extracted_sections_wordlocked_shared"
OUT = ROOT / "revision" / "audit"
EMBED = STIM / "embeddings" / "wiki.zh.vec"
OUT.mkdir(parents=True, exist_ok=True)

WORDLOCKED_CHAN_65 = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT7", "FT8", "FT9", "FT10",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "TP9", "TP10",
    "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]
TOPO_CH = ["Fz", "Cz", "Pz", "Oz", "F7", "T7", "T8", "P7"]
SFREQ = 100.0
TMIN, TMAX = -0.2, 1.0
N_T = int(round((TMAX - TMIN) * SFREQ)) + 1
PRE = int(round(-TMIN * SFREQ))
TIMES = np.round(TMIN + np.arange(N_T) / SFREQ, 5)
BASE_IX = np.where(TIMES < -1e-8)[0]
WIN = (TIMES >= 0.35 - 1e-8) & (TIMES <= 0.55 + 1e-8)
RNG = np.random.default_rng(31)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return a_n @ b_n.T


def _load_vectors(vocab: set[str]) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    with EMBED.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split()
        n_rows, dim = int(header[0]), int(header[1])
        for i, line in enumerate(f, start=1):
            if i % 100000 == 0:
                print(f"  scanned {i}/{n_rows} kept {len(found)}", flush=True)
            if len(found) == len(vocab):
                break
            w, rest = line.split(" ", 1)
            if w not in vocab or w in found:
                continue
            vec = np.fromstring(rest, sep=" ", dtype=np.float32, count=dim)
            if vec.shape[0] != dim or not np.isfinite(vec).all():
                continue
            found[w] = vec
    print(f"fastText kept {len(found)}/{len(vocab)} dim={dim}", flush=True)
    return found


def _assoc_from_context(words, content, vectors, transform=None):
    """Mean cosine to the previous up to 3 content words. Target is appended after scoring."""
    assoc = np.full(len(words), np.nan)
    n_ctx = np.zeros(len(words), dtype=int)
    self_index_in_ctx = np.zeros(len(words), dtype=bool)
    same_string_in_ctx = np.zeros(len(words), dtype=bool)
    prev_idx: list[int] = []
    prev_vec: list[np.ndarray | None] = []
    for i, (word, is_content) in enumerate(zip(words, content)):
        vec = vectors.get(word)
        if transform is not None and vec is not None:
            vec = transform(vec)
        if prev_idx and vec is not None:
            take_i = prev_idx[-3:]
            take_v = [v for v in prev_vec[-3:] if v is not None]
            self_index_in_ctx[i] = i in take_i
            same_string_in_ctx[i] = any(words[j] == word for j in take_i)
            n_ctx[i] = len(take_v)
            if take_v:
                sims = []
                vn = float(np.linalg.norm(vec))
                for v in take_v:
                    nn = float(np.linalg.norm(v))
                    if vn < 1e-8 or nn < 1e-8:
                        continue
                    sims.append(float(np.dot(vec, v) / (vn * nn)))
                if sims:
                    assoc[i] = float(np.mean(sims))
                    n_ctx[i] = len(sims)
        if is_content:
            usable = None
            if vec is not None and float(np.linalg.norm(vec)) > 1e-8:
                usable = vec
            prev_idx.append(i)
            prev_vec.append(usable)
    return assoc, n_ctx, self_index_in_ctx, same_string_in_ctx


def _pair_cosines(vecs: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    a = vecs[i]
    b = vecs[j]
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.clip(an * bn, 1e-8, None)


def audit_association(words: pd.DataFrame) -> dict:
    print("association audit", flush=True)
    vocab = set(words["word"].astype(str))
    vectors = _load_vectors(vocab)
    w = words["word"].astype(str).tolist()
    content = words["content"].astype(bool).to_numpy()
    raw, n_ctx, self_ix, same_str = _assoc_from_context(w, content, vectors)
    stored = words["association"].to_numpy(float)
    both = np.isfinite(raw) & np.isfinite(stored)
    max_abs = float(np.max(np.abs(raw[both] - stored[both]))) if both.any() else float("nan")
    # Analysis set matches the original complete-predictor content rows.
    analysis = words["analysis_complete"].astype(bool).to_numpy() & np.isfinite(raw)
    ctx_vals = raw[analysis]

    # In-vocabulary story types.
    type_rows = []
    for word, g in words.groupby(words["word"].astype(str)):
        if word not in vectors:
            continue
        type_rows.append((word, float(g["log_frequency"].mean())))
    type_words = [t[0] for t in type_rows]
    type_freq = np.array([t[1] for t in type_rows], dtype=float)
    type_index = {word: i for i, word in enumerate(type_words)}
    X = np.stack([vectors[word] for word in type_words]).astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    shrink = 1e-4 * (np.trace(cov) / cov.shape[0])
    cov = cov + shrink * np.eye(cov.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-8, None)
    W = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    Xw = Xc @ W

    def transform_center(v, mu=mu):
        return np.asarray(v, dtype=np.float64) - mu

    def transform_white(v, mu=mu, W=W):
        return (np.asarray(v, dtype=np.float64) - mu) @ W

    # Map transforms through a dict of precomputed vectors so OOV stays missing.
    vec_c = {word: Xc[i] for i, word in enumerate(type_words)}
    vec_w = {word: Xw[i] for i, word in enumerate(type_words)}
    assoc_c, _, _, _ = _assoc_from_context(w, content, vec_c)
    assoc_w, _, _, _ = _assoc_from_context(w, content, vec_w)
    ac = assoc_c[analysis]
    aw = assoc_w[analysis]

    # Context membership: token index -> set of type indices in its context.
    # Rebuild lightly for the null pairs (type-level, not token-level).
    neighbor_types: set[tuple[int, int]] = set()
    prev_types: list[int] = []
    for i, (word, is_content) in enumerate(zip(w, content)):
        ti = type_index.get(word)
        if ti is not None and prev_types:
            for pj in prev_types[-3:]:
                a, b = (ti, pj) if ti < pj else (pj, ti)
                if a != b:
                    neighbor_types.add((a, b))
        if is_content and ti is not None:
            prev_types.append(ti)

    n_types = len(type_words)
    # Random unrelated pairs.
    rand_i = []
    rand_j = []
    tries = 0
    while len(rand_i) < 5000 and tries < 200000:
        tries += 1
        a = int(RNG.integers(0, n_types))
        b = int(RNG.integers(0, n_types))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in neighbor_types:
            continue
        rand_i.append(key[0])
        rand_j.append(key[1])
    rand_i = np.asarray(rand_i)
    rand_j = np.asarray(rand_j)
    rand_cos = _pair_cosines(X, rand_i, rand_j)
    rand_c = _pair_cosines(Xc, rand_i, rand_j)
    rand_w = _pair_cosines(Xw, rand_i, rand_j)

    # Frequency-matched: same decile, not neighbors.
    dec = pd.qcut(type_freq, 10, labels=False, duplicates="drop")
    dec = np.asarray(dec, dtype=int)
    buckets: dict[int, np.ndarray] = {}
    for d in np.unique(dec):
        buckets[int(d)] = np.flatnonzero(dec == d)
    freq_i = []
    freq_j = []
    tries = 0
    dec_ids = list(buckets)
    while len(freq_i) < 5000 and tries < 300000:
        tries += 1
        d = int(dec_ids[int(RNG.integers(0, len(dec_ids)))])
        pool = buckets[d]
        if pool.size < 2:
            continue
        a, b = RNG.choice(pool, size=2, replace=False)
        a, b = int(a), int(b)
        key = (a, b) if a < b else (b, a)
        if key in neighbor_types:
            continue
        freq_i.append(key[0])
        freq_j.append(key[1])
    freq_i = np.asarray(freq_i)
    freq_j = np.asarray(freq_j)
    freq_cos = _pair_cosines(X, freq_i, freq_j)
    freq_c = _pair_cosines(Xc, freq_i, freq_j)
    freq_w = _pair_cosines(Xw, freq_i, freq_j)

    def _summ(x: np.ndarray) -> dict:
        x = x[np.isfinite(x)]
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)) if x.size else None,
            "sd": float(np.std(x, ddof=1)) if x.size > 1 else None,
            "p05": float(np.quantile(x, 0.05)) if x.size else None,
            "p50": float(np.quantile(x, 0.50)) if x.size else None,
            "p95": float(np.quantile(x, 0.95)) if x.size else None,
        }

    corr_c = float(np.corrcoef(ctx_vals, ac)[0, 1])
    corr_w = float(np.corrcoef(ctx_vals, aw)[0, 1])
    summary = {
        "embedding": str(EMBED),
        "n_vocab_requested": len(vocab),
        "n_vectors": len(vectors),
        "n_story_types_with_vectors": n_types,
        "recompute_vs_stored_max_abs": max_abs,
        "recompute_vs_stored_n": int(both.sum()),
        "target_index_in_own_context_n": int(self_ix.sum()),
        "same_string_previous_content_n": int(same_str.sum()),
        "analysis_raw": _summ(ctx_vals),
        "analysis_mean_centred": _summ(ac),
        "analysis_whitened": _summ(aw),
        "corr_raw_vs_mean_centred": corr_c,
        "corr_raw_vs_whitened": corr_w,
        "random_unrelated_raw": _summ(rand_cos),
        "random_unrelated_mean_centred": _summ(rand_c),
        "random_unrelated_whitened": _summ(rand_w),
        "freq_matched_raw": _summ(freq_cos),
        "freq_matched_mean_centred": _summ(freq_c),
        "freq_matched_whitened": _summ(freq_w),
        "compression_real": bool(
            np.mean(ctx_vals) > 0.8
            and np.mean(ctx_vals) > np.mean(rand_cos) + 0.05
        ),
        "context_sample": _context_sample(words),
        "note": (
            "Target token index is scored against prev and only then appended. "
            "same_string counts a repeated earlier content word, which is real context. "
            "context_sample lists the target string and the previous content strings."
        ),
    }
    pd.DataFrame(
        {
            "item_id": words.loc[analysis, "item_id"].to_numpy(),
            "association_raw": ctx_vals,
            "association_mean_centred": ac,
            "association_whitened": aw,
        }
    ).to_csv(OUT / "association_recompute_analysis.csv", index=False)
    (OUT / "association_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "recompute_vs_stored_max_abs", "analysis_raw", "random_unrelated_raw",
        "freq_matched_raw", "analysis_mean_centred", "analysis_whitened",
        "target_index_in_own_context_n", "compression_real",
    )}, indent=2), flush=True)
    return summary


def _ols(y, X):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[keep]
    X = X[keep]
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - k
    sigma2 = float(resid @ resid / df) if df > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return {
        "n": int(n),
        "df": int(df),
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t": t.tolist(),
        "p": p.tolist(),
    }


def audit_coverage() -> dict:
    print("coverage audit", flush=True)
    rej = pd.read_csv(STIM / "reject_summary.csv")
    slopes = pd.read_csv(STIM / "subject_slopes_cp.csv")
    slopes = slopes.rename(columns={"participant": "subject"})
    if "subject" not in slopes.columns:
        slopes = slopes.rename(columns={slopes.columns[0]: "subject"})
    df = rej.merge(slopes, on="subject", how="left", suffixes=("", "_slope"))
    # group column may duplicate
    if "group_slope" in df.columns:
        df["group"] = df["group"].fillna(df["group_slope"])
    df["group"] = df["group"].astype(str)
    rows = []
    for g, sub in df.groupby("group"):
        rows.append({
            "group": g,
            "n_subjects": int(len(sub)),
            "n_kept_mean": float(sub["n_kept"].mean()),
            "n_kept_sd": float(sub["n_kept"].std(ddof=1)),
            "n_kept_min": int(sub["n_kept"].min()),
            "n_kept_max": int(sub["n_kept"].max()),
            "artifact_rate_mean": float(sub["reject_rate_artifact_among_candidates"].mean()),
            "artifact_rate_sd": float(sub["reject_rate_artifact_among_candidates"].std(ddof=1)),
        })
    by_group = pd.DataFrame(rows)
    by_group.to_csv(OUT / "coverage_by_group.csv", index=False)
    df.to_csv(OUT / "coverage_by_child.csv", index=False)

    groups = sorted(df["group"].unique())
    # Kruskal-Wallis
    kw_kept = stats.kruskal(*[df.loc[df.group == g, "n_kept"].to_numpy() for g in groups])
    kw_rate = stats.kruskal(*[
        df.loc[df.group == g, "reject_rate_artifact_among_candidates"].to_numpy() for g in groups
    ])
    # OLS: slope ~ n_kept + group dummies (TD reference if present)
    y = df["sur_slope_cp"].to_numpy(float) if "sur_slope_cp" in df.columns else df.filter(like="slope").iloc[:, 0].to_numpy(float)
    slope_col = "sur_slope_cp" if "sur_slope_cp" in df.columns else df.filter(like="slope").columns[0]
    ref = "TD" if "TD" in groups else groups[0]
    dummies = [(df["group"] == g).astype(float).to_numpy() for g in groups if g != ref]
    names_g = [g for g in groups if g != ref]
    X_kept = np.column_stack([np.ones(len(df)), df["n_kept"].to_numpy(float), *dummies])
    X_rate = np.column_stack([
        np.ones(len(df)),
        df["reject_rate_artifact_among_candidates"].to_numpy(float),
        *dummies,
    ])
    fit_kept = _ols(y, X_kept)
    fit_rate = _ols(y, X_rate)
    # Also slope ~ predictor alone
    fit_kept_only = _ols(y, np.column_stack([np.ones(len(df)), df["n_kept"].to_numpy(float)]))
    fit_rate_only = _ols(y, np.column_stack([
        np.ones(len(df)),
        df["reject_rate_artifact_among_candidates"].to_numpy(float),
    ]))
    summary = {
        "slope_column": slope_col,
        "group_reference": ref,
        "group_dummies": names_g,
        "kruskal_n_kept": {"H": float(kw_kept.statistic), "p": float(kw_kept.pvalue)},
        "kruskal_artifact_rate": {"H": float(kw_rate.statistic), "p": float(kw_rate.pvalue)},
        "slope_on_n_kept_plus_group": fit_kept,
        "slope_on_artifact_rate_plus_group": fit_rate,
        "slope_on_n_kept_only": fit_kept_only,
        "slope_on_artifact_rate_only": fit_rate_only,
        "by_group": rows,
    }
    (OUT / "coverage_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "kruskal_n_kept", "kruskal_artifact_rate", "slope_on_n_kept_only", "slope_on_artifact_rate_only"
    )}, indent=2), flush=True)
    return summary


def audit_topography() -> dict:
    print("topography audit", flush=True)
    trials = pd.read_csv(STIM / "trials_all.csv", usecols=["subject", "section", "word_index", "keep"])
    kept = trials.loc[trials["keep"] == 1, ["subject", "section", "word_index"]]
    words = pd.read_csv(STIM / "word_table.csv", usecols=["section", "word_index", "onset_sample"])
    kept = kept.merge(words, on=["section", "word_index"], how="left")
    ch_ix = {ch: WORDLOCKED_CHAN_65.index(ch) for ch in TOPO_CH}
    # subject-equal grand mean
    acc = []
    subjects = sorted(kept["subject"].unique())
    window_rows = []
    for si, subject in enumerate(subjects, start=1):
        sub = kept.loc[kept["subject"] == subject]
        waves = []
        for section, g in sub.groupby("section"):
            sec_dir = SHARED / subject / f"section_{int(section):03d}"
            eeg = np.load(sec_dir / "eeg_data.npy", mmap_mode="r")
            if eeg.shape[0] != 65:
                raise SystemExit(f"{subject} section {section} n_ch {eeg.shape[0]}")
            n_samples = eeg.shape[1]
            ix = np.array([ch_ix[ch] for ch in TOPO_CH])
            for onset in g["onset_sample"].to_numpy(int):
                start = int(onset) - PRE
                end = start + N_T
                if start < 0 or end > n_samples:
                    continue
                ep = np.asarray(eeg[ix, start:end], dtype=np.float64)
                ep = ep - ep[:, BASE_IX].mean(axis=1, keepdims=True)
                waves.append(ep)
        if not waves:
            print(f"  {subject} no epochs", flush=True)
            continue
        mean_ep = np.mean(np.stack(waves, axis=0), axis=0)  # ch x time
        acc.append(mean_ep)
        win_mean = mean_ep[:, WIN].mean(axis=1)
        window_rows.append({"subject": subject, **{ch: float(win_mean[i]) for i, ch in enumerate(TOPO_CH)}})
        if si % 10 == 0:
            print(f"  topography {si}/{len(subjects)}", flush=True)
    grand = np.mean(np.stack(acc, axis=0), axis=0)
    grand_win = grand[:, WIN].mean(axis=1)
    # posterior vs frontal
    name_to_i = {ch: i for i, ch in enumerate(TOPO_CH)}
    posterior = float(np.mean([grand_win[name_to_i["Pz"]], grand_win[name_to_i["Oz"]]]))
    frontal = float(grand_win[name_to_i["Fz"]])
    lateral = float(np.mean([grand_win[name_to_i[ch]] for ch in ("F7", "T7", "T8", "P7")]))
    summary = {
        "n_subjects": len(acc),
        "channels": TOPO_CH,
        "indices": ch_ix,
        "montage_matches_decoding_interpretation": True,
        "window_s": [0.35, 0.55],
        "grand_window_mean": {ch: float(grand_win[i]) for i, ch in enumerate(TOPO_CH)},
        "pz_oz_mean": posterior,
        "fz": frontal,
        "lateral_mean_F7_T7_T8_P7": lateral,
        "pz_oz_minus_fz": posterior - frontal,
        "pz_minus_fz": float(grand_win[name_to_i["Pz"]] - grand_win[name_to_i["Fz"]]),
        "oz_minus_fz": float(grand_win[name_to_i["Oz"]] - grand_win[name_to_i["Fz"]]),
        "note": (
            "Subject-equal mean of baseline-corrected kept content-word epochs. "
            "A classic N400 is a relative posterior negativity, so Pz/Oz minus Fz should be negative."
        ),
    }
    np.savez_compressed(
        OUT / "topography_grand.npz",
        times=TIMES.astype(np.float32),
        channels=np.array(TOPO_CH),
        grand=grand.astype(np.float32),
    )
    pd.DataFrame(window_rows).to_csv(OUT / "topography_subject_window.csv", index=False)
    (OUT / "topography_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["grand_window_mean"], indent=2), flush=True)
    print("pz_oz_minus_fz", summary["pz_oz_minus_fz"], flush=True)
    return summary


def audit_eeg_provenance() -> dict:
    # Amplitude on three subjects, both sections. Extractor behaviour is documented
    # from encoding_final.ipynb (SectionExtractorWordLocked.load_eeg_data).
    samples = []
    for subject in ("RN102", "D001d", "D019d"):
        for section in (1, 2):
            eeg = np.load(SHARED / subject / f"section_{section:03d}" / "eeg_data.npy", mmap_mode="r")
            meta = json.loads((SHARED / subject / f"section_{section:03d}" / "metadata.json").read_text())
            sl = np.asarray(eeg[:, ::100], dtype=np.float64)
            samples.append({
                "subject": subject,
                "section": section,
                "shape": list(eeg.shape),
                "dtype": str(eeg.dtype),
                "sfreq": meta.get("sfreq"),
                "channel_names_in_metadata": "channel_names" in meta,
                "std": float(np.std(sl)),
                "mean": float(np.mean(sl)),
                "min": float(np.min(sl)),
                "max": float(np.max(sl)),
            })
    summary = {
        "extractor": (
            "encoding_final.ipynb cell 0, class SectionExtractorWordLocked, "
            "methods load_eeg_data / process_participant / save_sections. "
            "MAIN sets target_sfreq=100, trigger_delay=1.0, output_dir=extracted_sections_wordlocked_shared."
        ),
        "load_path_preference": "preprocessed_for_isc.set if present, else data.bdf",
        "channels_dropped_if_present": ["1", "2", "3", "4", "5", "6", "A1", "A2"],
        "extractor_rereference": False,
        "extractor_bandpass": False,
        "extractor_resample_hz": 100,
        "units_rule": "if std(raw.get_data()) < 1e-3, multiply by 1e6 (V to uV); else leave as in the .set",
        "channel_names_saved": False,
        "wordlocked_chan_65_equals_decoding_interpretation_CHANNEL_NAMES": True,
        "indices": {"Fz": 7, "Cz": 27, "Pz": 46, "Oz": 62},
        "upstream_set_on_pool": False,
        "amplitude_samples": samples,
        "units_judgement": (
            "Stored amplitudes have SD about 20-40 and peaks of a few hundred, "
            "which matches microvolts after the extractor's Volt-to-microvolt rule, "
            "not raw volts. The extractor does not re-reference or bandpass. "
            "Any reference or filter is inside preprocessed_for_isc.set and is not in metadata.json. "
            "That .set file was not found on the pool."
        ),
    }
    (OUT / "eeg_provenance.json").write_text(json.dumps(summary, indent=2))
    return summary


def audit_behaviour_direction() -> dict:
    """Direction from test notes only. Never from the correlation with 识字量."""
    print("behaviour direction audit", flush=True)
    xlsx = POOL / "dyslexia_natualistics_listing" / "behavioral_data.xlsx"
    searched = [
        str(xlsx),
        str(POOL / "dyslexia_natualistics_listing" / "README.md"),
        str(POOL / "dyslexia_natualistics_listing" / "b2b_decoding" / "README.md"),
    ]
    notes_found = []
    header_units = {}
    if xlsx.is_file():
        raw = pd.read_excel(xlsx, sheet_name="Sheet1")
        raw = raw.rename(columns={c: str(c).strip() for c in raw.columns})
        measure = raw["测试项目"].astype(str).str.strip()
        targets = [
            "快速命名-数字",
            "快速命名-混合",
            "快速命名-物体",
            "快速命名-颜色",
            "短文阅读",
        ]
        for name in targets:
            sub = raw.loc[measure == name]
            remarks = []
            if "备注" in raw.columns:
                remarks = sorted({str(v).strip() for v in sub["备注"].dropna().unique() if str(v).strip()})
            category = sorted({str(v).strip() for v in sub["测试类别"].dropna().unique()}) if "测试类别" in raw.columns else []
            header_units[name] = {
                "n_rows": int(len(sub)),
                "category": category,
                "header_contains_unit": any(tok in name for tok in ("时间", "秒", "正确", "分数", "门限")),
                "n_nonempty_remarks": len(remarks),
                "remark_examples": remarks[:8],
            }
            # Remarks that state a scoring rule, not a child's percent or attention note.
            for text in remarks:
                if any(tok in text for tok in ("越高", "越低", "时间", "秒", "正确率", "分数越高", "lower", "higher")):
                    notes_found.append({"measure": name, "text": text})
    summary = {
        "searched": searched,
        "workbook_columns": ["测试类别", "测试项目", "测试结果", "备注"],
        "header_units": header_units,
        "scoring_notes_found": notes_found,
        "disallowed_source": "correlation with 识字量 is not used to set direction",
        "快速命名": {
            "direction": "unresolved",
            "reason": (
                "Sheet1 headers do not state a unit. 备注 has no scoring rule for these rows. "
                "No test manual was found next to behavioral_data.xlsx. "
                "The original results inferred lower-better from a seconds-like range plus the "
                "识字量 correlation; that correlation is not used here."
            ),
        },
        "短文阅读": {
            "direction": "unresolved",
            "reason": (
                "Sheet1 category is 阅读能力 and the header has no unit. "
                "No scoring note was found. Direction is not taken from the 识字量 correlation."
            ),
        },
    }
    (OUT / "behaviour_direction.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("快速命名", "短文阅读")}, ensure_ascii=False, indent=2), flush=True)
    return summary


def _context_sample(words: pd.DataFrame, n: int = 8) -> list[dict]:
    """Show that the current token is scored against earlier content words, then appended."""
    w = words["word"].astype(str).tolist()
    content = words["content"].astype(bool).to_numpy()
    analysis = words["analysis_complete"].astype(bool).to_numpy()
    prev: list[str] = []
    rows = []
    for i, (word, is_content, keep) in enumerate(zip(w, content, analysis)):
        ctx = prev[-3:]
        if keep and ctx and len(rows) < n:
            rows.append({
                "item_id": str(words["item_id"].iloc[i]),
                "target": word,
                "context": ctx,
                "target_is_in_context_strings": word in ctx,
                "target_index_appended_before_score": False,
            })
        if is_content:
            prev.append(word)
    return rows


def main() -> None:
    import os
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refusing to audit outside Slurm. Submit run_audit.sbatch.")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "provenance"):
        audit_eeg_provenance()
    if which in ("all", "coverage"):
        audit_coverage()
    if which in ("all", "association"):
        words = pd.read_csv(STIM / "word_table.csv")
        audit_association(words)
    if which in ("all", "topography"):
        audit_topography()
    if which in ("all", "direction"):
        audit_behaviour_direction()
    print("audit done", flush=True)


if __name__ == "__main__":
    main()
