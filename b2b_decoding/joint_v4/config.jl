# =====================================================================
# Joint-family B2B v4 configuration (ENV-overridable paths and knobs)
# Whisper Large V2 + onset-locked EEG; independent family PCA by default
# =====================================================================

using SHA
using Dates
using JSON3
using CSV
using DataFrames

const B2B_EXTRACTOR_DIR = get(ENV, "B2B_EXTRACTOR_DIR",
    "/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared")
const B2B_OUTDIR = get(ENV, "B2B_OUTDIR",
    "/home/haolun52/orcd/pool/encoding_results_b2b_largev2_onset_joint_k8")
const B2B_BASIS_DIR = get(ENV, "B2B_BASIS_DIR",
    joinpath(B2B_OUTDIR, "_basis"))

const B2B_MODE = lowercase(get(ENV, "B2B_MODE", "observed"))  # observed|null|both
const B2B_PCA_K = parse(Int, get(ENV, "B2B_PCA_K", "8"))
const B2B_PCA_FIT_MAX = parse(Int, get(ENV, "B2B_PCA_FIT_MAX", "16"))
const B2B_N_NULL = parse(Int, get(ENV, "B2B_N_NULL", "200"))
const B2B_NULL_START = parse(Int, get(ENV, "B2B_NULL_START", "1"))
const B2B_NULL_COUNT = parse(Int, get(ENV, "B2B_NULL_COUNT", string(B2B_N_NULL)))
const B2B_SEED = parse(Int, get(ENV, "B2B_SEED", "42"))

const SECTIONS = (1, 2)

# Configurable Whisper Large V2 feature keys (text-only is sensitivity-only).
# Lexical King-style arm reuses these three slots, e.g.:
#   B2B_FEAT_ACOUSTIC=lexical_duration
#   B2B_FEAT_SPEECH=lexical_frequency
#   B2B_FEAT_LANGUAGE=gpt2cn_surprisal
#   B2B_FAMILY_NAMES=duration,frequency,surprisal
#   B2B_PCA_K=1
const FEAT_ACOUSTIC = get(ENV, "B2B_FEAT_ACOUSTIC", "whisperlargev2_acoustic")
const FEAT_SPEECH   = get(ENV, "B2B_FEAT_SPEECH", "whisperlargev2_speech")
const FEAT_LANGUAGE = get(ENV, "B2B_FEAT_LANGUAGE", "whisperlargev2_language_audio_fused")
const FEAT_LANGUAGE_TEXTONLY = get(ENV, "B2B_FEAT_LANGUAGE_TEXTONLY",
    "whisperlargev2_language_text_only")
# Override the three FEAT_* slots with an explicit comma-separated list
# (HDC pilot: B2B_FEATURE_SETS=hdc_phonetic,hdc_word_form,...).
const FEATURE_SETS = begin
    raw = strip(get(ENV, "B2B_FEATURE_SETS", ""))
    if isempty(raw)
        [FEAT_ACOUSTIC, FEAT_SPEECH, FEAT_LANGUAGE]
    else
        # Accept comma / semicolon / colon (colon preferred for sbatch --export).
        String[strip(x) for x in split(raw, r"[,;:]") if !isempty(strip(x))]
    end
end
const FAMILY_NAMES  = begin
    raw = strip(get(ENV, "B2B_FAMILY_NAMES", "acoustic,speech,language"))
    fam_names = String[strip(x) for x in split(raw, r"[,;:]") if !isempty(strip(x))]
    length(fam_names) == length(FEATURE_SETS) ||
        error("B2B_FAMILY_NAMES ($fam_names) must match FEATURE_SETS length $(length(FEATURE_SETS))")
    fam_names
end
# Skip PCA: standardize each family and keep all columns (HDC linguistic features).
const B2B_PASSTHROUGH = lowercase(get(ENV, "B2B_PASSTHROUGH", "0")) in
    ("1", "true", "yes")
# Optional TG (train G at t, evaluate at t'). 1 = every 2nd sample (~50 Hz).
const B2B_DO_TG = lowercase(get(ENV, "B2B_DO_TG", "0")) in ("1", "true", "yes")
const B2B_TG_STRIDE = parse(Int, get(ENV, "B2B_TG_STRIDE", "2"))
# Ridge on H only for collinear linguistic designs (Gwilliams 2025). Whisper
# path keeps unregularized OLS. 0 = OLS; >0 = shrink until cond(X'X+λI) ≤ this.
const B2B_H_RIDGE_KAPPA = parse(Float64, get(ENV, "B2B_H_RIDGE_KAPPA",
    B2B_PASSTHROUGH ? "32" : "0"))
# Low-SNR EEG preprocessing (child 20-min recordings). Defaults preserve legacy.
# Spatial: none | pca | dss (DSS uses word-locked average as bias filter).
const B2B_SPATIAL_DENOISE = lowercase(get(ENV, "B2B_SPATIAL_DENOISE", "none"))
(B2B_SPATIAL_DENOISE in ("none", "pca", "dss")) ||
    error("B2B_SPATIAL_DENOISE must be none|pca|dss; got $(B2B_SPATIAL_DENOISE)")
const B2B_SPATIAL_N_COMP = parse(Int, get(ENV, "B2B_SPATIAL_N_COMP", "40"))
B2B_SPATIAL_N_COMP >= 1 || error("B2B_SPATIAL_N_COMP must be >= 1")
# Temporal bin width in ms (0 = off). 20 ms → 50 Hz effective rate at 100 Hz sfreq.
const B2B_TIME_BIN_MS = parse(Float64, get(ENV, "B2B_TIME_BIN_MS", "0"))
B2B_TIME_BIN_MS >= 0 || error("B2B_TIME_BIN_MS must be >= 0")
# When spatial denoise is on: fit once on all kept epochs (default), or re-fit
# per outer partition on the train half only (leakage control).
const B2B_DSS_FIT_SCOPE = lowercase(get(ENV, "B2B_DSS_FIT_SCOPE", "all"))
(B2B_DSS_FIT_SCOPE in ("all", "train_half")) ||
    error("B2B_DSS_FIT_SCOPE must be all|train_half; got $(B2B_DSS_FIT_SCOPE)")
# If set, tune ridge α on every time sample (more stable λ under low SNR).
const B2B_LAMBDA_TUNE_ALL = lowercase(get(ENV, "B2B_LAMBDA_TUNE_ALL", "0")) in
    ("1", "true", "yes")
# Phase 3b/3c: filter continuous EEG before epoching; optional channel subset.
# broad = no extra filter (frozen default). delta = 0.5–4 Hz, theta = 4–8 Hz.
const B2B_EEG_BAND = lowercase(get(ENV, "B2B_EEG_BAND", "broad"))
(B2B_EEG_BAND in ("broad", "delta", "theta")) ||
    error("B2B_EEG_BAND must be broad|delta|theta; got $(B2B_EEG_BAND)")
const B2B_CHAN_SET = lowercase(get(ENV, "B2B_CHAN_SET", "all"))
(B2B_CHAN_SET in ("all", "left_temporal", "right_temporal")) ||
    error("B2B_CHAN_SET must be all|left_temporal|right_temporal; got $(B2B_CHAN_SET)")
const B2B_ALLOW_INPLACE_NULLS = lowercase(get(ENV, "B2B_ALLOW_INPLACE_NULLS", "0")) in
    ("1", "true", "yes")

const EXPECTED_FEAT_DIMS = Dict{String,Int}(
    "whisperlargev2_acoustic" => 12800,
    "whisperlargev2_speech" => 12800,
    "whisperlargev2_acoustic_wpca" => 1280,
    "whisperlargev2_speech_wpca" => 1280,
    "whisperlargev2_language_audio_fused" => 1280,
    "whisperlargev2_language_text_only" => 1280,
    "whisperlargev3_acoustic" => 12800,
    "whisperlargev3_speech" => 12800,
    "whisperlargev3_language_audio_fused" => 1280,
    "whisperlargev3_language_text_only" => 1280,
    "whispertiny_acoustic" => 3840,
    "whispertiny_speech" => 3840,
    "whispertiny_acoustic_wpca" => 384,
    "whispertiny_speech_wpca" => 384,
    "whispertiny_language_audio_fused" => 384,
    "whispertiny_language_text_only" => 384,
    # Lexical King-style scalars (onset-locked duration / logfreq / GPT2CN surprisal)
    "lexical_duration" => 1,
    "lexical_frequency" => 1,
    "gpt2cn_surprisal" => 1,
    "word_onset" => 4,
    "word_offset" => 4,
    "offset" => 4,
    "tone" => 5,   # T1–T4 one-hot + F0-contour RMSE
    "tone_v2" => 6,  # T1–T4 + within-tone z-RMSE + valid flag
    "tone_v3" => 6,  # T1–T4 + Legendre-space z-dev + valid flag
    "ctx_controls" => 39,
    "ctx_pitch" => 5,
    "ctx_evidence" => 3,
    "ctx_resid" => 3,
    "pitch" => 13,
    # Static / LM embeddings (speech vs GPT2-CN vs GloVe arm)
    "glove" => 300,
    "gpt2_l24" => 1600,
    "qwen_l18" => 4096,
    "qwen_l24" => 4096,
    "qwen_l36" => 4096,
    # HDC pilot (wordinfo + GloVe PCA-10; pos_PUNCT dropped)
    "hdc_phonetic" => 13,   # ph_vowel dropped (manner sum-to-1)
    "hdc_word_form" => 4,   # wf_n_morphemes dropped (≡ n_syllables)
    "hdc_lexical_syntactic" => 14,
    "hdc_syntax_proxy" => 3,
    "hdc_semantic" => 10,
    "hdc_syntactic_operation" => 3,
    "hdc_syntactic_state" => 7,
)
# Tagged / pruned HDC variants (dims filled at prepare time; allow soft check).
const B2B_STRICT_FEAT_DIMS = lowercase(get(ENV, "B2B_STRICT_FEAT_DIMS", "1")) in
    ("1", "true", "yes")

# Scalar / 1-d predictors may legitimately be ~0 (e.g. near-certain surprisal).
# HDC POS one-hots can be all-zero on a row after rare-tag drop.
const B2B_ALLOW_ZERO_FEATURE_ROWS = lowercase(get(ENV, "B2B_ALLOW_ZERO_FEATURE_ROWS",
    (any(f -> get(EXPECTED_FEAT_DIMS, f, 0) == 1, FEATURE_SETS) || B2B_PASSTHROUGH) ? "1" : "0")) in
    ("1", "true", "yes")

const B2B_EPOCH_ANCHOR = lowercase(get(ENV, "B2B_EPOCH_ANCHOR", "onset"))  # onset|offset
(B2B_EPOCH_ANCHOR in ("onset", "offset")) ||
    error("B2B_EPOCH_ANCHOR must be onset|offset; got $(B2B_EPOCH_ANCHOR)")

const B2B_BASIS_MODE = lowercase(get(ENV, "B2B_BASIS_MODE", "independent"))  # independent|ordered
(B2B_BASIS_MODE in ("independent", "ordered")) ||
    error("B2B_BASIS_MODE must be independent|ordered; got $(B2B_BASIS_MODE)")

const B2B_INCLUDE_NUISANCE = lowercase(get(ENV, "B2B_INCLUDE_NUISANCE", "0")) in
    ("1", "true", "yes")
const NUISANCE_NAMES = ["word_duration", "log_frequency", "story_position",
                        "section_indicator", "char_length"]

const TMIN_S = parse(Float64, get(ENV, "B2B_TMIN_S", "-0.2"))
const TMAX_S = parse(Float64, get(ENV, "B2B_TMAX_S", "1.0"))
const EXPECTED_SFREQ = 100.0
const EXPECTED_N_CH  = 65
# Defaults are the frozen word-locked campaign. Nucleus-locked tone_ctx_v1
# overrides via B2B_EXPECTED_N_WORDS / B2B_EXPECTED_SEC_LENGTHS.
const EXPECTED_N_WORDS = parse(Int, get(ENV, "B2B_EXPECTED_N_WORDS", "3553"))
const EXPECTED_SEC_LENGTHS = begin
    raw = strip(get(ENV, "B2B_EXPECTED_SEC_LENGTHS", "1753,1800"))
    vals = [parse(Int, strip(x)) for x in split(raw, r"[,;:]") if !isempty(strip(x))]
    length(vals) == 2 || error("B2B_EXPECTED_SEC_LENGTHS must have 2 integers; got $(vals)")
    (vals[1], vals[2])
end
const SAMPLE_INDEX_BASE = 0

const PTP_THRESH_UV   = 200.0
const PTP_SOFT_UV     = 150.0
const MAX_BAD_CH_FRAC = 0.20
const ZSCORE_ABS_MAX  = 6.0
const MIN_KEEP_WORDS  = 200
const REQUIRE_FULL_WINDOW_MASK = true
const BASELINE_CORRECT = false
const BASELINE_WIN_S   = (-0.2, 0.0)

const N_BLOCKS_PER_SECTION = parse(Int, get(ENV, "B2B_N_BLOCKS", "8"))
const GUARD_WORDS          = parse(Int, get(ENV, "B2B_GUARD_WORDS", "5"))
const N_OUTER_PARTITIONS   = parse(Int, get(ENV, "B2B_N_PARTITIONS", "20"))
const N_INNER_FOLDS        = 4
const MIN_VALID_PARTITIONS = 16
const KAPPA_WARN  = 32.0
const KAPPA_INVALID = 85.0
const B2B_DRIFT_KNOTS = parse(Int, get(ENV, "B2B_DRIFT_KNOTS", "40"))

# Normalized ridge grid: λ = α * trace(Y'Y) / n_channels
const ALPHA_GRID = exp10.(collect(-8.0:0.5:8.0))
# Include epoch endpoints when B2B_TMIN_S / B2B_TMAX_S differ from defaults.
const LAMBDA_TUNE_TIMES_S = begin
    base = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    times = unique(sort(vcat(base, [TMIN_S, TMAX_S])))
    filter(t -> TMIN_S - 1e-9 <= t <= TMAX_S + 1e-9, times)
end

const NULL_MIN_SHIFT_WORDS = parse(Int, get(ENV, "B2B_NULL_MIN_SHIFT", "80"))
const CONFIRM_WIN_S = (0.0, 0.8)
# Negative-control window tracks pre-onset by default; override with
# B2B_NEGCTRL_WIN="t0,t1" (e.g. "-0.2,0.0").
const NEGCTRL_WIN_S = begin
    raw = strip(get(ENV, "B2B_NEGCTRL_WIN", ""))
    if isempty(raw)
        (TMIN_S, 0.0)
    else
        parts = [strip(x) for x in split(raw, ",") if !isempty(strip(x))]
        length(parts) == 2 || error("B2B_NEGCTRL_WIN must be \"t0,t1\"; got $raw")
        (parse(Float64, parts[1]), parse(Float64, parts[2]))
    end
end

const COHORT_CSV = get(ENV, "B2B_COHORT_CSV",
    joinpath(@__DIR__, "cohort_groups.csv"))

const DO_PLOT = lowercase(get(ENV, "B2B_NO_PLOT", "0")) ∉ ("1", "true", "yes")

function load_cohort(path::AbstractString=COHORT_CSV)
    df = CSV.read(path, DataFrame)
    required = ["participant", "group", "include_primary"]
    for c in required
        c in names(df) || error("cohort CSV missing column: $c")
    end
    df = df[Bool.(df.include_primary), :]
    participants = String.(df.participant)
    length(unique(participants)) == length(participants) ||
        error("Duplicate participants in cohort table")
    groups = String.(df.group)
    counts = Dict(g => count(==(g), groups) for g in unique(groups))
    expected = Dict(
        "TD" => 24,
        "dyslexia_normal_CAP" => 19,
        "dyslexia_atypical_CAP" => 20,
    )
    for (g, n) in expected
        get(counts, g, 0) == n ||
            error("Group $g has $(get(counts, g, 0)) participants; expected $n")
    end
    length(participants) == 63 ||
        error("Expected 63 primary participants; got $(length(participants))")
    return df, participants
end

function file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function cohort_hash(path::AbstractString=COHORT_CSV)
    isfile(path) || error("Cohort CSV not found: $path")
    return file_sha256(path)
end

function git_commit_hash(; repo_dir::AbstractString=joinpath(@__DIR__, "..", ".."))
    try
        return strip(read(`git -C $repo_dir rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

function slurm_job_id()
    get(ENV, "SLURM_JOB_ID", get(ENV, "SLURM_ARRAY_JOB_ID", ""))
end

function software_versions()
    Dict(
        "julia" => string(VERSION),
        "date_utc" => string(Dates.now(Dates.UTC)),
        "hostname" => gethostname(),
        "git_commit" => git_commit_hash(),
        "slurm_job_id" => slurm_job_id(),
    )
end

"""Locked analysis config used to refuse mismatched appends to an outdir."""
function make_config_fingerprint()
    Dict{String,Any}(
        "pipeline_version" => "joint_v4",
        "feat_acoustic" => FEAT_ACOUSTIC,
        "feat_speech" => FEAT_SPEECH,
        "feat_language" => FEAT_LANGUAGE,
        "family_names" => FAMILY_NAMES,
        "feature_sets" => FEATURE_SETS,
        "epoch_anchor" => B2B_EPOCH_ANCHOR,
        "tmin_s" => TMIN_S,
        "tmax_s" => TMAX_S,
        "confirm_win_s" => collect(CONFIRM_WIN_S),
        "negctrl_win_s" => collect(NEGCTRL_WIN_S),
        "basis_mode" => B2B_BASIS_MODE,
        "pca_k" => B2B_PCA_K,
        "pca_fit_max" => B2B_PCA_FIT_MAX,
        "passthrough" => B2B_PASSTHROUGH,
        "h_ridge_kappa" => B2B_H_RIDGE_KAPPA,
        # do_tg is additive (diagonal-only outdir may later grow TG files).
        "tg_stride" => B2B_TG_STRIDE,
        "spatial_denoise" => B2B_SPATIAL_DENOISE,
        "spatial_n_comp" => B2B_SPATIAL_N_COMP,
        "time_bin_ms" => B2B_TIME_BIN_MS,
        "dss_fit_scope" => B2B_DSS_FIT_SCOPE,
        "lambda_tune_all" => B2B_LAMBDA_TUNE_ALL,
        "eeg_band" => B2B_EEG_BAND,
        "chan_set" => B2B_CHAN_SET,
        "drift_n_knots" => B2B_DRIFT_KNOTS,
        "include_nuisance" => B2B_INCLUDE_NUISANCE,
        "nuisance_names" => B2B_INCLUDE_NUISANCE ? NUISANCE_NAMES : String[],
        "n_null" => B2B_N_NULL,
        "seed" => B2B_SEED,
        "n_partitions" => N_OUTER_PARTITIONS,
        "guard_words" => GUARD_WORDS,
        "n_blocks_per_section" => N_BLOCKS_PER_SECTION,
        "cohort_hash" => isfile(COHORT_CSV) ? cohort_hash() : "",
        "extractor_dir" => B2B_EXTRACTOR_DIR,
        "claim" => "conditional linear alignment of multichannel EEG with retained " *
                   "acoustic/speech/language embedding subspaces (independent PCA default)",
    )
end

function _canon_val(x)
    if x === nothing
        return "null"
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x isa Number
        return string(Float64(x))
    else
        return string(x)
    end
end

function _fingerprint_equal(a::AbstractDict, b::AbstractDict)
    keys_a = sort(string.(collect(keys(a))))
    keys_b = sort(string.(collect(keys(b))))
    keys_a == keys_b || return false
    for k in keys_a
        va = a[k]
        vb = haskey(b, k) ? b[k] : b[Symbol(k)]
        if va isa AbstractDict && vb isa AbstractDict
            _fingerprint_equal(Dict{String,Any}(string(kk) => vv for (kk, vv) in va),
                               Dict{String,Any}(string(kk) => vv for (kk, vv) in vb)) || return false
        elseif va isa AbstractVector && vb isa AbstractVector
            length(va) == length(vb) || return false
            for (x, y) in zip(va, vb)
                _canon_val(x) == _canon_val(y) || return false
            end
        else
            _canon_val(va) == _canon_val(vb) || return false
        end
    end
    return true
end

"""Write or validate the outdir-level config manifest. Refuse mismatched appends."""
function assert_outdir_manifest!(outdir::AbstractString=B2B_OUTDIR)
    mkpath(outdir)
    path = joinpath(outdir, "_analysis_manifest.json")
    current = make_config_fingerprint()
    current_full = merge(Dict{String,Any}(current), software_versions())
    current_full["basis_dir"] = B2B_BASIS_DIR
    current_full["outdir"] = outdir
    existing = nothing
    if isfile(path) && filesize(path) > 0
        try
            existing = JSON3.read(read(path, String), Dict{String,Any})
        catch err
            @warn "Corrupt analysis manifest; rewriting" path = path exception = err
        end
    end
    if existing === nothing
        write_json(path, current_full)
    else
        # Compare locked scientific fields only
        locked_keys = keys(make_config_fingerprint())
        ex_locked = Dict{String,Any}(string(k) => existing[k] for k in locked_keys if haskey(existing, k))
        # Backward-compat defaults for knobs added after some outdirs were created.
        get!(ex_locked, "spatial_denoise", "none")
        get!(ex_locked, "spatial_n_comp", 40)
        get!(ex_locked, "time_bin_ms", 0.0)
        get!(ex_locked, "dss_fit_scope", "all")
        get!(ex_locked, "lambda_tune_all", false)
        get!(ex_locked, "eeg_band", "broad")
        get!(ex_locked, "chan_set", "all")
        cur_locked = make_config_fingerprint()
        if !_fingerprint_equal(ex_locked, cur_locked)
            error("Refusing to append to $outdir: analysis manifest differs from current config.\n" *
                  "Existing: $path\n" *
                  "Use a new B2B_OUTDIR for a different configuration.")
        end
    end
    return path
end

function make_run_manifest(; kwargs...)
    d = Dict{String,Any}(string(k) => v for (k, v) in kwargs)
    merge!(d, software_versions())
    merge!(d, make_config_fingerprint())
    d["extractor_dir"] = B2B_EXTRACTOR_DIR
    d["outdir"] = B2B_OUTDIR
    d["basis_dir"] = B2B_BASIS_DIR
    d["mode"] = B2B_MODE
    d["null_start"] = B2B_NULL_START
    d["null_count"] = B2B_NULL_COUNT
    d["alpha_grid"] = ALPHA_GRID
    d["lambda_tune_times_s"] = LAMBDA_TUNE_TIMES_S
    if isfile(COHORT_CSV)
        d["group_table_hash"] = cohort_hash()
        d["group_table_path"] = COHORT_CSV
    end
    meta_path = joinpath(B2B_BASIS_DIR, "feature_basis_meta.json")
    if isfile(meta_path)
        d["feature_basis_hash"] = file_sha256(meta_path)
        npz_path = joinpath(B2B_BASIS_DIR, "feature_basis.npz")
        isfile(npz_path) && (d["feature_basis_npz_hash"] = file_sha256(npz_path))
        meta = JSON3.read(read(meta_path, String), Dict{String,Any})
        if haskey(meta, "feature_checksums")
            d["feature_checksums"] = meta["feature_checksums"]
        end
    end
    return d
end

function write_json(path::AbstractString, obj)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    tmp = joinpath(dir, "." * basename(path) * ".tmp.$(getpid()).$(time_ns())")
    open(tmp, "w") do io
        JSON3.write(io, obj)
    end
    mv(tmp, path; force=true)
end
