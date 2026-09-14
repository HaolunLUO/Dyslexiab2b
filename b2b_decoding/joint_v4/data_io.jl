# =====================================================================
# Data loading + epoch QC (joint B2B v4)
# Epoch anchor selectable via B2B_EPOCH_ANCHOR=onset|offset
# =====================================================================

using DataFrames
using NPZ
using JSON3
using CSV
using Statistics
using Printf
using SHA

to_julia_sample(s::Integer) = Int(s) - SAMPLE_INDEX_BASE + 1

function load_section_eeg(extractor_dir::AbstractString, participant::AbstractString,
                          section_id::Integer)
    sec_dir = joinpath(extractor_dir, participant, @sprintf("section_%03d", section_id))
    isdir(sec_dir) || error("Section dir not found: $sec_dir")
    eeg_path = joinpath(sec_dir, "eeg_data.npy")
    isfile(eeg_path) || error("EEG not found: $eeg_path")
    eeg = Float64.(NPZ.npzread(eeg_path))
    ndims(eeg) == 2 || error("EEG has unexpected shape: $(size(eeg))")
    if size(eeg, 1) > size(eeg, 2)
        eeg = permutedims(eeg)
    end
    train_mask = nothing
    mask_path = joinpath(sec_dir, "train_keep_mask.npy")
    if isfile(mask_path)
        train_mask = Bool.(vec(NPZ.npzread(mask_path)))
    end
    meta_path = joinpath(sec_dir, "metadata.json")
    meta = isfile(meta_path) ? JSON3.read(read(meta_path, String), Dict) : Dict()
    return eeg, train_mask, meta
end

function load_section_word_timing(extractor_dir, participant, section_id; sfreq=100.0)
    sec_dir = joinpath(extractor_dir, participant, @sprintf("section_%03d", section_id))
    csv_path = joinpath(sec_dir, "word_events.csv")
    df = if isfile(csv_path)
        CSV.read(csv_path, DataFrame)
    else
        shared_csv = joinpath(extractor_dir, "_shared_wordlocked_features",
                              @sprintf("section_%03d", section_id), "word_timing_relative.csv")
        isfile(shared_csv) || error("No word timing for $participant section $section_id")
        CSV.read(shared_csv, DataFrame)
    end
    if !("onset_sample" in names(df)) && ("onset_relative" in names(df))
        df.onset_sample = round.(Int, df.onset_relative .* sfreq)
    end
    if !("offset_sample" in names(df)) && ("offset_relative" in names(df))
        df.offset_sample = round.(Int, df.offset_relative .* sfreq)
    end
    "onset_sample" in names(df) || error("Word timing missing onset_sample")
    "offset_sample" in names(df) || error("Word timing missing offset_sample")
    on = Int.(df.onset_sample)
    off = Int.(df.offset_sample)
    all(diff(on) .>= 0) || error("onset_sample not monotonic in section $section_id")
    all(diff(off) .>= 0) || error("offset_sample not monotonic in section $section_id")
    return df
end

function load_shared_word_timing(extractor_dir::AbstractString, section_id::Integer)
    shared_csv = joinpath(extractor_dir, "_shared_wordlocked_features",
                          @sprintf("section_%03d", section_id), "word_timing_relative.csv")
    isfile(shared_csv) || error("Shared word timing not found: $shared_csv")
    return CSV.read(shared_csv, DataFrame)
end

"""Validate a feature matrix: finite, not all-zero rows, expected dims."""
function validate_feature_matrix!(X::AbstractMatrix, feat::AbstractString, section_id::Integer)
    n, p = size(X)
    any(!isfinite, X) &&
        error("Feature $feat section $section_id contains non-finite values")
    # Fail on all-zero rows (missing extraction). Scalar lexical predictors may
    # legitimately be zero, so skip when B2B_ALLOW_ZERO_FEATURE_ROWS is set.
    if !B2B_ALLOW_ZERO_FEATURE_ROWS
        row_norms = [sum(abs2, @view X[i, :]) for i in 1:n]
        zrows = findall(iszero, row_norms)
        isempty(zrows) ||
            error("Feature $feat section $section_id has $(length(zrows)) all-zero rows " *
                  "(first at $(zrows[1]))")
    end
    expected_n = EXPECTED_SEC_LENGTHS[section_id]
    n == expected_n ||
        error("Feature $feat section $section_id: n_words=$n, expected $expected_n")
    if haskey(EXPECTED_FEAT_DIMS, feat)
        if p != EXPECTED_FEAT_DIMS[feat]
            msg = "Feature $feat section $section_id: dim=$p, expected $(EXPECTED_FEAT_DIMS[feat])"
            B2B_STRICT_FEAT_DIMS ? error(msg) : @warn msg
        end
    end
    return X
end

function load_section_features(extractor_dir, section_id, feature_sets;
                               strict::Bool=true)
    shared_dir = joinpath(extractor_dir, "_shared_wordlocked_features",
                          @sprintf("section_%03d", section_id))
    X_word  = Dict{String, Matrix{Float64}}()
    X_names = Dict{String, Vector{String}}()
    checksums = Dict{String,String}()
    for feat in feature_sets
        path = joinpath(shared_dir, "X_word_$(feat).npy")
        isfile(path) || error("Shared feature not found: $path")
        checksums[feat] = file_sha256(path)
        X = Float64.(NPZ.npzread(path))
        if strict
            validate_feature_matrix!(X, feat, section_id)
        else
            # Legacy soft path — still refuse NaN/Inf rather than zero-filling
            any(!isfinite, X) &&
                error("Feature $feat section $section_id contains non-finite values")
        end
        X_word[feat] = X
        npath = joinpath(shared_dir, "X_word_$(feat)_feature_names.txt")
        X_names[feat] = isfile(npath) ? readlines(npath) :
                        ["$(feat)_$(i)" for i in 1:size(X, 2)]
    end
    return X_word, X_names, checksums
end

"""Build standardized nuisance predictors from shared timing tables (both sections)."""
function build_nuisance_matrix(extractor_dir::AbstractString=B2B_EXTRACTOR_DIR;
                               sfreq::Float64=EXPECTED_SFREQ)
    blocks = Matrix{Float64}[]
    for sid in SECTIONS
        df = load_shared_word_timing(extractor_dir, sid)
        n = nrow(df)
        duration = if "onset_relative" in names(df) && "offset_relative" in names(df)
            Float64.(df.offset_relative .- df.onset_relative)
        else
            Float64.(df.offset_sample .- df.onset_sample) ./ sfreq
        end
        logfreq = if "logfreq" in names(df)
            Float64.(df.logfreq)
        else
            zeros(n)
        end
        # Rare OOV lemmas are coded as -Inf; impute with finite-column minimum
        finite_lf = filter(isfinite, logfreq)
        isempty(finite_lf) && error("No finite logfreq values in section $sid")
        lf_floor = minimum(finite_lf)
        n_imp = count(!isfinite, logfreq)
        logfreq = ifelse.(isfinite.(logfreq), logfreq, lf_floor)
        story_pos = n > 1 ? collect(range(0.0, 1.0; length=n)) : zeros(n)
        section_ind = fill(Float64(sid - 1), n)
        char_len = Float64[length(string(w)) for w in df.word]
        push!(blocks, hcat(duration, logfreq, story_pos, section_ind, char_len))
        n_imp > 0 && @info "Imputed $n_imp non-finite logfreq values in section $sid with $lf_floor"
    end
    N = vcat(blocks...)
    size(N, 1) == EXPECTED_N_WORDS ||
        error("Nuisance rows $(size(N, 1)) != expected $EXPECTED_N_WORDS")
    any(!isfinite, N) && error("Nuisance matrix still contains non-finite values")
    # Standardize each column
    μ = vec(mean(N; dims=1))
    σ = vec(std(N; dims=1, corrected=true))
    σ = ifelse.(σ .> 1e-10, σ, 1.0)
    Nstd = (N .- μ') ./ σ'
    return Nstd, copy(NUISANCE_NAMES), Dict("mean" => μ, "scale" => σ)
end

function zscore_channels(eeg::AbstractMatrix; eps=1e-10)
    μ = mean(eeg, dims=2)
    σ = std(eeg, dims=2)
    σ = ifelse.(isfinite.(σ) .& (σ .> eps), σ, 1.0)
    out = (eeg .- μ) ./ σ
    replace!(out, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    return out
end

"""
Build word-locked epochs.

`anchor` selects the timing column: `:onset` → onset_sample, `:offset` → offset_sample.
t=0 in the returned epochs indexes the chosen anchor.
"""
function build_epochs_qc(eeg_sections::Vector{<:AbstractMatrix},
                         eeg_raw_sections::Vector{<:AbstractMatrix},
                         word_dfs::Vector{<:AbstractDataFrame},
                         train_masks::Vector,
                         sfreq::Float64, tmin_s::Float64, tmax_s::Float64;
                         anchor::Symbol=Symbol(B2B_EPOCH_ANCHOR))
    anchor in (:onset, :offset) || error("anchor must be :onset or :offset")
    col = anchor === :onset ? :onset_sample : :offset_sample

    s_pre  = round(Int, tmin_s * sfreq)
    s_post = round(Int, tmax_s * sfreq)
    n_times = s_post - s_pre + 1
    n_ch = size(eeg_sections[1], 1)
    n_words_total = sum(nrow.(word_dfs))

    dat      = Array{Float64}(undef, n_ch, n_times, n_words_total)
    dat_raw  = Array{Float64}(undef, n_ch, n_times, n_words_total)
    keep     = falses(n_words_total)
    sec_ids  = Vector{Int}(undef, n_words_total)
    word_idx_in_sec = Vector{Int}(undef, n_words_total)
    lock_julia_vec  = Vector{Int}(undef, n_words_total)
    reject_reason   = fill("", n_words_total)
    epoch_ptp_max   = fill(NaN, n_words_total)
    epoch_z_max     = fill(NaN, n_words_total)
    epoch_var_mean  = fill(NaN, n_words_total)

    mask_informative = [m !== nothing && !all(m) && !all(.!m) for m in train_masks]

    w = 0
    for (sec_idx, (eeg, eeg_raw, wt, tmask)) in enumerate(
            zip(eeg_sections, eeg_raw_sections, word_dfs, train_masks))
        T = size(eeg, 2)
        col in propertynames(wt) || col in Symbol.(names(wt)) ||
            error("Word timing missing $col for epoch anchor=$anchor")
        anchors = Int.(wt[!, col])
        for (i, lock0) in enumerate(anchors)
            w += 1
            lock = to_julia_sample(lock0)
            a = lock + s_pre
            b = lock + s_post
            sec_ids[w] = sec_idx
            word_idx_in_sec[w] = i
            lock_julia_vec[w] = lock

            if a < 1 || b > T
                dat[:, :, w] .= 0
                dat_raw[:, :, w] .= 0
                reject_reason[w] = "out_of_bounds"
                continue
            end

            ep = eeg[:, a:b]
            ep_raw = eeg_raw[:, a:b]
            dat[:, :, w] = ep
            dat_raw[:, :, w] = ep_raw

            if REQUIRE_FULL_WINDOW_MASK && tmask !== nothing && mask_informative[sec_idx]
                if a > length(tmask) || b > length(tmask) || !all(@view tmask[a:b])
                    reject_reason[w] = "mask_window"
                    continue
                end
            end

            # Peak-to-peak diagnostics only (saved for QC tables). Absolute µV
            # thresholds are NOT applied: eeg_data.npy is already bandpassed /
            # re-referenced and unit-scaled, so 150–200 µV cutoffs are invalid.
            ptp = vec(maximum(ep_raw; dims=2) .- minimum(ep_raw; dims=2))
            epoch_ptp_max[w] = maximum(ptp)

            zmax = maximum(abs, ep)
            epoch_z_max[w] = zmax
            if zmax > ZSCORE_ABS_MAX
                reject_reason[w] = "zscore"
                continue
            end

            epoch_var_mean[w] = mean(var(ep; dims=2))
            keep[w] = true
            reject_reason[w] = "keep"
        end
    end

    qc = (
        keep = keep,
        section_id = sec_ids,
        word_idx_in_sec = word_idx_in_sec,
        lock_julia = lock_julia_vec,
        reject_reason = reject_reason,
        epoch_ptp_max = epoch_ptp_max,
        epoch_z_max = epoch_z_max,
        epoch_var_mean = epoch_var_mean,
        s_pre = s_pre,
        s_post = s_post,
        n_times = n_times,
        mask_informative = mask_informative,
        epoch_anchor = String(anchor),
        anchor_column = String(col),
    )
    return dat, dat_raw, qc
end

function channel_qc_metrics(dat::AbstractArray{<:Real,3}, keep::AbstractVector{Bool},
                            s_pre::Integer, sfreq::Real;
                            noise_win::Tuple{<:Real,<:Real}=BASELINE_WIN_S)
    n_ch, n_times, _ = size(dat)
    kept = findall(keep)
    isempty(kept) && return (ch_var=fill(NaN, n_ch), residual_noise=fill(NaN, n_ch),
                             n_kept=0)
    X = dat[:, :, kept]
    ch_var = vec(var(reshape(X, n_ch, :); dims=2))
    bi0 = clamp(round(Int, noise_win[1] * sfreq) - s_pre + 1, 1, n_times)
    bi1 = clamp(round(Int, noise_win[2] * sfreq) - s_pre + 1, 1, n_times)
    bi0 > bi1 && ((bi0, bi1) = (bi1, bi0))
    base = X[:, bi0:bi1, :]
    base = base .- mean(base; dims=2)
    residual_noise = vec(sqrt.(mean(base .^ 2; dims=(2, 3))))
    return (ch_var=ch_var, residual_noise=residual_noise, n_kept=length(kept))
end
