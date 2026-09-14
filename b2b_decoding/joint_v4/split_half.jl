# =====================================================================
# Phase 2 split-half: interleaved ~60 s odd/even blocks per story.
# Independent observed B2B on each half; null-sub uses Phase 1 medians.
# =====================================================================

const SPLIT_BLOCK_S = parse(Float64, get(ENV, "B2B_SPLIT_BLOCK_S", "60"))

"""Shared stimulus onset times aligned to the concatenated word axis."""
function load_shared_onsets()
    onsets = Float64[]
    secs = Int[]
    for sid in SECTIONS
        df = load_shared_word_timing(B2B_EXTRACTOR_DIR, sid)
        "onset_relative" in names(df) || error("word_timing_relative missing onset_relative")
        append!(onsets, Float64.(df.onset_relative))
        append!(secs, fill(sid, nrow(df)))
    end
    return secs, onsets
end

"""Odd/even masks from ~60 s time blocks within each story (not 1st/2nd half)."""
function interleaved_60s_masks(sec_ids::AbstractVector{Int},
                               onset_rel::AbstractVector{<:Real};
                               win::Real=SPLIT_BLOCK_S)
    length(sec_ids) == length(onset_rel) || error("section / onset length mismatch")
    odd = falses(length(sec_ids))
    even = falses(length(sec_ids))
    for sid in unique(sec_ids)
        rows = findall(==(sid), sec_ids)
        t = onset_rel[rows]
        t0 = minimum(t)
        blk = floor.(Int, (t .- t0) ./ win)
        odd[rows[.!iseven.(blk)]] .= true
        even[rows[iseven.(blk)]] .= true
    end
    return odd, even
end

function peak_latency(times::AbstractVector{<:Real}, trace::AbstractVector{<:Real};
                      win=CONFIRM_WIN_S)
    m = findall(t -> win[1] - 1e-12 <= t <= win[2] + 1e-12, times)
    isempty(m) && return NaN
    i = m[argmax(trace[m])]
    return Float64(times[i])
end

"""50% fractional-area latency on positive part of the confirm-window trace."""
function frac_area_latency(times::AbstractVector{<:Real}, trace::AbstractVector{<:Real};
                           frac::Real=0.5, win=CONFIRM_WIN_S)
    m = findall(t -> win[1] - 1e-12 <= t <= win[2] + 1e-12, times)
    isempty(m) && return NaN
    t = times[m]
    y = max.(trace[m], 0.0)
    tot = sum(y)
    tot <= 0 && return NaN
    c = cumsum(y)
    target = frac * tot
    j = findfirst(>=(target), c)
    j === nothing && return Float64(t[end])
    return Float64(t[j])
end

const PITCH_LEVEL_COLS = 1:1          # f0_mean
const PITCH_DYNAMIC_COLS = 2:13       # std, range, t00–t09
const TONE_ONEHOT_COLS = 1:4
const TONE_DEV_COLS = 5:6

function pitch_subfamily_cols(feature_set, ncols)
    if feature_set == "ctx_pitch"
        ncols == 5 || error("ctx_pitch requires 5 columns; got $ncols")
        return (("level", 1:1), ("dynamic", 2:5))
    end
    ncols == 13 || error("Legacy pitch subfamilies require 13 columns; got $ncols")
    return (("level", PITCH_LEVEL_COLS), ("dynamic", PITCH_DYNAMIC_COLS))
end

function _confirm_mask(times::AbstractVector{<:Real}; win=CONFIRM_WIN_S)
    return (times .>= win[1]) .& (times .<= win[2])
end

function _sum_block(Sμ::AbstractMatrix, times, col0::Int, cols)
    return [sum(Sμ[t, col0 .+ cols]) for t in 1:length(times)]
end

"""Observed B2B on a keep-mask subset. Skips degenerate partitions."""
function run_observed_on_mask!(bundle, keep_mask::AbstractVector{Bool};
                               label::AbstractString)
    p = bundle.participant
    k = bundle.k
    fks = bundle.family_ks
    times = bundle.times
    qc = bundle.qc
    keep = qc.keep .& keep_mask
    n_keep = count(keep)
    n_keep < MIN_KEEP_WORDS &&
        error("$p $label: too few usable words after split ($n_keep)")

    fam_traces = Dict(f => Vector{Vector{Float64}}() for f in FAMILY_NAMES)
    valid = falses(length(bundle.partitions))
    S_acc = zeros(length(times), sum(fks))
    n_ok = 0

    for (pid, part) in enumerate(bundle.partitions)
        half1, half2 = make_partition_halves(
            qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, keep, part;
            n_blocks=N_BLOCKS_PER_SECTION, guard=GUARD_WORDS)
        if sum(half1) < 40 || sum(half2) < 40
            println("    [$label part $pid] skip degenerate halves $(sum(half1))/$(sum(half2))")
            continue
        end
        snr_meta = hasproperty(bundle, :snr_meta) ? bundle.snr_meta : nothing
        dat_use = maybe_partition_spatial(bundle.dat_3d, half1, snr_meta)
        S, fam, diag, _, _ = b2b_joint_partition(
            dat_use, bundle.X_joint, half1, half2,
            qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, times;
            k=k, family_ks=fks)
        diag.invalid && continue
        valid[pid] = true
        n_ok += 1
        S_acc .+= S
        for fam_name in FAMILY_NAMES
            push!(fam_traces[fam_name], fam.traces[fam_name])
        end
        println("    [$label part $pid] ok  maxκ=$(round(diag.max_kappa; sigdigits=3))")
    end
    n_ok == 0 && error("$p $label: no valid partitions")
    Sμ = S_acc ./ n_ok

    cwin = _confirm_mask(times)
    rows = NamedTuple[]
    col0 = 0
    for (fi, fam_name) in enumerate(FAMILY_NAMES)
        mats = fam_traces[fam_name]
        M = reduce(hcat, mats)'
        μ = vec(mean(M; dims=1))
        push!(rows, (
            participant=p, split=label, family=fam_name, subfamily="all",
            confirm_mean=mean(μ[cwin]),
            peak_lat=peak_latency(times, μ),
            fa50_lat=frac_area_latency(times, μ),
            n_valid_splits=n_ok, n_keep=n_keep,
        ))
        if fam_name == "pitch"
            for (sub, cols) in pitch_subfamily_cols(FEATURE_SETS[fi], fks[fi])
                tr = _sum_block(Sμ, times, col0, cols)
                push!(rows, (
                    participant=p, split=label, family=fam_name, subfamily=sub,
                    confirm_mean=mean(tr[cwin]),
                    peak_lat=peak_latency(times, tr),
                    fa50_lat=frac_area_latency(times, tr),
                    n_valid_splits=n_ok, n_keep=n_keep,
                ))
            end
        elseif fam_name == "tone"
            for (sub, cols) in (("onehot", TONE_ONEHOT_COLS), ("dev", TONE_DEV_COLS))
                tr = _sum_block(Sμ, times, col0, cols)
                push!(rows, (
                    participant=p, split=label, family=fam_name, subfamily=sub,
                    confirm_mean=mean(tr[cwin]),
                    peak_lat=peak_latency(times, tr),
                    fa50_lat=frac_area_latency(times, tr),
                    n_valid_splits=n_ok, n_keep=n_keep,
                ))
            end
        end
        col0 += fks[fi]
    end
    return DataFrame(rows), Sμ, n_ok
end

function run_split_half!(participant::AbstractString)
    println("\n", "#"^80, "\n# SPLIT-HALF  $participant  block=$(SPLIT_BLOCK_S)s\n", "#"^80)
    assert_outdir_manifest!(B2B_OUTDIR)
    out_part = joinpath(B2B_OUTDIR, participant)
    mkpath(out_part)
    done = joinpath(out_part, "$(participant)_splithalf_scores.csv")
    if isfile(done) && lowercase(get(ENV, "B2B_FORCE", "0")) ∉ ("1", "true", "yes")
        println("  [skip] $done exists")
        return nothing
    end

    bundle = load_participant_bundle(participant; k=B2B_PCA_K)
    secs, onsets = load_shared_onsets()
    length(secs) == length(bundle.qc.keep) ||
        error("Shared onset length $(length(secs)) != words $(length(bundle.qc.keep))")
    odd, even = interleaved_60s_masks(secs, onsets; win=SPLIT_BLOCK_S)
    println("  odd keep=$(count(odd .& bundle.qc.keep))  even keep=$(count(even .& bundle.qc.keep))")

    odd_df, _, n_odd = run_observed_on_mask!(bundle, odd; label="odd")
    even_df, _, n_even = run_observed_on_mask!(bundle, even; label="even")
    scores = vcat(odd_df, even_df)
    CSV.write(done, scores)
    write_json(joinpath(out_part, "$(participant)_splithalf_meta.json"), Dict(
        "participant" => participant,
        "block_s" => SPLIT_BLOCK_S,
        "n_valid_odd" => n_odd,
        "n_valid_even" => n_even,
        "n_keep_odd" => count(odd .& bundle.qc.keep),
        "n_keep_even" => count(even .& bundle.qc.keep),
        "epoch_anchor" => B2B_EPOCH_ANCHOR,
        "tmin_s" => TMIN_S,
        "tmax_s" => TMAX_S,
        "confirm_win_s" => collect(CONFIRM_WIN_S),
    ))
    println("  wrote $done")
    return scores
end
