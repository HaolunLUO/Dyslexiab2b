# =====================================================================
# Joint-family B2B v4 per-participant driver
# =====================================================================

using DataFrames
using CSV
using Statistics
using StatsBase
using Random
using Printf
using JSON3
using NPZ

_csv_safe(df::DataFrame) = begin
    out = copy(df)
    for c in names(out)
        col = out[!, c]
        if eltype(col) === Nothing || all(isnothing, col)
            select!(out, Not(c))
        elseif any(isnothing, col)
            out[!, c] = [x === nothing ? missing : x for x in col]
        end
    end
    return out
end

function _participant_seed(participant::AbstractString, extra::Integer=0)
    return B2B_SEED + sum(Int.(codeunits(participant))) + 10007 * extra
end

"""Load participant EEG/QC and attach frozen joint PC design (+ optional nuisance)."""
function load_participant_bundle(participant::AbstractString; k::Int=B2B_PCA_K)
    Z_by_fam, sec_lengths_shared, basis_meta, _ = load_feature_basis(; k=k)

    eeg_z_sections   = Matrix{Float64}[]
    eeg_raw_sections = Matrix{Float64}[]
    train_masks      = Any[]
    word_dfs         = DataFrame[]
    sfreqs           = Float64[]
    n_chs            = Int[]

    for sid in SECTIONS
        eeg, tmask, meta = load_section_eeg(B2B_EXTRACTOR_DIR, participant, sid)
        sfreq = Float64(get(meta, "sfreq", EXPECTED_SFREQ))
        eeg, _ch_used = preprocess_continuous_eeg(eeg, sfreq, meta)
        push!(sfreqs, sfreq)
        push!(n_chs, size(eeg, 1))
        if abs(sfreq - EXPECTED_SFREQ) > 1e-6
            @warn "Unexpected sfreq" participant sid sfreq
        end
        if B2B_CHAN_SET == "all" && size(eeg, 1) != EXPECTED_N_CH
            @warn "Unexpected n_channels" participant sid n_ch=size(eeg, 1)
        end
        push!(eeg_raw_sections, copy(eeg))
        push!(eeg_z_sections, zscore_channels(eeg))
        push!(train_masks, tmask)
        wt = load_section_word_timing(B2B_EXTRACTOR_DIR, participant, sid; sfreq=sfreq)
        push!(word_dfs, wt)
        nrow(wt) == sec_lengths_shared[sid] ||
            error("Word count mismatch $participant sec $sid: $(nrow(wt)) vs $(sec_lengths_shared[sid])")
    end
    allequal(sfreqs) || error("sfreq differs across sections: $sfreqs")
    allequal(n_chs)  || error("n_channels differs across sections: $n_chs")
    sfreq = sfreqs[1]
    n_words_per_sec = [nrow(wt) for wt in word_dfs]

    dat_3d, dat_raw, qc = build_epochs_qc(
        eeg_z_sections, eeg_raw_sections, word_dfs, train_masks, sfreq, TMIN_S, TMAX_S;
        anchor=Symbol(B2B_EPOCH_ANCHOR))
    sum(qc.keep) < MIN_KEEP_WORDS &&
        error("Too few usable words ($(sum(qc.keep))) for $participant")

    times = collect(range(TMIN_S, TMAX_S; length=qc.n_times))
    # Channel QC on original sensor epochs (before spatial compress / binning).
    ch_qc = channel_qc_metrics(dat_3d, qc.keep, qc.s_pre, sfreq)
    dat_3d, times, sfreq_eff, snr_meta = preprocess_epochs_snr(
        dat_3d, qc.keep, times, sfreq)

    nuisance = nothing
    nuisance_names = String[]
    if B2B_INCLUDE_NUISANCE
        nuisance, nuisance_names, _ = build_nuisance_matrix(B2B_EXTRACTOR_DIR; sfreq=sfreq)
    end
    X_joint, feat_names = joint_design(Z_by_fam; k=k, nuisance=nuisance,
                                       nuisance_names=nuisance_names)
    size(X_joint, 1) == length(qc.keep) ||
        error("Joint design / word axis mismatch")
    fks = family_ks_of(Z_by_fam)

    partitions, part_counts = generate_balanced_partitions(
        N_OUTER_PARTITIONS; n_blocks=N_BLOCKS_PER_SECTION, seed=B2B_SEED)

    return (
        participant=participant,
        dat_3d=dat_3d,
        qc=qc,
        ch_qc=ch_qc,
        X_joint=X_joint,
        Z_by_fam=Z_by_fam,
        nuisance=nuisance,
        nuisance_names=nuisance_names,
        feat_names=feat_names,
        family_ks=fks,
        times=times,
        sfreq=sfreq_eff,
        sfreq_raw=sfreq,
        n_ch=size(dat_3d, 1),
        n_ch_raw=n_chs[1],
        n_words_per_sec=n_words_per_sec,
        partitions=partitions,
        part_counts=part_counts,
        basis_meta=basis_meta,
        snr_meta=snr_meta,
        k=k,
    )
end

function save_word_qc(bundle, out_part)
    p = bundle.participant
    qc = bundle.qc
    word_qc = DataFrame(
        word_global = 1:length(qc.keep),
        section = qc.section_id,
        word_in_section = qc.word_idx_in_sec,
        lock_sample_julia = qc.lock_julia,
        epoch_anchor = qc.epoch_anchor,
        keep = qc.keep,
        reject_reason = qc.reject_reason,
        epoch_ptp_max = qc.epoch_ptp_max,
        epoch_z_max = qc.epoch_z_max,
        epoch_var_mean = qc.epoch_var_mean,
    )
    CSV.write(joinpath(out_part, "$(p)_word_qc.csv"), word_qc)
    CSV.write(joinpath(out_part, "$(p)_usable_words.csv"), word_qc[word_qc.keep, :])
    CSV.write(joinpath(out_part, "$(p)_rejected_words.csv"), word_qc[.!word_qc.keep, :])
    ch_df = DataFrame(
        channel = 1:length(bundle.ch_qc.ch_var),
        channel_variance = bundle.ch_qc.ch_var,
        residual_noise = bundle.ch_qc.residual_noise,
    )
    CSV.write(joinpath(out_part, "$(p)_channel_qc.csv"), ch_df)
    return word_qc
end

"""Run observed joint B2B across all partitions."""
function run_observed!(bundle; out_part::AbstractString)
    p = bundle.participant
    k = bundle.k
    fks = bundle.family_ks
    times = bundle.times
    qc = bundle.qc
    n_fam_cols = sum(fks)

    pc_rows = NamedTuple[]
    fam_rows = NamedTuple[]
    diag_rows = NamedTuple[]
    # Primary score traces (trace/K) for aggregation / null comparison
    fam_traces = Dict(f => Vector{Vector{Float64}}() for f in FAMILY_NAMES)
    valid_partition = falses(length(bundle.partitions))

    assign_rows = partition_assignment_rows(p, bundle.partitions; n_blocks=N_BLOCKS_PER_SECTION)
    CSV.write(joinpath(out_part, "$(p)_partition_assignments.csv"), DataFrame(assign_rows))

    for (pid, part) in enumerate(bundle.partitions)
        half1, half2 = make_partition_halves(
            qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, qc.keep, part;
            n_blocks=N_BLOCKS_PER_SECTION, guard=GUARD_WORDS)
        println("  [partition $pid] half1=$(sum(half1)) half2=$(sum(half2))")
        (sum(half1) < 40 || sum(half2) < 40) && error("Degenerate halves on partition $pid")

        snr_meta = hasproperty(bundle, :snr_meta) ? bundle.snr_meta : nothing
        dat_use = maybe_partition_spatial(bundle.dat_3d, half1, snr_meta)
        S, fam, diag, d1, d2 = b2b_joint_partition(
            dat_use, bundle.X_joint, half1, half2,
            qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, times;
            k=k, family_ks=fks)

        valid_partition[pid] = !diag.invalid
        for fam_name in FAMILY_NAMES
            push!(fam_traces[fam_name], fam.scores[fam_name])
        end

        push!(diag_rows, (
            participant=p, pca_k=k, split=pid, direction=0,
            n_half1=diag.n_half1, n_half2=diag.n_half2,
            alpha_1=diag.alpha_1, alpha_2=diag.alpha_2,
            lambda_1=diag.lambda_1, lambda_2=diag.lambda_2,
            cv_loss_1=diag.cv_loss_1, cv_loss_2=diag.cv_loss_2,
            on_boundary_1=diag.on_boundary_1, on_boundary_2=diag.on_boundary_2,
            max_kappa=diag.max_kappa, kappa_1=diag.kappa_1, kappa_2=diag.kappa_2,
            warn_kappa=diag.warn_kappa, invalid=diag.invalid,
            rank_ok=diag.rank_ok, usable_trials=sum(half1 .| half2),
        ))
        for (dir, dd) in ((1, d1), (2, d2))
            push!(diag_rows, (
                participant=p, pca_k=k, split=pid, direction=dir,
                n_half1=dd.n_train, n_half2=dd.n_test,
                alpha_1=dd.alpha, alpha_2=NaN,
                lambda_1=dd.lambda, lambda_2=NaN,
                cv_loss_1=dd.cv_loss, cv_loss_2=NaN,
                on_boundary_1=dd.on_boundary, on_boundary_2=false,
                max_kappa=dd.max_kappa, kappa_1=dd.max_kappa, kappa_2=NaN,
                warn_kappa=dd.warn_kappa, invalid=dd.invalid,
                rank_ok=!dd.invalid, usable_trials=sum(half1 .| half2),
            ))
        end

        for (ti, t) in enumerate(times)
            for (j, fname) in enumerate(bundle.feat_names)
                if j <= n_fam_cols
                    fam_name, pc_idx = family_and_pc(j, fks)
                    push!(pc_rows, (
                        participant=p, pca_k=k, split=pid, direction=0,
                        time=t, family=fam_name, PC=pc_idx, coefficient=S[ti, j],
                        is_nuisance=false,
                    ))
                else
                    push!(pc_rows, (
                        participant=p, pca_k=k, split=pid, direction=0,
                        time=t, family="nuisance", PC=j - n_fam_cols,
                        coefficient=S[ti, j], is_nuisance=true,
                    ))
                end
            end
            for fam_name in FAMILY_NAMES
                push!(fam_rows, (
                    participant=p, pca_k=k, split=pid, direction=0,
                    time=t, family=fam_name,
                    family_trace=fam.traces[fam_name][ti],
                    family_score=fam.scores[fam_name][ti],  # primary = trace/K
                ))
            end
        end
        println("    invalid=$(diag.invalid)  maxκ=$(round(diag.max_kappa; sigdigits=3))  " *
                "boundary=$(diag.on_boundary_1 || diag.on_boundary_2)")
    end

    n_valid = count(valid_partition)
    println("  Valid partitions: $n_valid / $(length(bundle.partitions))")

    pc_df = DataFrame(pc_rows)
    fam_df = DataFrame(fam_rows)
    diag_df = DataFrame(diag_rows)
    CSV.write(joinpath(out_part, "$(p)_b2b_pc_splits.csv"), _csv_safe(pc_df))
    CSV.write(joinpath(out_part, "$(p)_b2b_family_splits.csv"), _csv_safe(fam_df))
    CSV.write(joinpath(out_part, "$(p)_diagnostics.csv"), _csv_safe(diag_df))

    # Aggregate over valid partitions (primary = family_score = trace/K_family)
    agg_rows = NamedTuple[]
    for (fi, fam_name) in enumerate(FAMILY_NAMES)
        mats = [fam_traces[fam_name][i] for i in 1:length(bundle.partitions) if valid_partition[i]]
        # Also aggregate raw traces
        if isempty(mats)
            μ = fill(NaN, length(times))
            σ = fill(NaN, length(times))
            μ_tr = fill(NaN, length(times))
        else
            M = reduce(hcat, mats)'   # n_valid × n_times
            μ = vec(mean(M; dims=1))
            σ = size(M, 1) > 1 ? vec(std(M; dims=1)) : zeros(length(times))
            μ_tr = μ .* fks[fi]
        end
        for (ti, t) in enumerate(times)
            push!(agg_rows, (
                participant=p, time=t, family=fam_name,
                mean_score=μ[ti],           # primary: trace/K
                mean_trace=μ_tr[ti],        # raw trace(H_family)
                split_sd=σ[ti], n_valid_splits=n_valid,
            ))
        end
    end
    agg_df = DataFrame(agg_rows)
    CSV.write(joinpath(out_part, "$(p)_b2b_family_agg.csv"), _csv_safe(agg_df))

    if B2B_DO_TG
        run_temporal_generalization!(bundle, valid_partition; out_part=out_part)
    end

    return (
        pc_df=pc_df, fam_df=fam_df, diag_df=diag_df, agg_df=agg_df,
        fam_traces=fam_traces, valid_partition=valid_partition, n_valid=n_valid,
    )
end

"""Average TG across valid partitions (both directions) and write duration/sustain."""
function run_temporal_generalization!(bundle, valid_partition; out_part::AbstractString)
    p = bundle.participant
    k = bundle.k
    fks = bundle.family_ks
    times = bundle.times
    qc = bundle.qc
    stride = B2B_TG_STRIDE
    t_idx = collect(1:stride:length(times))
    times_tg = times[t_idx]
    n_t = length(t_idx)
    acc = Dict(f => zeros(n_t, n_t) for f in FAMILY_NAMES)
    n_ok = 0
    println("  [TG] stride=$stride  n_t=$n_t"); flush(stdout)
    for (pid, part) in enumerate(bundle.partitions)
        valid_partition[pid] || continue
        half1, half2 = make_partition_halves(
            qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, qc.keep, part;
            n_blocks=N_BLOCKS_PER_SECTION, guard=GUARD_WORDS)
        snr_meta = hasproperty(bundle, :snr_meta) ? bundle.snr_meta : nothing
        dat_full = maybe_partition_spatial(bundle.dat_3d, half1, snr_meta)
        use = half1 .| half2
        idx = findall(use)
        Y = dat_full[:, :, idx]
        X = bundle.X_joint[idx, :]
        h1 = half1[idx]; h2 = half2[idx]
        sec_u = qc.section_id[idx]
        wi_u = qc.word_idx_in_sec[idx]
        blk_u = [word_block_id(wi_u[j], bundle.n_words_per_sec[sec_u[j]], N_BLOCKS_PER_SECTION)
                 for j in eachindex(idx)]
        TG1, d1 = b2b_one_direction_tg(Y, X, h1, h2, sec_u, blk_u, times; stride=stride)
        TG2, d2 = b2b_one_direction_tg(Y, X, h2, h1, sec_u, blk_u, times; stride=stride)
        (d1.invalid || d2.invalid) && continue
        TGavg = 0.5 .* (TG1 .+ TG2)
        famM = family_tg_from_TG(TGavg, fks)
        for f in FAMILY_NAMES
            acc[f] .+= famM[f]
        end
        n_ok += 1
        println("    TG partition $pid ok  n_ok=$n_ok"); flush(stdout)
    end
    n_ok == 0 && (@warn "No valid TG partitions for $p"; return nothing)
    metric_rows = NamedTuple[]
    for f in FAMILY_NAMES
        M = acc[f] ./ n_ok
        npz_path = joinpath(out_part, "$(p)_tg_$(f).npy")
        NPZ.npzwrite(npz_path, M)
        dur, sus = tg_duration_sustain(M, times_tg; thresh=0.0)
        push!(metric_rows, (
            participant=p, family=f, n_valid_splits=n_ok,
            duration_s=dur, sustain_s=sus,
            tg_stride=stride, n_times_tg=n_t,
            tmin_s=first(times_tg), tmax_s=last(times_tg),
        ))
    end
    CSV.write(joinpath(out_part, "$(p)_tg_metrics.csv"), _csv_safe(DataFrame(metric_rows)))
    write_json(joinpath(out_part, "$(p)_tg_times.json"), Dict(
        "t_idx" => t_idx, "times_tg" => collect(times_tg), "stride" => stride,
        "n_valid_splits" => n_ok, "family_ks" => collect(fks),
        "families" => collect(FAMILY_NAMES),
    ))
    println("  [TG] wrote metrics for $p  n_ok=$n_ok")
    return metric_rows
end

"""Null max-over-window statistic for one family trace."""
function _max_abs_in_window(trace::AbstractVector{<:Real}, times, win)
    m = (times .>= win[1]) .& (times .<= win[2])
    return maximum(abs, trace[m]; init=0.0)
end

function run_null_chunk!(bundle, obs_result; out_part::AbstractString,
                         null_start::Int=B2B_NULL_START,
                         null_count::Int=B2B_NULL_COUNT,
                         n_null_total::Int=B2B_N_NULL)
    p = bundle.participant
    k = bundle.k
    times = bundle.times
    qc = bundle.qc
    null_end = min(null_start + null_count - 1, n_null_total)
    null_start > null_end && return nothing

    chunk_tag = @sprintf("null_%04d_%04d", null_start, null_end)
    println("  [null chunk] reps $null_start:$null_end of $n_null_total")

    obs_max = Dict{String,Float64}()
    for fam in FAMILY_NAMES
        mats = [obs_result.fam_traces[fam][i]
                for i in 1:length(bundle.partitions) if obs_result.valid_partition[i]]
        isempty(mats) && error("No valid partitions for null comparison")
        μ = vec(mean(reduce(hcat, mats)'; dims=1))
        obs_max[fam] = _max_abs_in_window(μ, times, CONFIRM_WIN_S)
    end
    obs_max_family = maximum(values(obs_max))

    trace_rows = NamedTuple[]
    summary_rows = Dict{String,Any}[]

    for rep in null_start:null_end
        rng = Random.MersenneTwister(_participant_seed(p, rep))
        shifts = Dict{Int,Int}()
        for sid in 1:length(SECTIONS)
            nw = bundle.n_words_per_sec[sid]
            lo = NULL_MIN_SHIFT_WORDS
            hi = max(lo + 1, nw - NULL_MIN_SHIFT_WORDS)
            shifts[sid] = rand(rng, lo:hi)
        end
        # Embedding families shift together; nuisance stays unshifted
        Z_shift = circular_shift_pcs(bundle.Z_by_fam, qc.section_id, shifts)
        X_shift, _ = joint_design(Z_shift; k=k,
                                  nuisance=bundle.nuisance,
                                  nuisance_names=bundle.nuisance_names)

        fam_acc = Dict(f => zeros(length(times)) for f in FAMILY_NAMES)
        n_ok = 0
        for (pid, part) in enumerate(bundle.partitions)
            half1, half2 = make_partition_halves(
                qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, qc.keep, part;
                n_blocks=N_BLOCKS_PER_SECTION, guard=GUARD_WORDS)
            snr_meta = hasproperty(bundle, :snr_meta) ? bundle.snr_meta : nothing
            dat_use = maybe_partition_spatial(bundle.dat_3d, half1, snr_meta)
            S, fam, diag, _, _ = b2b_joint_partition(
                dat_use, X_shift, half1, half2,
                qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, times;
                k=k, family_ks=bundle.family_ks)
            diag.invalid && continue
            n_ok += 1
            for f in FAMILY_NAMES
                fam_acc[f] .+= fam.scores[f]
            end
        end
        n_ok == 0 && @warn "Null rep $rep: no valid partitions"
        null_max = Dict{String,Float64}()
        for f in FAMILY_NAMES
            μ = n_ok > 0 ? fam_acc[f] ./ n_ok : fill(NaN, length(times))
            null_max[f] = _max_abs_in_window(μ, times, CONFIRM_WIN_S)
            for (ti, t) in enumerate(times)
                push!(trace_rows, (
                    participant=p, pca_k=k, null_rep=rep, time=t,
                    family=f, family_score=μ[ti],
                ))
            end
        end
        null_max_family = maximum(values(null_max))
        row = Dict{String,Any}(
            "participant" => p, "pca_k" => k, "null_rep" => rep,
            "max_over_family" => null_max_family,
            "n_valid_partitions" => n_ok,
            "shift_sec1" => shifts[1], "shift_sec2" => shifts[2],
        )
        for f in FAMILY_NAMES
            row["max_$f"] = null_max[f]
        end
        push!(summary_rows, row)
        if rep == null_start || rep % 5 == 0 || rep == null_end
            println("    null rep $rep  max_family=$(round(null_max_family; digits=4))")
        end
    end

    traces_df = DataFrame(trace_rows)
    sum_df = DataFrame(summary_rows)
    CSV.write(joinpath(out_part, "$(p)_$(chunk_tag)_traces.csv"), _csv_safe(traces_df))
    CSV.write(joinpath(out_part, "$(p)_$(chunk_tag)_summary.csv"), _csv_safe(sum_df))

    write_json(joinpath(out_part, "$(p)_$(chunk_tag)_manifest.json"), Dict(
        "participant" => p,
        "null_start" => null_start,
        "null_end" => null_end,
        "n_null_total" => n_null_total,
        "obs_max" => obs_max,
        "obs_max_family" => obs_max_family,
        "n_rows" => nrow(traces_df),
        "epoch_anchor" => B2B_EPOCH_ANCHOR,
        "basis_mode" => B2B_BASIS_MODE,
    ))
    return traces_df, sum_df
end

"""Detect missing or duplicate null replicate chunks; aggregate finals."""
function aggregate_participant_nulls(participant::AbstractString, out_part::AbstractString;
                                     n_null_total::Int=B2B_N_NULL, k::Int=B2B_PCA_K)
    agg_path = joinpath(out_part, "$(participant)_b2b_family_agg.csv")
    isfile(agg_path) || error("Missing observed agg: $agg_path")
    agg = CSV.read(agg_path, DataFrame)
    times = sort(unique(agg.time))

    obs_max = Dict{String,Float64}()
    for fam in FAMILY_NAMES
        sub = sort(agg[agg.family .== fam, :], :time)
        obs_max[fam] = _max_abs_in_window(sub.mean_score, times, CONFIRM_WIN_S)
    end
    obs_max_family = maximum(values(obs_max))

    trace_files = filter(f -> occursin(r"_null_\d+_\d+_traces\.csv$", f), readdir(out_part))
    sum_files = filter(f -> occursin(r"_null_\d+_\d+_summary\.csv$", f), readdir(out_part))
    isempty(sum_files) && error("No null chunk summaries in $out_part")

    sums = vcat([CSV.read(joinpath(out_part, f), DataFrame) for f in sort(sum_files)]...)
    # Detect duplicates before collapsing
    n_raw = nrow(sums)
    n_unique_reps = length(unique(sums.null_rep))
    if n_raw != n_unique_reps
        @warn "Duplicate null replicate rows for $participant: raw=$n_raw unique=$n_unique_reps"
    end
    sort!(sums, [:null_rep])
    sums = combine(groupby(sums, :null_rep), names(sums) .=> last .=> names(sums))
    n_reps = nrow(sums)
    missing_reps = setdiff(1:n_null_total, Int.(sums.null_rep))
    if !isempty(missing_reps)
        error("Null aggregation incomplete for $participant: missing reps " *
              "$(missing_reps[1:min(10, end)])… ($(length(missing_reps)) missing)")
    end
    n_reps == n_null_total ||
        error("Null rep count $n_reps != expected $n_null_total for $participant")

    if !isempty(trace_files)
        traces = vcat([CSV.read(joinpath(out_part, f), DataFrame) for f in sort(trace_files)]...)
        n_tr_raw = nrow(traces)
        traces = combine(groupby(traces, [:null_rep, :family, :time]),
                         names(traces) .=> last .=> names(traces))
        if nrow(traces) != n_tr_raw
            @warn "Duplicate null trace rows collapsed for $participant: " *
                  "raw=$n_tr_raw unique=$(nrow(traces))"
        end
        CSV.write(joinpath(out_part, "$(participant)_null_traces.csv"), _csv_safe(traces))
    end

    function emp_p(obs, null_vals)
        N = length(null_vals)
        exc = count(>=(obs), null_vals)
        return (exc + 1) / (N + 1)
    end

    null_summary = NamedTuple[]
    for fam in FAMILY_NAMES
        col = Symbol("max_$fam")
        null_vals = Float64.(sums[!, col])
        all(isfinite, null_vals) ||
            error("Non-finite null traces for $participant family=$fam")
        push!(null_summary, (
            participant=participant, pca_k=k, family=fam,
            obs_max_confirm=obs_max[fam],
            null_max_mean=mean(null_vals),
            null_max_sd=std(null_vals),
            p_max=(emp_p(obs_max[fam], null_vals)),
            n_null=length(null_vals),
        ))
    end
    p_fam = emp_p(obs_max_family, Float64.(sums.max_over_family))
    ns = DataFrame(null_summary)
    ns[!, :obs_max_over_family] .= obs_max_family
    ns[!, :p_max_over_family] .= p_fam
    CSV.write(joinpath(out_part, "$(participant)_null_summary.csv"), _csv_safe(ns))
    CSV.write(joinpath(out_part, "$(participant)_null_rep_stats.csv"), _csv_safe(sums))

    # Completeness sidecar
    write_json(joinpath(out_part, "$(participant)_null_completeness.json"), Dict(
        "participant" => participant,
        "n_null_expected" => n_null_total,
        "n_null_found" => n_reps,
        "n_raw_summary_rows" => n_raw,
        "duplicate_collapsed" => n_raw != n_unique_reps,
        "complete" => true,
    ))
    return ns
end

function run_subject_v4(participant::AbstractString;
                        mode::AbstractString=B2B_MODE,
                        k::Int=B2B_PCA_K)
    println("\n", "#"^80, "\n# JOINT B2B v4  $participant  mode=$mode  K=$k  " *
            "anchor=$(B2B_EPOCH_ANCHOR)  basis=$(B2B_BASIS_MODE)\n", "#"^80)
    Random.seed!(_participant_seed(participant))

    assert_outdir_manifest!(B2B_OUTDIR)
    out_part = joinpath(B2B_OUTDIR, participant)
    mkpath(out_part)

    bundle = load_participant_bundle(participant; k=k)
    save_word_qc(bundle, out_part)

    qc = bundle.qc
    qc_row = Dict{String,Any}(
        "participant" => participant,
        "sfreq" => bundle.sfreq,
        "sfreq_raw" => get(bundle, :sfreq_raw, bundle.sfreq),
        "n_channels" => bundle.n_ch,
        "n_channels_raw" => get(bundle, :n_ch_raw, bundle.n_ch),
        "n_words_total" => length(qc.keep),
        "n_usable" => sum(qc.keep),
        "n_rejected" => sum(.!qc.keep),
        "keep_frac" => mean(qc.keep),
        "n_usable_sec1" => sum(qc.keep .& (qc.section_id .== 1)),
        "n_usable_sec2" => sum(qc.keep .& (qc.section_id .== 2)),
        "mean_channel_var" => mean(bundle.ch_qc.ch_var),
        "median_channel_var" => median(bundle.ch_qc.ch_var),
        "mean_residual_noise" => mean(bundle.ch_qc.residual_noise),
        "median_residual_noise" => median(bundle.ch_qc.residual_noise),
        "pca_k" => k,
        "n_partitions" => length(bundle.partitions),
        "mode" => mode,
        "epoch_anchor" => qc.epoch_anchor,
        "anchor_column" => qc.anchor_column,
        "basis_mode" => B2B_BASIS_MODE,
        "include_nuisance" => B2B_INCLUDE_NUISANCE,
        "spatial_denoise" => B2B_SPATIAL_DENOISE,
        "spatial_n_comp" => B2B_SPATIAL_N_COMP,
        "time_bin_ms" => B2B_TIME_BIN_MS,
        "dss_fit_scope" => B2B_DSS_FIT_SCOPE,
        "eeg_band" => B2B_EEG_BAND,
        "chan_set" => B2B_CHAN_SET,
        "n_times" => length(bundle.times),
        "feat_acoustic" => FEAT_ACOUSTIC,
        "feat_speech" => FEAT_SPEECH,
        "feat_language" => FEAT_LANGUAGE,
        "tmin_s" => TMIN_S,
        "tmax_s" => TMAX_S,
        "confirm_win_s" => collect(CONFIRM_WIN_S),
        "cohort_hash" => cohort_hash(),
    )
    if hasproperty(bundle, :snr_meta) && bundle.snr_meta !== nothing
        # Scalars only in qc CSV; full meta (incl. eigenvalues) in JSON sidecar.
        for kmeta in ("spatial_denoise", "spatial_n_comp", "time_bin_ms",
                      "sfreq_in", "sfreq_out", "n_ch_in", "n_ch_out",
                      "n_times_in", "n_times_out", "n_samp_per_bin",
                      "spatial_var_explained", "n_ch_spatial")
            haskey(bundle.snr_meta, kmeta) || continue
            qc_row["snr_$(kmeta)"] = bundle.snr_meta[kmeta]
        end
        write_json(joinpath(out_part, "$(participant)_snr_preprocess.json"), bundle.snr_meta)
    end

    obs_result = nothing
    obs_path = joinpath(out_part, "$(participant)_b2b_family_agg.csv")
    tg_path = joinpath(out_part, "$(participant)_tg_metrics.csv")
    skip_obs_for_tg = isfile(obs_path) && B2B_DO_TG && !isfile(tg_path) &&
                      (mode == "observed") && lowercase(get(ENV, "B2B_FORCE", "0")) ∉ ("1", "true", "yes")
    if skip_obs_for_tg
        println("  [skip observed] $obs_path exists; running TG only")
        fam_df = CSV.read(joinpath(out_part, "$(participant)_b2b_family_splits.csv"), DataFrame)
        diag_df = CSV.read(joinpath(out_part, "$(participant)_diagnostics.csv"), DataFrame)
        d0 = sort(diag_df[diag_df.direction .== 0, :], :split)
        valid = .!Bool.(d0.invalid)
        length(valid) == length(bundle.partitions) ||
            error("diagnostics splits $(length(valid)) != n_partitions $(length(bundle.partitions))")
        obs_result = (
            fam_traces=Dict{String,Vector{Vector{Float64}}}(),
            valid_partition=valid,
            n_valid=count(valid),
            diag_df=diag_df,
            agg_df=CSV.read(obs_path, DataFrame),
            pc_df=DataFrame(), fam_df=fam_df,
        )
        qc_row["n_valid_partitions"] = obs_result.n_valid
        run_temporal_generalization!(bundle, valid; out_part=out_part)
    elseif mode == "observed" || mode == "both" || !isfile(obs_path)
        obs_result = run_observed!(bundle; out_part=out_part)
        qc_row["n_valid_partitions"] = obs_result.n_valid
        qc_row["min_valid_partitions_ok"] = obs_result.n_valid >= MIN_VALID_PARTITIONS
        d0 = obs_result.diag_df[obs_result.diag_df.direction .== 0, :]
        qc_row["any_lambda_boundary"] = any(d0.on_boundary_1 .| d0.on_boundary_2)
        qc_row["any_kappa_warn"] = any(d0.warn_kappa)
        qc_row["any_invalid"] = any(d0.invalid)
        qc_row["max_kappa"] = maximum(d0.max_kappa)

        # Do NOT silently fall back to K=4 — report and continue; operator decides
        if obs_result.n_valid < MIN_VALID_PARTITIONS || qc_row["any_invalid"] ||
           qc_row["max_kappa"] > KAPPA_INVALID
            @warn "Conditioning / validity gates failed for $participant " *
                  "(n_valid=$(obs_result.n_valid), maxκ=$(qc_row["max_kappa"])). " *
                  "Do not silently switch K or regularize H; test K=4 in a separate outdir."
        end
        if qc_row["max_kappa"] > KAPPA_WARN
            @warn "Held-out design κ > KAPPA_WARN ($(KAPPA_WARN)) for $participant: " *
                  "$(qc_row["max_kappa"]) — investigate if above 32, refuse if above 85."
        end
    else
        fam_df = CSV.read(joinpath(out_part, "$(participant)_b2b_family_splits.csv"), DataFrame)
        diag_df = CSV.read(joinpath(out_part, "$(participant)_diagnostics.csv"), DataFrame)
        d0 = diag_df[diag_df.direction .== 0, :]
        valid = .!Bool.(d0.invalid)
        fam_traces = Dict(f => Vector{Vector{Float64}}() for f in FAMILY_NAMES)
        for pid in sort(unique(fam_df.split))
            for f in FAMILY_NAMES
                sub = sort(fam_df[(fam_df.split .== pid) .& (fam_df.family .== f), :], :time)
                # Prefer primary score column; fall back to legacy family_score-as-trace
                col = "family_score" in names(sub) ? sub.family_score : sub[!, end]
                push!(fam_traces[f], Vector{Float64}(col))
            end
        end
        obs_result = (
            fam_traces=fam_traces,
            valid_partition=valid,
            n_valid=count(valid),
            diag_df=diag_df,
            agg_df=CSV.read(obs_path, DataFrame),
            pc_df=DataFrame(), fam_df=fam_df,
        )
        qc_row["n_valid_partitions"] = obs_result.n_valid
    end

    if mode in ("null", "both")
        run_null_chunk!(bundle, obs_result; out_part=out_part)
        sum_files = filter(f -> occursin(r"_null_\d+_\d+_summary\.csv$", f), readdir(out_part))
        n_done = 0
        for f in sum_files
            df = CSV.read(joinpath(out_part, f), DataFrame)
            n_done += length(unique(df.null_rep))
        end
        if n_done >= B2B_N_NULL
            ns = aggregate_participant_nulls(participant, out_part; k=k)
            qc_row["null_aggregated"] = true
            qc_row["p_max_over_family"] = ns.p_max_over_family[1]
        else
            qc_row["null_aggregated"] = false
            qc_row["null_reps_done"] = n_done
        end
    end

    manifest = make_run_manifest(
        participant=participant,
        keep_frac=qc_row["keep_frac"],
        n_valid_partitions=get(qc_row, "n_valid_partitions", missing),
        pca_k_used=k,
        epoch_anchor=qc.epoch_anchor,
        anchor_column=qc.anchor_column,
        nuisance_names=bundle.nuisance_names,
    )
    write_json(joinpath(out_part, "$(participant)_run_manifest.json"), manifest)
    write_json(joinpath(out_part, "$(participant)_qc_summary.json"), qc_row)
    CSV.write(joinpath(out_part, "$(participant)_qc_summary.csv"), DataFrame([qc_row]))
    println("Done $participant → $out_part")
    return qc_row
end

# Back-compat alias
const run_subject_v2 = run_subject_v4
