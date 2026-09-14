# =====================================================================
# Phase 1 existence nulls: circular-shift the joint design, reuse α.
# Stage-1 G = (Y'Y + λI)⁻¹ Y'X. Cache the EEG-only factor once per
# partition / direction / time; each shift is one Y'X product + cheap H.
# =====================================================================

using Random
using Statistics

const EXISTENCE_MIN_SHIFT = parse(Int, get(ENV, "B2B_EXISTENCE_MIN_SHIFT", "20"))
const EXISTENCE_N_NULL = parse(Int, get(ENV, "B2B_EXISTENCE_N_NULL", string(B2B_N_NULL)))
const EXISTENCE_SEED_OFFSET = parse(Int, get(ENV, "B2B_EXISTENCE_SEED_OFFSET", "100000"))
const B2B_OBSERVED_OUTDIR = get(ENV, "B2B_OBSERVED_OUTDIR", "")
# whole_design = joint circular shift of all X (legacy / tone_v3).
# freedman_lane = circular-shift residuals of the last family after OLS on
# the reduced (nested) design; keeps X_red–X_add covariance.
const EXISTENCE_NULL_METHOD = lowercase(get(ENV, "B2B_EXISTENCE_NULL", "whole_design"))
EXISTENCE_NULL_METHOD in ("whole_design", "freedman_lane") ||
    error("B2B_EXISTENCE_NULL must be whole_design|freedman_lane")

"""Unique circular shifts whose wrap-distance is ≥ min_abs."""
function allowed_circular_shifts(n_words::Integer; min_abs::Integer=EXISTENCE_MIN_SHIFT)
    n_words > 2 * min_abs ||
        error("section length $n_words too short for min |k|=$min_abs")
    return collect(min_abs:(n_words - min_abs))
end

function draw_section_shift(rng::AbstractRNG, n_words::Integer;
                            min_abs::Integer=EXISTENCE_MIN_SHIFT)
    return rand(rng, allowed_circular_shifts(n_words; min_abs=min_abs))
end

"""Shift every column of X jointly within each story (preserves collinearity)."""
function circular_shift_design(X::AbstractMatrix, sec_ids::AbstractVector{Int},
                               shifts::Dict{Int,Int})
    Xs = Matrix{Float64}(X)
    for sid in unique(sec_ids)
        rows = findall(==(sid), sec_ids)
        k = shifts[sid]
        Xs[rows, :] = circshift(view(X, rows, :), (k, 0))
    end
    return Xs
end

"""OLS residualize the last family on the leading reduced columns; shift residuals."""
function freedman_lane_prepare(X::AbstractMatrix, n_red::Integer)
    n_red >= 1 || error("Freedman–Lane needs a nonempty reduced design")
    n_red < size(X, 2) || error("Freedman–Lane needs an added family")
    Xr = X[:, 1:n_red]
    Xa = X[:, (n_red + 1):end]
    B = Xr \ Xa
    fit = Xr * B
    resid = Xa .- fit
    return (n_red=n_red, fit=fit, resid=resid)
end

function freedman_lane_shift(X::AbstractMatrix, sec_ids::AbstractVector{Int},
                             shifts::Dict{Int,Int}, fl)
    Xs = Matrix{Float64}(X)
    Rs = circular_shift_design(fl.resid, sec_ids, shifts)
    Xs[:, (fl.n_red + 1):end] = fl.fit .+ Rs
    return Xs
end

"""P = (Y'Y + λI)⁻¹ Y' so G = P * Xtrc. Y-only; reuse across shifts."""
function prepare_eeg_ridge_cache(Y::AbstractArray{<:Real,3},
                                 train_idx::AbstractVector{Bool},
                                 test_idx::AbstractVector{Bool},
                                 α::Real)
    n_ch, n_times, _ = size(Y)
    cache = Vector{NamedTuple{(:P, :Ytec, :λ),
        Tuple{Matrix{Float64},Matrix{Float64},Float64}}}(undef, n_times)
    for t in 1:n_times
        Ytr = Matrix(transpose(Y[:, t, train_idx]))
        Yte = Matrix(transpose(Y[:, t, test_idx]))
        μY = vec(mean(Ytr; dims=1))
        Ytrc = Ytr .- μY'
        Ytec = Yte .- μY'
        λ = α * LinearAlgebra.tr(Ytrc' * Ytrc) / n_ch
        F = svd(Ytrc; full=false)
        scale = F.S ./ (F.S .^ 2 .+ λ)
        P = F.Vt' * (scale .* F.U')
        cache[t] = (P=P, Ytec=Ytec, λ=λ)
    end
    return cache
end

"""One-direction B2B with cached EEG factor and frozen α (no λ retune)."""
function b2b_one_direction_cached(cache, X::AbstractMatrix,
                                  train_idx::AbstractVector{Bool},
                                  test_idx::AbstractVector{Bool})
    n_times = length(cache)
    n_feat = size(X, 2)
    Xtr = X[train_idx, :]
    Xte = X[test_idx, :]
    μX = vec(mean(Xtr; dims=1))
    Xtrc = Xtr .- μX'
    Xtec = Xte .- vec(mean(Xte; dims=1))'
    Hfac = prepare_H_factor(Xtec)
    S = fill(NaN, n_times, n_feat)
    if !Hfac.full_rank
        return S, Hfac
    end
    for t in 1:n_times
        G = cache[t].P * Xtrc
        Xhat = cache[t].Ytec * G
        Xhatc = Xhat .- vec(mean(Xhat; dims=1))'
        H, _, _, ok = ols_H_diagnostics(Xtec, Xhatc; pre=Hfac)
        if ok
            @inbounds for k in 1:n_feat
                S[t, k] = H[k, k]
            end
        end
    end
    return S, Hfac
end

function load_observed_alphas(obs_dir::AbstractString, participant::AbstractString)
    path = joinpath(obs_dir, participant, "$(participant)_diagnostics.csv")
    isfile(path) || error("Missing observed diagnostics: $path")
    diag = CSV.read(path, DataFrame)
    alphas = Dict{Tuple{Int,Int},Float64}()
    valid = Dict{Int,Bool}()
    for row in eachrow(diag)
        split = Int(row.split)
        dir = Int(row.direction)
        if dir == 0
            valid[split] = !Bool(row.invalid)
        elseif dir == 1 || dir == 2
            α = Float64(row.alpha_1)
            isfinite(α) || error("Non-finite α for $participant split=$split dir=$dir")
            alphas[(split, dir)] = α
        end
    end
    return alphas, valid
end

function confirm_mask(times::AbstractVector{<:Real}; win=CONFIRM_WIN_S)
    return (times .>= win[1]) .& (times .<= win[2])
end

function family_trace_from_S(S::AbstractMatrix, family_ks::AbstractVector{<:Integer})
    n_times = size(S, 1)
    n_fam = length(family_ks)
    T = fill(NaN, n_times, n_fam)
    col0 = 0
    for (i, kk) in enumerate(family_ks)
        cols = (col0 + 1):(col0 + kk)
        col0 += kk
        for t in 1:n_times
            T[t, i] = sum(S[t, cols])
        end
    end
    return T
end

function write_existence_outputs(out_part::AbstractString, participant::AbstractString;
                                 times, feat_names, family_ks, family_names,
                                 S_null, fam_trace, shift_sec1, shift_sec2, n_valid,
                                 alphas_used, n_ok_partitions)
    mkpath(out_part)
    cwin = confirm_mask(times)
    n_shifts, n_times, n_feat = size(S_null)
    n_fam = length(family_ks)
    confirm = fill(NaN, n_shifts, n_feat)
    fam_confirm = fill(NaN, n_shifts, n_fam)
    for s in 1:n_shifts
        for j in 1:n_feat
            confirm[s, j] = mean(S_null[s, cwin, j])
        end
        for f in 1:n_fam
            fam_confirm[s, f] = mean(fam_trace[s, cwin, f])
        end
    end

    NPZ.npzwrite(joinpath(out_part, "$(participant)_existence_S_null.npy"), S_null)
    NPZ.npzwrite(joinpath(out_part, "$(participant)_existence_family_trace.npy"), fam_trace)
    NPZ.npzwrite(joinpath(out_part, "$(participant)_existence_confirm.npy"), confirm)
    NPZ.npzwrite(joinpath(out_part, "$(participant)_existence_family_confirm.npy"), fam_confirm)
    NPZ.npzwrite(joinpath(out_part, "$(participant)_existence_times.npy"), collect(Float64, times))

    rows = NamedTuple[]
    for s in 1:n_shifts
        col0 = 0
        for (fi, fam) in enumerate(family_names)
            kk = family_ks[fi]
            for j in 1:kk
                col = col0 + j
                push!(rows, (
                    participant=participant, shift=s, family=fam, column=j,
                    feat_name=feat_names[col],
                    confirm_mean=confirm[s, col],
                    shift_sec1=shift_sec1[s], shift_sec2=shift_sec2[s],
                    n_valid_partitions=n_valid[s],
                ))
            end
            col0 += kk
            push!(rows, (
                participant=participant, shift=s, family=fam, column=0,
                feat_name="$(fam)_trace",
                confirm_mean=fam_confirm[s, fi],
                shift_sec1=shift_sec1[s], shift_sec2=shift_sec2[s],
                n_valid_partitions=n_valid[s],
            ))
        end
    end
    CSV.write(joinpath(out_part, "$(participant)_existence_confirm.csv"), DataFrame(rows))

    write_json(joinpath(out_part, "$(participant)_existence_meta.json"), Dict(
        "participant" => participant,
        "n_shifts" => n_shifts,
        "n_times" => n_times,
        "n_feat" => n_feat,
        "family_names" => collect(family_names),
        "family_ks" => collect(Int, family_ks),
        "feat_names" => collect(String, feat_names),
        "confirm_win_s" => collect(CONFIRM_WIN_S),
        "min_shift" => EXISTENCE_MIN_SHIFT,
        "seed_offset" => EXISTENCE_SEED_OFFSET,
        "reuse_lambda" => true,
        "n_ok_partitions_observed" => n_ok_partitions,
        "alphas" => Dict("$(s)_$(d)" => α for ((s, d), α) in alphas_used),
        "method" => "circular_shift_joint_design_cached_YTY",
        "null_family" => EXISTENCE_NULL_METHOD,
        "test" => "one_sided_positive_group_mean",
        "p_formula" => "(1 + #{T* >= T}) / (B + 1)",
        "epoch_anchor" => B2B_EPOCH_ANCHOR,
        "tmin_s" => TMIN_S,
        "tmax_s" => TMAX_S,
        "observed_outdir" => B2B_OBSERVED_OUTDIR,
        "outdir" => B2B_OUTDIR,
    ))
    return confirm, fam_confirm
end

function run_existence_nulls!(participant::AbstractString;
                              n_null::Integer=EXISTENCE_N_NULL,
                              min_shift::Integer=EXISTENCE_MIN_SHIFT)
    isempty(B2B_OBSERVED_OUTDIR) &&
        error("B2B_OBSERVED_OUTDIR must point at the observed outdir")
    if occursin("passthrough", B2B_OUTDIR) && !occursin("existence", B2B_OUTDIR)
        error("Refusing existence writes into frozen passthrough outdir: $B2B_OUTDIR")
    end
    if B2B_OUTDIR == B2B_OBSERVED_OUTDIR && !B2B_ALLOW_INPLACE_NULLS
        error("B2B_OUTDIR equals observed outdir; set B2B_ALLOW_INPLACE_NULLS=1 for band jobs")
    end
    println("\n", "#"^80, "\n# EXISTENCE NULLS  $participant  n_null=$n_null  " *
            "min|k|=$min_shift  reuse α\n", "#"^80)

    assert_outdir_manifest!(B2B_OUTDIR)
    out_part = joinpath(B2B_OUTDIR, participant)
    mkpath(out_part)
    done_csv = joinpath(out_part, "$(participant)_existence_confirm.csv")
    if isfile(done_csv) && lowercase(get(ENV, "B2B_FORCE", "0")) ∉ ("1", "true", "yes")
        println("  [skip] $done_csv exists")
        return nothing
    end

    bundle = load_participant_bundle(participant; k=B2B_PCA_K)
    alphas, valid_obs = load_observed_alphas(B2B_OBSERVED_OUTDIR, participant)
    qc = bundle.qc
    times = bundle.times
    fks = bundle.family_ks
    n_feat = size(bundle.X_joint, 2)
    n_fam = length(FAMILY_NAMES)
    n_times = length(times)
    n_part = length(bundle.partitions)
    n_ok_obs = count(get(valid_obs, i, false) for i in 1:n_part)
    n_ok_obs >= MIN_VALID_PARTITIONS ||
        error("$participant has $n_ok_obs valid observed partitions")

    shifts = Vector{Dict{Int,Int}}(undef, n_null)
    shift_sec1 = zeros(Int, n_null)
    shift_sec2 = zeros(Int, n_null)
    for s in 1:n_null
        rng = Random.MersenneTwister(_participant_seed(participant, EXISTENCE_SEED_OFFSET + s))
        d = Dict{Int,Int}()
        for sid in 1:length(SECTIONS)
            d[sid] = draw_section_shift(rng, bundle.n_words_per_sec[sid]; min_abs=min_shift)
        end
        shifts[s] = d
        shift_sec1[s] = d[1]
        shift_sec2[s] = d[2]
    end

    n_red = sum(fks[1:max(0, n_fam - 1)])
    fl = if EXISTENCE_NULL_METHOD == "freedman_lane"
        n_fam >= 2 || error("freedman_lane needs a nested added family")
        println("  Null method = Freedman–Lane residual shift of last family  n_red=$n_red")
        freedman_lane_prepare(bundle.X_joint, n_red)
    else
        println("  Null method = whole-design circular shift")
        nothing
    end
    # Stream shifts (do not store n_null full X copies; B=5000 would be ~10 GB).
    acc = zeros(Float64, n_null, n_times, n_feat)
    n_valid = zeros(Int, n_null)
    alphas_used = Dict{Tuple{Int,Int},Float64}()

    for (pid, part) in enumerate(bundle.partitions)
        get(valid_obs, pid, false) || continue
        haskey(alphas, (pid, 1)) && haskey(alphas, (pid, 2)) ||
            error("Missing observed α for $participant split=$pid")
        α1 = alphas[(pid, 1)]
        α2 = alphas[(pid, 2)]
        alphas_used[(pid, 1)] = α1
        alphas_used[(pid, 2)] = α2

        half1, half2 = make_partition_halves(
            qc.section_id, qc.word_idx_in_sec, bundle.n_words_per_sec, qc.keep, part;
            n_blocks=N_BLOCKS_PER_SECTION, guard=GUARD_WORDS)
        snr_meta = hasproperty(bundle, :snr_meta) ? bundle.snr_meta : nothing
        dat_use = maybe_partition_spatial(bundle.dat_3d, half1, snr_meta)
        use = half1 .| half2
        idx = findall(use)
        Y = dat_use[:, :, idx]
        h1 = half1[idx]
        h2 = half2[idx]

        println("  [partition $pid] cache EEG SVD  half1=$(sum(h1)) half2=$(sum(h2))  " *
                "α1=$(round(α1; sigdigits=4)) α2=$(round(α2; sigdigits=4))")
        flush(stdout)
        cache1 = prepare_eeg_ridge_cache(Y, h1, h2, α1)
        cache2 = prepare_eeg_ridge_cache(Y, h2, h1, α2)

        for s in 1:n_null
            Xfull = if fl === nothing
                circular_shift_design(bundle.X_joint, qc.section_id, shifts[s])
            else
                freedman_lane_shift(bundle.X_joint, qc.section_id, shifts[s], fl)
            end
            Xsh = Xfull[idx, :]
            S1, fac1 = b2b_one_direction_cached(cache1, Xsh, h1, h2)
            S2, fac2 = b2b_one_direction_cached(cache2, Xsh, h2, h1)
            (fac1.full_rank && fac2.full_rank) || continue
            acc[s, :, :] .+= 0.5 .* (S1 .+ S2)
            n_valid[s] += 1
        end
        if pid == 1 || pid % 5 == 0 || pid == n_part
            println("    done partition $pid  mean n_valid=$(round(mean(n_valid); digits=1))")
            flush(stdout)
        end
    end

    any(n_valid .== 0) && @warn "Some shifts had zero valid partitions" participant
    S_null = similar(acc)
    for s in 1:n_null
        if n_valid[s] > 0
            S_null[s, :, :] .= acc[s, :, :] ./ n_valid[s]
        else
            S_null[s, :, :] .= NaN
        end
    end

    fam_trace = fill(NaN, n_null, n_times, n_fam)
    for s in 1:n_null
        fam_trace[s, :, :] = family_trace_from_S(S_null[s, :, :], fks)
    end

    write_existence_outputs(out_part, participant;
                            times=times, feat_names=bundle.feat_names,
                            family_ks=fks, family_names=FAMILY_NAMES,
                            S_null=S_null, fam_trace=fam_trace,
                            shift_sec1=shift_sec1, shift_sec2=shift_sec2,
                            n_valid=n_valid, alphas_used=alphas_used,
                            n_ok_partitions=n_ok_obs)
    println("  wrote $out_part")
    return out_part
end
