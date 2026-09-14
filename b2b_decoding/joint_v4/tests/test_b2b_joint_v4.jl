#!/usr/bin/env julia
# Validation tests for joint B2B v4 (integrity without EEG where possible)
# Run: julia --project=.. tests/test_b2b_joint_v4.jl

using Test
using LinearAlgebra
using Statistics
using Random
using JSON3
using SHA
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "data_io.jl"))
include(joinpath(@__DIR__, "..", "eeg_band.jl"))
include(joinpath(@__DIR__, "..", "spatial_denoise.jl"))
include(joinpath(@__DIR__, "..", "partitions.jl"))
include(joinpath(@__DIR__, "..", "estimator.jl"))
include(joinpath(@__DIR__, "..", "feature_basis.jl"))
include(joinpath(@__DIR__, "..", "pipeline.jl"))

@testset "cohort table counts" begin
    df, parts = load_cohort(joinpath(@__DIR__, "..", "cohort_groups.csv"))
    @test length(parts) == 63
    @test count(df.group .== "TD") == 24
    @test count(df.group .== "dyslexia_normal_CAP") == 19
    @test count(df.group .== "dyslexia_atypical_CAP") == 20
    @test cohort_hash() == file_sha256(joinpath(@__DIR__, "..", "cohort_groups.csv"))
end

@testset "config defaults (Large V2 / onset / independent)" begin
    @test startswith(FEAT_ACOUSTIC, "whisperlargev2")
    @test FEAT_LANGUAGE == "whisperlargev2_language_audio_fused"
    @test B2B_EPOCH_ANCHOR == "onset"
    @test B2B_BASIS_MODE == "independent"
    @test TMIN_S == -0.2
    @test TMAX_S == 1.0
    @test CONFIRM_WIN_S == (0.0, 0.8)
    @test B2B_PCA_K == 8
end

@testset "shared Large V2 feature shapes and integrity" begin
    checksums_all = Dict{String,String}()
    for sid in SECTIONS
        Xw, Xn, csums = load_section_features(B2B_EXTRACTOR_DIR, sid, FEATURE_SETS)
        @test size(Xw[FEAT_ACOUSTIC]) == (EXPECTED_SEC_LENGTHS[sid], 12800)
        @test size(Xw[FEAT_SPEECH]) == (EXPECTED_SEC_LENGTHS[sid], 12800)
        @test size(Xw[FEAT_LANGUAGE]) == (EXPECTED_SEC_LENGTHS[sid], 1280)
        merge!(checksums_all, Dict("s$(sid)_$k" => v for (k, v) in csums))
        # text-only present as sensitivity feature
        Xt, _, _ = load_section_features(B2B_EXTRACTOR_DIR, sid, [FEAT_LANGUAGE_TEXTONLY])
        @test size(Xt[FEAT_LANGUAGE_TEXTONLY]) == (EXPECTED_SEC_LENGTHS[sid], 1280)
    end
    @test sum(EXPECTED_SEC_LENGTHS) == EXPECTED_N_WORDS == 3553
    @test length(checksums_all) == 6  # 3 features × 2 sections
end

@testset "feature validation failures" begin
    # all-zero row
    Xbad = randn(10, 4)
    Xbad[3, :] .= 0
    @test_throws ErrorException validate_feature_matrix!(Xbad, "dummy", 1)
    # non-finite
    Xnan = randn(10, 4)
    Xnan[2, 1] = NaN
    @test_throws ErrorException validate_feature_matrix!(Xnan, "dummy", 1)
    # wrong n_words for section 1
    Xwrong = randn(100, 12800)
    @test_throws ErrorException validate_feature_matrix!(Xwrong, "whisperlargev2_acoustic", 1)
end

@testset "synthetic EEG impulse: t=0 indexes onset not offset" begin
    Random.seed!(7)
    n_ch, T = 4, 500
    sfreq = 100.0
    # Word duration 0.1 s so onset stays inside the −0.2 s pre-window when offset-locked
    onset0, offset0 = 100, 110
    eeg = zeros(n_ch, T)
    onset_j = to_julia_sample(onset0)
    eeg[:, onset_j] .= 1.0
    wt = DataFrame(onset_sample=[onset0], offset_sample=[offset0])
    tmin, tmax = -0.2, 1.0
    dat_on, _, qc_on = build_epochs_qc(
        [eeg], [eeg], [wt], [nothing], sfreq, tmin, tmax; anchor=:onset)
    times = collect(range(tmin, tmax; length=qc_on.n_times))
    t0 = argmin(abs.(times .- 0.0))
    @test abs(times[t0]) < 1e-9
    @test maximum(abs, dat_on[:, t0, 1]) ≈ 1.0
    # Offset-locked: impulse at t = onset − offset = −0.1 s, not at t=0
    dat_off, _, qc_off = build_epochs_qc(
        [eeg], [eeg], [wt], [nothing], sfreq, tmin, tmax; anchor=:offset)
    times_off = collect(range(tmin, tmax; length=qc_off.n_times))
    t0_off = argmin(abs.(times_off .- 0.0))
    @test maximum(abs, dat_off[:, t0_off, 1]) ≈ 0.0 atol=1e-12
    t_imp = argmin(abs.(times_off .- ((onset0 - offset0) / sfreq)))
    @test abs(times_off[t_imp] - (onset0 - offset0) / sfreq) < 1e-9
    @test maximum(abs, dat_off[:, t_imp, 1]) ≈ 1.0
    @test qc_on.epoch_anchor == "onset"
    @test qc_on.anchor_column == "onset_sample"
    @test qc_off.anchor_column == "offset_sample"
end

@testset "partition balance" begin
    parts, counts = generate_balanced_partitions(20; seed=42)
    @test length(parts) == 20
    for sec in 1:2, b in 1:8
        @test 8 <= counts[sec][b] <= 12
    end
end

@testset "family_scores trace and trace/K consistency" begin
    Random.seed!(9)
    n_times, k = 20, 8
    S = randn(n_times, 3k + 2)  # +2 nuisance columns
    fam = family_scores_from_S(S, k)
    for (i, f) in enumerate(FAMILY_NAMES)
        cols = ((i - 1) * k + 1):(i * k)
        tr = [sum(S[t, cols]) for t in 1:n_times]
        @test fam.traces[f] ≈ tr
        @test fam.scores[f] ≈ tr ./ k
    end
    # K=4/8/16 consistency of primary score definition
    for kk in (4, 8, 16)
        Sk = randn(n_times, 3kk)
        fkk = family_scores_from_S(Sk, kk)
        for f in FAMILY_NAMES
            @test fkk.scores[f] ≈ fkk.traces[f] ./ kk
        end
    end
end

@testset "independent mode does not force cross-family correlations to zero" begin
    Random.seed!(11)
    n, p, k = 400, 40, 4
    X_ac = randn(n, p)
    X_sp = 0.8 .* X_ac .+ 0.2 .* randn(n, p)  # strongly related
    X_lg = randn(n, p)
    Z_ac, _ = fit_stimulus_pca(X_ac; n_components=8, seed=1)
    Z_sp, _ = fit_stimulus_pca(X_sp; n_components=8, seed=2)
    Z_lg, _ = fit_stimulus_pca(X_lg; n_components=8, seed=3)
    Z_by = Dict("acoustic" => Z_ac, "speech" => Z_sp, "language" => Z_lg)
    gate = basis_gate_b_diagnostics(Z_by; k=k)
    # Independent PCA on related embeddings must retain nonzero canon corr
    @test gate.canon_ac_sp > 0.3
end

@testset "ordered mode forces near-zero cross-family canon" begin
    Random.seed!(12)
    n, p, k = 400, 40, 4
    X_ac = randn(n, p)
    X_sp = 0.8 .* X_ac .+ 0.2 .* randn(n, p)
    X_lg = 0.5 .* X_ac .+ 0.5 .* X_sp .+ 0.2 .* randn(n, p)
    Z_ac, _ = fit_stimulus_pca(X_ac; n_components=8, seed=1)
    Z_ac_k = Z_ac[:, 1:k]
    Z_sp, _, _ = _fit_orthogonal_family(X_sp, Z_ac_k; n_components=8, seed=2)
    Z_sp_k = Z_sp[:, 1:k]
    Z_lg, _, _ = _fit_orthogonal_family(X_lg, hcat(Z_ac_k, Z_sp_k); n_components=8, seed=3)
    Z_by = Dict("acoustic" => Z_ac, "speech" => Z_sp, "language" => Z_lg)
    gate = basis_gate_b_diagnostics(Z_by; k=k)
    @test maximum(abs, (gate.canon_ac_sp, gate.canon_ac_lg, gate.canon_sp_lg)) < 0.05
    @test gate.cond_joint ≈ 1.0 rtol=0.1
end

@testset "nuisance unshifted while embeddings shift together" begin
    Random.seed!(13)
    n1, n2, k = 50, 60, 4
    n = n1 + n2
    Z = Dict(
        "acoustic" => randn(n, k),
        "speech" => randn(n, k),
        "language" => randn(n, k),
    )
    sec = vcat(fill(1, n1), fill(2, n2))
    Nuis, nnames, _ = begin
        # Synthetic standardized nuisance
        N = randn(n, length(NUISANCE_NAMES))
        N, NUISANCE_NAMES, nothing
    end
    shifts = Dict(1 => 7, 2 => 11)
    Zs = circular_shift_pcs(Z, sec, shifts)
    X0, names0 = joint_design(Z; k=k, nuisance=Nuis, nuisance_names=nnames)
    Xs, namess = joint_design(Zs; k=k, nuisance=Nuis, nuisance_names=nnames)
    @test names0 == namess
    # Embedding block changed
    @test X0[:, 1:3k] != Xs[:, 1:3k]
    # Nuisance block identical (unshifted)
    @test X0[:, 3k+1:end] == Xs[:, 3k+1:end]
    # Common shift across families
    @test Zs["acoustic"][1:n1, :] == Z["acoustic"][circshift(1:n1, shifts[1]), :]
    @test Zs["speech"][1:n1, :] == Z["speech"][circshift(1:n1, shifts[1]), :]
end

@testset "nuisance builder from shared timing" begin
    N, names, fit = build_nuisance_matrix(B2B_EXTRACTOR_DIR)
    @test size(N) == (EXPECTED_N_WORDS, length(NUISANCE_NAMES))
    @test names == NUISANCE_NAMES
    @test all(isfinite, N)
    # Standardized
    @test all(abs.(vec(mean(N; dims=1))) .< 1e-8)
    @test all(isapprox.(vec(std(N; dims=1, corrected=true)), 1.0; atol=1e-6))
end

@testset "manifest fingerprint refuse-append" begin
    tmp = mktempdir()
    fp = make_config_fingerprint()
    write_json(joinpath(tmp, "_analysis_manifest.json"), merge(Dict{String,Any}(fp), software_versions()))
    existing = JSON3.read(read(joinpath(tmp, "_analysis_manifest.json"), String), Dict{String,Any})
    locked = make_config_fingerprint()
    ex_locked = Dict{String,Any}()
    for k in keys(locked)
        ks = string(k)
        haskey(existing, ks) || error("missing key $ks in written manifest")
        ex_locked[ks] = existing[ks]
    end
    @test _fingerprint_equal(ex_locked, locked)
    existing["basis_mode"] = "ordered"
    mutated = Dict{String,Any}(string(k) => existing[string(k)] for k in keys(locked))
    @test !_fingerprint_equal(mutated, locked)
end

@testset "null aggregation detects missing/duplicate chunks" begin
    tmp = mktempdir()
    # Minimal observed agg
    times = collect(range(-0.2, 1.0; length=5))
    rows = []
    for t in times, f in FAMILY_NAMES
        push!(rows, (participant="TEST", time=t, family=f,
                     mean_score=0.1, mean_trace=0.8, split_sd=0.01, n_valid_splits=20))
    end
    CSV.write(joinpath(tmp, "TEST_b2b_family_agg.csv"), DataFrame(rows))
    # Only 2 of expected null reps
    sum_rows = [
        (participant="TEST", pca_k=8, null_rep=1,
         max_acoustic=0.1, max_speech=0.1, max_language=0.1,
         max_over_family=0.1, n_valid_partitions=20, shift_sec1=10, shift_sec2=12),
        (participant="TEST", pca_k=8, null_rep=2,
         max_acoustic=0.1, max_speech=0.1, max_language=0.1,
         max_over_family=0.1, n_valid_partitions=20, shift_sec1=11, shift_sec2=13),
        # duplicate rep 1
        (participant="TEST", pca_k=8, null_rep=1,
         max_acoustic=0.2, max_speech=0.2, max_language=0.2,
         max_over_family=0.2, n_valid_partitions=20, shift_sec1=10, shift_sec2=12),
    ]
    CSV.write(joinpath(tmp, "TEST_null_0001_0002_summary.csv"), DataFrame(sum_rows))
    @test_throws Exception aggregate_participant_nulls("TEST", tmp; n_null_total=200, k=8)
end

@testset "drift basis spans constant" begin
    onset = cumsum(rand(80) .+ 0.05)
    B = _drift_basis(onset; n_knots=20)
    ones_col = ones(size(B, 1))
    @test isapprox(B * (B \ ones_col), ones_col; rtol=1e-10)
end

@testset "joint B2B recovers active families" begin
    Random.seed!(1)
    n_ch, n_times, n_trials = 20, 30, 400
    k = 4
    X_ac = randn(n_trials, k)
    X_sp = 0.6 .* X_ac .+ 0.4 .* randn(n_trials, k)
    X_lg = randn(n_trials, k)
    X = hcat(X_ac, X_sp, X_lg)
    W_ac = randn(n_ch, k) ./ 5
    W_sp = randn(n_ch, k) ./ 5
    Y = Array{Float64}(undef, n_ch, n_times, n_trials)
    for t in 1:n_times
        signal = W_ac * X_ac' .+ W_sp * X_sp'
        Y[:, t, :] = signal .+ 0.5 .* randn(n_ch, n_trials)
    end
    times = range(-0.2, 0.8; length=n_times) |> collect
    half1 = falses(n_trials); half1[1:2:end] .= true
    half2 = .!half1
    sec = ones(Int, n_trials)
    blk = [mod1(i, 8) for i in 1:n_trials]
    S1, d1 = b2b_one_direction_joint(Y, X, half1, half2, sec, blk, times)
    S2, d2 = b2b_one_direction_joint(Y, X, half2, half1, sec, blk, times)
    S = 0.5 .* (S1 .+ S2)
    fam = family_scores_from_S(S, k)
    mid = (times .>= 0.0) .& (times .<= 0.8)
    μ = Dict(f => mean(fam.scores[f][mid]) for f in FAMILY_NAMES)
    println("  synthetic family mean scores (trace/K): ", μ)
    @test μ["acoustic"] > 0.01
    @test μ["speech"] > 0.01
    @test abs(μ["language"]) < 0.5 * min(μ["acoustic"], μ["speech"])
end

println("\nAll joint B2B v4 tests finished.")
