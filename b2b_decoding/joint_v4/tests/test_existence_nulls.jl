#!/usr/bin/env julia
# Unit tests for Phase 1 existence-null machinery (no EEG).
#   julia --project=.. tests/test_existence_nulls.jl

using Test
using LinearAlgebra
using Statistics
using Random

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "estimator.jl"))
include(joinpath(@__DIR__, "..", "existence_nulls.jl"))

@testset "allowed circular shifts exclude |k| < 20" begin
    allowed = allowed_circular_shifts(1753; min_abs=20)
    @test minimum(allowed) == 20
    @test maximum(allowed) == 1753 - 20
    @test all(k -> min(k, 1753 - k) >= 20, allowed)
    @test 19 ∉ allowed
    @test 1734 ∉ allowed
    Random.seed!(1)
    rng = MersenneTwister(0)
    for _ in 1:200
        k = draw_section_shift(rng, 1800; min_abs=20)
        @test abs(k) >= 20
        @test min(k, 1800 - k) >= 20
    end
end

@testset "joint design shift preserves column Gram" begin
    Random.seed!(2)
    n1, n2, p = 40, 50, 6
    X = randn(n1 + n2, p)
    sec = vcat(fill(1, n1), fill(2, n2))
    Xs = circular_shift_design(X, sec, Dict(1 => 7, 2 => -11))
    # Within-story column correlations are a circular permutation of rows.
    G0 = X[1:n1, :]' * X[1:n1, :]
    G1 = Xs[1:n1, :]' * Xs[1:n1, :]
    @test G0 ≈ G1 atol=1e-10
    @test X ≉ Xs
end

@testset "cached P*Xtrc matches SVD ridge" begin
    Random.seed!(3)
    n, n_ch, n_feat = 80, 8, 5
    Ytr = randn(n, n_ch)
    Xtr = randn(n, n_feat)
    μY = vec(mean(Ytr; dims=1))
    μX = vec(mean(Xtr; dims=1))
    Ytrc = Ytr .- μY'
    Xtrc = Xtr .- μX'
    α = 10.0
    λ = α * LinearAlgebra.tr(Ytrc' * Ytrc) / n_ch
    F = svd(Ytrc; full=false)
    G_svd = ridge_coef_svd(F.U, F.S, F.Vt, Xtrc, λ)
    scale = F.S ./ (F.S .^ 2 .+ λ)
    P = F.Vt' * (scale .* F.U')
    G_cached = P * Xtrc
    @test G_cached ≈ G_svd atol=1e-10
    @test G_cached ≈ ridge_coef(Ytrc, Xtrc, λ) atol=1e-8
end

@testset "cached one-direction matches uncached G then H" begin
    Random.seed!(4)
    n_ch, n_times, n_words, n_feat = 6, 4, 60, 3
    Y = randn(n_ch, n_times, n_words)
    X = randn(n_words, n_feat)
    train = vcat(trues(30), falses(30))
    test = .!train
    α = 3.1622776601683795
    cache = prepare_eeg_ridge_cache(Y, train, test, α)
    S, fac = b2b_one_direction_cached(cache, X, train, test)
    @test fac.full_rank
    @test size(S) == (n_times, n_feat)
    @test all(isfinite, S)

    t = 2
    Ytr = Matrix(transpose(Y[:, t, train]))
    Yte = Matrix(transpose(Y[:, t, test]))
    Xtr = X[train, :]
    Xte = X[test, :]
    μY = vec(mean(Ytr; dims=1))
    μX = vec(mean(Xtr; dims=1))
    Ytrc = Ytr .- μY'
    Xtrc = Xtr .- μX'
    Ytec = Yte .- μY'
    Xtec = Xte .- vec(mean(Xte; dims=1))'
    λ = α * LinearAlgebra.tr(Ytrc' * Ytrc) / n_ch
    F = svd(Ytrc; full=false)
    G = ridge_coef_svd(F.U, F.S, F.Vt, Xtrc, λ)
    Xhatc = (Ytec * G)
    Xhatc .-= vec(mean(Xhatc; dims=1))'
    H, _, _, ok = ols_H_diagnostics(Xtec, Xhatc)
    @test ok
    @test S[t, :] ≈ diag(H) atol=1e-8
end

@testset "family trace is sum of column diags" begin
    S = [1.0 2.0 3.0 4.0; 0.5 0.5 0.5 0.5]
    T = family_trace_from_S(S, [2, 2])
    @test T[1, 1] ≈ 3.0
    @test T[1, 2] ≈ 7.0
    @test T[2, 1] ≈ 1.0
end
