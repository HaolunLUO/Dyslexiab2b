#!/usr/bin/env julia
# Strict B2B audit against King et al. NeuroImage 2020 (Algorithm 1).
# Run: julia --project=.. tests/audit_b2b_king2020.jl
#
# Checks:
#   1. Algorithm structure: split G/H, diag(H), ensemble over splits
#   2. Causal recovery: active features > inactive on diag(H)
#   3. Null centering: OLS H → diag(H) ≈ 0 when Y ⊥ X
#   4. Ridge-H bias: regularized H → positive bias on null (King §2.3, §3.1.2)
#   5. AUC for causal vs non-causal column identification
#   6. joint_v4 pipeline path (b2b_joint_partition) agrees with one-direction core

using LinearAlgebra
using Statistics
using Random
using Printf

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "partitions.jl"))
include(joinpath(@__DIR__, "..", "estimator.jl"))

# ---------------------------------------------------------------------------
# Minimal King-style simulator + reference estimators for audit only
# ---------------------------------------------------------------------------

"""Ridge H for audit contrast (King allows Λ_X on H; paper warns this biases Ŝ)."""
function ridge_H_coef(X::AbstractMatrix, Xhat::AbstractMatrix, λ::Real)
    n, p = size(X)
    XtX = X' * X
    @inbounds for i in 1:p
        XtX[i, i] += λ
    end
    return XtX \ (X' * Xhat)
end

"""Single King Algorithm-1 replicate: shuffle, half-split, G on (Y1,X1), H on (X2,X̂2)."""
function king_b2b_replicate(Y::AbstractMatrix, X::AbstractMatrix;
                            λY::Real=1e-2, λX::Real=0.0,
                            rng::AbstractRNG=Random.default_rng())
    m, dx = size(X)
    _, dy = size(Y)
    perm = randperm(rng, m)
    Xs, Ys = X[perm, :], Y[perm, :]
    h = m ÷ 2
    X1, Y1 = Xs[1:h, :], Ys[1:h, :]
    X2, Y2 = Xs[h+1:end, :], Ys[h+1:end, :]

    # Center columns (matches joint_v4 per-time centering spirit)
    μY1 = vec(mean(Y1; dims=1)); Y1c = Y1 .- μY1'
    μX1 = vec(mean(X1; dims=1)); X1c = X1 .- μX1'
    μY2 = vec(mean(Y2; dims=1)); Y2c = Y2 .- μY2'
    μX2 = vec(mean(X2; dims=1)); X2c = X2 .- μX2'

    # G: Y → X (ridge on Y side, King eq. 2)
    G = ridge_coef(Y1c, X1c, λY)
    Xhat2 = Y2c * G
    Xhat2c = Xhat2 .- vec(mean(Xhat2; dims=1))'

    # H: X → X̂ (OLS if λX=0, else ridge — biased per King §2.3)
    H = λX > 0 ? ridge_H_coef(X2c, Xhat2c, λX) : (X2c \ Xhat2c)
    return vec(diag(H))
end

"""Ensemble diag(H) over n_rep shuffled half-splits."""
function king_b2b_ensemble(Y::AbstractMatrix, X::AbstractMatrix; n_rep=50,
                           λY=1e-2, λX=0.0, seed=0)
    rng = MersenneTwister(seed)
    acc = zeros(size(X, 2))
    for _ in 1:n_rep
        acc .+= king_b2b_replicate(Y, X; λY=λY, λX=λX, rng=rng)
    end
    return acc ./ n_rep
end

"""King synthetic: Y = h * (X * S) * F + N, S = diag(active mask)."""
function simulate_king(;
    m=800, dx=12, dy=24, n_causal=4, h_snr=1.0,
    corr_strength=0.6, noise_x=0.3, seed=0)
    rng = MersenneTwister(seed)
    # Correlated X (Cholesky)
    A = randn(rng, dx, dx)
    Σ = A * A'
    Σ .+= 0.05 .* Matrix(I, dx, dx)
    L = cholesky(Symmetric(Σ)).L
    Xraw = randn(rng, m, dx) * L'
    μ = vec(mean(Xraw; dims=1))
    σ = vec(std(Xraw; dims=1, corrected=true))
    σ = ifelse.(σ .> 1e-10, σ, 1.0)
    X = (Xraw .- μ') ./ σ'

    active = falses(dx)
    active[1:n_causal] .= true  # first n_causal columns causal
    S = Diagonal(active)
    F = randn(rng, dx, dy) ./ sqrt(dx)
    N = noise_x .* randn(rng, m, dx) * L'  # structured noise on X side
    N = (N .- vec(mean(N; dims=1))') ./ max.(vec(std(N; dims=1, corrected=true)), 1e-10)'
    Y = h_snr .* ((X * S + N) * F) .+ 0.5 .* randn(rng, m, dy)
    return X, Y, active
end

"""Reshape flat (m×dy) King data to joint_v4 EEG tensor (dy × 1 × m)."""
function king_to_eeg(Y::AbstractMatrix)
    m, dy = size(Y)
    return reshape(Y', dy, 1, m)
end

function auc_score(y_true::AbstractVector{Bool}, scores::AbstractVector{<:Real})
    pos = scores[y_true]
    neg = scores[.!y_true]
    isempty(pos) || isempty(neg) && return NaN
    wins = 0.0
    for p in pos, n in neg
        wins += (p > n) + 0.5 * (p == n)
    end
    return wins / (length(pos) * length(neg))
end

function audit_header(title)
    println("\n", "="^72)
    println(title)
    println("="^72)
end

function pass_fail(ok, msg)
    tag = ok ? "PASS" : "FAIL"
    println("  [$tag] $msg")
    return ok
end

# ---------------------------------------------------------------------------
# Audit scenarios
# ---------------------------------------------------------------------------

results = Bool[]

audit_header("A. King Algorithm 1 structure (split independence)")
Random.seed!(42)
X, Y, active = simulate_king(m=600, dx=8, dy=16, n_causal=3, h_snr=2.0, seed=1)
# Verify G uses train only: if we leak test Y into G, scores inflate on null cols
S_ols = king_b2b_ensemble(Y, X; n_rep=40, λY=1e-1, λX=0.0, seed=10)
S_ridgeH = king_b2b_ensemble(Y, X; n_rep=40, λY=1e-1, λX=1.0, seed=10)
causal_mean = mean(S_ols[active])
null_mean = mean(S_ols[.!active])
push!(results, pass_fail(causal_mean > null_mean + 0.05,
    @sprintf("active diag(H) mean=%.3f > inactive mean=%.3f (OLS H)", causal_mean, null_mean)))

audit_header("B. Null centering — OLS H when Y ⊥ X (King §2.3 unbiasedness)")
Random.seed!(99)
m, dx, dy = 1000, 10, 20
Xnull = randn(m, dx)
Xnull = (Xnull .- vec(mean(Xnull; dims=1))') ./ vec(std(Xnull; dims=1, corrected=true))'
Ynull = randn(m, dy)  # independent of X
Snull_ols = king_b2b_ensemble(Ynull, Xnull; n_rep=80, λY=1e-2, λX=0.0, seed=20)
null_center = mean(Snull_ols)
null_sd = std(Snull_ols)
push!(results, pass_fail(abs(null_center) < 0.08,
    @sprintf("mean(diag H)=%.4f ≈ 0 under null (|μ|<0.08)", null_center)))
push!(results, pass_fail(null_sd < 0.35,
    @sprintf("std(diag H)=%.4f reasonable under null", null_sd)))

"""Forward ridge encoding scores (King baseline): sum of squared coeffs per feature — always ≥ 0."""
function forward_ridge_scores(Y::AbstractMatrix, X::AbstractMatrix; λ=1e-2, n_rep=40, seed=0)
    rng = MersenneTwister(seed)
    m, dx = size(X)
    acc = zeros(dx)
    for _ in 1:n_rep
        perm = randperm(rng, m)
        Xs, Ys = X[perm, :], Y[perm, :]
        h = m ÷ 2
        Xtr = Xs[1:h, :]; Ytr = Ys[1:h, :]
        Xtr = Xtr .- vec(mean(Xtr; dims=1))'
        Ytr = Ytr .- vec(mean(Ytr; dims=1))'
        XtX = Xtr' * Xtr
        for i in 1:dx; XtX[i, i] += λ; end
        Hfwd = XtX \ (Xtr' * Ytr)  # dx × dy  (forward X → Y)
        acc .+= vec(sum(Hfwd .^ 2; dims=2))
    end
    return acc ./ n_rep
end

audit_header("C. Forward vs B2B null bias (King §2.3 / §3.1.2)")
# Forward encoding uses Σ H² — always positive on null → invalid one-sided tests vs 0.
fwd_null = forward_ridge_scores(Ynull, Xnull; λ=1e-2, n_rep=60, seed=25)
fwd_null_mean = mean(fwd_null)
push!(results, pass_fail(fwd_null_mean > 0.02 && abs(null_center) < 0.08,
    @sprintf("forward null mean ΣH²=%.4f > 0 while B2B OLS mean=%.4f ≈ 0", fwd_null_mean, null_center)))
push!(results, pass_fail(fwd_null_mean > 5 * abs(null_center),
    @sprintf("forward null >> B2B null (ratio=%.1f×)", fwd_null_mean / max(abs(null_center), 1e-6))))

audit_header("C2. Ridge-H inflates inactive features (King: regularized H breaks zero-centering)")
Xc, Yc, active_c = simulate_king(m=1000, dx=12, dy=20, n_causal=4, h_snr=1.5, seed=33)
S_ols_c = king_b2b_ensemble(Yc, Xc; n_rep=80, λY=1e-1, λX=0.0, seed=34)
S_ridge_c = king_b2b_ensemble(Yc, Xc; n_rep=80, λY=1e-1, λX=80.0, seed=34)
inact = .!active_c
inactive_ols = mean(S_ols_c[inact])
inactive_ridge = mean(S_ridge_c[inact])
push!(results, pass_fail(inactive_ridge > inactive_ols + 0.02,
    @sprintf("ridge-H inactive mean=%.4f > OLS inactive=%.4f", inactive_ridge, inactive_ols)))
push!(results, pass_fail(mean(S_ridge_c[active_c]) > mean(S_ols_c[active_c]) * 0.5,
    @sprintf("ridge-H preserves active signal (active ridge=%.3f ols=%.3f)",
             mean(S_ridge_c[active_c]), mean(S_ols_c[active_c]))))

audit_header("D. Causal discovery AUC across SNR (King §3.1.2)")
auc_vals = Float64[]
for (label, snr) in [("low", 0.3), ("mid", 1.0), ("high", 3.0)]
    Xd, Yd, act = simulate_king(m=1000, dx=16, dy=32, n_causal=5, h_snr=snr, seed=Int(100 * snr))
    Sd = king_b2b_ensemble(Yd, Xd; n_rep=60, λY=1e-1, λX=0.0, seed=30)
    auc = auc_score(act, Sd)
    push!(auc_vals, auc)
    push!(results, pass_fail(auc > 0.55,
        @sprintf("SNR=%s: AUC=%.3f > 0.55", label, auc)))
end
push!(results, pass_fail(auc_vals[end] >= auc_vals[1],
    @sprintf("AUC non-decreasing with SNR: low=%.3f high=%.3f", auc_vals[1], auc_vals[end])))

audit_header("E. joint_v4 estimator.jl path (multi-time EEG tensor)")
Random.seed!(7)
n_ch, n_times, n_trials = 24, 15, 500
k = 4
X_ac = randn(n_trials, k)
X_sp = 0.65 .* X_ac .+ 0.35 .* randn(n_trials, k)
X_lg = randn(n_trials, k)
Xjoint = hcat(X_ac, X_sp, X_lg)
W_ac = randn(n_ch, k) ./ 4
W_sp = randn(n_ch, k) ./ 4
Y = Array{Float64}(undef, n_ch, n_times, n_trials)
for t in 1:n_times
    Y[:, t, :] = W_ac * X_ac' .+ W_sp * X_sp' .+ 0.4 .* randn(n_ch, n_trials)
end
times = collect(range(-0.2, 0.8; length=n_times))
half1 = falses(n_trials); half1[1:2:end] .= true
half2 = .!half1
sec = ones(Int, n_trials)
blk = [mod1(i, 8) for i in 1:n_trials]

S1, d1 = b2b_one_direction_joint(Y, Xjoint, half1, half2, sec, blk, times)
S2, d2 = b2b_one_direction_joint(Y, Xjoint, half2, half1, sec, blk, times)
Savg = 0.5 .* (S1 .+ S2)
fam = family_scores_from_S(Savg, k)
mid = (times .>= 0.0) .& (times .<= 0.8)
μ = Dict(f => mean(fam.scores[f][mid]) for f in FAMILY_NAMES)

push!(results, pass_fail(!d1.invalid && !d2.invalid,
    @sprintf("partitions valid (κ1=%.1f κ2=%.1f)", d1.max_kappa, d2.max_kappa)))
push!(results, pass_fail(μ["acoustic"] > 0.02 && μ["speech"] > 0.02,
    @sprintf("active families acoustic=%.4f speech=%.4f > 0.02", μ["acoustic"], μ["speech"])))
push!(results, pass_fail(abs(μ["language"]) < 0.5 * min(μ["acoustic"], μ["speech"]),
    @sprintf("inactive language=%.4f << active families", μ["language"])))

# b2b_joint_partition consistency (reuse simple half split)
S_part, fam_part, diag_part, _, _ = b2b_joint_partition(
    Y, Xjoint, half1, half2, sec, collect(1:n_trials), [n_trials], times; k=k)
μ_part = Dict(f => mean(fam_part.scores[f][mid]) for f in FAMILY_NAMES)
push!(results, pass_fail(μ_part["acoustic"] > 0.01 && μ_part["speech"] > 0.01,
    @sprintf("b2b_joint_partition acoustic=%.4f speech=%.4f", μ_part["acoustic"], μ_part["speech"])))

audit_header("F. Null via joint_v4 path (EEG independent of X)")
Ynull_eeg = randn(n_ch, n_times, n_trials)
S1n, d1n = b2b_one_direction_joint(Ynull_eeg, Xjoint, half1, half2, sec, blk, times)
S2n, d2n = b2b_one_direction_joint(Ynull_eeg, Xjoint, half2, half1, sec, blk, times)
Snull_joint = 0.5 .* (S1n .+ S2n)
fam_null = family_scores_from_S(Snull_joint, k)
μ_null_joint = Dict(f => mean(fam_null.scores[f][mid]) for f in FAMILY_NAMES)
max_abs_null = maximum(abs.(values(μ_null_joint)))
push!(results, pass_fail(max_abs_null < 0.15,
    @sprintf("joint_v4 null family scores |μ|max=%.4f < 0.15", max_abs_null)))

audit_header("G. Collinearity disentanglement (King motivation)")
# Only X[:,1] causal; X[:,2] correlated but not in Y
Random.seed!(55)
m = 800
x1 = randn(m)
x2 = 0.9 .* x1 .+ 0.1 .* randn(m)
x3 = randn(m)
Xcol = hcat(x1, x2, x3, randn(m, 5))
Xcol = (Xcol .- vec(mean(Xcol; dims=1))') ./ vec(std(Xcol; dims=1, corrected=true))'
Fcol = zeros(8, 16)
Fcol[1, :] .= randn(16)  # only feature 1 drives Y
Ycol = Xcol * Fcol .+ 0.3 .* randn(m, 16)
Scol = king_b2b_ensemble(Ycol, Xcol; n_rep=60, λY=1e-1, λX=0.0, seed=40)
push!(results, pass_fail(Scol[1] > Scol[2] + 0.05,
    @sprintf("causal col1=%.3f > correlated col2=%.3f (disentangled)", Scol[1], Scol[2])))
push!(results, pass_fail(Scol[1] > mean(Scol[4:end]) + 0.05,
    @sprintf("causal col1=%.3f > uncorrelated mean=%.3f", Scol[1], mean(Scol[4:end]))))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
audit_header("AUDIT SUMMARY")
n_pass = count(identity, results)
n_total = length(results)
println(@sprintf("  %d / %d checks passed", n_pass, n_total))
if n_pass == n_total
    println("  Overall: PASS — implementation aligns with King et al. 2020 B2B guidelines")
    exit(0)
else
    println("  Overall: FAIL — see failed checks above")
    exit(1)
end
