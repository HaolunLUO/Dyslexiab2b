# =====================================================================
# Correct joint B2B estimator (one H for acoustic+speech+language)
# Scores: family_trace = sum(diag(H_family)); primary = family_trace / K
# Nuisance diagonals (if present) are excluded from family scores.
# =====================================================================

using LinearAlgebra
using Statistics

"""Ridge G: predictors A (n×p), targets B (n×q) → p×q. λ is absolute."""
function ridge_coef(A::AbstractMatrix, B::AbstractMatrix, λ::Real)
    AtA = A' * A
    @inbounds for i in 1:size(AtA, 1)
        AtA[i, i] += λ
    end
    return AtA \ (A' * B)
end

"""Ridge via precomputed economy SVD of centered A = U S Vt."""
function ridge_coef_svd(U::AbstractMatrix, S::AbstractVector, Vt::AbstractMatrix,
                        B::AbstractMatrix, λ::Real)
    # A = U * Diagonal(S) * Vt,  coef = V * diag(s/(s^2+λ)) * U' B
    scale = S ./ (S .^ 2 .+ λ)
    return Vt' * (scale .* (U' * B))
end

"""Condition number of design X (n×p), not of X'X."""
function cond_X(X::AbstractMatrix)
    n, p = size(X)
    n < 1 && return Inf
    F = svd(X; full=false)
    s = F.S
    smax = maximum(s)
    smin = minimum(s)
    (smin <= 0 || !isfinite(smin)) && return Inf
    return smax / smin
end

"""H: X → Xhat. OLS when B2B_H_RIDGE_KAPPA==0; else smallest ridge that
brings the shrunk design condition down to that target (HDC / Gwilliams).
Returns (H, rank, kappa, full_rank).
"""
function ols_H_diagnostics(X::AbstractMatrix, Xhat::AbstractMatrix;
                           pre::Union{Nothing,NamedTuple}=nothing)
    n, p = size(X)
    size(Xhat, 2) == p || error("X/Xhat feature mismatch")
    if pre === nothing
        pre = prepare_H_factor(X)
    end
    if !pre.full_rank
        return fill(NaN, p, p), pre.rank, pre.kappa, false
    end
    H = _h_from_svd(pre.F, Xhat, pre.lambda)
    return H, pre.rank, pre.kappa, true
end

function _h_ridge_lambda(s::AbstractVector)
    smax = maximum(s)
    smin = minimum(s)
    κ = (smin > 0 && isfinite(smin)) ? (smax / smin) : Inf
    target = B2B_H_RIDGE_KAPPA
    if target <= 0
        ok = isfinite(κ) && κ <= KAPPA_INVALID
        return 0.0, κ, ok
    end
    if isfinite(κ) && κ <= target
        return 0.0, κ, true
    end
    t2 = target^2
    λ = (smax^2 - t2 * smin^2) / max(t2 - 1, eps(Float64))
    λ = max(λ, 0.0)
    κ_eff = sqrt((smax^2 + λ) / (smin^2 + λ))
    return λ, κ_eff, isfinite(κ_eff)
end

function _h_from_svd(F, Xhat::AbstractMatrix, λ::Real)
    scale = F.S ./ (F.S .^ 2 .+ λ)
    return F.Vt' * (scale .* (F.U' * Xhat))
end

function prepare_H_factor(X::AbstractMatrix)
    n, p = size(X)
    F = svd(X; full=false)
    tol = maximum(size(X)) * eps(Float64) * maximum(F.S)
    rankX = count(>(tol), F.S)
    κ = (minimum(F.S) > 0) ? (maximum(F.S) / minimum(F.S)) : Inf
    λ, κ_eff, ok = _h_ridge_lambda(F.S)
    full_rank = rankX >= 1 && n >= p && ok
    return (rank=rankX, kappa=κ, kappa_eff=κ_eff, lambda=λ,
            full_rank=full_rank, F=F, QR=nothing)
end

function time_indices(times::AbstractVector{<:Real}, targets::AbstractVector{<:Real})
    idxs = Int[]
    for t in targets
        (t < first(times) - 1e-9 || t > last(times) + 1e-9) && continue
        push!(idxs, argmin(abs.(times .- t)))
    end
    return unique(idxs)
end

function composite_inner_folds(sec_ids_train::AbstractVector{Int},
                               block_ids_train::AbstractVector{Int};
                               n_folds::Integer=4)
    by_sec = Dict{Int, Vector{Int}}()
    for (s, b) in zip(sec_ids_train, block_ids_train)
        push!(get!(by_sec, s, Int[]), b)
    end
    for s in keys(by_sec)
        by_sec[s] = sort(unique(by_sec[s]))
    end
    secs = sort(collect(keys(by_sec)))
    fold_holdout = Dict{Int, Vector{Tuple{Int,Int}}}()
    for f in 1:n_folds
        fold_holdout[f] = Tuple{Int,Int}[]
    end
    for s in secs
        blocks = by_sec[s]
        for (i, b) in enumerate(blocks)
            f = mod1(i, n_folds)
            push!(fold_holdout[f], (s, b))
        end
    end
    folds = zeros(Int, length(sec_ids_train))
    for (j, (s, b)) in enumerate(zip(sec_ids_train, block_ids_train))
        for f in 1:n_folds
            if (s, b) in fold_holdout[f]
                folds[j] = f
                break
            end
        end
        folds[j] == 0 && (folds[j] = 1)
    end
    return folds
end

"""
Tune normalized ridge α for G on training half.
Precomputes SVD per (fold, tune-time) so the α grid is cheap.
"""
function tune_normalized_lambda(Y::AbstractArray{<:Real,3},
                                X::AbstractMatrix,
                                train_mask::AbstractVector{Bool},
                                sec_ids::AbstractVector{Int},
                                block_ids::AbstractVector{Int},
                                times::AbstractVector{<:Real};
                                alphas=ALPHA_GRID,
                                tune_times_s=LAMBDA_TUNE_TIMES_S,
                                n_folds::Integer=N_INNER_FOLDS)
    train_rows = findall(train_mask)
    Ytr = Y[:, :, train_rows]
    Xtr = X[train_rows, :]
    sec_tr = sec_ids[train_rows]
    blk_tr = block_ids[train_rows]
    folds = composite_inner_folds(sec_tr, blk_tr; n_folds=n_folds)
    t_idxs = time_indices(times, tune_times_s)
    isempty(t_idxs) && error("No tune times inside epoch window")
    n_ch = size(Ytr, 1)

    # Precompute per (fold, time): SVD of A_trc, B_trc, A_tec, B_tec_c, scale0=tr(A'A)/n_ch
    jobs = NamedTuple[]
    for f in 1:n_folds
        te = folds .== f
        trn = .!te
        sum(trn) < n_ch && continue
        sum(te) < 1 && continue
        for ti in t_idxs
            A = Matrix(transpose(Ytr[:, ti, :]))
            A_tr = A[trn, :]; A_te = A[te, :]
            B_tr = Xtr[trn, :]; B_te = Xtr[te, :]
            μA = vec(mean(A_tr; dims=1))
            μB = vec(mean(B_tr; dims=1))
            A_trc = A_tr .- μA'
            B_trc = B_tr .- μB'
            A_tec = A_te .- μA'
            B_tec = B_te .- μB'
            F = svd(A_trc; full=false)
            scale0 = sum(F.S .^ 2) / n_ch   # tr(A'A)/n_ch
            push!(jobs, (
                U=F.U, S=F.S, Vt=F.Vt, B_trc=B_trc,
                A_tec=A_tec, B_tec=B_tec, scale0=scale0,
            ))
        end
    end
    isempty(jobs) && error("No valid inner folds for λ tuning")

    best_α, best_loss = alphas[1], Inf
    losses = Dict{Float64,Float64}()
    for α in alphas
        err = 0.0
        for job in jobs
            λ = α * job.scale0
            G = ridge_coef_svd(job.U, job.S, job.Vt, job.B_trc, λ)
            pred = job.A_tec * G
            err += mean((pred .- job.B_tec) .^ 2)
        end
        err /= length(jobs)
        losses[α] = err
        if err < best_loss
            best_loss = err
            best_α = α
        end
    end
    # Plateau rule: among α within atol of the best loss, take the smallest α.
    # Avoids floating-point drift to the grid boundary on flat CV curves.
    atol = max(1e-8, 1e-6 * abs(best_loss))
    plateau = sort([a for (a, e) in losses if e <= best_loss + atol])
    best_α = first(plateau)
    best_loss = losses[best_α]

    ti = t_idxs[clamp(length(t_idxs) ÷ 2, 1, length(t_idxs))]
    A = Matrix(transpose(Ytr[:, ti, :]))
    μA = vec(mean(A; dims=1))
    Ac = A .- μA'
    λ_rep = best_α * LinearAlgebra.tr(Ac' * Ac) / n_ch
    on_boundary = (best_α == first(alphas) || best_α == last(alphas))
    return best_α, λ_rep, best_loss, on_boundary, losses
end

"""
One-direction joint B2B at all time points with correct centering.
X (held-out) factorized once; G retuned α applied per time via SVD.
"""
function b2b_one_direction_joint(Y::AbstractArray{<:Real,3},
                                 X::AbstractMatrix,
                                 train_idx::AbstractVector{Bool},
                                 test_idx::AbstractVector{Bool},
                                 sec_ids::AbstractVector{Int},
                                 block_ids::AbstractVector{Int},
                                 times::AbstractVector{<:Real})
    n_ch, n_times, _ = size(Y)
    n_feat = size(X, 2)
    S = fill(NaN, n_times, n_feat)
    ranks = fill(0, n_times)
    kappas = fill(NaN, n_times)
    valid_t = falses(n_times)

    α, λ_rep, cv_loss, on_boundary, _ = tune_normalized_lambda(
        Y, X, train_idx, sec_ids, block_ids, times;
        tune_times_s = B2B_LAMBDA_TUNE_ALL ? collect(Float64, times) : LAMBDA_TUNE_TIMES_S)

    # Held-out X is time-invariant → factor once
    Xte = X[test_idx, :]
    Xtec = Xte .- vec(mean(Xte; dims=1))'
    Hfac = prepare_H_factor(Xtec)

    for t in 1:n_times
        Ytr = Matrix(transpose(Y[:, t, train_idx]))
        Yte = Matrix(transpose(Y[:, t, test_idx]))
        Xtr = X[train_idx, :]

        μY = vec(mean(Ytr; dims=1))
        μX = vec(mean(Xtr; dims=1))
        Ytrc = Ytr .- μY'
        Xtrc = Xtr .- μX'
        Ytec = Yte .- μY'

        λ = α * LinearAlgebra.tr(Ytrc' * Ytrc) / n_ch
        # SVD ridge for speed at each time
        F = svd(Ytrc; full=false)
        G = ridge_coef_svd(F.U, F.S, F.Vt, Xtrc, λ)
        Xhat = Ytec * G
        Xhatc = Xhat .- vec(mean(Xhat; dims=1))'

        H, rankX, κ, full_rank = ols_H_diagnostics(Xtec, Xhatc; pre=Hfac)
        ranks[t] = rankX
        kappas[t] = κ
        if full_rank
            for k in 1:n_feat
                S[t, k] = H[k, k]
            end
            valid_t[t] = true
        end
    end

    n_valid_t = count(valid_t)
    max_κ = Hfac.kappa_eff
    warn_κ = max_κ > KAPPA_WARN
    invalid_κ = !Hfac.full_rank || n_valid_t < length(times) ÷ 2
    diag = (
        alpha=α, lambda=λ_rep, cv_loss=cv_loss, on_boundary=on_boundary,
        max_kappa=max_κ, raw_kappa=Hfac.kappa, h_ridge_lambda=Hfac.lambda,
        warn_kappa=warn_κ, invalid=invalid_κ,
        n_valid_times=n_valid_t, ranks=ranks, kappas=kappas,
        n_train=sum(train_idx), n_test=sum(test_idx),
    )
    return S, diag
end

"""Family block traces and primary scores (trace / K) from diag(H) matrix S.

`family_ks` gives the width of each family block in column order of FAMILY_NAMES.
Any trailing columns are nuisance and are excluded from family scores.
When `family_ks` is omitted, families are assumed equal-width `k`.
"""
function family_scores_from_S(S::AbstractMatrix, k::Integer;
                              n_families::Integer=length(FAMILY_NAMES),
                              family_ks::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    ks = family_ks === nothing ? fill(Int(k), n_families) : Int.(collect(family_ks))
    length(ks) == n_families || error("family_ks length $(length(ks)) != n_families $n_families")
    n_times = size(S, 1)
    n_fam_cols = sum(ks)
    size(S, 2) >= n_fam_cols ||
        error("S has $(size(S, 2)) cols but need ≥ $n_fam_cols for families $ks")
    traces = Dict{String, Vector{Float64}}()
    scores = Dict{String, Vector{Float64}}()  # primary = trace / K_family
    col0 = 0
    for (i, fam) in enumerate(FAMILY_NAMES[1:n_families])
        cols = (col0 + 1):(col0 + ks[i])
        col0 += ks[i]
        tr = [sum(S[t, cols]) for t in 1:n_times]
        traces[fam] = tr
        scores[fam] = tr ./ ks[i]
    end
    return (traces=traces, scores=scores)
end

function family_and_pc(j::Integer, family_ks::AbstractVector{<:Integer})
    acc = 0
    for (i, kk) in enumerate(family_ks)
        if j <= acc + kk
            return FAMILY_NAMES[i], j - acc
        end
        acc += kk
    end
    return "nuisance", j - acc
end

function b2b_joint_partition(Y_all::AbstractArray{<:Real,3},
                             X_joint::AbstractMatrix,
                             half1::AbstractVector{Bool},
                             half2::AbstractVector{Bool},
                             sec_ids::AbstractVector{Int},
                             word_idx_in_sec::AbstractVector{Int},
                             n_words_per_sec::AbstractVector{Int},
                             times::AbstractVector{<:Real};
                             k::Integer=B2B_PCA_K,
                             n_blocks::Integer=N_BLOCKS_PER_SECTION,
                             family_ks::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    use = half1 .| half2
    idx = findall(use)
    isempty(idx) && error("No trials in blocked halves")
    Y = Y_all[:, :, idx]
    X = X_joint[idx, :]
    h1 = half1[idx]
    h2 = half2[idx]
    sec_u = sec_ids[idx]
    wi_u = word_idx_in_sec[idx]
    blk_u = [word_block_id(wi_u[j], n_words_per_sec[sec_u[j]], n_blocks)
             for j in eachindex(idx)]

    S1, d1 = b2b_one_direction_joint(Y, X, h1, h2, sec_u, blk_u, times)
    S2, d2 = b2b_one_direction_joint(Y, X, h2, h1, sec_u, blk_u, times)

    S = 0.5 .* (S1 .+ S2)
    invalid = d1.invalid || d2.invalid
    max_κ = max(d1.max_kappa, d2.max_kappa)
    warn_κ = max_κ > KAPPA_WARN
    fam = family_scores_from_S(S, k; family_ks=family_ks)
    diag = (
        alpha_1=d1.alpha, alpha_2=d2.alpha,
        lambda_1=d1.lambda, lambda_2=d2.lambda,
        cv_loss_1=d1.cv_loss, cv_loss_2=d2.cv_loss,
        on_boundary_1=d1.on_boundary, on_boundary_2=d2.on_boundary,
        max_kappa=max_κ, warn_kappa=warn_κ, invalid=invalid,
        rank_ok=!invalid,
        n_half1=d1.n_train, n_half2=d1.n_test,
        kappa_1=d1.max_kappa, kappa_2=d2.max_kappa,
    )
    return S, fam, diag, d1, d2
end

"""Train G at times `t_idx`, apply at all `t_idx` (temporal generalization).

Returns TG of shape (n_t, n_t, n_feat) with diag(H) at each train/test pair,
plus the shared ridge diagnostics from the first direction's λ tune.
"""
function b2b_one_direction_tg(Y::AbstractArray{<:Real,3},
                              X::AbstractMatrix,
                              train_idx::AbstractVector{Bool},
                              test_idx::AbstractVector{Bool},
                              sec_ids::AbstractVector{Int},
                              block_ids::AbstractVector{Int},
                              times::AbstractVector{<:Real};
                              stride::Integer=B2B_TG_STRIDE)
    n_ch, n_times, _ = size(Y)
    n_feat = size(X, 2)
    t_idx = collect(1:stride:n_times)
    n_t = length(t_idx)
    TG = fill(NaN, n_t, n_t, n_feat)

    α, λ_rep, cv_loss, on_boundary, _ = tune_normalized_lambda(
        Y, X, train_idx, sec_ids, block_ids, times;
        tune_times_s = B2B_LAMBDA_TUNE_ALL ? collect(Float64, times) : LAMBDA_TUNE_TIMES_S)

    Xte = X[test_idx, :]
    Xtec = Xte .- vec(mean(Xte; dims=1))'
    Hfac = prepare_H_factor(Xtec)
    if !Hfac.full_rank
        diag = (
            alpha=α, lambda=λ_rep, cv_loss=cv_loss, on_boundary=on_boundary,
            max_kappa=Hfac.kappa_eff, warn_kappa=Hfac.kappa_eff > KAPPA_WARN,
            invalid=true, n_valid_times=0, n_train=sum(train_idx), n_test=sum(test_idx),
            t_idx=t_idx,
        )
        return TG, diag
    end

    Xtr = X[train_idx, :]
    μX = vec(mean(Xtr; dims=1))
    Xtrc = Xtr .- μX'

    Gs = Vector{Matrix{Float64}}(undef, n_t)
    for (i, t) in enumerate(t_idx)
        Ytr = Matrix(transpose(Y[:, t, train_idx]))
        μY = vec(mean(Ytr; dims=1))
        Ytrc = Ytr .- μY'
        λ = α * LinearAlgebra.tr(Ytrc' * Ytrc) / n_ch
        F = svd(Ytrc; full=false)
        Gs[i] = ridge_coef_svd(F.U, F.S, F.Vt, Xtrc, λ)
    end

    Yte_c = Vector{Matrix{Float64}}(undef, n_t)
    for (j, tp) in enumerate(t_idx)
        Yte = Matrix(transpose(Y[:, tp, test_idx]))
        # Center with the training μY of... we need per-train-time μY.
        # Store uncentered Yte; subtract per-train μ below.
        Yte_c[j] = Yte
    end

    for (i, t) in enumerate(t_idx)
        Ytr = Matrix(transpose(Y[:, t, train_idx]))
        μY = vec(mean(Ytr; dims=1))
        G = Gs[i]
        for j in 1:n_t
            Ytec = Yte_c[j] .- μY'
            Xhat = Ytec * G
            Xhatc = Xhat .- vec(mean(Xhat; dims=1))'
            H, _, _, full_rank = ols_H_diagnostics(Xtec, Xhatc; pre=Hfac)
            if full_rank
                for k in 1:n_feat
                    TG[i, j, k] = H[k, k]
                end
            end
        end
    end

    diag = (
        alpha=α, lambda=λ_rep, cv_loss=cv_loss, on_boundary=on_boundary,
        max_kappa=Hfac.kappa_eff, warn_kappa=Hfac.kappa_eff > KAPPA_WARN,
        invalid=!Hfac.full_rank, n_valid_times=n_t,
        n_train=sum(train_idx), n_test=sum(test_idx), t_idx=t_idx,
    )
    return TG, diag
end

function family_tg_from_TG(TG::AbstractArray{<:Real,3}, family_ks::AbstractVector{<:Integer})
    n_t = size(TG, 1)
    out = Dict{String, Array{Float64,2}}()
    col0 = 0
    for (i, fam) in enumerate(FAMILY_NAMES)
        cols = (col0 + 1):(col0 + family_ks[i])
        col0 += family_ks[i]
        M = fill(NaN, n_t, n_t)
        for a in 1:n_t, b in 1:n_t
            M[a, b] = mean(TG[a, b, cols])
        end
        out[fam] = M
    end
    return out
end

"""Duration (diagonal length) and sustain (mean off-diagonal width) in seconds."""
function tg_duration_sustain(M::AbstractMatrix, times_tg::AbstractVector{<:Real};
                             thresh::Real=0.0)
    n = size(M, 1)
    n == length(times_tg) || error("TG matrix / times mismatch")
    dt = n > 1 ? mean(diff(times_tg)) : 0.0
    diagv = [M[i, i] for i in 1:n]
    above = diagv .> thresh
    duration = count(above) * dt
    widths = Float64[]
    for i in 1:n
        row_above = M[i, :] .> thresh
        any(row_above) || continue
        push!(widths, count(row_above) * dt)
    end
    sustain = isempty(widths) ? 0.0 : mean(widths)
    return duration, sustain
end
