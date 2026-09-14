# =====================================================================
# Frozen EEG-independent embedding transform (joint B2B v4)
# Default: drift removal → independent PCA → whiten per family
# Optional ordered mode: sequential acoustic → speech → language residualization
# =====================================================================

using LinearAlgebra
using Statistics
using NPZ
using JSON3
using SHA
using Random
using CSV
using DataFrames
using Printf

"""JSON3 rejects NaN/Inf; omit non-finite scalars from metadata sidecars."""
_json_val(x) = (x isa Real && !isfinite(x)) ? nothing : x

"""
Top-k PCA via randomized rangefinder + small SVD (Halko et al.).
Avoids full dense SVD/eigen on large designs (e.g. 3553×12800).
"""
function _randomized_pca(Xstd::AbstractMatrix, n_comp::Integer; n_oversample::Integer=10,
                         n_iter::Integer=2, rng::AbstractRNG=Random.default_rng())
    n, p = size(Xstd)
    k = min(n_comp, n - 1, p)
    q = min(k + n_oversample, min(n, p))
    Ω = randn(rng, p, q)
    Y = Xstd * Ω
    for _ in 1:n_iter
        Y = Xstd * (Xstd' * Y)
    end
    Q = Matrix(qr(Y).Q)[:, 1:min(size(Y, 2), n)]
    B = Q' * Xstd                       # q × p
    F = svd(B; full=false)
    kk = min(k, length(F.S))
    loadings = Matrix(F.Vt[1:kk, :]')   # p × k   (V from B = U S Vt)
    svals = F.S[1:kk]
    scores = Xstd * loadings            # n × k
    eigenvalues = (svals .^ 2) ./ max(n - 1, 1)
    return loadings, scores, eigenvalues, kk
end

function _drift_basis(onset_rel::AbstractVector{<:Real}; n_knots::Int=B2B_DRIFT_KNOTS)
    k = range(minimum(onset_rel), maximum(onset_rel); length=n_knots)
    w = step(k)
    B = [exp(-0.5 * ((t - kj) / w)^2) for t in onset_rel, kj in k]
    return hcat(ones(length(onset_rel)), B)   # n × (n_knots+1)
end

function remove_slow_drift(X::AbstractMatrix, onset_rel::AbstractVector{<:Real};
                           n_knots::Int=B2B_DRIFT_KNOTS)
    # n_knots <= 0 disables drift removal (used for lexical scalar predictors).
    n_knots <= 0 && return Matrix{Float64}(X)
    B = _drift_basis(onset_rel; n_knots=n_knots)
    return X - B * (B \ X)
end

function load_section_onset_relative(extractor_dir::AbstractString, section_id::Integer)
    csv_path = joinpath(extractor_dir, "_shared_wordlocked_features",
                        @sprintf("section_%03d", section_id), "word_timing_relative.csv")
    isfile(csv_path) || error("Shared word timing not found: $csv_path")
    df = CSV.read(csv_path, DataFrame)
    "onset_relative" in names(df) || error("word_timing_relative.csv missing onset_relative: $csv_path")
    return Float64.(df.onset_relative)
end

function residualize_columns(X::AbstractMatrix, Z_pred::AbstractMatrix)
    isempty(Z_pred) && return Matrix(X)
    B = hcat(ones(size(X, 1)), Z_pred)
    return X - B * (B \ X)
end

"""
Fit standardize + PCA + whiten on the full stimulus design (all words).
Returns Z (n_words × n_comp) and portable fit dict.
"""
function fit_stimulus_pca(X::AbstractMatrix;
                          n_components::Int=16,
                          standardize::Bool=true,
                          seed::Integer=B2B_SEED)
    n, p = size(X)
    if standardize
        μ = vec(mean(X; dims=1))
        σ = vec(std(X; dims=1, corrected=true))
        σ = ifelse.(σ .> 1e-10, σ, 1.0)
        Xstd = (X .- μ') ./ σ'
    else
        μ = zeros(p); σ = ones(p)
        Xstd = Matrix(X)
    end

    rng = Random.MersenneTwister(seed + n + p + n_components)
    loadings, scores, eigenvalues, n_comp = _randomized_pca(Xstd, n_components; rng=rng)

    wscale = vec(std(scores; dims=1, corrected=true))
    wscale = ifelse.(wscale .> 1e-10, wscale, 1.0)
    Z = scores ./ wscale'

    fit = Dict{String,Any}(
        "mean" => μ,
        "scale" => σ,
        "loadings" => loadings,
        "eigenvalues" => eigenvalues,
        "whiten_scale" => wscale,
        "n_components" => n_comp,
        "n_words" => n,
        "n_raw_dims" => p,
        "pca_method" => "randomized",
    )
    return Z, fit
end

"""Standardize only (identity 'PCA'). Used for HDC linguistic features."""
function fit_passthrough(X::AbstractMatrix; standardize::Bool=true)
    n, p = size(X)
    if standardize
        μ = vec(mean(X; dims=1))
        σ = vec(std(X; dims=1, corrected=true))
        σ = ifelse.(σ .> 1e-10, σ, 1.0)
        Xstd = (X .- μ') ./ σ'
    else
        μ = zeros(p); σ = ones(p)
        Xstd = Matrix(X)
    end
    loadings = Matrix{Float64}(I, p, p)
    eigenvalues = vec(var(Xstd; dims=1, corrected=true))
    wscale = ones(p)
    fit = Dict{String,Any}(
        "mean" => μ,
        "scale" => σ,
        "loadings" => loadings,
        "eigenvalues" => eigenvalues,
        "whiten_scale" => wscale,
        "n_components" => p,
        "n_words" => n,
        "n_raw_dims" => p,
        "pca_method" => "passthrough",
    )
    return Xstd, fit
end

function family_ks_of(Z_by_family::Dict{String,<:AbstractMatrix})
    return Int[size(Z_by_family[fam], 2) for fam in FAMILY_NAMES]
end

function transform_stimulus_pca(X::AbstractMatrix, fit::Dict; k::Union{Nothing,Int}=nothing)
    μ = Vector{Float64}(fit["mean"])
    σ = Vector{Float64}(fit["scale"])
    loadings = Matrix{Float64}(fit["loadings"])
    wscale = Vector{Float64}(fit["whiten_scale"])
    kk = k === nothing ? size(loadings, 2) : min(k, size(loadings, 2))
    Xstd = (X .- μ') ./ σ'
    Z = (Xstd * loadings[:, 1:kk]) ./ wscale[1:kk]'
    return Z
end

"""Load both sections of shared embeddings, remove per-section drift, concatenate."""
function load_all_stimulus_embeddings(extractor_dir::AbstractString=B2B_EXTRACTOR_DIR;
                                      n_knots::Int=B2B_DRIFT_KNOTS,
                                      feature_sets=FEATURE_SETS)
    Xs = Dict(f => Matrix{Float64}[] for f in feature_sets)
    names = Dict{String, Vector{String}}()
    sec_lengths = Int[]
    drift_raw_var = Dict(f => 0.0 for f in feature_sets)
    drift_clean_var = Dict(f => 0.0 for f in feature_sets)
    checksums = Dict{String,Dict{String,String}}()
    for sid in SECTIONS
        Xw, Xn, csums = load_section_features(extractor_dir, sid, feature_sets)
        checksums["section_$(lpad(sid, 3, '0'))"] = csums
        onset_rel = load_section_onset_relative(extractor_dir, sid)
        for f in feature_sets
            n_rows = size(Xw[f], 1)
            length(onset_rel) == n_rows ||
                error("onset_relative length $(length(onset_rel)) != n_words $n_rows (section $sid, $f)")
            X_clean = remove_slow_drift(Xw[f], onset_rel; n_knots=n_knots)
            drift_raw_var[f] += sum(var(Xw[f]; dims=1, corrected=true))
            drift_clean_var[f] += sum(var(X_clean; dims=1, corrected=true))
            push!(Xs[f], X_clean)
            names[f] = Xn[f]
        end
        push!(sec_lengths, size(Xw[feature_sets[1]], 1))
    end
    Xfull = Dict(f => vcat(Xs[f]...) for f in feature_sets)
    n_total = size(Xfull[feature_sets[1]], 1)
    n_total == EXPECTED_N_WORDS ||
        error("Expected $EXPECTED_N_WORDS stimulus words; got $n_total")
    sec_lengths == collect(EXPECTED_SEC_LENGTHS) ||
        error("Section lengths $sec_lengths != expected $(collect(EXPECTED_SEC_LENGTHS))")
    drift_retained = Dict(f => drift_clean_var[f] / max(drift_raw_var[f], 1e-30) for f in feature_sets)
    return Xfull, names, sec_lengths, drift_retained, checksums
end

function _canonical_correlations(A::AbstractMatrix, B::AbstractMatrix)
    qa = Matrix(qr(A).Q)
    qb = Matrix(qr(B).Q)
    k = min(size(qa, 2), size(qb, 2))
    qa = qa[:, 1:k]
    qb = qb[:, 1:k]
    return svdvals(qa' * qb)
end

function _max_pc_r2(target::AbstractMatrix, predictors::AbstractMatrix)
    B = hcat(ones(size(predictors, 1)), predictors)
    maximum([begin
        y = target[:, j]
        β = B \ y
        ŷ = B * β
        ss_res = sum((y .- ŷ) .^ 2)
        ss_tot = sum((y .- mean(y)) .^ 2)
        ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0
    end for j in 1:size(target, 2)])
end

"""Gate diagnostics on drift-removed independent PCAs (before any orthogonalization)."""
function basis_gate_a_diagnostics(Xfull::Dict{String,<:AbstractMatrix};
                                    n_fit::Int=B2B_PCA_FIT_MAX,
                                    k::Int=B2B_PCA_K,
                                    feature_sets=FEATURE_SETS)
    Z_indep = Dict{String, Matrix{Float64}}()
    fits_indep = Dict{String, Dict{String,Any}}()
    for f in feature_sets
        if B2B_PASSTHROUGH
            Z, fit = fit_passthrough(Xfull[f])
            Z_indep[f] = Z
        else
            Z, fit = fit_stimulus_pca(Xfull[f]; n_components=n_fit)
            kk = min(k, size(Z, 2))
            Z_indep[f] = Z[:, 1:kk]
        end
        fits_indep[f] = fit
    end
    n_fam = length(FAMILY_NAMES)
    Z_by = Dict(FAMILY_NAMES[i] => Z_indep[feature_sets[i]] for i in 1:n_fam)
    X_joint, _ = joint_design(Z_by; k=k)
    canon_lead = if n_fam >= 2
        _canonical_correlations(Z_indep[feature_sets[1]], Z_indep[feature_sets[2]])[1]
    else
        NaN
    end
    if n_fam >= 3
        others = hcat(Z_indep[feature_sets[2]], Z_indep[feature_sets[3]])
    elseif n_fam >= 2
        others = Z_indep[feature_sets[2]]
    else
        others = ones(size(Z_indep[feature_sets[1]], 1), 1)
    end
    max_r2_ac = _max_pc_r2(Z_indep[feature_sets[1]], others)
    evr1_ac = fits_indep[feature_sets[1]]["eigenvalues"][1] /
              max(fits_indep[feature_sets[1]]["n_raw_dims"], 1)
    return (
        cond_joint = cond(X_joint),
        canon_ac_sp_leading = canon_lead,
        max_r2_acoustic = max_r2_ac,
        acoustic_evr1 = evr1_ac,
        Z_indep = Z_indep,
        fits_indep = fits_indep,
    )
end

"""Gate diagnostics on final joint design (independent or ordered)."""
function basis_gate_b_diagnostics(Z_by_family::Dict{String,<:AbstractMatrix}; k::Int=B2B_PCA_K)
    X_joint, _ = joint_design(Z_by_family; k=k)
    n_fam = length(FAMILY_NAMES)
    function _fam_mat(i)
        M = Z_by_family[FAMILY_NAMES[i]]
        kk = B2B_PASSTHROUGH ? size(M, 2) : min(k, size(M, 2))
        return M[:, 1:kk]
    end
    c12 = n_fam >= 2 ? _canonical_correlations(_fam_mat(1), _fam_mat(2))[1] : NaN
    c13 = n_fam >= 3 ? _canonical_correlations(_fam_mat(1), _fam_mat(3))[1] : NaN
    c23 = n_fam >= 3 ? _canonical_correlations(_fam_mat(2), _fam_mat(3))[1] : NaN
    col_var = vec(var(X_joint; dims=1, corrected=true))
    return (
        cond_joint = cond(X_joint),
        canon_ac_sp = c12,
        canon_ac_lg = c13,
        canon_sp_lg = c23,
        col_var = col_var,
    )
end

function _fit_orthogonal_family(X::AbstractMatrix, Z_pred::AbstractMatrix;
                                n_components::Int, seed::Integer=B2B_SEED)
    X_resid = residualize_columns(X, Z_pred)
    Z, fit = fit_stimulus_pca(X_resid; n_components=n_components, seed=seed)
    fit["orthogonalize_against_dim"] = size(Z_pred, 2)
    return Z, fit, X_resid
end

"""
Prepare and save the frozen feature basis under B2B_BASIS_DIR.

`basis_mode`:
  - `independent` (default): PCA each drift-cleaned family separately; no
    cross-family residualization. Canonical correlations between families are
    NOT forced to zero.
  - `ordered`: sequential acoustic → speech → language residualization
    (sensitivity comparison only).
"""
function prepare_feature_basis(; extractor_dir=B2B_EXTRACTOR_DIR,
                                 basis_dir=B2B_BASIS_DIR,
                                 n_fit=B2B_PCA_FIT_MAX,
                                 n_knots::Int=B2B_DRIFT_KNOTS,
                                 k_orth::Int=B2B_PCA_K,
                                 basis_mode::AbstractString=B2B_BASIS_MODE)
    basis_mode = lowercase(basis_mode)
    basis_mode in ("independent", "ordered") ||
        error("basis_mode must be independent|ordered; got $basis_mode")
    mkpath(basis_dir)
    Xfull, names, sec_lengths, drift_retained, checksums =
        load_all_stimulus_embeddings(extractor_dir; n_knots=n_knots)

    gate_a = basis_gate_a_diagnostics(Xfull; n_fit=n_fit, k=k_orth)
    n_joint = sum(B2B_PASSTHROUGH ? size(Xfull[f], 2) : k_orth for f in FEATURE_SETS)
    println("Gate A (drift-removed, $(B2B_PASSTHROUGH ? "passthrough" : "independent PCA"), k=$k_orth):")
    println("  cond(joint $n_joint) = ", round(gate_a.cond_joint; digits=4))
    if length(FAMILY_NAMES) >= 2
        println("  canon($(FAMILY_NAMES[1]),$(FAMILY_NAMES[2])) leading = ",
                round(gate_a.canon_ac_sp_leading; digits=4))
    end
    println("  max per-PC R² $(FAMILY_NAMES[1])  = ", round(gate_a.max_r2_acoustic; digits=4))
    println("  $(FAMILY_NAMES[1]) evr1           = ", round(gate_a.acoustic_evr1; digits=4))

    Z_all = Dict{String, Matrix{Float64}}()
    fits = Dict{String, Dict{String,Any}}()

    if basis_mode == "independent"
        println("Basis mode = independent (no cross-family residualization" *
                (B2B_PASSTHROUGH ? "; PASSTHROUGH standardize-only)" : ")"))
        for (i, f) in enumerate(FEATURE_SETS)
            if B2B_PASSTHROUGH
                println("Passthrough standardize for $f  shape=$(size(Xfull[f]))")
                Z, fit = fit_passthrough(Xfull[f])
            else
                println("Fitting PCA for $f  shape=$(size(Xfull[f]))  k_fit=$n_fit")
                Z, fit = fit_stimulus_pca(Xfull[f]; n_components=n_fit,
                                          seed=B2B_SEED + i)
            end
            Z_all[f] = Z
            fits[f] = fit
            println("  retained=$(fit["n_components"])  eig1=$(round(fit["eigenvalues"][1]; sigdigits=4))")
        end
    else
        B2B_PASSTHROUGH && error("B2B_PASSTHROUGH is incompatible with ordered residualization")
        println("Basis mode = ordered (acoustic → speech → language residualization)")
        println("Fitting PCA for $(FEAT_ACOUSTIC)  shape=$(size(Xfull[FEAT_ACOUSTIC]))  k_fit=$n_fit")
        Z_ac, fit_ac = fit_stimulus_pca(Xfull[FEAT_ACOUSTIC]; n_components=n_fit)
        Z_all[FEAT_ACOUSTIC] = Z_ac
        fits[FEAT_ACOUSTIC] = fit_ac
        Z_ac_k = Z_ac[:, 1:k_orth]
        println("  retained=$(fit_ac["n_components"])  eig1=$(round(fit_ac["eigenvalues"][1]; sigdigits=4))")

        println("Fitting PCA for $(FEAT_SPEECH)  (orthogonalized vs acoustic k=$k_orth)")
        Z_sp, fit_sp, _ = _fit_orthogonal_family(Xfull[FEAT_SPEECH], Z_ac_k;
                                                   n_components=n_fit,
                                                   seed=B2B_SEED + 1)
        Z_all[FEAT_SPEECH] = Z_sp
        fits[FEAT_SPEECH] = fit_sp
        Z_sp_k = Z_sp[:, 1:k_orth]
        println("  retained=$(fit_sp["n_components"])  eig1=$(round(fit_sp["eigenvalues"][1]; sigdigits=4))")

        println("Fitting PCA for $(FEAT_LANGUAGE)  (orthogonalized vs acoustic+speech)")
        Z_lg, fit_lg, _ = _fit_orthogonal_family(Xfull[FEAT_LANGUAGE], hcat(Z_ac_k, Z_sp_k);
                                                   n_components=n_fit,
                                                   seed=B2B_SEED + 2)
        Z_all[FEAT_LANGUAGE] = Z_lg
        fits[FEAT_LANGUAGE] = fit_lg
        println("  retained=$(fit_lg["n_components"])  eig1=$(round(fit_lg["eigenvalues"][1]; sigdigits=4))")
    end

    gate_b = basis_gate_b_diagnostics(Dict(FAMILY_NAMES[i] => Z_all[FEATURE_SETS[i]]
                                           for i in 1:length(FAMILY_NAMES));
                                      k=k_orth)
    println("Gate B (basis_mode=$basis_mode, k=$k_orth, passthrough=$B2B_PASSTHROUGH):")
    println("  cond(joint) = ", round(gate_b.cond_joint; digits=4))
    if length(FAMILY_NAMES) >= 2
        println("  canon($(FAMILY_NAMES[1]),$(FAMILY_NAMES[2])) = ", gate_b.canon_ac_sp)
    end
    if length(FAMILY_NAMES) >= 3
        println("  canon($(FAMILY_NAMES[1]),$(FAMILY_NAMES[3])) = ", gate_b.canon_ac_lg)
        println("  canon($(FAMILY_NAMES[2]),$(FAMILY_NAMES[3])) = ", gate_b.canon_sp_lg)
    end
    if basis_mode == "independent"
        # Independent mode must NOT force cross-family correlations to zero
        println("  (independent: cross-family canon need not be ~0)")
    end
    println("  col variances           = ", gate_b.col_var)
    # Interpretable passthrough + OLS H: refuse unidentified designs instead of
    # silently writing a basis that will force ridge-H or invalid partitions.
    if B2B_PASSTHROUGH && B2B_H_RIDGE_KAPPA <= 0
        Xj, _ = joint_design(Dict(FAMILY_NAMES[i] => Z_all[FEATURE_SETS[i]]
                                  for i in 1:length(FAMILY_NAMES)); k=k_orth)
        n_j, p_j = size(Xj)
        Fj = svd(Xj; full=false)
        tol = maximum(size(Xj)) * eps(Float64) * maximum(Fj.S)
        rank_j = count(>(tol), Fj.S)
        cond_j = gate_b.cond_joint
        if rank_j < p_j || n_j < p_j || !isfinite(cond_j) || cond_j > KAPPA_INVALID
            error("OLS-H passthrough basis unidentified: n=$n_j p=$p_j rank=$rank_j " *
                  "cond=$(cond_j). Fix feature redundancy before analysis " *
                  "(do not silently enable ridge-H).")
        end
    end

    npz_path = joinpath(basis_dir, "feature_basis.npz")
    arrays = Dict{String,Any}()
    for (i, f) in enumerate(FEATURE_SETS)
        prefix = FAMILY_NAMES[i]
        fit = fits[f]
        arrays["$(prefix)_mean"] = fit["mean"]
        arrays["$(prefix)_scale"] = fit["scale"]
        arrays["$(prefix)_loadings"] = fit["loadings"]
        arrays["$(prefix)_eigenvalues"] = fit["eigenvalues"]
        arrays["$(prefix)_whiten_scale"] = fit["whiten_scale"]
        arrays["$(prefix)_Z"] = Z_all[f]
    end
    arrays["section_lengths"] = Float64.(sec_lengths)
    NPZ.npzwrite(npz_path, arrays)

    feature_order = String[]
    for (i, fam) in enumerate(FAMILY_NAMES)
        ncomp = fits[FEATURE_SETS[i]]["n_components"]
        raw_names = names[FEATURE_SETS[i]]
        for kk in 1:ncomp
            if B2B_PASSTHROUGH && kk <= length(raw_names)
                push!(feature_order, "$(fam)::$(raw_names[kk])")
            else
                push!(feature_order, "$(fam)_PC$(kk)")
            end
        end
    end

    drift_retained_named = Dict(FAMILY_NAMES[i] => drift_retained[FEATURE_SETS[i]]
                              for i in 1:length(FAMILY_NAMES))

    # Flatten checksums for manifest
    flat_checksums = Dict{String,String}()
    for (sec, d) in checksums
        for (feat, h) in d
            flat_checksums["$(sec)/$(feat)"] = h
        end
    end

    meta = Dict{String,Any}(
        "basis_version" => "v4",
        "basis_mode" => basis_mode,
        "families" => FAMILY_NAMES,
        "raw_feature_keys" => FEATURE_SETS,
        "feature_checksums" => flat_checksums,
        "n_fit_max" => n_fit,
        "n_components_per_family" => Dict(FAMILY_NAMES[i] => fits[FEATURE_SETS[i]]["n_components"]
                                         for i in 1:length(FAMILY_NAMES)),
        "n_raw_dims" => Dict(FAMILY_NAMES[i] => fits[FEATURE_SETS[i]]["n_raw_dims"]
                            for i in 1:length(FAMILY_NAMES)),
        "section_lengths" => sec_lengths,
        "n_words_total" => sum(sec_lengths),
        "feature_order_all" => feature_order,
        "primary_k" => B2B_PASSTHROUGH ? 0 : k_orth,
        "passthrough" => B2B_PASSTHROUGH,
        "family_ks" => Dict(FAMILY_NAMES[i] => fits[FEATURE_SETS[i]]["n_components"]
                            for i in 1:length(FAMILY_NAMES)),
        "extractor_dir" => extractor_dir,
        "whiten" => true,
        "standardize" => true,
        "drift_removed" => true,
        "drift_n_knots" => n_knots,
        "drift_variance_retained" => drift_retained_named,
        "cohort_hash" => cohort_hash(),
        "git_commit" => git_commit_hash(),
        "slurm_job_id" => slurm_job_id(),
        "gate_a" => Dict(
            "cond_joint" => gate_a.cond_joint,
            "canon_ac_sp_leading" => _json_val(gate_a.canon_ac_sp_leading),
            "max_r2_acoustic" => _json_val(gate_a.max_r2_acoustic),
            "acoustic_evr1" => _json_val(gate_a.acoustic_evr1),
        ),
        "gate_b" => Dict(
            "cond_joint" => gate_b.cond_joint,
            "canon_ac_sp" => _json_val(gate_b.canon_ac_sp),
            "canon_ac_lg" => _json_val(gate_b.canon_ac_lg),
            "canon_sp_lg" => _json_val(gate_b.canon_sp_lg),
        ),
    )
    if basis_mode == "ordered"
        meta["orthogonalization_order"] = FAMILY_NAMES
        meta["orthogonalization_k"] = k_orth
        meta["claim"] = "family result = incremental contribution of retained K-dimensional " *
                        "subspace orthogonalized acoustic → speech → language"
    else
        meta["orthogonalization_order"] = nothing
        meta["orthogonalization_k"] = nothing
        meta["claim"] = B2B_PASSTHROUGH ?
            "family result = mean diag(H) over linguistic features in the family (no PCA)" :
            "family result = conditional linear alignment of retained " *
            "K-dimensional subspace (independently PCA'd per family)"
    end
    for (i, f) in enumerate(FEATURE_SETS)
        meta["$(FAMILY_NAMES[i])_n_names"] = length(names[f])
        if haskey(fits[f], "orthogonalize_against_dim")
            meta["$(FAMILY_NAMES[i])_orthogonalize_against_dim"] = fits[f]["orthogonalize_against_dim"]
        end
    end
    meta_path = joinpath(basis_dir, "feature_basis_meta.json")
    open(meta_path, "w") do io
        JSON3.write(io, meta)
    end

    open(joinpath(basis_dir, "feature_basis_hashes.json"), "w") do io
        JSON3.write(io, Dict(
            "npz_sha256" => bytes2hex(open(sha256, npz_path)),
            "meta_sha256" => bytes2hex(open(sha256, meta_path)),
        ))
    end

    # Data-only diagnostics sidecar
    diag_path = joinpath(basis_dir, "basis_diagnostics.json")
    open(diag_path, "w") do io
        JSON3.write(io, Dict(
            "basis_mode" => basis_mode,
            "k" => k_orth,
            "gate_a" => meta["gate_a"],
            "gate_b" => meta["gate_b"],
            "drift_variance_retained" => drift_retained_named,
            "n_raw_dims" => meta["n_raw_dims"],
            "feature_checksums" => flat_checksums,
        ))
    end
    println("Wrote $npz_path")
    println("Wrote $meta_path")
    println("Wrote $diag_path")
    return Z_all, fits, meta
end

"""Load frozen basis; return Z matrices truncated to k PCs per family.

For ordered bases, requesting k ≠ orthogonalization_k is a hard error.
For independent bases, k must match primary_k stored at fit time (separate
basis dirs for K-sensitivity).
"""
function load_feature_basis(; basis_dir=B2B_BASIS_DIR, k::Int=B2B_PCA_K)
    npz_path = joinpath(basis_dir, "feature_basis.npz")
    meta_path = joinpath(basis_dir, "feature_basis_meta.json")
    isfile(npz_path) || error("Feature basis NPZ not found: $npz_path\nRun prepare_feature_basis.jl first.")
    isfile(meta_path) || error("Feature basis meta not found: $meta_path")
    arrays = NPZ.npzread(npz_path)
    meta = JSON3.read(read(meta_path, String), Dict{String,Any})

    basis_mode = string(get(meta, "basis_mode", "ordered"))
    passthrough = Bool(get(meta, "passthrough", false)) || B2B_PASSTHROUGH
    if !passthrough
        if basis_mode == "ordered" && meta["orthogonalization_k"] !== nothing
            k_orth = Int(meta["orthogonalization_k"])
            k == k_orth || error(
                "Requested k=$k but basis was orthogonalized at k=$k_orth " *
                "(basis_dir=$basis_dir). Build a separate basis for this K.")
        elseif haskey(meta, "primary_k") && meta["primary_k"] !== nothing
            k_prim = Int(meta["primary_k"])
            k_prim == 0 || k == k_prim || error(
                "Requested k=$k but independent basis primary_k=$k_prim " *
                "(basis_dir=$basis_dir). Build a separate basis for this K.")
        end
    end

    # Feature key sanity
    if haskey(meta, "raw_feature_keys")
        stored = String.(meta["raw_feature_keys"])
        stored == FEATURE_SETS ||
            error("Basis feature keys $stored != configured FEATURE_SETS $FEATURE_SETS")
    end

    Z = Dict{String, Matrix{Float64}}()
    for fam in FAMILY_NAMES
        Zfull = Float64.(arrays["$(fam)_Z"])
        if passthrough
            Z[fam] = Zfull
        else
            kk = min(k, size(Zfull, 2))
            Z[fam] = Zfull[:, 1:kk]
        end
    end
    sec_lengths = Int.(round.(arrays["section_lengths"]))
    return Z, sec_lengths, meta, arrays
end

"""Build joint design matrix ordered acoustic|speech|language PCs (+ optional nuisance)."""
function joint_design(Z_by_family::Dict{String,<:AbstractMatrix};
                      k::Int=B2B_PCA_K,
                      nuisance::Union{Nothing,AbstractMatrix}=nothing,
                      nuisance_names::Vector{String}=String[])
    mats = Matrix{Float64}[]
    names = String[]
    for fam in FAMILY_NAMES
        M = Z_by_family[fam]
        kk = B2B_PASSTHROUGH ? size(M, 2) : min(k, size(M, 2))
        push!(mats, M[:, 1:kk])
        for j in 1:kk
            push!(names, B2B_PASSTHROUGH ? "$(fam)_$(j)" : "$(fam)_PC$(j)")
        end
    end
    X = hcat(mats...)
    if nuisance !== nothing && size(nuisance, 2) > 0
        size(nuisance, 1) == size(X, 1) ||
            error("Nuisance rows $(size(nuisance, 1)) != design rows $(size(X, 1))")
        X = hcat(X, Matrix{Float64}(nuisance))
        append!(names, isempty(nuisance_names) ?
                ["nuisance_$i" for i in 1:size(nuisance, 2)] : nuisance_names)
    end
    return X, names
end

"""Section-wise circular shift of embedding PCs (common shift across families).

Nuisance predictors are NOT shifted — pass them separately and keep aligned.
"""
function circular_shift_pcs(Z_by_family::Dict{String,<:AbstractMatrix},
                            sec_ids::AbstractVector{Int},
                            shift_by_sec::Dict{Int,Int})
    out = Dict{String, Matrix{Float64}}()
    for (fam, Z) in Z_by_family
        W = similar(Z)
        for sid in unique(sec_ids)
            rows = findall(==(sid), sec_ids)
            s = mod(shift_by_sec[sid], length(rows))
            W[rows, :] = Z[rows[circshift(collect(1:length(rows)), s)], :]
        end
        out[fam] = W
    end
    return out
end
