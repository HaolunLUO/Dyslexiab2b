# =====================================================================
# Spatial denoising (PCA / DSS) + temporal binning for low-SNR EEG
# Fit filters on kept epochs only; apply to the full epoch tensor.
# =====================================================================

using LinearAlgebra
using Statistics

"""Channel-space PCA on kept epochs. Returns (W, μ, eigenvalues, method)."""
function fit_spatial_pca(dat::AbstractArray{<:Real,3}, keep::AbstractVector{Bool};
                         n_comp::Integer=B2B_SPATIAL_N_COMP)
    n_ch, n_times, _ = size(dat)
    kept = findall(keep)
    isempty(kept) && error("No kept epochs for spatial PCA")
    X = reshape(dat[:, :, kept], n_ch, :)
    μ = vec(mean(X; dims=2))
    Xc = X .- μ
    C = Symmetric((Xc * Xc') / size(Xc, 2))
    F = eigen(C)
    ord = sortperm(F.values; rev=true)
    vals = F.values[ord]
    n_keep = clamp(Int(n_comp), 1, n_ch)
    W = F.vectors[:, ord[1:n_keep]]
    return (W=W, μ=μ, eigenvalues=vals[1:n_keep], method="pca",
            var_explained=sum(vals[1:n_keep]) / max(sum(vals), eps()))
end

"""
Denoising Source Separation with the word-locked average as bias filter
(de Cheveigné & Simon style: whiten → average → PCA of whitened ERP).
"""
function fit_spatial_dss(dat::AbstractArray{<:Real,3}, keep::AbstractVector{Bool};
                         n_comp::Integer=B2B_SPATIAL_N_COMP,
                         reg_frac::Float64=1e-6)
    n_ch, n_times, _ = size(dat)
    kept = findall(keep)
    isempty(kept) && error("No kept epochs for DSS")
    X = reshape(dat[:, :, kept], n_ch, :)
    μ = vec(mean(X; dims=2))
    Xc = X .- μ
    Cxx = Symmetric((Xc * Xc') / size(Xc, 2))
    Fxx = eigen(Cxx)
    ord = sortperm(Fxx.values; rev=true)
    evals = max.(Fxx.values[ord], 0.0)
    V = Fxx.vectors[:, ord]
    floor = reg_frac * (maximum(evals) + eps())
    evals_r = max.(evals, floor)
    Dinvsqrt = 1.0 ./ sqrt.(evals_r)

    erp = dropdims(mean(dat[:, :, kept]; dims=3); dims=3)
    erpc = erp .- μ
    Z = Diagonal(Dinvsqrt) * (V' * erpc)          # whitened ERP (n_ch × n_times)
    Cz = Symmetric((Z * Z') / n_times)
    Fz = eigen(Cz)
    ordz = sortperm(Fz.values; rev=true)
    n_keep = clamp(Int(n_comp), 1, n_ch)
    U = Fz.vectors[:, ordz[1:n_keep]]
    # Sensor → component: y = U' D^{-1/2} V' (x − μ)
    W = V * (Diagonal(Dinvsqrt) * U)
    for j in 1:n_keep
        nj = norm(@view W[:, j])
        nj > 0 && (W[:, j] ./= nj)
    end
    dss_vals = max.(Fz.values[ordz[1:n_keep]], 0.0)
    return (W=W, μ=μ, eigenvalues=dss_vals, method="dss",
            var_explained=sum(dss_vals) / max(sum(max.(Fz.values, 0.0)), eps()))
end

function fit_spatial_filter(dat::AbstractArray{<:Real,3}, keep::AbstractVector{Bool};
                            method::AbstractString=B2B_SPATIAL_DENOISE,
                            n_comp::Integer=B2B_SPATIAL_N_COMP)
    m = lowercase(method)
    m == "none" && return nothing
    m == "pca" && return fit_spatial_pca(dat, keep; n_comp=n_comp)
    m == "dss" && return fit_spatial_dss(dat, keep; n_comp=n_comp)
    error("B2B_SPATIAL_DENOISE must be none|pca|dss; got $method")
end

"""Project (n_ch × n_times × n_words) → (n_comp × n_times × n_words)."""
function apply_spatial_filter(dat::AbstractArray{<:Real,3}, filt)
    filt === nothing && return dat
    n_ch, n_times, n_words = size(dat)
    size(filt.W, 1) == n_ch ||
        error("Spatial filter channels $(size(filt.W, 1)) != data $n_ch")
    n_comp = size(filt.W, 2)
    X = reshape(dat, n_ch, :)
    Y = filt.W' * (X .- filt.μ)
    return reshape(Y, n_comp, n_times, n_words)
end

"""
Average adjacent samples into bins of width `bin_ms`.

Returns (dat_binned, times_binned, sfreq_eff, n_samp_per_bin).
`bin_ms ≤ 0` leaves data unchanged.
Leftover samples at the end of the epoch (not filling a full bin) are dropped.
"""
function bin_epochs_time(dat::AbstractArray{<:Real,3},
                         times::AbstractVector{<:Real};
                         bin_ms::Real=B2B_TIME_BIN_MS,
                         sfreq::Real)
    bin_ms <= 0 && return dat, collect(Float64, times), Float64(sfreq), 1
    n_samp = max(1, round(Int, bin_ms * sfreq / 1000.0))
    n_samp == 1 && return dat, collect(Float64, times), Float64(sfreq), 1
    n_ch, n_times, n_words = size(dat)
    length(times) == n_times || error("times length $(length(times)) != n_times $n_times")
    n_bins = n_times ÷ n_samp
    n_bins < 2 && error("B2B_TIME_BIN_MS=$bin_ms yields <2 bins over epoch ($(n_times) samples)")
    out = Array{Float64}(undef, n_ch, n_bins, n_words)
    new_times = Vector{Float64}(undef, n_bins)
    for b in 1:n_bins
        i0 = (b - 1) * n_samp + 1
        i1 = b * n_samp
        out[:, b, :] = dropdims(mean(@view(dat[:, i0:i1, :]); dims=2); dims=2)
        new_times[b] = mean(@view times[i0:i1])
    end
    sfreq_eff = Float64(sfreq) / n_samp
    return out, new_times, sfreq_eff, n_samp
end

"""Apply configured spatial denoise then temporal binning; return updated arrays + meta.

When `B2B_DSS_FIT_SCOPE=train_half` and spatial denoise is on, spatial filtering is
deferred: only temporal binning is applied here, and callers must call
`apply_partition_spatial` once per outer partition on the train-half keep mask.
"""
function preprocess_epochs_snr(dat::AbstractArray{<:Real,3},
                               keep::AbstractVector{Bool},
                               times::AbstractVector{<:Real},
                               sfreq::Real)
    meta = Dict{String,Any}(
        "spatial_denoise" => B2B_SPATIAL_DENOISE,
        "spatial_n_comp" => B2B_SPATIAL_N_COMP,
        "time_bin_ms" => B2B_TIME_BIN_MS,
        "dss_fit_scope" => B2B_DSS_FIT_SCOPE,
        "spatial_deferred" => false,
        "sfreq_in" => Float64(sfreq),
        "n_ch_in" => size(dat, 1),
        "n_times_in" => size(dat, 2),
    )
    defer = (B2B_DSS_FIT_SCOPE == "train_half") && (B2B_SPATIAL_DENOISE != "none")
    dat1 = dat
    if defer
        meta["spatial_deferred"] = true
        println("  [SNR] spatial=$(B2B_SPATIAL_DENOISE) DEFERRED (fit_scope=train_half); " *
                "binning sensors first")
    else
        filt = fit_spatial_filter(dat, keep)
        dat1 = apply_spatial_filter(dat, filt)
        if filt !== nothing
            meta["spatial_eigenvalues"] = collect(filt.eigenvalues)
            meta["spatial_var_explained"] = filt.var_explained
            meta["n_ch_spatial"] = size(dat1, 1)
            println("  [SNR] spatial=$(filt.method)  n_comp=$(size(dat1, 1))  " *
                    "var/bias-explained=$(round(filt.var_explained; digits=4))")
        end
    end
    dat2, times2, sfreq2, n_samp = bin_epochs_time(dat1, times; sfreq=sfreq)
    meta["n_samp_per_bin"] = n_samp
    meta["sfreq_out"] = sfreq2
    meta["n_ch_out"] = size(dat2, 1)
    meta["n_times_out"] = size(dat2, 2)
    if n_samp > 1
        println("  [SNR] time_bin=$(B2B_TIME_BIN_MS) ms  n_samp/bin=$n_samp  " *
                "n_times=$(size(dat2, 2))  sfreq_eff=$(round(sfreq2; digits=4)) Hz")
    end
    return dat2, times2, sfreq2, meta
end

"""Fit spatial filter on `keep_train` and apply to full `dat` (partition-scoped DSS/PCA)."""
function apply_partition_spatial(dat::AbstractArray{<:Real,3},
                                 keep_train::AbstractVector{Bool})
    filt = fit_spatial_filter(dat, keep_train)
    filt === nothing && return dat, nothing
    return apply_spatial_filter(dat, filt), filt
end

"""Return dat for one partition: identity unless spatial was deferred to train_half."""
function maybe_partition_spatial(dat::AbstractArray{<:Real,3},
                                 keep_train::AbstractVector{Bool},
                                 snr_meta)
    deferred = snr_meta !== nothing && get(snr_meta, "spatial_deferred", false) == true
    deferred || return dat
    sum(keep_train) < 10 && error("Too few train-half epochs for deferred spatial fit")
    dat_out, filt = apply_partition_spatial(dat, keep_train)
    if filt !== nothing
        println("    [SNR] partition spatial=$(filt.method) n_comp=$(size(dat_out, 1)) " *
                "fit_on=$(sum(keep_train)) epochs")
    end
    return dat_out
end
