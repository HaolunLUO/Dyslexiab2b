#!/usr/bin/env julia
# Smoke-test spatial DSS/PCA + temporal binning (no EEG I/O).
#   julia --project=.. tests/test_snr_preprocess.jl

using Test
using LinearAlgebra
using Statistics
using Random

# Force SNR knobs before including config
ENV["B2B_SPATIAL_DENOISE"] = "dss"
ENV["B2B_SPATIAL_N_COMP"] = "20"
ENV["B2B_TIME_BIN_MS"] = "20"

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "spatial_denoise.jl"))

@testset "spatial PCA/DSS + binning" begin
    Random.seed!(0)
    n_ch, n_t, n_w = 65, 121, 300
    times = collect(range(-0.2, 1.0; length=n_t))
    dat = randn(n_ch, n_t, n_w)
    sig = exp.(-((times .- 0.3) .^ 2) ./ (2 * 0.05^2))
    for w in 1:n_w
        amp = 0.8 + 0.2 * randn()   # same-sign so the average ERP is nonzero
        @views dat[1:5, :, w] .+= amp .* sig'
    end
    keep = trues(n_w)
    keep[1:10] .= false

    filt_pca = fit_spatial_pca(dat, keep; n_comp=20)
    @test size(filt_pca.W) == (n_ch, 20)
    out_pca = apply_spatial_filter(dat, filt_pca)
    @test size(out_pca) == (20, n_t, n_w)

    filt_dss = fit_spatial_dss(dat, keep; n_comp=20)
    @test size(filt_dss.W) == (n_ch, 20)
    out_dss = apply_spatial_filter(dat, filt_dss)
    @test size(out_dss) == (20, n_t, n_w)
    # Top DSS component should track the injected ERP shape
    erp_dss = dropdims(mean(out_dss[:, :, keep]; dims=3); dims=3)
    r = cor(vec(erp_dss[1, :]), sig)
    @test r > 0.5

    binned, t2, sf2, ns = bin_epochs_time(out_dss, times; bin_ms=20.0, sfreq=100.0)
    @test ns == 2
    @test size(binned, 2) == n_t ÷ 2
    @test length(t2) == size(binned, 2)
    @test sf2 ≈ 50.0

    dat2, t3, sf3, meta = preprocess_epochs_snr(dat, keep, times, 100.0)
    @test meta["spatial_denoise"] == "dss"
    @test meta["n_ch_out"] == 20
    @test meta["n_samp_per_bin"] == 2
    @test size(dat2) == (20, n_t ÷ 2, n_w)
    @test sf3 ≈ 50.0
end

println("SNR preprocess smoke OK")
