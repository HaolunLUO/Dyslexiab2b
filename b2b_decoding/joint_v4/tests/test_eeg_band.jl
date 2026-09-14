#!/usr/bin/env julia
#   julia --project=.. tests/test_eeg_band.jl

using Test
using LinearAlgebra
using Statistics
using JSON3

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "eeg_band.jl"))

function _tone_amp(y, fs, f)
    t = range(0, length=length(y), step=1 / fs)
    return abs(dot(y, sin.(2π * f .* t))) / length(y)
end

@testset "delta keeps 2 Hz, attenuates 6 Hz" begin
    fs = 100.0
    t = range(0, 20; step=1 / fs)
    x = zeros(1, length(t))
    x[1, :] = sin.(2π .* 2.0 .* t) .+ sin.(2π .* 6.0 .* t)
    y = vec(bandpass_continuous(x, fs; band="delta"))
    @test _tone_amp(y, fs, 2.0) > 5 * _tone_amp(y, fs, 6.0)
end

@testset "theta keeps 6 Hz, attenuates 2 Hz" begin
    fs = 100.0
    t = range(0, 20; step=1 / fs)
    x = zeros(1, length(t))
    x[1, :] = sin.(2π .* 2.0 .* t) .+ sin.(2π .* 6.0 .* t)
    y = vec(bandpass_continuous(x, fs; band="theta"))
    @test _tone_amp(y, fs, 6.0) > 5 * _tone_amp(y, fs, 2.0)
end

@testset "broad is identity" begin
    x = randn(3, 200)
    @test bandpass_continuous(x, 100.0; band="broad") == x
end

@testset "temporal subsets" begin
    names = ["Fz", "C5", "T7", "TP7", "P7", "C6", "T8", "TP8", "P8", "Cz"]
    meta = Dict("channel_names" => names)
    eeg = randn(length(names), 50)
    L, ln = select_chan_set(eeg, meta; chan_set="left_temporal")
    R, rn = select_chan_set(eeg, meta; chan_set="right_temporal")
    @test ln == LEFT_TEMPORAL
    @test rn == RIGHT_TEMPORAL
    @test size(L, 1) == 4 && size(R, 1) == 4
end

@testset "extractor montage has laterality sites" begin
    meta_path = joinpath("/orcd/pool/005/haolun52",
        "extracted_sections_wordlocked_mne_ica_v1_envpitch_b2b",
        "D001d", "section_001", "metadata.json")
    if isfile(meta_path)
        meta = JSON3.read(read(meta_path, String), Dict)
        names = channel_names_from_meta(meta)
        for w in vcat(LEFT_TEMPORAL, RIGHT_TEMPORAL)
            @test w in names
        end
    else
        @test_skip "extractor metadata not on this host"
    end
end
