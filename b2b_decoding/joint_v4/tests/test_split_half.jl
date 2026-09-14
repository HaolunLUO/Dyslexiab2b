#!/usr/bin/env julia
#   julia --project=.. tests/test_split_half.jl

using Test

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "split_half.jl"))

@testset "pitch subfamilies respect the feature basis" begin
    @test pitch_subfamily_cols("ctx_pitch", 5) == (("level", 1:1), ("dynamic", 2:5))
    @test pitch_subfamily_cols("pitch", 13) == (("level", 1:1), ("dynamic", 2:13))
    @test_throws ErrorException pitch_subfamily_cols("ctx_pitch", 13)
    S = reshape(collect(1.0:44.0), 1, 44)
    cols = pitch_subfamily_cols("ctx_pitch", 5)[2][2]
    @test _sum_block(S, [0.0], 39, cols) == [sum(41.0:44.0)]
end

@testset "interleaved 60 s is not first/second half" begin
    # 180 s story: blocks 0,1,2 at 60 s
    sec = fill(1, 9)
    t = [5.0, 20, 40, 70, 80, 100, 130, 150, 170]
    odd, even = interleaved_60s_masks(sec, t; win=60)
    # t in [0,60) block 0 even; [60,120) block 1 odd; [120,180) block 2 even
    @test even == [true, true, true, false, false, false, true, true, true]
    @test odd == .!even
    @test any(odd[1:3]) == false          # early words not all in "first half"
    @test count(odd) == 3 && count(even) == 6
end

@testset "two stories wrap independently" begin
    sec = vcat(fill(1, 4), fill(2, 4))
    t = [10.0, 70, 10, 70, 10, 70, 10, 70]
    odd, even = interleaved_60s_masks(sec, t; win=60)
    @test even == [true, false, true, false, true, false, true, false]
end

@testset "peak and fractional-area latency" begin
    times = collect(0.0:0.1:0.8)
    tr = zeros(length(times))
    tr[4] = 1.0   # 0.3 s
    @test peak_latency(times, tr) ≈ 0.3
    # rectangle 0.2–0.6
    tr .= 0
    tr[(times .>= 0.2) .& (times .<= 0.6)] .= 1
    fa = frac_area_latency(times, tr; frac=0.5)
    @test 0.35 <= fa <= 0.45
    @test isnan(frac_area_latency(times, fill(-1.0, length(times))))
end
