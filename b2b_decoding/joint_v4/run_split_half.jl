#!/usr/bin/env julia
# Phase 2 split-half observed B2B (odd/even 60 s blocks).
#   julia --project=.. run_split_half.jl 1

const BLAS_THREADS = parse(Int, get(ENV, "B2B_BLAS_THREADS",
                                     string(max(1, Sys.CPU_THREADS))))
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(BLAS_THREADS)
println("BLAS threads: ", BLAS.get_num_threads())

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "data_io.jl"))
include(joinpath(@__DIR__, "eeg_band.jl"))
include(joinpath(@__DIR__, "spatial_denoise.jl"))
include(joinpath(@__DIR__, "feature_basis.jl"))
include(joinpath(@__DIR__, "partitions.jl"))
include(joinpath(@__DIR__, "estimator.jl"))
include(joinpath(@__DIR__, "pipeline.jl"))
include(joinpath(@__DIR__, "split_half.jl"))

function resolve_participant()
    raw = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "B2B_SUBJECT", "")
    isempty(raw) && error("No participant given (ARG or B2B_SUBJECT).")
    _, participants = load_cohort()
    idx = tryparse(Int, raw)
    if idx !== nothing
        (1 <= idx <= length(participants)) ||
            error("Index $idx out of range 1:$(length(participants))")
        return participants[idx]
    end
    raw in participants || error("Unknown participant: $raw")
    return raw
end

function main()
    mkpath(B2B_OUTDIR)
    participant = resolve_participant()
    println("="^60)
    println("Phase 2 split-half (interleaved 60 s)")
    println("="^60)
    println("  Participant : ", participant)
    println("  Anchor      : ", B2B_EPOCH_ANCHOR)
    println("  Epoch       : ", TMIN_S, " .. ", TMAX_S, " s")
    println("  Outdir      : ", B2B_OUTDIR)
    println("  Basis       : ", B2B_BASIS_DIR)
    run_split_half!(participant)
end

isinteractive() || main()
