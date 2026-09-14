#!/usr/bin/env julia
# Phase 1 existence nulls (cached Y'Y). Does not retune λ.
#   julia --project=.. run_existence_nulls.jl RN109
#   julia --project=.. run_existence_nulls.jl 1
# Required env: B2B_OUTDIR (existence), B2B_OBSERVED_OUTDIR (frozen observed).

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
include(joinpath(@__DIR__, "existence_nulls.jl"))

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
    isempty(B2B_OBSERVED_OUTDIR) &&
        error("Set B2B_OBSERVED_OUTDIR to the frozen observed outdir")
    isdir(B2B_OBSERVED_OUTDIR) ||
        error("B2B_OBSERVED_OUTDIR does not exist: $B2B_OBSERVED_OUTDIR")
    if B2B_OUTDIR == B2B_OBSERVED_OUTDIR && !B2B_ALLOW_INPLACE_NULLS
        error("B2B_OUTDIR must be a new existence outdir, not the observed outdir")
    end
    occursin("tonev3_onset_tmin03_passthrough", B2B_OUTDIR) &&
        occursin("existence", B2B_OUTDIR) == false &&
        error("Refusing to write into frozen observed outdir: $B2B_OUTDIR")
    occursin("tonev3_woffset_tmin05_passthrough", B2B_OUTDIR) &&
        occursin("existence", B2B_OUTDIR) == false &&
        error("Refusing to write into frozen observed outdir: $B2B_OUTDIR")
    mkpath(B2B_OUTDIR)
    participant = resolve_participant()
    println("="^60)
    println("Phase 1 existence nulls (cached YTY, reuse α)")
    println("="^60)
    println("  Participant : ", participant)
    println("  Anchor      : ", B2B_EPOCH_ANCHOR)
    println("  Epoch       : ", TMIN_S, " .. ", TMAX_S, " s")
    println("  Features    : ", FEATURE_SETS)
    println("  Families    : ", FAMILY_NAMES)
    println("  N null      : ", EXISTENCE_N_NULL)
    println("  Min |k|     : ", EXISTENCE_MIN_SHIFT)
    println("  Null method : ", EXISTENCE_NULL_METHOD)
    println("  Observed    : ", B2B_OBSERVED_OUTDIR)
    println("  Outdir      : ", B2B_OUTDIR)
    println("  Basis       : ", B2B_BASIS_DIR)
    println("  Extractor   : ", B2B_EXTRACTOR_DIR)
    println("  Git         : ", git_commit_hash())
    isfile(joinpath(B2B_BASIS_DIR, "feature_basis.npz")) ||
        error("Missing feature basis. Point B2B_BASIS_DIR at the observed _basis.")
    run_existence_nulls!(participant)
end

isinteractive() || main()
