#!/usr/bin/env julia
# =====================================================================
# Joint-family B2B v4 Slurm / CLI entry point
# Usage:
#   julia --project=.. b2b_joint_v4_pipeline.jl RN109
#   julia --project=.. b2b_joint_v4_pipeline.jl 1
# Env: B2B_MODE=observed|null|both, B2B_PCA_K, B2B_N_NULL,
#      B2B_NULL_START, B2B_NULL_COUNT, B2B_SEED, B2B_OUTDIR,
#      B2B_EPOCH_ANCHOR, B2B_BASIS_MODE, B2B_FEAT_*, B2B_INCLUDE_NUISANCE, …
# =====================================================================

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
    assert_outdir_manifest!(B2B_OUTDIR)
    participant = resolve_participant()
    println("="^60)
    println("Joint-family B2B decoding v4 (Large V2 / onset-locked)")
    println("="^60)
    println("  Participant : ", participant)
    println("  Mode        : ", B2B_MODE)
    println("  PCA K       : ", B2B_PCA_K)
    println("  Anchor      : ", B2B_EPOCH_ANCHOR)
    println("  Basis mode  : ", B2B_BASIS_MODE)
    println("  Features    : ", FEATURE_SETS)
    println("  Nuisance    : ", B2B_INCLUDE_NUISANCE)
    println("  Epoch       : ", TMIN_S, " .. ", TMAX_S, " s")
    println("  Spatial     : ", B2B_SPATIAL_DENOISE, "  n_comp=", B2B_SPATIAL_N_COMP,
            "  fit_scope=", B2B_DSS_FIT_SCOPE)
    println("  Time bin    : ", B2B_TIME_BIN_MS, " ms")
    println("  Confirm win : ", CONFIRM_WIN_S)
    println("  Outdir      : ", B2B_OUTDIR)
    println("  Basis dir   : ", B2B_BASIS_DIR)
    println("  Null        : start=$(B2B_NULL_START) count=$(B2B_NULL_COUNT) total=$(B2B_N_NULL)")
    println("  Seed        : ", B2B_SEED)
    println("  Cohort hash : ", cohort_hash())
    println("  Git commit  : ", git_commit_hash())

    isfile(joinpath(B2B_BASIS_DIR, "feature_basis.npz")) ||
        error("Missing feature basis. Run prepare_feature_basis.jl first.")

    run_subject_v4(participant; mode=B2B_MODE, k=B2B_PCA_K)
end

isinteractive() || main()
