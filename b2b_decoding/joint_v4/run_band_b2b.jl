#!/usr/bin/env julia
# Phase 3b/3c: observed + 200 existence nulls on band-limited (and optional
# channel-subset) EEG. Writes into B2B_OUTDIR only; never the frozen observed.
#   julia --project=.. run_band_b2b.jl 1

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
    occursin("passthrough", B2B_OUTDIR) &&
        error("Refusing band/laterality writes into a passthrough (frozen) outdir: $B2B_OUTDIR")
    B2B_OBSERVED_OUTDIR == B2B_OUTDIR ||
        error("Band jobs must export B2B_OBSERVED_OUTDIR=\$B2B_OUTDIR before julia starts. " *
              "Got observed=$(B2B_OBSERVED_OUTDIR) out=$(B2B_OUTDIR)")
    B2B_ALLOW_INPLACE_NULLS ||
        error("Band jobs require B2B_ALLOW_INPLACE_NULLS=1 at julia start")
    mkpath(B2B_OUTDIR)
    participant = resolve_participant()
    println("="^60)
    println("Phase 3b/3c band B2B (observed + existence nulls)")
    println("="^60)
    println("  Participant : ", participant)
    println("  Band        : ", B2B_EEG_BAND)
    println("  Chan set    : ", B2B_CHAN_SET)
    println("  Anchor      : ", B2B_EPOCH_ANCHOR)
    println("  Epoch       : ", TMIN_S, " .. ", TMAX_S, " s")
    println("  Outdir      : ", B2B_OUTDIR)
    println("  Basis       : ", B2B_BASIS_DIR)
    isfile(joinpath(B2B_BASIS_DIR, "feature_basis.npz")) ||
        error("Missing feature basis")

    obs_csv = joinpath(B2B_OUTDIR, participant, "$(participant)_b2b_family_agg.csv")
    diag_csv = joinpath(B2B_OUTDIR, participant, "$(participant)_diagnostics.csv")
    if isfile(obs_csv) && isfile(diag_csv) &&
       lowercase(get(ENV, "B2B_FORCE", "0")) ∉ ("1", "true", "yes")
        println("  [skip observed] $obs_csv exists")
    else
        run_subject_v4(participant; mode="observed", k=B2B_PCA_K)
    end
    run_existence_nulls!(participant)
end

isinteractive() || main()
