#!/usr/bin/env julia
# Prepare frozen EEG-independent feature basis (joint B2B v4).
# Defaults: Whisper Large V2 features, independent PCA, 40-knot drift removal.
#
#   julia --project=.. prepare_feature_basis.jl
#
# Suggested Slurm: 8 CPUs, 48 GB, 4 h (Large V2 acoustic/speech are 12800-d).

const BLAS_THREADS = parse(Int, get(ENV, "B2B_BLAS_THREADS",
                                     string(max(1, Sys.CPU_THREADS))))
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(BLAS_THREADS)

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "data_io.jl"))
include(joinpath(@__DIR__, "feature_basis.jl"))

function main()
    println("Preparing feature basis")
    println("  extractor : ", B2B_EXTRACTOR_DIR)
    println("  basis_dir : ", B2B_BASIS_DIR)
    println("  features  : ", FEATURE_SETS)
    println("  basis_mode: ", B2B_BASIS_MODE)
    println("  K         : ", B2B_PCA_K, "  fit_max=", B2B_PCA_FIT_MAX)
    println("  drift_knots: ", B2B_DRIFT_KNOTS)
    assert_outdir_manifest!(B2B_OUTDIR)
    prepare_feature_basis()
end

isinteractive() || main()
