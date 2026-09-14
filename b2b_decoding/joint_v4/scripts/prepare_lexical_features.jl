#!/usr/bin/env julia
# =====================================================================
# Build lexical scalar features for King-style B2B:
#   duration   = word acoustic duration (onset → offset)
#   frequency  = logfreq (imputed if -Inf)
#   surprisal  = GPT2CN surprisal (must already exist as npy)
#
# Writes into extractor _shared_wordlocked_features/section_XXX/:
#   X_word_lexical_duration.npy
#   X_word_lexical_frequency.npy
#   (surprisal expected as X_word_gpt2cn_surprisal.npy from compute script)
#
# Usage:
#   julia --project=.. scripts/prepare_lexical_features.jl
# =====================================================================

using Statistics
using CSV
using DataFrames
using NPZ
using Printf
using JSON3

const EXTRACTOR = get(ENV, "B2B_EXTRACTOR_DIR",
    "/home/haolun52/orcd/pool/extracted_sections_wordlocked_shared")
const SECTIONS = (1, 2)
const EXPECTED = (1753, 1800)
const SFREQ = 100.0

function _impute_logfreq(v::AbstractVector)
    x = Float64.(v)
    finite = filter(isfinite, x)
    isempty(finite) && error("No finite logfreq")
    floor = minimum(finite)
    n_imp = count(!isfinite, x)
    x = ifelse.(isfinite.(x), x, floor)
    return x, n_imp, floor
end

function main()
    println("Preparing lexical scalar features under $EXTRACTOR")
    for (i, sid) in enumerate(SECTIONS)
        shared = joinpath(EXTRACTOR, "_shared_wordlocked_features",
                          @sprintf("section_%03d", sid))
        csv_path = joinpath(shared, "word_timing_relative.csv")
        isfile(csv_path) || error("Missing $csv_path")
        df = CSV.read(csv_path, DataFrame)
        nrow(df) == EXPECTED[i] ||
            error("Section $sid: $(nrow(df)) words != $(EXPECTED[i])")

        duration = if "onset_relative" in names(df) && "offset_relative" in names(df)
            Float64.(df.offset_relative .- df.onset_relative)
        else
            Float64.(df.offset_sample .- df.onset_sample) ./ SFREQ
        end
        any(duration .<= 0) && @warn "Non-positive durations in section $sid" n=count(<=(0), duration)

        logfreq, n_imp, floor = _impute_logfreq(df.logfreq)
        n_imp > 0 && println("  sec$sid: imputed $n_imp logfreq with floor=$floor")

        surp_path = joinpath(shared, "X_word_gpt2cn_surprisal.npy")
        isfile(surp_path) || error(
            "Missing GPT2CN surprisal: $surp_path\n" *
            "Run: python scripts/compute_gpt2cn_surprisal.py --timing-root … --out-root …")

        surp = Float64.(NPZ.npzread(surp_path))
        size(surp, 1) == nrow(df) ||
            error("Surprisal rows $(size(surp,1)) != words $(nrow(df))")
        size(surp, 2) == 1 || error("Expected surprisal n×1, got $(size(surp))")
        any(!isfinite, surp) && error("Non-finite surprisal in section $sid")

        # Save duration / frequency as n×1 float32-compatible float64 arrays
        for (name, col) in (
            ("lexical_duration", duration),
            ("lexical_frequency", logfreq),
        )
            X = reshape(Float64.(col), :, 1)
            out = joinpath(shared, "X_word_$(name).npy")
            NPZ.npzwrite(out, X)
            open(joinpath(shared, "X_word_$(name)_feature_names.txt"), "w") do io
                println(io, name)
            end
            meta = Dict(
                "feature" => name,
                "section_id" => sid,
                "n_words" => nrow(df),
                "mean" => mean(col),
                "std" => std(col; corrected=true),
                "source_csv" => csv_path,
            )
            open(joinpath(shared, "X_word_$(name)_meta.json"), "w") do io
                JSON3.write(io, meta)
            end
            println("  wrote $out  mean=$(round(mean(col); digits=4))")
        end

        # Correlation snapshot (raw, unstandardized)
        C = cor(hcat(duration, logfreq, vec(surp)))
        println("  sec$sid corr [dur,freq,surp]:")
        println("    ", round.(C; digits=3))
    end
    println("Lexical feature files ready.")
end

isinteractive() || main()
