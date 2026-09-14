#!/usr/bin/env julia
# Aggregate null chunks and/or QC for joint B2B v4.
# Usage:
#   julia --project=.. aggregate_v4.jl [outdir]
#   julia --project=.. aggregate_v4.jl --nulls-only

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "data_io.jl"))
include(joinpath(@__DIR__, "eeg_band.jl"))
include(joinpath(@__DIR__, "spatial_denoise.jl"))
include(joinpath(@__DIR__, "feature_basis.jl"))
include(joinpath(@__DIR__, "partitions.jl"))
include(joinpath(@__DIR__, "estimator.jl"))
include(joinpath(@__DIR__, "pipeline.jl"))

using CSV
using DataFrames
using JSON3
using Statistics

function main()
    outdir = B2B_OUTDIR
    nulls_only = "--nulls-only" in ARGS
    for a in ARGS
        startswith(a, "-") || (outdir = a)
    end
    _, participants = load_cohort()
    qc_dir = joinpath(outdir, "_qc")
    mkpath(qc_dir)

    rows = DataFrame[]
    missing_p = String[]
    null_complete = String[]
    null_incomplete = String[]
    for p in participants
        part = joinpath(outdir, p)
        if !isdir(part)
            push!(missing_p, p)
            continue
        end
        if nulls_only || !isfile(joinpath(part, "$(p)_null_summary.csv"))
            sum_files = filter(f -> occursin(r"_null_\d+_\d+_summary\.csv$", f),
                               isdir(part) ? readdir(part) : String[])
            if !isempty(sum_files)
                try
                    aggregate_participant_nulls(p, part)
                    push!(null_complete, p)
                catch e
                    @warn "Null aggregate failed for $p" exception=e
                    push!(null_incomplete, p)
                end
            else
                push!(null_incomplete, p)
            end
        else
            push!(null_complete, p)
        end
        qpath = joinpath(part, "$(p)_qc_summary.csv")
        if isfile(qpath)
            push!(rows, CSV.read(qpath, DataFrame))
        else
            push!(missing_p, p)
        end
        mpath = joinpath(part, "$(p)_run_manifest.json")
        isfile(mpath) || @warn "Missing run manifest for $p"
        # Cohort hash consistency
        if isfile(mpath)
            man = JSON3.read(read(mpath, String), Dict{String,Any})
            if haskey(man, "cohort_hash") && string(man["cohort_hash"]) != cohort_hash()
                @warn "Cohort hash mismatch for $p" stored=man["cohort_hash"] current=cohort_hash()
            end
        end
    end

    if !isempty(rows)
        qc = vcat(rows...; cols=:union)
        cohort, _ = load_cohort()
        gmap = Dict(String(r.participant) => String(r.group) for r in eachrow(cohort))
        qc.group = [get(gmap, string(p), "unknown") for p in qc.participant]
        CSV.write(joinpath(qc_dir, "participant_qc.csv"), qc)

        # Onset-locked retention by group
        if "keep_frac" in names(qc) && "group" in names(qc)
            by_g = combine(groupby(qc, :group),
                           :keep_frac => mean => :mean_keep,
                           :keep_frac => median => :median_keep,
                           nrow => :n)
            CSV.write(joinpath(qc_dir, "keep_frac_by_group.csv"), by_g)
            println(by_g)
        end

        flow = DataFrame(
            stage = ["listed", "with_qc", "missing",
                     "valid_partitions_ge_16", "keep_frac_ge_0.5",
                     "null_complete"],
            n = [
                length(participants),
                nrow(qc),
                length(unique(missing_p)),
                count(x -> !ismissing(x) && x >= 16,
                      "n_valid_partitions" in names(qc) ? qc.n_valid_partitions : fill(missing, nrow(qc))),
                count(x -> !ismissing(x) && x >= 0.5, qc.keep_frac),
                length(unique(null_complete)),
            ],
        )
        CSV.write(joinpath(qc_dir, "cohort_flow.csv"), flow)
        println(flow)
    end
    if !isempty(missing_p)
        CSV.write(joinpath(qc_dir, "missing_qc_participants.csv"),
                  DataFrame(participant=unique(missing_p)))
    end
    write_json(joinpath(qc_dir, "null_completeness_report.json"), Dict(
        "n_expected" => length(participants),
        "n_complete" => length(unique(null_complete)),
        "complete" => sort(unique(null_complete)),
        "incomplete" => sort(unique(null_incomplete)),
        "cohort_hash" => cohort_hash(),
    ))
    println("QC aggregation written under $qc_dir")
end

isinteractive() || main()
