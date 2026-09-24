# MixedModels fallback. Called from run_lmm.sbatch only after lme4 fails a required rung.
# New project environment. Does not touch the B2B Project.toml.

using CSV
using CategoricalArrays
using DataFrames
using MixedModels
using Statistics

const INCSV = "/orcd/pool/005/haolun52/n400_storytime_v1/stimulus/lmm_input.csv"

function age_center(df)
    sub = combine(groupby(df, :subject), :age => (x -> x[1]) => :age)
    sub = sub[isfinite.(sub.age), :]
    mean(sub.age), nrow(sub)
end

function add_codes!(df, center)
    g1 = fill(NaN, nrow(df))
    g2 = fill(NaN, nrow(df))
    for i in 1:nrow(df)
        g = df.group[i]
        if g == "TD"
            g1[i] = 2 / 3
            g2[i] = 0.0
        elseif g == "dyslexia_normal_CAP"
            g1[i] = -1 / 3
            g2[i] = 0.5
        elseif g == "dyslexia_atypical_CAP"
            g1[i] = -1 / 3
            g2[i] = -0.5
        end
    end
    df.g1 = g1
    df.g2 = g2
    df.age_c = df.age .- center
    df.group = string.(df.group)
    df.subject = string.(df.subject)
    df.item = string.(df.item_id)
    df.section = categorical(string.(df.section))
    df.subject = categorical(df.subject)
    df.item = categorical(df.item)
    df.trial_uid = categorical(string.(df.subject) .* "::" .* string.(df.item))
    df
end

function write_coef(m, path)
    ct = DataFrame(coeftable(m))
    CSV.write(path, ct)
    open(replace(path, ".csv" => "_opt.txt"), "w") do io
        println(io, "engine MixedModels")
        println(io, "objective ", objective(m))
        println(io, "reterms ", join(string.(m.reterms), " | "))
    end
end

function main()
    isempty(get(ENV, "SLURM_JOB_ID", "")) && error("Refusing to fit outside Slurm")
    out = length(ARGS) >= 1 ? ARGS[1] : "/orcd/pool/005/haolun52/n400_storytime_v1/revision/lmm"
    mkpath(out)
    raw = CSV.read(INCSV, DataFrame)
    raw = raw[(raw.keep .== 1) .& isfinite.(raw.dv_cp) .& isfinite.(raw.dv_fz), :]
    center, n_age = age_center(raw)
    println("age_center $center n_age $n_age")
    df = add_codes!(raw, center)
    d_age = df[isfinite.(df.age), :]

    status_path = joinpath(out, "STATUS.txt")
    status = isfile(status_path) ? read(status_path, String) : ""
    # (1 + sur_z || subject) + (1 | item). zerocorr drops the slope-intercept correlation.
    if occursin("NEED_MIXEDMODELS", status)
        f = @formula(
            dv_cp ~ sur_z * age_c + sur_z * g1 + sur_z * g2 +
                duration_z + position_z + section +
                sur_m1_z + sur_p1_z + freq_m1_z + freq_p1_z +
                assoc_m1_z + assoc_p1_z + freq_z + assoc_z +
                zerocorr(1 + sur_z | subject) + (1 | item)
        )
        println("MixedModels core floor")
        m = fit(MixedModel, f, d_age; REML = true, progress = false)
        write_coef(m, joinpath(out, "core_mixed_coef.csv"))
        if any(stderror(m) .<= 0)
            error("MixedModels core zeroed a standard error; Bayesian step still required")
        end
        open(joinpath(out, "core_mixed_OK.txt"), "w") do io
            println(io, "MixedModels REML core floor converged")
        end
    end

    if isfile(joinpath(out, "region_FAILED.txt"))
        cp = copy(d_age)
        cp.dv = cp.dv_cp
        cp.region = fill(0.5, nrow(cp))
        fz = copy(d_age)
        fz.dv = fz.dv_fz
        fz.region = fill(-0.5, nrow(fz))
        long = vcat(cp, fz)
        f = @formula(
            dv ~ sur_z * region * g1 + sur_z * region * g2 +
                sur_z * age_c + region * age_c +
                duration_z + position_z + section +
                sur_m1_z + sur_p1_z + freq_m1_z + freq_p1_z +
                assoc_m1_z + assoc_p1_z + freq_z + assoc_z +
                zerocorr(1 + sur_z | subject) + (1 | item)
        )
        println("MixedModels region without trial_uid")
        m = fit(MixedModel, f, long; REML = true, progress = false)
        write_coef(m, joinpath(out, "region_mixed_coef.csv"))
        open(joinpath(out, "region_mixed_OK.txt"), "w") do io
            println(io, "MixedModels region without trial_uid")
            println(io, "lme4 reason: ", read(joinpath(out, "region_FAILED.txt"), String))
        end
    end
    println("MixedModels fallback done")
end

main()
