# Unfold deconvolution for one child. New project env only.
# Do not activate the B2B project.
#
# Usage (from the unfold julia env):
#   julia --project=.../unfold/julia_env fit_unfold.jl SUBJECT
# Skips a subject whose unfold/<ID>/fit_done.json already exists.

using Unfold
using DataFrames
using CSV
using NPZ
using JSON3
using Statistics
using StatsModels

const POOL = "/orcd/pool/005/haolun52"
const SHARED = joinpath(POOL, "extracted_sections_wordlocked_shared")
const WORD_CSV = joinpath(POOL, "n400_storytime_v1", "stimulus", "word_table.csv")
const SCALER_JSON = joinpath(POOL, "n400_storytime_v1", "stimulus", "predictor_scaler.json")
const FEAT = joinpath(POOL, "stimulus_features_fs100", "trf_auditory_v1")
const OUTROOT = joinpath(POOL, "n400_storytime_v1", "unfold")
const SFREQ = 100.0
const TAU = (-0.2, 1.0)
# 0-based row indices in WORDLOCKED_CHAN_65.
const CP_ROWS0 = (46, 27, 62)  # Pz, Cz, Oz
const FZ_ROW0 = 7

function zapply(x, mean, sd)
    (Float64(x) - Float64(mean)) / Float64(sd)
end

function load_words()
    words = CSV.read(WORD_CSV, DataFrame)
    scaler = JSON3.read(read(SCALER_JSON, String))
    words.sur_z = map(x -> zapply(x, scaler.surprisal.mean, scaler.surprisal.sd), words.surprisal)
    words.freq_z = map(x -> zapply(x, scaler.log_frequency.mean, scaler.log_frequency.sd), words.log_frequency)
    words.duration_z = map(x -> zapply(x, scaler.duration.mean, scaler.duration.sd), words.duration)
    words.assoc_z = map(words.association) do x
        ismissing(x) || !isfinite(x) ? 0.0 : zapply(x, scaler.association.mean, scaler.association.sd)
    end
    words.sur_p1_z = map(words.sur_p1) do x
        ismissing(x) || !isfinite(x) ? 0.0 : zapply(x, scaler.sur_p1.mean, scaler.sur_p1.sd)
    end
    words.assoc_missing = map(x -> (ismissing(x) || !isfinite(x)) ? 1.0 : 0.0, words.association)
    words.sur_p1_missing = map(x -> (ismissing(x) || !isfinite(x)) ? 1.0 : 0.0, words.sur_p1)
    words.content_flag = map(x -> x == true || x == 1 || x == "true" || x == "True" ? 1.0 : 0.0, words.content)
    words
end

function cp_fz(eeg::AbstractMatrix)
    # npy is channels x time, 0-based indices above.
    cp = (eeg[CP_ROWS0[1] + 1, :] .+ eeg[CP_ROWS0[2] + 1, :] .+ eeg[CP_ROWS0[3] + 1, :]) ./ 3
    fz = eeg[FZ_ROW0 + 1, :]
    return Matrix{Float64}(hcat(cp, fz)')  # 2 x time
end

function fit_section(words, section::Int, eeg::AbstractMatrix, env::AbstractVector, onset::AbstractVector)
    T = size(eeg, 2)
    length(env) == T || error("envelope length $(length(env)) != EEG $T")
    length(onset) == T || error("onset-envelope length $(length(onset)) != EEG $T")
    sub = words[words.section .== section, :]
    # Finite current surprisal, frequency, duration. Association and N+1 may be filled with 0.
    ok = trues(nrow(sub))
    for i in 1:nrow(sub)
        s0 = Int(sub.onset_sample[i])
        ok[i] = s0 >= 0 && (s0 + 1) <= T &&
            isfinite(sub.surprisal[i]) && isfinite(sub.log_frequency[i]) && isfinite(sub.duration[i])
    end
    sub = sub[ok, :]
    word = DataFrame(
        event = fill(:word, nrow(sub)),
        latency = Float64.(Int.(sub.onset_sample) .+ 1),
        sur_z = Float64.(sub.sur_z),
        freq_z = Float64.(sub.freq_z),
        assoc_z = Float64.(sub.assoc_z),
        duration_z = Float64.(sub.duration_z),
        content = Float64.(sub.content_flag),
        sur_p1_z = Float64.(sub.sur_p1_z),
        envelope = zeros(nrow(sub)),
        onset_envelope = zeros(nrow(sub)),
    )
    # Sample-aligned acoustics: one event per sample, no intercept (the word onset carries it).
    ac = DataFrame(
        event = fill(:ac, T),
        latency = collect(1.0:Float64(T)),
        sur_z = zeros(T),
        freq_z = zeros(T),
        assoc_z = zeros(T),
        duration_z = zeros(T),
        content = zeros(T),
        sur_p1_z = zeros(T),
        envelope = Float64.(env),
        onset_envelope = Float64.(onset),
    )
    events = vcat(word, ac)
    # Separate basis objects: Unfold mutates the basis name to the event name.
    b_word = firbasis(TAU, SFREQ, :word)
    b_ac = firbasis(TAU, SFREQ, :ac)
    f_word = @formula(0 ~ 1 + sur_z + freq_z + assoc_z + duration_z + content + sur_p1_z)
    f_ac = @formula(0 ~ 0 + envelope + onset_envelope)
    y = cp_fz(eeg)
    println("fitting section=$section words=$(nrow(sub)) T=$T y=$(size(y))")
    flush(stdout)
    t0 = time()
    model = fit(
        UnfoldModel,
        [:word => (f_word, b_word), :ac => (f_ac, b_ac)],
        events,
        y;
        eventcolumn = :event,
        solver = (x, yy) -> Unfold.solver_default(x, yy; stderror = true),
        show_progress = false,
    )
    elapsed = time() - t0
    ct = coeftable(model)
    return ct, elapsed, nrow(sub), count(sub.assoc_missing .== 1), count(sub.sur_p1_missing .== 1)
end

function coef_series(ct::DataFrame, name::String, channel::Int)
    ev = string.(ct.eventname)
    sub = ct[(ct.coefname .== name) .& (ct.channel .== channel) .& (ev .== "word"), :]
    sub = sort(sub, :time)
    Float64.(sub.estimate), Float64.(sub.stderror), Float64.(sub.time)
end

function window_mean(times, beta)
    m = (times .>= 0.35 - 1e-8) .& (times .<= 0.55 + 1e-8)
    mean(beta[m]), count(m)
end

function main()
    if isempty(get(ENV, "SLURM_JOB_ID", ""))
        error("Refusing to fit Unfold outside Slurm. Submit run_unfold.sbatch.")
    end
    length(ARGS) >= 1 || error("usage: fit_unfold.jl SUBJECT")
    subject = ARGS[1]
    outdir = joinpath(OUTROOT, subject)
    done = joinpath(outdir, "fit_done.json")
    if isfile(done)
        println("$subject already done, skip")
        return
    end
    mkpath(outdir)
    words = load_words()
    pieces = []
    meta = Dict{String,Any}("subject" => subject, "basis" => "fir", "tau" => [-0.2, 1.0], "sfreq" => SFREQ)
    for section in (1, 2)
        sec = "section_$(lpad(section, 3, '0'))"
        eeg = NPZ.npzread(joinpath(SHARED, subject, sec, "eeg_data.npy"))
        eeg = eeg isa AbstractMatrix ? eeg : error("$subject $sec eeg not a matrix")
        # NPZ may return time x channel; the extractor saved channels x time.
        if size(eeg, 1) != 65 && size(eeg, 2) == 65
            eeg = permutedims(eeg)
        end
        size(eeg, 1) == 65 || error("$subject $sec n_ch $(size(eeg))")
        env = vec(NPZ.npzread(joinpath(FEAT, sec, "X_envelope.npy")))
        onset = vec(NPZ.npzread(joinpath(FEAT, sec, "X_onset_envelope.npy")))
        ct, elapsed, n_word, n_assoc_fill, n_p1_fill = fit_section(words, section, eeg, env, onset)
        # coeftable can carry an all-nothing :group column, which CSV cannot write.
        ct.eventname = string.(ct.eventname)
        for c in names(ct)
            if eltype(ct[!, c]) === Nothing
                ct[!, c] = Vector{Union{Missing, String}}(missing, nrow(ct))
            end
        end
        CSV.write(joinpath(outdir, "$(sec)_coef.csv"), ct)
        sur_cp, se_cp, times = coef_series(ct, "sur_z", 1)
        sur_fz, se_fz, times_fz = coef_series(ct, "sur_z", 2)
        p1_cp, se_p1, _ = coef_series(ct, "sur_p1_z", 1)
        times == times_fz || error("time grids differ")
        w_cp, nwin = window_mean(times, sur_cp)
        w_fz, _ = window_mean(times, sur_fz)
        w_p1, _ = window_mean(times, p1_cp)
        push!(pieces, (; times, sur_cp, se_cp, sur_fz, se_fz, p1_cp, se_p1, n_word, w_cp, w_fz, w_p1, nwin, elapsed))
        meta[sec] = Dict(
            "n_word_events" => n_word,
            "n_assoc_filled_0" => n_assoc_fill,
            "n_sur_p1_filled_0" => n_p1_fill,
            "elapsed_s" => elapsed,
            "window_sur_cp" => w_cp,
            "window_sur_fz" => w_fz,
            "window_sur_p1_cp" => w_p1,
            "n_window_samples" => nwin,
        )
        println("$subject $sec words=$n_word window_cp=$w_cp elapsed=$elapsed")
        flush(stdout)
    end
    # Event-weighted average of the two section rERPs.
    w = [p.n_word for p in pieces]
    wsum = sum(w)
    times = pieces[1].times
    sur_cp = (pieces[1].sur_cp .* w[1] .+ pieces[2].sur_cp .* w[2]) ./ wsum
    sur_fz = (pieces[1].sur_fz .* w[1] .+ pieces[2].sur_fz .* w[2]) ./ wsum
    p1_cp = (pieces[1].p1_cp .* w[1] .+ pieces[2].p1_cp .* w[2]) ./ wsum
    w_cp, nwin = window_mean(times, sur_cp)
    w_fz, _ = window_mean(times, sur_fz)
    w_p1, _ = window_mean(times, p1_cp)
    NPZ.npzwrite(
        joinpath(outdir, "rerp.npz"),
        Dict(
            "times" => collect(times),
            "sur_cp" => sur_cp,
            "sur_fz" => sur_fz,
            "sur_p1_cp" => p1_cp,
        ),
    )
    meta["combined"] = Dict(
        "window_sur_cp" => w_cp,
        "window_sur_fz" => w_fz,
        "window_sur_p1_cp" => w_p1,
        "n_window_samples" => nwin,
        "weight" => "n_word_events",
        "channels" => ["CP_mean_Pz_Cz_Oz", "Fz"],
    )
    open(done, "w") do io
        JSON3.write(io, meta)
    end
    println("$subject done window_cp=$w_cp window_p1=$w_p1")
end

main()
