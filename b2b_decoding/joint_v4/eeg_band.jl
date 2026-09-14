# =====================================================================
# Phase 3b/3c: band-limit continuous EEG and optional temporal subsets.
# Filter is applied to the continuous section (before epoching).
# =====================================================================

using DSP

const EEG_BAND_HZ = Dict(
    "delta" => (0.5, 4.0),
    "theta" => (4.0, 8.0),
)
const LEFT_TEMPORAL = ["T7", "TP7", "P7", "C5"]
const RIGHT_TEMPORAL = ["T8", "TP8", "P8", "C6"]

function channel_names_from_meta(meta)::Vector{String}
    raw = nothing
    for k in ("channel_names", :channel_names)
        if haskey(meta, k)
            raw = meta[k]
            break
        end
    end
    raw === nothing && error("metadata.json missing channel_names")
    return String[string(x) for x in raw]
end

"""Zero-phase Butterworth bandpass, per channel. No-op for band=broad."""
function bandpass_continuous(eeg::AbstractMatrix, sfreq::Real; band::AbstractString=B2B_EEG_BAND)
    band == "broad" && return Matrix{Float64}(eeg)
    haskey(EEG_BAND_HZ, band) || error("Unknown EEG band: $band")
    lo, hi = EEG_BAND_HZ[band]
    hi < sfreq / 2 || error("band high $hi exceeds Nyquist $(sfreq/2)")
    n_ch, n_t = size(eeg)
    n_t < 16 && error("continuous EEG too short to filter: $n_t")
    # DSP 0.8: Bandpass is Hz; pass fs to digitalfilter (not Bandpass).
    filt = digitalfilter(Bandpass(lo, hi), Butterworth(4); fs=Float64(sfreq))
    out = Matrix{Float64}(undef, n_ch, n_t)
    for ch in 1:n_ch
        out[ch, :] = filtfilt(filt, Float64.(eeg[ch, :]))
    end
    return out
end

function select_chan_set(eeg::AbstractMatrix, meta; chan_set::AbstractString=B2B_CHAN_SET)
    if chan_set == "all"
        names = try
            channel_names_from_meta(meta)
        catch
            String["ch$(i)" for i in 1:size(eeg, 1)]
        end
        return Matrix{Float64}(eeg), names
    end
    names = channel_names_from_meta(meta)
    want = chan_set == "left_temporal" ? LEFT_TEMPORAL :
           chan_set == "right_temporal" ? RIGHT_TEMPORAL :
           error("Unknown chan_set: $chan_set")
    idxs = Int[]
    for w in want
        i = findfirst(==(w), names)
        i === nothing && error("Channel $w not in montage: $(names)")
        push!(idxs, i)
    end
    return Matrix{Float64}(eeg[idxs, :]), String[names[i] for i in idxs]
end

"""Filter then subset. Returns (eeg, channel_names_used)."""
function preprocess_continuous_eeg(eeg::AbstractMatrix, sfreq::Real, meta)
    eeg_f = bandpass_continuous(eeg, sfreq; band=B2B_EEG_BAND)
    return select_chan_set(eeg_f, meta; chan_set=B2B_CHAN_SET)
end
