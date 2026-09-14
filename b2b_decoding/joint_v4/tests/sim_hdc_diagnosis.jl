# =====================================================================
# sim_hdc_diagnosis.jl
# Validate the joint B2B estimator on synthetic data with KNOWN encoding.
# Mirrors Gwilliams et al. SI §1.12 (Gaussian response simulation), but
# calls the actual production functions (estimator.jl / partitions.jl).
#
# Design:
#   - 2 sections x 600 words; 3 families x 3 standardized features.
#   - Family A (cols 1:3): autocorrelated, ENCODED on EEG channels 1:4
#     via a Gaussian bump peaking at +0.30 s relative to (offset) anchor.
#   - Family B (cols 4:6): correlated with A (r ~ 0.8) but NOT encoded.
#   - Family C (cols 7:9): independent, NOT encoded.
# Pass criteria:
#   1. S[:, 1:3] shows a positive bump peaking near 0.30 s.
#   2. S[:, 4:6] stays ~0 (B2B disentangles correlated non-encoded features).
#   3. S[:, 7:9] ~ 0.
#   4. Circular-shift null => max|S| small (~0).
# =====================================================================

using Random
using Statistics
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "partitions.jl"))
include(joinpath(@__DIR__, "..", "estimator.jl"))

function smooth_randn(rng, n; kern=9)
    x = randn(rng, n + kern)
    y = [mean(x[i:i+kern-1]) for i in 1:n]
    return y
end

function zscore_col(x)
    return (x .- mean(x)) ./ std(x)
end

function main()
    rng = Random.MersenneTwister(1234)
    n_sec = 2
    n_per = 600
    n_words = n_sec * n_per
    n_ch = 65
    sfreq = 100.0
    tmin, tmax = TMIN_S, TMAX_S          # -0.2 .. 1.0
    times = collect(range(tmin, tmax; length=round(Int, (tmax - tmin) * sfreq) + 1))
    n_times = length(times)

    # ---- Features ----
    A = hcat([zscore_col(smooth_randn(rng, n_per)) for _ in 1:3]...)   # encoded
    B = hcat([zscore_col(0.8 .* A[:, j] .+ 0.6 .* smooth_randn(rng, n_per)) for j in 1:3]...)  # correlated w/ A
    C = hcat([zscore_col(smooth_randn(rng, n_per)) for _ in 1:3]...)   # independent
    # stack sections (independent draws per section)
    A = vcat(A, hcat([zscore_col(smooth_randn(rng, n_per)) for _ in 1:3]...))
    B = vcat(B, hcat([zscore_col(0.8 .* A[n_per+1:end, j] .+ 0.6 .* smooth_randn(rng, n_per)) for j in 1:3]...))
    C = vcat(C, hcat([zscore_col(smooth_randn(rng, n_per)) for _ in 1:3]...))
    X = hcat(A, B, C)  # 1200 x 9
    X = mapslices(zscore_col, X; dims=1)

    # ---- EEG with known encoding of family A ----
    bump = exp.(-0.5 .* ((times .- 0.30) ./ 0.08) .^ 2)   # Gaussian at +0.30 s
    Gtrue = zeros(n_ch, 3)
    Gtrue[1:4, :] .= randn(rng, 4, 3) .* 1.0
    amp = 0.35
    Y = randn(rng, n_ch, n_times, n_words)               # unit noise
    for w in 1:n_words
        sig = Gtrue * A[w, :]                            # n_ch x 3 -> n_ch
        Y[:, :, w] .+= amp .* (sig * bump')              # rank-1 bump
    end

    # ---- Metadata for partitions ----
    sec_ids = vcat(fill(1, n_per), fill(2, n_per))
    word_idx_in_sec = vcat(collect(1:n_per), collect(1:n_per))
    n_words_per_sec = [n_per, n_per]
    keep = trues(n_words)

    partitions, _ = generate_balanced_partitions(N_OUTER_PARTITIONS;
        n_blocks=N_BLOCKS_PER_SECTION, seed=B2B_SEED)
    half1, half2 = make_partition_halves(sec_ids, word_idx_in_sec, n_words_per_sec,
                                         keep, partitions[1];
                                         n_blocks=N_BLOCKS_PER_SECTION, guard=GUARD_WORDS)
    println("half1=$(sum(half1)) half2=$(sum(half2))")

    fks = [3, 3, 3]
    S, fam, diag, d1, d2 = b2b_joint_partition(Y, X, half1, half2,
        sec_ids, word_idx_in_sec, n_words_per_sec, times; k=1, family_ks=fks)

    println(@sprintf("alpha=(%.2e, %.2e)  kappa=%.1f  invalid=%s",
                     diag.alpha_1, diag.alpha_2, diag.max_kappa, diag.invalid))

    # ---- Evaluate ----
    function colstats(cols, label)
        for c in cols
            pk = argmax(S[:, c])
            @printf("  col %d (%s): peak=%+.4f @ %.2fs  mean_pre0=%+.4f  mean|.|=%.4f\n",
                    c, label, S[pk, c], times[pk],
                    mean(S[times .< 0, c]), mean(abs, S[:, c]))
        end
    end
    println("\nEncoded family A (expect bump @0.30s):");   colstats(1:3, "A")
    println("Correlated family B (expect ~0):");          colstats(4:6, "B")
    println("Independent family C (expect ~0):");         colstats(7:9, "C")

    famA = [sum(S[t, 1:3]) / 3 for t in 1:n_times]
    famB = [sum(S[t, 4:6]) / 3 for t in 1:n_times]
    famC = [sum(S[t, 7:9]) / 3 for t in 1:n_times]
    pkA = argmax(famA)
    @printf("\nFamily means: A peak=%+.4f @ %.2fs ; B max|.|=%.4f ; C max|.|=%.4f\n",
            famA[pkA], times[pkA], maximum(abs, famB), maximum(abs, famC))

    # ---- Circular-shift null ----
    rng_null = Random.MersenneTwister(99)
    Xnull = similar(X)
    for sid in 1:n_sec
        rows = findall(==(sid), sec_ids)
        s = 100 + rand(rng_null, 1:200)
        Xnull[rows, :] = X[rows[circshift(collect(1:length(rows)), s)], :]
    end
    Sn, _, dn, _, _ = b2b_joint_partition(Y, Xnull, half1, half2,
        sec_ids, word_idx_in_sec, n_words_per_sec, times; k=1, family_ks=fks)
    @printf("Null: max|S|=%.4f (encoded A region max|S|=%.4f)  invalid=%s\n",
            maximum(abs, Sn), maximum(abs, Sn[:, 1:3]), dn.invalid)

    # ---- Verdict ----
    ok1 = times[pkA] > 0.15 && times[pkA] < 0.45 && famA[pkA] > 0.05
    ok2 = maximum(abs, famB) < 0.35 * famA[pkA]
    ok3 = maximum(abs, famC) < 0.35 * famA[pkA]
    ok4 = maximum(abs, Sn) < 0.35 * famA[pkA]
    println("\nPASS recovery=$(ok1)  disentangle_B=$(ok2)  null_C=$(ok3)  shift_null=$(ok4)")
    println(ok1 && ok2 && ok3 && ok4 ? "ALL PASS" : "SOME CHECKS FAILED")
end

main()
