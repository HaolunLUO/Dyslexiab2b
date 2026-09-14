# =====================================================================
# Deterministic balanced blocked partitions for joint B2B v2
# =====================================================================

using Random

function word_block_id(word_idx::Integer, n_words::Integer, n_blocks::Integer)
    n_blocks = min(n_blocks, n_words)
    edges = round.(Int, range(0, n_words; length=n_blocks + 1))
    for b in 1:n_blocks
        if word_idx > edges[b] && word_idx <= edges[b + 1]
            return b
        end
    end
    return n_blocks
end

function in_guard_zone(word_idx::Integer, n_words::Integer, n_blocks::Integer, guard::Integer)
    guard <= 0 && return false
    edges = round.(Int, range(0, n_words; length=n_blocks + 1))
    for e in edges[2:end-1]
        if abs(word_idx - e) <= guard || abs(word_idx - (e + 1)) <= guard
            return true
        end
    end
    return false
end

function _combinations(n::Integer, k::Integer)
    out = Vector{Int}[]
    function rec(start, chosen)
        if length(chosen) == k
            push!(out, copy(chosen))
            return
        end
        for i in start:n
            push!(chosen, i)
            rec(i + 1, chosen)
            pop!(chosen)
        end
    end
    rec(1, Int[])
    return out
end

function partition_key(part::Vector{Vector{Int}}, n_blocks::Integer=8)
    a = [sort(unique(Int.(s))) for s in part]
    b = [sort(setdiff(collect(1:n_blocks), s)) for s in a]
    return a <= b ? a : b
end

function complement_partition(part::Vector{Vector{Int}}, n_blocks::Integer=8)
    return [sort(setdiff(collect(1:n_blocks), s)) for s in part]
end

"""
Build one section's 20 half-assignments with each block appearing exactly 10 times.
Uses deterministic backtracking over shuffled candidates.
"""
function _balanced_section_halves(n_partitions::Integer=20, n_blocks::Integer=8;
                                  seed::Integer=42, lo::Integer=8, hi::Integer=12)
    k = n_blocks ÷ 2
    # All orientations (not modulo complement) so we can balance freely
    opts = _combinations(n_blocks, k)
    rng = Random.MersenneTwister(seed)
    shuffle!(rng, opts)

    selected = Vector{Int}[]
    counts = zeros(Int, n_blocks)

    function feasible(comb, remaining_slots)
        for b in comb
            c = counts[b] + 1
            c > hi && return false
            # Even if we put this block in every remaining partition, must reach lo
            # After adding: max possible = c + remaining_slots
            # min possible for others handled globally at end
        end
        # Check all blocks can still reach lo
        for b in 1:n_blocks
            will = counts[b] + (b in comb ? 1 : 0)
            max_reach = will + remaining_slots
            max_reach < lo && return false
            will > hi && return false
        end
        return true
    end

    function bt()
        length(selected) == n_partitions && return true
        rem = n_partitions - length(selected) - 1
        for comb in opts
            feasible(comb, rem) || continue
            # Reject exact duplicate of already selected (same half1 set)
            any(comb == s for s in selected) && continue
            # Also reject if complement already selected as half1 (same partition mod complement for single section)
            comp = sort(setdiff(collect(1:n_blocks), comb))
            any(comp == s for s in selected) && continue
            push!(selected, comb)
            for b in comb
                counts[b] += 1
            end
            if bt()
                return true
            end
            for b in comb
                counts[b] -= 1
            end
            pop!(selected)
        end
        return false
    end

    bt() || error("Failed to find balanced section halves")
    return selected, counts
end

"""
Generate `n_partitions` deterministic balanced blocked partitions.

Each half receives 4 of 8 blocks from each section. Assignments are unique
modulo complements. Every block appears in half1 between 8 and 12 times.
"""
function generate_balanced_partitions(n_partitions::Integer=20;
                                      n_blocks::Integer=8,
                                      n_sections::Integer=2,
                                      seed::Integer=42)
    n_sections == 2 || error("Only n_sections=2 implemented")
    n_partitions == 20 && n_blocks == 8 ||
        @warn "Balance constraints tuned for n_partitions=20, n_blocks=8"

    # Independent balanced schedules per section with different seeds
    h1, c1 = _balanced_section_halves(n_partitions, n_blocks; seed=seed, lo=8, hi=12)
    h2, c2 = _balanced_section_halves(n_partitions, n_blocks; seed=seed + 1009, lo=8, hi=12)

    selected = Vector{Vector{Vector{Int}}}()
    seen = Set{Vector{Vector{Int}}}()
    for i in 1:n_partitions
        part = [copy(h1[i]), copy(h2[i])]
        key = partition_key(part, n_blocks)
        # If duplicate key, try flipping section-2 orientation only... shouldn't happen often
        if key in seen
            part = [copy(h1[i]), sort(setdiff(collect(1:n_blocks), h2[i]))]
            key = partition_key(part, n_blocks)
        end
        key in seen && error("Duplicate partition at index $i after flip")
        push!(seen, key)
        push!(selected, part)
    end

    # Recompute counts from selected orientations
    counts = [zeros(Int, n_blocks) for _ in 1:n_sections]
    for part in selected
        for sec in 1:n_sections
            for b in part[sec]
                counts[sec][b] += 1
            end
        end
    end
    for sec in 1:n_sections, b in 1:n_blocks
        c = counts[sec][b]
        (8 <= c <= 12) ||
            error("Section $sec block $b appears in half1 $c times; require 8..12  counts=$counts")
    end
    return selected, counts
end

function make_partition_halves(sec_ids::AbstractVector{Int},
                               word_idx_in_sec::AbstractVector{Int},
                               n_words_per_sec::AbstractVector{Int},
                               keep::AbstractVector{Bool},
                               half1_blocks_by_sec::Vector{Vector{Int}};
                               n_blocks::Integer=8,
                               guard::Integer=5)
    n = length(keep)
    half1 = falses(n)
    half2 = falses(n)
    for w in 1:n
        keep[w] || continue
        sid = sec_ids[w]
        wi  = word_idx_in_sec[w]
        nw  = n_words_per_sec[sid]
        in_guard_zone(wi, nw, n_blocks, guard) && continue
        b = word_block_id(wi, nw, n_blocks)
        if b in half1_blocks_by_sec[sid]
            half1[w] = true
        else
            half2[w] = true
        end
    end
    return half1, half2
end

function partition_assignment_rows(participant::AbstractString,
                                   partitions::Vector{Vector{Vector{Int}}};
                                   n_blocks::Integer=8)
    rows = NamedTuple[]
    for (pid, part) in enumerate(partitions)
        for sec in 1:length(part)
            for b in 1:n_blocks
                half = b in part[sec] ? 1 : 2
                push!(rows, (
                    participant=participant,
                    partition=pid,
                    section=sec,
                    block=b,
                    half=half,
                ))
            end
        end
    end
    return rows
end
