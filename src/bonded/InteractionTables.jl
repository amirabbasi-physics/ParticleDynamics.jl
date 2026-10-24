# ------------------------------------------------------------------
# Unique bonded interaction tables and particle incidence
# ------------------------------------------------------------------

"""
    InteractionTable{K}

Unique K-body interactions on the device: atom columns `atoms[a][t]`,
parameter ids `ptype[t]`, and a CSR particle incidence (`offsets`, `entries`)
whose entries `e = (t-1)*K + a` address the interaction-local force slots.
Atom order is preserved exactly as given (it defines angle/torsion signs).
"""
struct InteractionTable{K}
    atoms::NTuple{K,CuVector{Int32}}
    ptype::CuVector{Int32}
    offsets::CuVector{Int32}
    entries::CuVector{Int32}
    n::Int
end

Base.length(t::InteractionTable) = t.n

"""
    build_interaction_table(N, tuples, ptype) -> InteractionTable{K}

Validate and upload `tuples::Vector{NTuple{K,<:Integer}}` (1-based) with their
1-based parameter ids. Rejects out-of-range or repeated atoms within a tuple
and index overflow.
"""
function build_interaction_table(N::Integer, tuples::AbstractVector{<:NTuple{K,Integer}},
                                 ptype::AbstractVector{<:Integer}) where {K}
    n = length(tuples)
    length(ptype) == n || throw(ArgumentError("one parameter id per interaction is required"))
    n * K < typemax(Int32) || throw(ArgumentError("too many $(K)-body interactions for Int32 indexing"))
    N < typemax(Int32) || throw(ArgumentError("particle count exceeds Int32 indexing"))
    cols = ntuple(_ -> Vector{Int32}(undef, n), K)
    deg = zeros(Int32, N)
    for (t, tup) in enumerate(tuples)
        for a in 1:K
            i = Int(tup[a])
            1 <= i <= N || throw(ArgumentError("interaction $(t): atom index $(i) out of range 1:$(N)"))
            for b in 1:(a - 1)
                Int(tup[b]) == i && throw(ArgumentError("interaction $(t): atom $(i) repeated within $(tup)"))
            end
            cols[a][t] = Int32(i)
            deg[i] += 1
        end
        ptype[t] >= 1 || throw(ArgumentError("interaction $(t): parameter id must be positive"))
    end
    offsets = zeros(Int32, N + 1)
    for i in 1:N
        offsets[i + 1] = offsets[i] + deg[i]
    end
    fillpos = copy(offsets)
    entries = Vector{Int32}(undef, n * K)
    for t in 1:n, a in 1:K
        i = Int(cols[a][t])
        fillpos[i] += 1
        entries[fillpos[i]] = Int32((t - 1) * K + a)
    end
    return InteractionTable{K}(map(CuArray, cols), CuArray(Int32.(ptype)), CuArray(offsets), CuArray(entries), n)
end
