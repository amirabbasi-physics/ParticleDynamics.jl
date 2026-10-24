# ------------------------------------------------------------------
# Bond topology
# ------------------------------------------------------------------

# Common interface for bond topology storage.
abstract type AbstractBondTopology end

# Sorted states: topology tables keep build-time particle ids. Kernels read the
# row of storage slot `i` through `tag` (slot → id) and partner ids through `slot_of`
# (id → slot); `nothing` means unsorted storage (slot = id).
@inline _brow(i, ::Nothing) = i
@inline _brow(i, tag) = @inbounds tag[i]
@inline _bslot(j, ::Nothing) = j
@inline _bslot(j, slot_of) = @inbounds slot_of[j]

"""
    BondList

GPU adjacency list describing bead connectivity in build-time particle-id order.
Construct it with [`build_bondlist`](@ref). Each bond is stored at both endpoints;
`index` gives each particle's zero-based starting offset in `flat`, and `counts` its degree.
"""
struct BondList <: AbstractBondTopology
    index::CuArray{Int32,1}
    flat::CuArray{Int32,1}
    counts::CuArray{Int32,1}
end

"""
    build_bondlist(N, bonds) -> BondList

Construct a GPU-ready bond list from a collection of `(i, j)` tuples (1-based).
`examples/2D_polymer_bonded.jl` builds its chains via:

```julia
chain = collect(zip(1:(n-1), 2:n))
bond_list = build_bondlist(n, chain)
```
"""
function build_bondlist(N::Integer, bonds)
    N = Int(N)
    deg = zeros(Int32, N)
    for (i, j) in bonds
        @assert 1 <= i <= N && 1 <= j <= N "bond index out of range"
        deg[Int(i)] += 1
        deg[Int(j)] += 1
    end
    index = similar(deg)
    offs = Int32(0)
    for i in 1:N
        index[i] = offs
        offs += deg[i]
    end
    total = Int(offs)
    flat = Vector{Int32}(undef, total)
    counts = zeros(Int32, N)
    for (i, j) in bonds
        ii = Int(i)
        jj = Int(j)
        bi = index[ii] + counts[ii]
        flat[Int(bi) + 1] = Int32(jj)
        counts[ii] += 1
        bj = index[jj] + counts[jj]
        flat[Int(bj) + 1] = Int32(ii)
        counts[jj] += 1
    end
    return BondList(CUDA.CuArray(index), CUDA.CuArray(flat), CUDA.CuArray(counts))
end
