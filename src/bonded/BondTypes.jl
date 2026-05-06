# ------------------------------------------------------------------
# Bond topology
# ------------------------------------------------------------------

# Placeholder extension point for future angles/dihedrals and richer
# all-atom topologies. Stage 14 keeps only BondList as the concrete topology.
abstract type AbstractBondTopology end

"""
CSR-style adjacency list describing bead connectivity.
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
