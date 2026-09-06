# Mixed coefficients share arithmetic across ELL and all-pairs traversal.
# Nothing selects all pairs; tuples carry only the ELL arrays actually read.
_mixed_neighbors(nbh::NeighborLists.AbstractNeighborMatrix) = (nbh.neighbors_flat, nbh.counts)
_mixed_neighbors(::NeighborLists.AllPairsNeighborMatrix) = nothing
_mixed_bonds(::Nothing) = nothing
_mixed_bonds(bonds::BondedForces.BondList) = (bonds.index, bonds.flat, bonds.counts)
@inline _mixed_count(::Nothing, i, N) = N
@inline _mixed_count(rows::Tuple, i, N) = Int(rows[2][i])
@inline _mixed_neighbor(::Nothing, i, t, N) = Int32(t + 1)
@inline _mixed_neighbor(rows::Tuple, i, t, N) = rows[1][_ell_index(i, t, N)]
@inline _mixed_excluded(::Nothing, i, j) = false
@inline _mixed_excluded(bonds::Tuple, i, j) = _is_bonded(Int32(i), j, bonds...)
