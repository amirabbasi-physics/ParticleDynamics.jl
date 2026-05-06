"""
Neighbor list builders and query utilities used by the force kernels.

`NeighborLists` implements three strategies:

- [`NeighborMatrix`](@ref) — dense cell lists with uniform cutoffs (default in
  `build_simulation`, parameter choices mirror `examples/2D_example.jl`).
- [`StencilNeighborMatrix`](@ref) — particle- or type-dependent cutoffs as used
  in `examples/3D_stencil_two_sizes*.jl`.
- [`AllPairsNeighborMatrix`](@ref) — sentinel representing O(N²) evaluation
  (`examples/2D_allpairs_quicktest.jl`).

All stores use a CSR-style `(neighbors_index, neighbors_flat, counts)` layout
so that kernels can iterate neighbors without branches.
"""
module NeighborLists

using CUDA

export AbstractNeighborMatrix,
       NeighborMatrix, StencilNeighborMatrix,
       build_neighbors_dense!, build_neighbors_stencil!,
       build_neighbors_stencil_by_types!,
       update_neighbors_inplace!, update_needed!,
       build_neighbors_allpairs!, AllPairsNeighborMatrix

include("neighbors/NeighborTypes.jl")
include("neighbors/AllPairsNeighbors.jl")
include("neighbors/DenseNeighborCUDA.jl")
include("neighbors/StencilNeighborCUDA.jl")
include("neighbors/NeighborAllocation.jl")
include("neighbors/NeighborUpdateCUDA.jl")

end # module NeighborLists
