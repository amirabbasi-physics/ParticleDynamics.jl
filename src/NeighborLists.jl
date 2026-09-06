"""
Neighbor list builders and query utilities used by the force kernels.

`NeighborLists` implements three strategies:

- [`NeighborMatrix`](@ref) — dense cell lists with uniform cutoffs (default in
  `build_simulation`, parameter choices mirror `examples/2D_example.jl`).
- [`StencilNeighborMatrix`](@ref) — particle- or type-dependent cutoffs as used
  in `examples/3D_stencil_two_sizes*.jl`.
- [`AllPairsNeighborMatrix`](@ref) — sentinel representing O(N²) evaluation
  (`examples/2D_allpairs_quicktest.jl`).

Dense and stencil lists use slot-major ELL storage: slot `t` (zero based) of
particle `i` (one based) is at `t*N + i` in `neighbors_flat`. Bond adjacency
is a separate CSR structure. The all-pairs sentinel stores no neighbor rows.
"""
module NeighborLists

using CUDA

export AbstractNeighborMatrix,
       NeighborMatrix, StencilNeighborMatrix,
       build_neighbors_dense!, build_neighbors_stencil!,
       build_neighbors_stencil_by_types!,
       update_neighbors_inplace!, update_needed!,
       build_neighbors_allpairs!, AllPairsNeighborMatrix,
       NeighborCapacityError

include("neighbors/NeighborTypes.jl")
include("neighbors/AllPairsNeighbors.jl")
include("neighbors/DenseNeighborCUDA.jl")
include("neighbors/StencilNeighborCUDA.jl")
include("neighbors/NeighborAllocation.jl")
include("neighbors/NeighborUpdateCUDA.jl")

end # module NeighborLists
