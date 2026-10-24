"""
Cluster-pair GPU nonbonded engine.

`ClusterPairs` evaluates the short-range nonbonded forces (Lennard-Jones and
the real-space part of a Coulomb split) in one fused kernel over a
cluster-pair list:

* particles are grouped into spatially compact clusters of 8 and
  super-clusters of 64;
* the list stores, per super-cluster, the j-clusters within the list radius
  with one 8-bit interaction mask per j-cluster half (a warp), so each
  interacting pair appears once (Newton's third law);
* a thread block of 64 threads evaluates an 8×8 tile per i-cluster and
  j-cluster, keeps the i-atoms in shared memory, and reduces j-forces with
  warp shuffles;
* excluded and scaled pairs are bit masks in the tile, so the kernel computes
  their correction directly instead of in a separate pass;
* the Coulomb term is `C qᵢqⱼ (1/r + P(r²))` with a polynomial `P` fitted to
  the split (Ewald, prolate) or exact (reaction field), so no `erfc`/`exp`
  is evaluated per pair.

The pair physics (`ClusterPairs.pair_force`) is one device function shared
by every kernel variant.
"""
module ClusterPairs

using CUDA
using ..Definitions
using ..Definitions: periodic_delta, force_coordinate
using ..Electrostatics
const ES = Electrostatics

export ClusterPairEngine, CoulombPolynomial, coulomb_polynomial, coulomb_reaction_field

include("clusterpairs/PairFunctor.jl")
include("clusterpairs/List.jl")
include("clusterpairs/Kernels.jl")
include("clusterpairs/Engine.jl")

end # module ClusterPairs
