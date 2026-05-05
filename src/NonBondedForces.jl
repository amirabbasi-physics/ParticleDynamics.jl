module NonBondedForces

using CUDA
using ..Definitions
using ..NeighborLists  # so we can dispatch on NeighborLists.NeighborMatrix
using ..BondedForces   # for BondList in exclusions

export lj_forces_soa!, lj_forces_soa_noE!,
       wca_forces_soa!, wca_forces_soa_noE!,
       harmonic_rep_forces_soa!, harmonic_rep_forces_soa_noE!

include("nonbonded/PairMath.jl")
include("nonbonded/BondExclusionsCUDA.jl")
include("nonbonded/LJAllPairsCUDA.jl")
include("nonbonded/WCAAllPairsCUDA.jl")
include("nonbonded/SoftRepulsiveAllPairsCUDA.jl")
include("nonbonded/LJCSRCUDA.jl")
include("nonbonded/LJMixedCUDA.jl")
include("nonbonded/LJPairMatrixCUDA.jl")
include("nonbonded/WCACSRCUDA.jl")
include("nonbonded/WCAMixedCUDA.jl")
include("nonbonded/WCAPairMatrixCUDA.jl")
include("nonbonded/SoftRepulsiveCSRCUDA.jl")

end # module
