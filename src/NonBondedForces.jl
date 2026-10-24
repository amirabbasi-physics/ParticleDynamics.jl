module NonBondedForces
# Pair math and the kernels the device engine reuses: periodic differences, the
# Lennard-Jones pair term, half-virial rows, the serial single-lane LJ ELL kernels
# and the bonded-pair cache used by collision counting. Every pair force runs in the
# device engine (`ParticleDynamics.DeviceEngine`); the staged per-potential force
# kernels and their dispatch (`NonBondedInteractions`) were removed with the staged
# engine.

using CUDA
using ..Definitions
using ..Definitions: periodic_delta
using ..NeighborLists
using ..NeighborLists: _ell_index
using ..BondedForces   # for BondList in exclusions

include("nonbonded/PairMath.jl")
include("nonbonded/BondExclusionsCUDA.jl")
include("nonbonded/LJELLEngineKernels.jl")

end # module
