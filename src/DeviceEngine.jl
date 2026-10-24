"""
    DeviceEngine

Device-resident, graph-replayed MD engine that replaced the staged step engine.
It holds the
GPU neighbor control, spatial sorting, persistent global observables,
cooperative traversal and row construction, displacement-only maintenance and
control-kernel fusion.
"""
module DeviceEngine
using CUDA
import ..NeighborLists
import ..NonBondedForces
import ..SimulationCore
import ..BrownianIntegrators
import ..Collisions
import ..BondedForces
import ..Definitions
import ..ClusterPairs
import ..RigidWaterConstraints
import ..Electrostatics
const NL = NeighborLists
const NB = NonBondedForces
const SC = SimulationCore

include("engine/Ordering.jl")
include("engine/Observables.jl")
include("engine/Traversal.jl")
include("engine/PairModels.jl")
include("engine/PairTraversal.jl")
include("engine/Builder.jl")
include("engine/Policy.jl")
include("engine/Fusion.jl")
include("engine/Philox.jl")
include("engine/Features.jl")
include("engine/Unwrap.jl")
include("engine/Masses.jl")
include("engine/Bonded.jl")
include("engine/Freeze.jl")
include("engine/OUNoise.jl")
include("engine/CollisionCounting.jl")
include("engine/ExternalPotential.jl")
include("engine/RigidWater.jl")
include("engine/Langevin.jl")
include("engine/CSVR.jl")
include("engine/NHC.jl")
include("engine/Brownian.jl")
# Registered engine features (Features.jl). Adding a feature: its file + one entry.
# Order matters where hooks of several features run at the same point: after-force
# launches follow the staged force order (bonds, typed bonded terms, freeze spring).
const ENGINE_FEATURES=(UnwrapFeature,MassFeature,BondExclusionFeature,BondFeature,BondedTermsFeature,
    FreezeFeature,OUFeature,CollisionFeature,ExternalPotentialFeature,RigidWaterFeature)
include("engine/WaterFastPath.jl")
include("engine/ClusterBackend.jl")
include("engine/Run.jl")
include("engine/DeviceBarostat.jl")
include("engine/StateEngine.jl")
end
