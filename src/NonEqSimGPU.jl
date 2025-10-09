module NonEqSimGPU

using CUDA
using StaticArrays
using Printf
using DelimitedFiles

include("Definitions.jl")
include("Initialize.jl")
include("NeighborLists.jl")
include("BondedForces.jl")
include("NonBondedForces.jl")
include("LangevinIntegrators.jl")
include("BrownianIntegrators.jl")
include("Simulation.jl")
include("Filters.jl")
include("Writers.jl")
 
# Re-export commonly used APIs to simplify user code
using .Definitions: LJParams, SoftRepulsiveParams,
    HarmonicBondParams, FENEParams,
    BondPotential, HarmonicBond, FENEBond,
    StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime,
    harmonic_bond, fene_bond

using .Simulation: SimulationState, build_simulation, step!, step_graph!, zero_forces!
using .Simulation: IntegratorSpec, VVSpec, BAOABSpec, BrownianSpec, vv, baoab, brownian
using .Writers: InMemoryLogger, CSVWriter, XYZWriter,
    write_xyz!, write_observables_csv!, gsd_open, gsd_close, write_gsd_frame!, read_last_gsd
using .BondedForces: BondList, build_bondlist

export Filters, BondedForces,
       # Definitions / parameters
       LJParams, SoftRepulsiveParams,
       HarmonicBondParams, FENEParams,
       BondPotential, HarmonicBond, FENEBond,
       harmonic_bond, fene_bond,
       StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime,
       # Simulation helpers
       SimulationState, build_simulation, step!, step_graph!, zero_forces!,
       IntegratorSpec, VVSpec, BAOABSpec, BrownianSpec, vv, baoab, brownian,
       # Writers
       InMemoryLogger, CSVWriter, XYZWriter,
       write_xyz!, write_observables_csv!, gsd_open, gsd_close, write_gsd_frame!, read_last_gsd,
       # Bond list helper
       BondList, build_bondlist

println("##########################################################")
println("                  NonEqSimGPU Loaded                ")
println("##########################################################")

end # module NonEqSimGPU
