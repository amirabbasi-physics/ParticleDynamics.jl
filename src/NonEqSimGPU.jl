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
include("InitGenerators.jl")
 
# Re-export commonly used APIs to simplify user code
using .Definitions: LJParams, SoftRepulsiveParams,
    HarmonicBondParams, FENEParams,
    BondPotential, HarmonicBond, FENEBond,
    StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime,
    harmonic_bond, fene_bond

using .Simulation: SimulationState, build_simulation, step!, step_graph!, zero_forces!, accumulate_energies!
using .Simulation: IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama
using .Writers: InMemoryLogger, CSVWriter, XYZWriter,
    write_xyz!, write_observables_csv!, gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!
using .BondedForces: BondList, build_bondlist
using .InitGenerators: box_from_phi_2d, hex_random_2d, hex_circle_2d, hex_circle_plus_random_2d, hex_sites_in_box_2d, hex_circle_in_box_2d

export Filters, BondedForces,
       # Definitions / parameters
       LJParams, SoftRepulsiveParams,
       HarmonicBondParams, FENEParams,
       BondPotential, HarmonicBond, FENEBond,
       harmonic_bond, fene_bond,
       StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime,
       # Simulation helpers
       SimulationState, build_simulation, step!, step_graph!, zero_forces!, accumulate_energies!,
       IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, velocityverlet, baoab, baoa, gsm, eulermaruyama,
       # Writers
       InMemoryLogger, CSVWriter, XYZWriter,
       write_xyz!, write_observables_csv!, gsd_open, gsd_close, write_gsd_frame!, read_last_gsd,
       # Bond list helper
       BondList, build_bondlist,
       # Initial configuration generators (2D)
       box_from_phi_2d, hex_random_2d, hex_circle_2d, hex_circle_plus_random_2d, hex_sites_in_box_2d, hex_circle_in_box_2d

println("##########################################################")
println("                  NonEqSimGPU Loaded                ")
println("##########################################################")

end # module NonEqSimGPU
