"""
GPU-accelerated non-equilibrium particle simulations with Langevin/Brownian/Molecular dynamics.

`ParticleDynamics` orchestrates SoA GPU buffers, neighbor lists, nonbonded and
bonded force kernels, integrators, collision counters, and writers so that
research scripts can copy validated parameter sets from `examples/` and run
production simulations without touching CUDA code. The top-level module keeps a
curated public API (`SimulationState`, `build_simulation`, integrator builders,
filters, writers, and setup helpers), while lower-level execution helpers stay
under qualified submodules such as `ParticleDynamics.SimulationCore`.

# Example
The snippet below mirrors `examples/2D_allpairs_quicktest.jl`, which checks the
all-pairs WCA path with `N = 256` particles and a WCA cutoff of `2^(1/6)σ`.

```julia
using ParticleDynamics

N = 256
box = (80.0f0, 80.0f0)
dt = 2.0f-4
rcut = Float32(2^(1/6))

st = build_simulation(N=N, box=box, dt=dt,
                      cutoff=rcut, skin=0.4f0, cap=Int32(64),
                      neigh_interval=10, use_neighborlist=false,
                      epsilon=10f0, sigma=1f0,
                      gamma=50f0, temperature=1f0,
                      nonbonded=:wca, precision=:f32)

vv = velocityverlet(st; gamma=50f0, temperature=1f0, dt=dt)
for _ in 1:50
    step!(st, vv, dt; compute_energy=true)
end
```

See the README and the scripts under `examples/` for richer setups (two-temperature
filters, bonded polymers, Brownian dynamics, collision histograms, etc.).
"""
module ParticleDynamics

using CUDA

include("Definitions.jl")
include("Initialize.jl")
include("Backends.jl")
include("NeighborLists.jl")
include("BondedForces.jl")
include("NonBondedForces.jl")
include("NonBondedInteractions.jl")
include("LangevinIntegrators.jl")
include("BrownianIntegrators.jl")
include("IntegratorInterfaces.jl")
include("Collisions.jl")
include("ParticleGroups.jl")
include("Thermostats.jl")
include("Simulation.jl")
include("Filters.jl")
include("Writers.jl")
include("InitGenerators.jl")
include("Workflow.jl")
 
# Re-export commonly used APIs to simplify user code
using .Definitions: LJParams, SoftRepulsiveParams,
    HarmonicBondParams, FENEParams,
    BondPotential, HarmonicBond, FENEBond,
    StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime,
    harmonic_bond, fene_bond
using .IntegratorInterfaces: AbstractIntegratorSpec

using .ParticleGroups: ParticleSelection, ParticleGroup, All, TypeIDs, Indices,
    materialize, apply_scalar!, apply_values!, gather, sum_values
using .Thermostats: AbstractThermostat, ThermostatState,
    NoseHooverChainThermostat, CSVRThermostat,
    n_baths, target_temperature, response_time,
    set_target_temperature!, set_response_time!, cumulative_energy_exchange

using .SimulationCore: SimulationState, build_simulation, step!, step_graph!, sync_unwrapped!, accumulate_virial!, virial_components, virial_tensor
using .SimulationCore: collect_step_observables, reset_bath_exchange_accumulators!
using .SimulationCore: IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec
using .SimulationCore: velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nosehooverchain, csvr
using .Writers: InMemoryLogger, CSVWriter, XYZWriter,
    write_xyz!, write_observables_csv!, gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!
using .BondedForces: BondList, build_bondlist
using .InitGenerators: box_from_phi_2d, box_from_phi_3d,
    hex_random_2d, hex_circle_2d, hex_circle_plus_random_2d, hex_sites_in_box_2d, hex_circle_in_box_2d,
    hex_slab_coexistence_2d, fcc_sites_in_box_3d, fcc_random_3d, fcc_slab_coexistence_3d
using .Collisions: enable_collision_counting!, disable_collision_counting!,
    collisions_reset_counts!, collisions_read_counts!, set_collision_pair_cutoffs!
using .Workflow: Simulation, ParticleSystem, Particles, Topology, PeriodicBox,
    Group, Groups, AllSelection, TypeSelection, IndexSelection,
    Force, ForceField, PairTable, CellList, LennardJones, WCA, SoftRepulsive, HarmonicBondForce, FENEBondForce,
    Integrator, Method, ConstantVolume, Langevin, Brownian, ActiveOrnsteinUhlenbeck,
    Thermostat, CSVR, NoseHooverChain,
    Observable, ThermodynamicObservable, BathExchangeObservable, VirialObservable, CollisionObservable, MSDObservable, VACFObservable,
    Writer, TableWriter, GSDWriter,
    Every, AtSteps, Between,
    Stage, prepare!, run!, reset_step!, reset_observables!, state

export Filters, BondedForces, ParticleGroups, Thermostats, SimulationCore,
       # Definitions / parameters
       LJParams, SoftRepulsiveParams,
       HarmonicBondParams, FENEParams,
       BondPotential, HarmonicBond, FENEBond,
       harmonic_bond, fene_bond,
       StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime,
       # Particle groups and selection
       ParticleSelection, ParticleGroup, All, TypeIDs, Indices,
       materialize, apply_scalar!, apply_values!, gather, sum_values,
       # Thermostats
       AbstractThermostat, ThermostatState,
       NoseHooverChainThermostat, CSVRThermostat,
       n_baths, target_temperature, response_time,
       set_target_temperature!, set_response_time!, cumulative_energy_exchange,
       # Simulation helpers
       SimulationState, build_simulation, step!, step_graph!, sync_unwrapped!, accumulate_virial!, virial_components, virial_tensor,
       collect_step_observables, reset_bath_exchange_accumulators!,
       AbstractIntegratorSpec,
       IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec,
       velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nosehooverchain, csvr,
       # Writers
       InMemoryLogger, CSVWriter, XYZWriter,
       write_xyz!, write_observables_csv!, gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!,
       # Bond list helper
       BondList, build_bondlist,
       # Initial configuration generators (2D)
       box_from_phi_2d, box_from_phi_3d,
       hex_random_2d, hex_circle_2d, hex_circle_plus_random_2d, hex_sites_in_box_2d, hex_circle_in_box_2d,
       hex_slab_coexistence_2d, fcc_sites_in_box_3d, fcc_random_3d, fcc_slab_coexistence_3d,
       # Collisions API
       enable_collision_counting!, disable_collision_counting!,
       collisions_reset_counts!, collisions_read_counts!, set_collision_pair_cutoffs!,
       # High-level workflow facade
       Simulation, ParticleSystem, Particles, Topology, PeriodicBox,
       Group, Groups, AllSelection, TypeSelection, IndexSelection,
       Force, ForceField, PairTable, CellList, LennardJones, WCA, SoftRepulsive, HarmonicBondForce, FENEBondForce,
       Integrator, Method, ConstantVolume, Langevin, Brownian, ActiveOrnsteinUhlenbeck,
       Thermostat, CSVR, NoseHooverChain,
       Observable, ThermodynamicObservable, BathExchangeObservable, VirialObservable, CollisionObservable, MSDObservable, VACFObservable,
       Writer, TableWriter, GSDWriter,
       Every, AtSteps, Between,
       Stage,
       prepare!, run!, reset_step!, reset_observables!, state

if get(ENV, "PARTICLEDYNAMICS_VERBOSE_LOAD", "0") == "1"
    println("##########################################################")
    println("                  ParticleDynamics Loaded                ")
    println("##########################################################")
end

function __init__()
    Backends.initialize_backend_runtime!()
end

end # module ParticleDynamics
