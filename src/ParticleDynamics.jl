"""
GPU-accelerated non-equilibrium particle simulations with Langevin/Brownian/Molecular dynamics.

`ParticleDynamics` provides a high-level workflow API centered on
[`Simulation`](@ref), [`ParticleSystem`](@ref), workflow forces, methods,
observables, writers, and staged `run!` execution on CUDA-backed state. The
older low-level API (`build_simulation`, `step!`, explicit integrator specs,
and `SimulationCore`) remains available for expert use.

# Example
The snippet below mirrors the recommended workflow style used by the modern
examples.

```julia
using ParticleDynamics

N = 256
sigma = 1.0
cfg = hex_random_2d(N, sigma, 0.25; T=Float64)

system = ParticleSystem(cfg; types=[:A], typeids=fill(Int32(1), N), masses=Dict(:A => 1.0))
all_particles = Group(:all, AllSelection())
thermo = ThermodynamicObservable(all_particles; name=:all)

sim = Simulation(
    system;
    groups=Groups(all_particles),
    integrator=Integrator(
        dt=2.0e-4,
        scheme=VelocityVerlet(),
        forces=[WCA(epsilon=10.0, sigma=sigma, pairs=:all)],
        methods=[Langevin(all_particles; gamma=50.0, kT=1.0)],
    ),
    observables=[thermo],
    writers=[TableWriter("obs.csv"; every=50, observables=[thermo => [:temperature, :potential_energy]])],
)

run!(sim, Stage(:production, steps=200; progress=false))
```

Use the README and `examples/` for richer setups such as two-temperature baths,
bonded polymers, Brownian dynamics, active noise, collision histograms, and
restart workflows.
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
include("SimulationCore.jl")
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
using .SimulationCore: IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NVESpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec
using .SimulationCore: velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nve, nosehooverchain, csvr
using .SimulationCore: AbstractExternalPotential, external_forces!, attach_external_potential!, detach_external_potential!
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
    Integrator, Method, VelocityVerlet, BAOAB, BAOA, GSM, EulerHeun, EulerMaruyama, OUSpectrum,
    ConstantVolume, Langevin, Brownian, ActiveOrnsteinUhlenbeck,
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
       IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NVESpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec,
       velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nve, nosehooverchain, csvr,
       AbstractExternalPotential, external_forces!, attach_external_potential!, detach_external_potential!,
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
       Integrator, Method, VelocityVerlet, BAOAB, BAOA, GSM, EulerHeun, EulerMaruyama, OUSpectrum,
       ConstantVolume, Langevin, Brownian, ActiveOrnsteinUhlenbeck,
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
