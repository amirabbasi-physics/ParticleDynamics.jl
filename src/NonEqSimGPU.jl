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
include("Collisions.jl")
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
using .Collisions: enable_collision_counting!, disable_collision_counting!,
    collisions_reset_counts!, collisions_read_counts!, set_collision_pair_cutoffs!

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
       box_from_phi_2d, hex_random_2d, hex_circle_2d, hex_circle_plus_random_2d, hex_sites_in_box_2d, hex_circle_in_box_2d,
       # Collisions API
       enable_collision_counting!, disable_collision_counting!,
       collisions_reset_counts!, collisions_read_counts!, set_collision_pair_cutoffs!

println("##########################################################")
println("                  NonEqSimGPU Loaded                ")
println("##########################################################")

function __init__()
    _maybe_set_cuda_compat!()
end

# Allow legacy GPUs (e.g. Pascal cc 6.x) to run by pinning to a CUDA runtime
# below the 13.x cutoff on drivers that default to newer runtimes.
function _maybe_set_cuda_compat!()
    mode = get(ENV, "NEQSIMGPU_CUDA_COMPAT", "auto")
    mode == "off" && return

    target = try
        VersionNumber(get(ENV, "NEQSIMGPU_CUDA_LEGACY_VERSION", "12.4.1"))
    catch
        v"12.4.1"
    end

    force = mode == "force"

    try
        dev = CUDA.device()
        cap = CUDA.capability(dev)
        runtime = CUDA.runtime_version()
        needs_downgrade = cap < v"7.5" && runtime >= v"13.0.0"
        if (needs_downgrade || force) && runtime > target
            if !isdefined(CUDA, :set_runtime_version!)
                @warn "CUDA.set_runtime_version! not available; cannot adjust runtime automatically" device=CUDA.name(dev) capability=cap runtime=runtime
                return
            end
            @warn "Legacy GPU detected; attempting to pin CUDA runtime" device=CUDA.name(dev) capability=cap runtime=runtime target mode
            try
                CUDA.set_runtime_version!(target)
                new_runtime = CUDA.runtime_version()
                if new_runtime >= v"13.0.0"
                    @warn "CUDA runtime pin did not take effect; you may still see capability errors (driver may not ship compat libs)" device=CUDA.name(dev) capability=cap runtime=new_runtime target
                else
                    @info "CUDA runtime pinned for legacy GPU" device=CUDA.name(dev) capability=cap runtime=new_runtime target
                end
            catch err
                @warn "Failed to pin CUDA runtime for legacy GPU; driver may be too new or missing compatibility libraries" device=CUDA.name(dev) capability=cap runtime=runtime target error=err
            end
        end
    catch err
        @warn "Legacy CUDA compatibility probe failed; leaving defaults" error=err
    end
end

end # module NonEqSimGPU
