module Simulation

using CUDA
using ..Definitions
using ..NeighborLists
using ..NonBondedForces
using ..BondedForces
using ..LangevinIntegrators
using ..BrownianIntegrators
using ..Collisions
 

const NL_CHECK_STRIDE = 20  # only check NL rebuild every N steps to cut overhead
# Weeks-Chandler-Andersen cutoff factor: r_c = 2^(1/6) * σ ≈ 1.12246 σ
const WCA_RC_FACTOR = 1.122462048309373

# Nonbonded kind tags (host-side routing only)
const NB_KIND_LJ      = UInt8(1)
const NB_KIND_WCA     = UInt8(2)
const NB_KIND_SOFTREP = UInt8(3)

# Freeze modes
const FREEZE_NONE   = UInt8(0)
const FREEZE_HOLD   = UInt8(1)
const FREEZE_SPRING = UInt8(2)

export SimulationState, build_simulation, step!, step_graph!, zero_forces!, sync_unwrapped!, accumulate_virial!
export IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama

# =========================
#   Simulation state (SoA)
# =========================
"""
    SimulationState{T}

Structure-of-arrays storage for all GPU-resident buffers required to evolve a
non-equilibrium simulation. Users normally do not construct this type manually;
[`build_simulation`](@ref) assembles a fully initialized instance with concrete
CuArray fields and consistent dimensionality inferred from the `box` argument.

Key fields that user code may read or mutate:
- `rx, ry[, rz]`, `vx, vy[, vz]`, `fx, fy[, fz]`: positions, velocities, and
  force accumulators in GPU memory. Arrays are sized `N` and remain allocated
  for the entire simulation so that stepping is allocation-free.
- `rx_unwrap, ry_unwrap[, rz_unwrap]`: optional unwrapped positions that track
  continuous motion across periodic boundaries (enabled via `unwrapped_positions`).
- `nbh`: neighbor matrix (either dense cell list or sentinel all-pairs) built
  with [`NeighborLists.build_neighbors_*`](@ref) using the requested cutoff,
  skin, and capacity.
- `vv`: Langevin integrator parameters (`γ`, noise scale, per-particle OU state).
- `Epot`, `Ekin`, `virial`, `dq`, `dU` plus the corresponding `*_accum` buffers:
  energy, virial, and heat observables that can be sampled directly on the GPU.
- `typeid`: per-particle type ids used by `Filters` and mixed-size LJ kernels.
- `freeze_*`: optional buffers used by the freeze/tether helpers in `Filters`.
- `coll_*`: optional buffers that appear when collision counting is enabled via
  `enable_collision_counting!`.

All other fields are internal implementation details (neighbor rebuild state,
bond lists, cached LJ parameters, etc.) and should be treated as read-only.
"""
mutable struct SimulationState{T<:AbstractFloat}
    # SoA arrays
    rx::CuArray{T,1}; ry::CuArray{T,1}
    rz::Union{Nothing,CuArray{T,1}}
    rx_unwrap::Union{Nothing,CuArray{T,1}}
    ry_unwrap::Union{Nothing,CuArray{T,1}}
    rz_unwrap::Union{Nothing,CuArray{T,1}}
    vx::CuArray{T,1}; vy::CuArray{T,1}
    vz::Union{Nothing,CuArray{T,1}}
    fx::CuArray{T,1}; fy::CuArray{T,1}
    fz::Union{Nothing,CuArray{T,1}}

    # previous forces (to avoid per-step allocations)
    f0x::CuArray{T,1}; f0y::CuArray{T,1}
    f0z::Union{Nothing,CuArray{T,1}}

    # per-step random impulse (shared between pos/vel updates)
    rf_x::CuArray{T,1}; rf_y::CuArray{T,1}
    rf_z::Union{Nothing,CuArray{T,1}}
    # persistent OU noise state (used when corr_time is set)
    ou_x::Union{Nothing,CuArray{T,1}}
    ou_y::Union{Nothing,CuArray{T,1}}
    ou_z::Union{Nothing,CuArray{T,1}}

    # per-particle type id
    typeid::CuArray{Int32,1}

    # box (stored directly; no splatting)
    box2::Union{Definitions.Box2{T},Nothing}
    box3::Union{Definitions.Box3{T},Nothing}

    # neighbor list
    nbh::NeighborLists.AbstractNeighborMatrix
    neigh_interval::Int

    # pair params (global LJ)
    pair_lj::Definitions.LJParams{T}
    # optional per-particle size (mixed interactions)
    sigma_particle::Union{Nothing,CuArray{T,1}}
    rcut_factor::T
    # optional per-type pair parameters
    sigma_pair::Union{Nothing,CuArray{T,2}}
    epsilon_pair::Union{Nothing,CuArray{T,2}}
    rcut_pair::Union{Nothing,CuArray{T,2}}
    
    # bonded interactions (optional)
    bonds::Union{Nothing,BondedForces.BondList}
    bonding::Union{Nothing,Definitions.BondPotential{T}}

    # integrator params
    vv::LangevinIntegrators.VVParams{T}

    # observables buffers
    Epot::CuArray{T,1}
    dq::CuArray{T,1}
    dU::CuArray{T,1}
    Ekin::CuArray{T,1}
    virial::CuArray{T,1}
    # interval accumulators (GPU) to avoid host reductions each step
    Epot_accum::CuArray{T,1}
    Ekin_accum::CuArray{T,1}
    virial_accum::CuArray{T,1}

    # misc
    step::Int
    # last integrator used: 1=Langevin, 2=Brownian, 0=unknown
    last_integrator::UInt8
    nb_kind::UInt8
    softrep::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
    # freeze controls (optional)
    freeze_mode::UInt8
    freeze_until::Int
    freeze_include_energy::Bool
    freeze_mask::Union{Nothing,CuArray{UInt8,1}}
    freeze_k::T
    freeze_rx::Union{Nothing,CuArray{T,1}}
    freeze_ry::Union{Nothing,CuArray{T,1}}
    freeze_rz::Union{Nothing,CuArray{T,1}}
    # Collision counting (optional)
    coll_enabled::Bool
    coll_prev::Union{Nothing,CuArray{UInt8,1}}
    coll_counts::Union{Nothing,CuArray{Int64,1}}
    coll_bins::Union{Nothing,CuArray{Int32,2}}
end

# -------------------------
# Unified integrator specs
# -------------------------
"""
Marker type used to dispatch `step!(st, spec, dt)` onto specific integrators.
"""
abstract type IntegratorSpec{T<:AbstractFloat} end

"""
Wrapper storing a reference to `st.vv` so `step!(st, velocityverlet(st), dt)`
dispatches to the Langevin velocity-Verlet integrator (`examples/TwoT_2D_LD_VV.jl`).
"""
struct VVSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.VVParams{T}
end

"""
BAOAB splitting spec used in `examples/TwoT_2D_LD_BAOAB.jl`.
"""
struct BAOABSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
end

"""
BAOA splitting (no trailing B) for legacy scripts.
"""
struct BAOASpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
end

"""
Generalized simulation scheme (GSM) spec, which reuses the BAOAB parameter type.
"""
struct GSMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
end

"""
Midpoint Brownian integrator spec created by [`eulerheun`](@ref).
"""
struct BrownianSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianIntegrators.BrownianParams{T}
end

"""
Euler–Maruyama overdamped spec created by [`eulermaruyama`](@ref).
"""
struct EMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianIntegrators.EMParams{T}
end

"""
    velocityverlet(st) -> VVSpec

Convenience wrapper so `step!(st, velocityverlet(st), dt)` selects the GJF/Langevin
velocity-Verlet path.
"""
velocityverlet(st::SimulationState{T}) where {T<:AbstractFloat} = VVSpec{T}(st.vv)
"""
    baoab(st) -> BAOABSpec

Return a BAOAB spec built from `st.vv` buffers (gamma/noise_scale). Used in
`examples/TwoT_2D_LD_BAOAB.jl`.
"""
baoab(st::SimulationState{T}) where {T<:AbstractFloat} = BAOABSpec{T}(LangevinIntegrators.BAOABParams{T}(st.vv.gamma, st.vv.mass, st.vv.noise_scale, st.vv.corr_time))
"""
    baoa(st) -> BAOASpec

Legacy BAOA variant (no final B kick).
"""
baoa(st::SimulationState{T}) where {T<:AbstractFloat} = BAOASpec{T}(LangevinIntegrators.BAOABParams{T}(st.vv.gamma, st.vv.mass, st.vv.noise_scale, st.vv.corr_time))
"""
    gsm(st) -> GSMSpec

Construct a GSM spec (used by `examples/TwoT_2D_LD_GSM.jl`).
"""
gsm(st::SimulationState{T})  where {T<:AbstractFloat} = GSMSpec{T}(LangevinIntegrators.BAOABParams{T}(st.vv.gamma, st.vv.mass, st.vv.noise_scale, st.vv.corr_time))
"""
    eulerheun(st) -> BrownianSpec

Build a midpoint Brownian spec sharing `st.vv`'s gamma/noise-scale buffers.
"""
eulerheun(st::SimulationState{T}) where {T<:AbstractFloat} = BrownianSpec{T}(BrownianIntegrators.BrownianParams(st))
"""
    eulermaruyama(st) -> EMSpec

Return an Euler–Maruyama spec for overdamped dynamics (`examples/3D_BD.jl`).
"""
eulermaruyama(st::SimulationState{T}) where {T<:AbstractFloat} = EMSpec{T}(BrownianIntegrators.EMParams(st.vv.gamma, st.vv.noise_scale, st.vv.corr_time))
"""
BrownianIntegrators.BrownianParams(st)

Build a Brownian parameter container reusing the simulation's current
`gamma` and `noise_scale` buffers. Keeps element type `T` consistent.
"""
function BrownianIntegrators.BrownianParams(st::SimulationState{T}) where {T<:AbstractFloat}
    return BrownianIntegrators.BrownianParams{T}(st.vv.gamma, st.vv.noise_scale, st.vv.corr_time)
end

"""
    zero_forces!(st)

Fill `st.fx`, `st.fy`, (`st.fz` if present) with zeros. Called by the tests
before running isolated force kernels and safe to use between manual force
evaluations.
"""
function zero_forces!(st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(st.fx, zero(T)); fill!(st.fy, zero(T))
    st.fz === nothing || fill!(st.fz, zero(T))
    return nothing
end

"""
    sync_unwrapped!(st)

Copy the wrapped positions into the unwrapped buffers. Call this after manually
overwriting `st.rx`/`st.ry`/`st.rz` when `unwrapped_positions=true`.
"""
function sync_unwrapped!(st::SimulationState{T}) where {T<:AbstractFloat}
    st.rx_unwrap === nothing && return st
    copyto!(st.rx_unwrap, st.rx)
    copyto!(st.ry_unwrap, st.ry)
    if st.rz !== nothing && st.rz_unwrap !== nothing
        copyto!(st.rz_unwrap, st.rz)
    end
    return st
end

# Ensure OU state buffers exist when correlated noise is requested.
function _ensure_ou_state!(st::SimulationState{T},
                           corr_time::Union{Nothing,CuArray{T,1}}=st.vv.corr_time) where {T<:AbstractFloat}
    corr_time === nothing && return
    if st.ou_x === nothing
        st.ou_x = CUDA.CuArray{T}(undef, length(st.rx))
        st.ou_y = CUDA.CuArray{T}(undef, length(st.ry))
        fill!(st.ou_x, zero(T)); fill!(st.ou_y, zero(T))
        if st.rz === nothing
            st.ou_z = nothing
        else
            st.ou_z = CUDA.CuArray{T}(undef, length(st.rz))
            fill!(st.ou_z, zero(T))
        end
    elseif st.rz !== nothing && st.ou_z === nothing
        st.ou_z = CUDA.CuArray{T}(undef, length(st.rz))
        fill!(st.ou_z, zero(T))
    end
    return
end

@inline function _require_positive_gamma!(gamma::CuArray{T,1}, integrator::AbstractString) where {T<:AbstractFloat}
    gmin = minimum(gamma)
    if !(gmin > zero(T))
        throw(ArgumentError("$(integrator) integrator requires gamma > 0 for all particles."))
    end
    return nothing
end

# -------------------------
# Bond helpers (2D / 3D)
# -------------------------
function _apply_bonds2!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1}, E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    if (st.bonds === nothing) || (st.bonding === nothing)
        return
    end
    if st.bonding isa Definitions.HarmonicBond{T}
        p = (st.bonding::Definitions.HarmonicBond{T}).params
        if compute_energy && E !== nothing
            BondedForces.harmonic_forces_soa!(st.rx, st.ry, fx, fy, E, st.bonds, st.box2::Definitions.Box2{T}, p)
        else
            BondedForces.harmonic_forces_soa_noE!(st.rx, st.ry, fx, fy, st.bonds, st.box2::Definitions.Box2{T}, p)
        end
    elseif st.bonding isa Definitions.FENEBond{T}
        p = (st.bonding::Definitions.FENEBond{T}).params
        if compute_energy && E !== nothing
            BondedForces.fene_forces_soa!(st.rx, st.ry, fx, fy, E, st.bonds, st.box2::Definitions.Box2{T}, p)
        else
            BondedForces.fene_forces_soa_noE!(st.rx, st.ry, fx, fy, st.bonds, st.box2::Definitions.Box2{T}, p)
        end
    end
    return
end

function _apply_bonds3!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    if (st.bonds === nothing) || (st.bonding === nothing)
        return
    end
    if st.bonding isa Definitions.HarmonicBond{T}
        p = (st.bonding::Definitions.HarmonicBond{T}).params
        if compute_energy && E !== nothing
            BondedForces.harmonic_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, st.bonds, st.box3::Definitions.Box3{T}, p)
        else
            BondedForces.harmonic_forces_soa_noE!(st.rx, st.ry, st.rz, fx, fy, fz, st.bonds, st.box3::Definitions.Box3{T}, p)
        end
    elseif st.bonding isa Definitions.FENEBond{T}
        p = (st.bonding::Definitions.FENEBond{T}).params
        if compute_energy && E !== nothing
            BondedForces.fene_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, st.bonds, st.box3::Definitions.Box3{T}, p)
        else
            BondedForces.fene_forces_soa_noE!(st.rx, st.ry, st.rz, fx, fy, fz, st.bonds, st.box3::Definitions.Box3{T}, p)
        end
    end
    return
end

# -------------------------
# Freeze helpers
# -------------------------

@inline function _freeze_active!(st::SimulationState)
    if st.freeze_mode == FREEZE_NONE
        return false
    end
    if st.freeze_until >= 0 && st.step >= st.freeze_until
        st.freeze_mode = FREEZE_NONE
        st.freeze_until = -1
        return false
    end
    return true
end

function _freeze_hold2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                               mask::CuDeviceVector{UInt8},
                               ax::CuDeviceVector{T}, ay::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            rx[i] = ax[i]
            ry[i] = ay[i]
        end
    end
    return
end

function _freeze_hold3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                               mask::CuDeviceVector{UInt8},
                               ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            rx[i] = ax[i]
            ry[i] = ay[i]
            rz[i] = az[i]
        end
    end
    return
end

function _freeze_hold2!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        mask::CuArray{UInt8,1},
                        ax::CuArray{T,1}, ay::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _freeze_hold2_kernel!(rx, ry, mask, ax, ay)
    k(rx, ry, mask, ax, ay; threads, blocks)
    return nothing
end

function _freeze_hold3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        mask::CuArray{UInt8,1},
                        ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _freeze_hold3_kernel!(rx, ry, rz, mask, ax, ay, az)
    k(rx, ry, rz, mask, ax, ay, az; threads, blocks)
    return nothing
end

function _freeze_spring2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
        end
    end
    return
end

function _freeze_spring2_energy_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                        fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                        Epot::CuDeviceVector{T},
                                        mask::CuDeviceVector{UInt8},
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            Epot[i] += T(0.5) * k * (dx * dx + dy * dy)
        end
    end
    return
end

function _freeze_spring3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            dz = rz[i] - az[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            fz[i] -= k * dz
        end
    end
    return
end

function _freeze_spring3_energy_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                        fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                        Epot::CuDeviceVector{T},
                                        mask::CuDeviceVector{UInt8},
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            dz = rz[i] - az[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            fz[i] -= k * dz
            Epot[i] += T(0.5) * k * (dx * dx + dy * dy + dz * dz)
        end
    end
    return
end

function _freeze_spring2!(rx::CuArray{T,1}, ry::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring2_kernel!(rx, ry, fx, fy, mask, ax, ay, k)
    ker(rx, ry, fx, fy, mask, ax, ay, k; threads, blocks)
    return nothing
end

function _freeze_spring2_energy!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring2_energy_kernel!(rx, ry, fx, fy, Epot, mask, ax, ay, k)
    ker(rx, ry, fx, fy, Epot, mask, ax, ay, k; threads, blocks)
    return nothing
end

function _freeze_spring3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring3_kernel!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k)
    ker(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k; threads, blocks)
    return nothing
end

function _freeze_spring3_energy!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring3_energy_kernel!(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k)
    ker(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k; threads, blocks)
    return nothing
end

function _apply_freeze_hold!(st::SimulationState{T}, rx::CuArray{T,1}, ry::CuArray{T,1}) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    return _freeze_hold2!(rx, ry, mask, ax, ay)
end

function _apply_freeze_hold!(st::SimulationState{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    az = st.freeze_rz
    if mask === nothing || ax === nothing || ay === nothing || az === nothing
        return nothing
    end
    return _freeze_hold3!(rx, ry, rz, mask, ax, ay, az)
end

function _apply_freeze_hold_unwrap!(st::SimulationState{T}) where {T<:AbstractFloat}
    rxu = st.rx_unwrap
    ryu = st.ry_unwrap
    if rxu === nothing || ryu === nothing
        return nothing
    end
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    if st.rz_unwrap === nothing
        return _freeze_hold2!(rxu, ryu, mask, ax, ay)
    end
    az = st.freeze_rz
    az === nothing && return nothing
    return _freeze_hold3!(rxu, ryu, st.rz_unwrap, mask, ax, ay, az)
end

function _apply_freeze_hold_positions!(st::SimulationState{T}) where {T<:AbstractFloat}
    if st.rz === nothing
        _apply_freeze_hold!(st, st.rx, st.ry)
    else
        _apply_freeze_hold!(st, st.rx, st.ry, st.rz)
    end
    _apply_freeze_hold_unwrap!(st)
    return nothing
end

function _apply_freeze_spring!(st::SimulationState{T},
                               rx::CuArray{T,1}, ry::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1},
                               E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    k = st.freeze_k
    k <= zero(T) && return nothing
    if compute_energy && st.freeze_include_energy && E !== nothing
        return _freeze_spring2_energy!(rx, ry, fx, fy, E, mask, ax, ay, k)
    end
    return _freeze_spring2!(rx, ry, fx, fy, mask, ax, ay, k)
end

function _apply_freeze_spring!(st::SimulationState{T},
                               rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                               E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    az = st.freeze_rz
    if mask === nothing || ax === nothing || ay === nothing || az === nothing
        return nothing
    end
    k = st.freeze_k
    k <= zero(T) && return nothing
    if compute_energy && st.freeze_include_energy && E !== nothing
        return _freeze_spring3_energy!(rx, ry, rz, fx, fy, fz, E, mask, ax, ay, az, k)
    end
    return _freeze_spring3!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k)
end

# ==========================================
#  Top-level, non-capturing init kernels
#  (avoid nested functions / closures)
# ==========================================

function _init_vel2_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = randn(T) * sqrt(temperature_vec[i])
        vy[i] = randn(T) * sqrt(temperature_vec[i])
    end
    return
end

function _init_vel3_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    vz::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = randn(T) * sqrt(temperature_vec[i])
        vy[i] = randn(T) * sqrt(temperature_vec[i])
        vz[i] = randn(T) * sqrt(temperature_vec[i])
    end
    return
end

# Host launchers
function _init_vel2!(vx::CuArray{T,1}, vy::CuArray{T,1}, temperature_vec::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel2_kernel!(vx, vy, temperature_vec)
    CUDA.@sync k(vx, vy, temperature_vec; threads, blocks)
    return nothing
end

function _init_vel3!(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1}, temperature_vec::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel3_kernel!(vx, vy, vz, temperature_vec)
    CUDA.@sync k(vx, vy, vz, temperature_vec; threads, blocks)
    return nothing
end

# =========================
#   Build simulation
# =========================
"""
    build_simulation(; N, box, cutoff=1, skin=0.4, cap=Int32(96),
                      neigh_interval=20, use_neighborlist=true,
                      epsilon=1, sigma=1, gamma=1, temperature=1,
                      noise_corr_time=nothing, dt=0.001,
                      mass=1, bonds=nothing, bonding=nothing,
                      nonbonded=:lj, softrep_params=nothing,
                      precision=:f32, unwrapped_positions::Bool=false)

Construct a [`SimulationState`](@ref) with GPU-resident SoA arrays and a
neighbor list configured for the requested potential. All inputs are keyword
arguments so that scripts can copy known-good parameter sets directly from the
`examples/` directory without ambiguity.

Key behaviors:
- The simulation dimensionality (2D vs 3D) is inferred from the length of
  `box`. Positions/velocities/forces allocate the corresponding CuArrays.
- For `nonbonded = :wca` the neighbor cutoff is forced to the physical WCA
  value `r_c = 2^(1/6) σ` even if a larger `cutoff` was passed, guaranteeing
  that kernels reuse the validated parameter sets from the packaged examples.
- `gamma`, `temperature`, and `noise_corr_time` accept either scalars or
  length-`N` vectors. Scalars are broadcast on the GPU using the chosen
  floating-point precision (`:f32` or `:f64`).
- Initial velocities are drawn from a Maxwell–Boltzmann distribution at
  `temperature` and then centered to remove center-of-mass drift.
- When `unwrapped_positions=true`, additional `rx_unwrap`/`ry_unwrap`/`rz_unwrap`
  buffers track continuous positions across periodic boundaries.

Example (mirrors `examples/2D_example.jl`, scaled down to N=4096 for testing):

```julia
st = build_simulation(N=4096, box=(250f0, 250f0),
                      cutoff=Float32(2^(1/6)), skin=Float32(2^(1/6))/2,
                      cap=Int32(250), neigh_interval=50,
                      epsilon=1f4, sigma=1f0,
                      gamma=615f0, temperature=10f0,
                      dt=1f-5, nonbonded=:wca, precision=:f32)
step!(st, 1f-5; compute_energy=false)
```

Returns a fully initialized `SimulationState` ready for stepping with the
Langevin (Velocity Verlet / BAOAB / GSM) or Brownian integrators.
"""
function build_simulation(;N::Int,
                           box,
                           cutoff::Real=1.0,
                           skin::Real=0.4,
                           cap::Int32=Int32(96),
                           neigh_interval::Int=20,
                           use_neighborlist::Bool=true,
                           epsilon::Real=1,
                           sigma::Real=1,
                           gamma::Union{AbstractVector{<:Real},Real}=1,
                           temperature::Union{AbstractVector{<:Real},Real}=1,
                           noise_corr_time::Union{AbstractVector{<:Real},Real,Nothing}=nothing,
                           dt::Real=0.001,
                           mass::Real=1,
                           bonds::Union{Nothing,Vector{Tuple{Int32,Int32}}}=nothing,
                           bonding::Union{Nothing,Definitions.BondPotential}=nothing,
                           nonbonded::Symbol = :lj,
                           softrep_params::Union{Nothing,Definitions.SoftRepulsiveParams{<:Real}}=nothing,
                           precision::Symbol = :f32,
                           unwrapped_positions::Bool = false)

    # Dimension from box
    D = length(box)

    if precision == :f32
        T = Float32
    elseif precision == :f64
        T = Float64
    else
        error("Unknown precision=$(precision). Use :f32 or :f64")
    end

    epsilonT = T(epsilon)
    sigmaT   = T(sigma)
    requested_cutoff = T(cutoff)
    nb_cutoff = (nonbonded === :wca) ? (sigmaT * T(WCA_RC_FACTOR)) : requested_cutoff
    rcut_factor = sigmaT == zero(T) ? T(1) : nb_cutoff / sigmaT

    # Allocate SoA buffers
    rx = CUDA.CuArray{T}(undef, N); ry = CUDA.CuArray{T}(undef, N)
    vx = CUDA.CuArray{T}(undef, N); vy = CUDA.CuArray{T}(undef, N)
    fx = CUDA.CuArray{T}(undef, N); fy = CUDA.CuArray{T}(undef, N)
    rz = nothing; vz = nothing; fz = nothing
    rx_unwrap = unwrapped_positions ? CUDA.CuArray{T}(undef, N) : nothing
    ry_unwrap = unwrapped_positions ? CUDA.CuArray{T}(undef, N) : nothing
    rz_unwrap = nothing

    # previous forces
    f0x = CUDA.CuArray{T}(undef, N)
    f0y = CUDA.CuArray{T}(undef, N)
    f0z = nothing

    # Optional per-particle exponential correlation time (τ); if not provided, noise is white
    local corr_time_vec::Union{Nothing,CuArray{T,1}}
    if noise_corr_time === nothing
        corr_time_vec = nothing
    elseif noise_corr_time isa Real
        corr_time_vec = CUDA.fill(T(noise_corr_time), N)
    else
        @assert length(noise_corr_time) == N "noise_corr_time vector must have length N"
        corr_time_vec = CuArray(T.(noise_corr_time))
    end

    # per-step random impulse
    rf_x = CUDA.CuArray{T}(undef, N)
    rf_y = CUDA.CuArray{T}(undef, N)
    rf_z = nothing
    ou_x = nothing
    ou_y = nothing
    ou_z = nothing

    if D == 3
        rz  = CUDA.CuArray{T}(undef, N)
        vz  = CUDA.CuArray{T}(undef, N)
        fz  = CUDA.CuArray{T}(undef, N)
        f0z = CUDA.CuArray{T}(undef, N)
        rf_z = CUDA.CuArray{T}(undef, N)
        ou_z = corr_time_vec === nothing ? nothing : CUDA.CuArray{T}(undef, N)
        if unwrapped_positions
            rz_unwrap = CUDA.CuArray{T}(undef, N)
        end
    end
    if corr_time_vec !== nothing
        ou_x = CUDA.CuArray{T}(undef, N)
        ou_y = CUDA.CuArray{T}(undef, N)
    end

    fill!(rx, zero(T)); fill!(ry, zero(T))
    rz === nothing || fill!(rz, zero(T))
    rx_unwrap === nothing || fill!(rx_unwrap, zero(T))
    ry_unwrap === nothing || fill!(ry_unwrap, zero(T))
    rz_unwrap === nothing || fill!(rz_unwrap, zero(T))

    fill!(fx, zero(T)); fill!(fy, zero(T)); fz === nothing || fill!(fz, zero(T))
    fill!(f0x, zero(T)); fill!(f0y, zero(T)); f0z === nothing || fill!(f0z, zero(T))
    fill!(rf_x, zero(T)); fill!(rf_y, zero(T)); rf_z === nothing || fill!(rf_z, zero(T))
    ou_x === nothing || fill!(ou_x, zero(T)); ou_y === nothing || fill!(ou_y, zero(T)); ou_z === nothing || fill!(ou_z, zero(T))

    # Maxwell-Boltzmann initial velocities on GPU
    if temperature isa Real
        temperature_vec = CUDA.fill(T(temperature), N)
    else
        temperature_vec = CuArray(T.(temperature))
    end

    if D == 2
        _init_vel2!(vx, vy, temperature_vec)
    else
        _init_vel3!(vx, vy, vz, temperature_vec)
    end

    # Remove center-of-mass drift from the initial velocities
    if N > 0
        Vx = CUDA.sum(vx) / T(N)
        Vy = CUDA.sum(vy) / T(N)
        @. vx = vx - Vx
        @. vy = vy - Vy
        if D == 3
            Vz = CUDA.sum(vz) / T(N)
            @. vz = vz - Vz
        end
    end

    typeid = CUDA.fill(Int32(1), N)

    # Neighbors (dense cell-list or sentinel all-pairs)
    if use_neighborlist
        if D == 2
            nbh = NeighborLists.build_neighbors_dense!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=nb_cutoff, cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_dense!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=nb_cutoff, cap, skin=T(skin))
        end
    else
        if D == 2
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=nb_cutoff, cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=nb_cutoff, cap, skin=T(skin))
        end
    end

    lj = Definitions.LJParams{T}(epsilonT, sigmaT, nb_cutoff)

    if gamma isa Real
        gamma_vec = CUDA.fill(T(gamma), N)
    else
        gamma_vec = CuArray(T.(gamma))
    end

    noise_scale = CuArray(sqrt.(T(2) .* gamma_vec .* temperature_vec .* T(dt)))

    vv = LangevinIntegrators.VVParams{T}(gamma_vec, T(mass), noise_scale, corr_time_vec)

    Epot = CUDA.CuArray{T}(undef, N); fill!(Epot, zero(T))
    dq   = CUDA.CuArray{T}(undef, N); fill!(dq, zero(T))
    dU   = CUDA.CuArray{T}(undef, N); fill!(dU, zero(T))
    Ekin = CUDA.CuArray{T}(undef, N); fill!(Ekin, zero(T))
    virial = CUDA.CuArray{T}(undef, N); fill!(virial, zero(T))
    # interval accumulators (GPU)
    Epot_accum = CUDA.CuArray{T}(undef, N); fill!(Epot_accum, zero(T))
    Ekin_accum = CUDA.CuArray{T}(undef, N); fill!(Ekin_accum, zero(T))
    virial_accum = CUDA.CuArray{T}(undef, N); fill!(virial_accum, zero(T))

    # Build bonds (if provided)
    local bondlist
    if bonds === nothing
        bondlist = nothing
    else
        bondlist = BondedForces.build_bondlist(N, bonds)
    end

    # Determine nonbonded kind and soft-rep params
    local nb_tag::UInt8
    local srp::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
    if nonbonded === :lj
        nb_tag = NB_KIND_LJ
        srp = nothing
    elseif nonbonded === :wca
        nb_tag = NB_KIND_WCA
        srp = nothing
    elseif nonbonded === :soft_repulsive || nonbonded === :softrep || nonbonded === :soft
        nb_tag = NB_KIND_SOFTREP
        srp = softrep_params === nothing ? Definitions.SoftRepulsiveParams{T}(epsilonT, sigmaT) : softrep_params
    else
        error("Unknown nonbonded=:$(nonbonded). Use :lj, :wca, or :soft_repulsive")
    end

    # Resolve bonded potential from provided options (new unified or legacy)
    local bond_spec
    if bonding !== nothing
        # if provided, attempt to cast inner params to T
        if bonding isa Definitions.HarmonicBond
            p = (bonding::Definitions.HarmonicBond).params
            bond_spec = Definitions.HarmonicBond{T}(Definitions.HarmonicBondParams{T}(T(p.k), T(p.r0)))
        elseif bonding isa Definitions.FENEBond
            p = (bonding::Definitions.FENEBond).params
            bond_spec = Definitions.FENEBond{T}(Definitions.FENEParams{T}(T(p.k), T(p.R0)))
        else
            bond_spec = nothing
        end
    else
        bond_spec = nothing
    end

    # Construct with boxes set to nothing; assign after
    st = SimulationState(rx, ry, rz, rx_unwrap, ry_unwrap, rz_unwrap, vx, vy, vz, fx, fy, fz,
                         f0x, f0y, f0z,
                         rf_x, rf_y, rf_z,
                         ou_x, ou_y, ou_z,
                         typeid,
                         nothing,   # box2
                         nothing,   # box3
                         nbh, neigh_interval, lj,
                         nothing, rcut_factor,
                         nothing, nothing, nothing,
                         bondlist, bond_spec,
                         vv,
                         Epot, dq, dU, Ekin, virial, Epot_accum, Ekin_accum, virial_accum,
                         0, UInt8(0), nb_tag, srp,
                         FREEZE_NONE, -1, true, nothing, zero(T), nothing, nothing, nothing,
                         false, nothing, nothing, nothing) 

    # Assign the appropriate box directly (no extra tuple layer)
    if D == 2
        st.box2 = (T(box[1]), T(box[2]))
        st.box3 = nothing
    else
        st.box2 = nothing
        st.box3 = (T(box[1]), T(box[2]), T(box[3]))
    end

    return st
end

# =========================
#   Virial (GPU)
# =========================
function _virial2_kernel!(rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T},
                          fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                          virial::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rxu); if i > N; return; end
    @inbounds begin
        virial[i] = rxu[i] * fx[i] + ryu[i] * fy[i]
    end
    return
end

function _virial3_kernel!(rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T}, rzu::CuDeviceVector{T},
                          fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                          virial::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rxu); if i > N; return; end
    @inbounds begin
        virial[i] = rxu[i] * fx[i] + ryu[i] * fy[i] + rzu[i] * fz[i]
    end
    return
end

function _compute_virial!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1},
                          fz::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    use_unwrap = st.rx_unwrap !== nothing && st.ry_unwrap !== nothing &&
        (fz === nothing || st.rz_unwrap !== nothing)
    rxu = use_unwrap ? st.rx_unwrap : st.rx
    ryu = use_unwrap ? st.ry_unwrap : st.ry
    N = length(rxu)
    threads = min(256, N)
    blocks = cld(N, threads)
    if fz === nothing || (!use_unwrap && st.rz === nothing)
        k = CUDA.@cuda launch=false _virial2_kernel!(rxu, ryu, fx, fy, st.virial)
        k(rxu, ryu, fx, fy, st.virial; threads, blocks)
    else
        rzu = use_unwrap ? st.rz_unwrap : st.rz
        k = CUDA.@cuda launch=false _virial3_kernel!(rxu, ryu, rzu, fx, fy, fz, st.virial)
        k(rxu, ryu, rzu, fx, fy, fz, st.virial; threads, blocks)
    end
    return nothing
end

# =========================
#   Energy accumulation (GPU)
# =========================
function _accumulate_energies!(Ekin_accum, Epot_accum, Ekin, Epot)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(Ekin); if i > N; return; end
    @inbounds begin
        Ekin_accum[i] += Ekin[i]
        Epot_accum[i] += Epot[i]
    end
    return
end

"""
    accumulate_energies!(st)

Add the instantaneous `Ekin`/`Epot` buffers into their per-interval accumulators.
Called once per logging interval in `examples/TwoT_2D_LD_VV.jl` before computing
entropy production.
"""
function accumulate_energies!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.Ekin)
    threads = min(256, N)
    blocks  = cld(N, threads)
    k = CUDA.@cuda launch=false _accumulate_energies!(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot)
    k(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot; threads, blocks)
    return nothing
end

function _accumulate_virial!(virial_accum, virial)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        virial_accum[i] += virial[i]
    end
    return
end

"""
    accumulate_virial!(st)

Add the instantaneous `virial` buffer into the per-interval accumulator.
"""
function accumulate_virial!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.virial)
    threads = min(256, N)
    blocks  = cld(N, threads)
    k = CUDA.@cuda launch=false _accumulate_virial!(st.virial_accum, st.virial)
    k(st.virial_accum, st.virial; threads, blocks)
    return nothing
end

# =========================
#   One integrator step
# =========================
"""
    step!(st, dt; compute_energy=true)
    step!(st, spec::IntegratorSpec, dt; compute_energy=true)

Advance [`SimulationState`](@ref) by one time step. The default method reuses
the last integrator attached to `st` (Langevin VV or Brownian), while the
variant accepting an [`IntegratorSpec`](@ref) lets scripts such as
`examples/TwoT_2D_LD_BAOAB.jl` switch to BAOAB/GSM/Euler–Heun explicitly.

Pipeline overview:
1. Every `st.neigh_interval` steps call [`NeighborLists.update_needed!`](@ref)
   to test whether the displacement-based skin criterion was violated; rebuild
   in place (and reset collision state) when needed.
2. Swap the cached force buffers so the previous-step forces become `f₀`, or
   compute them from scratch if this is the first step.
3. Prepare correlated or white noise via `LangevinIntegrators.vv_prepare_noise!`
   (or the Brownian noise helpers) and advance positions on the GPU.
4. Optionally update collision statistics right after the position move.
5. Recompute nonbonded/bonded forces at `t + Δt` and update velocities;
   per-particle energy/heat accumulators are updated when `compute_energy=true`.

Pass `compute_energy=false` for production runs where observables are sampled
only every few thousand steps. All overloads reuse the same neighbor-list and
collision infrastructure, so switching integrators mid-run is inexpensive.

Examples mirroring `examples/2D_example.jl` and `examples/3D_example.jl`:

```julia
st2d = build_simulation(N=8192, box=(250f0, 250f0), cutoff=Float32(2^(1/6)),
                        skin=Float32(2^(1/6))/2, cap=Int32(250),
                        epsilon=1f4, sigma=1f0,
                        gamma=615f0, temperature=10f0,
                        dt=1f-5, nonbonded=:wca, precision=:f32)
step!(st2d, 1f-5; compute_energy=false)

st3d = build_simulation(N=4096, box=(250f0, 250f0, 250f0),
                        cutoff=Float32(2^(1/6)), skin=0.4f0,
                        cap=Int32(100), epsilon=10f0, sigma=1f0,
                        gamma=10f0, temperature=1f0, dt=5f-5)
step!(st3d, 5f-5; compute_energy=true)
```
"""
function step!(st::SimulationState{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    # Pipeline order: check/rebuild neighbor list → swap previous forces →
    # Langevin noise prep and position update → collision update hook →
    # new nonbonded/bonded forces → velocity update and optional energy accumulation.
    # Ensure the time step matches the simulation precision
    dtT = T(dt)
    freeze_active = _freeze_active!(st)
    freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING
    st.last_integrator = UInt8(1)
    D = st.rz === nothing ? 2 : 3

    # NL rebuild policy using new displacement-based algorithm only
    do_check = (st.step % st.neigh_interval == 0)
    rebuild_needed = false
    if do_check
        rebuild_needed = if D == 2
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
        else
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                        skin=st.nbh.skin,
                                        Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                        step=st.step)
        end
    end
    
    if rebuild_needed
        if D == 2
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
            # Reset collision contact state on rebuild
            _collisions_reinit_on_rebuild!(st)
        else
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
            _collisions_reinit_on_rebuild!(st)
        end
    end

    # Prepare forces at t (f0*): reuse last step's fx,* via buffer swap when available
    if st.step > 1
        st.f0x, st.fx = st.fx, st.f0x
        st.f0y, st.fy = st.fy, st.f0y
        if D == 3; st.f0z, st.fz = st.fz, st.f0z; end
    else
        if D == 2
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                           st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                    st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    end
                else
                    if compute_energy
                        NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                              st.nbh, st.box2::Definitions.Box2,
                                                              st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                   st.nbh, st.box2::Definitions.Box2,
                                                                   st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                            st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                 st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                      st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    end
                end
            else # NB_KIND_SOFTREP
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                       st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                      st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                          st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                end
            end
            # bonded contributions at t
            _apply_bonds2!(st, st.f0x, st.f0y, compute_energy ? st.Epot : nothing, compute_energy)
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.f0x, st.f0y,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                           st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                    st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    end
                else
                    if compute_energy
                        NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                              st.nbh, st.box3::Definitions.Box3,
                                                              st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                   st.nbh, st.box3::Definitions.Box3,
                                                                   st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                            st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                 st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                      st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                       st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                      st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                          st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                end
            end
            # bonded contributions at t
            _apply_bonds3!(st, st.f0x, st.f0y, st.f0z, compute_energy ? st.Epot : nothing, compute_energy)
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end
    end

    # Prepare noise ONCE for the step and reuse in both updates
    _ensure_ou_state!(st)
    if D == 2
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale;
                                              beta_z=nothing,
                                              corr_time=st.vv.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=nothing,
                                              dt=dtT)
        LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                              st.rf_x, st.rf_y, st.vv, dtT, st.box2::Definitions.Box2;
                                              unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
    else
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale;
                                              beta_z=st.rf_z,
                                              corr_time=st.vv.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=st.ou_z,
                                              dt=dtT)
        LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz,
                                              st.f0x, st.f0y, st.f0z,
                                              st.rf_x, st.rf_y, st.rf_z, st.vv, dtT, st.box3::Definitions.Box3;
                                              unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
    end

    # Forces at t + dt (write into fx,fy[,fz])
    if D == 2
        if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                          st.nbh, st.box2::Definitions.Box2,
                                                          st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                      st.nbh, st.box2::Definitions.Box2,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                           st.nbh, st.box2::Definitions.Box2,
                                                           st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                       st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                            st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                           st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy,
                                                                st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                        st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                             st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                            st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy,
                                                                  st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                                       st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                                      st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy,
                                                                          st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                end
            end
        end
        # bonded contributions at t+dt
        _apply_bonds2!(st, st.fx, st.fy, compute_energy ? st.Epot : nothing, compute_energy)
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
        if compute_energy
            _compute_virial!(st, st.fx, st.fy, nothing)
        end
        LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.f0x, st.f0y, st.fx, st.fy,
                                               st.rf_x, st.rf_y, st.dq, st.dU, st.Ekin, st.vv, dtT)
    else
        if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                          st.nbh, st.box3::Definitions.Box3,
                                                          st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                      st.nbh, st.box3::Definitions.Box3,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                           st.nbh, st.box3::Definitions.Box3,
                                                           st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                       st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                            st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                           st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                        st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                             st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                            st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                  st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                                       st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                      st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                          st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                end
            end
        end
        # bonded contributions at t+dt
        _apply_bonds3!(st, st.fx, st.fy, st.fz, compute_energy ? st.Epot : nothing, compute_energy)
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
        if compute_energy
            _compute_virial!(st, st.fx, st.fy, st.fz)
        end
        LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                               st.fx, st.fy, st.fz,
                                               st.rf_x, st.rf_y, st.rf_z,
                                               st.dq, st.dU, st.Ekin, st.vv, dtT)
    end

    if compute_energy
        if D == 2
            _compute_virial!(st, st.fx, st.fy, nothing)
        else
            _compute_virial!(st, st.fx, st.fy, st.fz)
        end
    end

    st.step += 1
    return nothing
end

"""
    step_graph!(st, dt; compute_energy=true)

Execute a single integrator step using a CUDA Graph-captured sequence for the steady
kernel chain (forces → noise → positions → forces → velocities). NL checks and rebuilds
are executed outside the graph when needed. The executable graph is cached and reused
across calls.
"""
function step_graph!(st::SimulationState{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    freeze_active = _freeze_active!(st)
    if freeze_active
        return step!(st, dt; compute_energy)
    end
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING
    dtT = T(dt)
    D = st.rz === nothing ? 2 : 3

    # NL rebuild decision outside graph
    do_check = (st.step % st.neigh_interval == 0)
    if do_check
        rebuild_needed = if D == 2
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
        else
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                        skin=st.nbh.skin,
                                        Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                        step=st.step)
        end
        if rebuild_needed
            if D == 2
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            end
        end
    end

    # Set f0 from previous fx via buffer swap if available; otherwise compute once before capture
    if st.step > 1
        st.f0x, st.fx = st.fx, st.f0x
        st.f0y, st.fy = st.fy, st.f0y
        if D == 3; st.f0z, st.fz = st.fz, st.f0z; end
    else
        if D == 2
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                       st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                        st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                             st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                              st.nbh, st.box2::Definitions.Box2, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                end
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                       st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                        st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                             st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                              st.nbh, st.box3::Definitions.Box3, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                end
            end
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end
    end

    _ensure_ou_state!(st)
    if D == 2
        CUDA.@captured begin
            LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale;
                                                  beta_z=nothing,
                                                  corr_time=st.vv.corr_time,
                                                  state_x=st.ou_x, state_y=st.ou_y, state_z=nothing,
                                                  dt=dtT)
        LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                              st.rf_x, st.rf_y, st.vv, dtT, st.box2::Definitions.Box2;
                                              unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        _collisions_update_after_positions!(st)
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                       st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                        st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                             st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                              st.nbh, st.box2::Definitions.Box2, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                end
            end
            if compute_energy
                _compute_virial!(st, st.fx, st.fy, nothing)
            end
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.f0x, st.f0y, st.fx, st.fy,
                                                   st.rf_x, st.rf_y, st.dq, st.dU, st.Ekin, st.vv, dtT)
        end
    else
        CUDA.@captured begin
            LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale;
                                                  beta_z=st.rf_z,
                                                  corr_time=st.vv.corr_time,
                                                  state_x=st.ou_x, state_y=st.ou_y, state_z=st.ou_z,
                                                  dt=dtT)
        LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz,
                                              st.f0x, st.f0y, st.f0z,
                                              st.rf_x, st.rf_y, st.rf_z, st.vv, dtT, st.box3::Definitions.Box3;
                                              unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        _collisions_update_after_positions!(st)
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                       st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                        st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                             st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                              st.nbh, st.box3::Definitions.Box3, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                end
            end
            if compute_energy
                _compute_virial!(st, st.fx, st.fy, st.fz)
            end
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                   st.fx, st.fy, st.fz,
                                                   st.rf_x, st.rf_y, st.rf_z,
                                                   st.dq, st.dU, st.Ekin, st.vv, dtT)
        end
    end

    st.step += 1
    return nothing
end

"""
    step!(st, vv::LangevinIntegrators.VVParams{T}, dt; compute_energy=true)

Run one Langevin (GJF/Velocity-Verlet style) step using the provided integrator
parameters instead of `st.vv`. This allows passing different noise scales or
parameters without rebuilding the simulation.
"""
function step!(st::SimulationState{T}, vv::LangevinIntegrators.VVParams{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    dtT = T(dt)
    old = st.vv
    st.vv = vv
    try
        return step!(st, dtT; compute_energy)
    finally
        st.vv = old
    end
end

# IntegratorSpec dispatch convenience
function step!(st::SimulationState{T}, spec::VVSpec{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, spec.params, dt; compute_energy)
end

function step!(st::SimulationState{T}, spec::BAOABSpec{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, spec.params, dt; compute_energy)
end

function step!(st::SimulationState{T}, spec::BAOASpec{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    # BAOA: B(1) → A(1/2) → O(1) → A(1/2)
    dtT = T(dt)
    freeze_active = _freeze_active!(st)
    freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING
    st.last_integrator = UInt8(1)
    D = st.rz === nothing ? 2 : 3

    # Neighbor rebuild check
    do_check = (st.step % st.neigh_interval == 0)
    if do_check
        rebuild_needed = if D == 2
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
        else
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                        skin=st.nbh.skin,
                                        Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                        step=st.step)
        end
        if rebuild_needed
            if D == 2
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            end
        end
    end

    # Prepare forces at t in f0*
    if st.step > 1
        st.f0x, st.fx = st.fx, st.f0x
        st.f0y, st.fy = st.fy, st.f0y
        if D == 3; st.f0z, st.fz = st.fz, st.f0z; end
    else
        if D == 2
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                           st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                    st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    end
                else
                    if compute_energy
                        NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                              st.nbh, st.box2::Definitions.Box2,
                                                              st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                   st.nbh, st.box2::Definitions.Box2,
                                                                   st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                            st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                 st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                      st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    end
                end
            else # NB_KIND_SOFTREP
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                       st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                      st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                          st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                end
            end
            # bonded contributions at t
            _apply_bonds2!(st, st.f0x, st.f0y, compute_energy ? st.Epot : nothing, compute_energy)
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.f0x, st.f0y,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                           st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                    st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    end
                else
                    if compute_energy
                        NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                              st.nbh, st.box3::Definitions.Box3,
                                                              st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                   st.nbh, st.box3::Definitions.Box3,
                                                                   st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                            st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                 st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                      st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                        end
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                                       st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                      st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                          st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                end
            end
            _apply_bonds3!(st, st.f0x, st.f0y, st.f0z, compute_energy ? st.Epot : nothing, compute_energy)
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end
    end

    _ensure_ou_state!(st, spec.params.corr_time)
    if D == 2
        # B(1): full kick using f(t)
        LangevinIntegrators.baoab_B_2d!(st.vx, st.vy, st.f0x, st.f0y, spec.params, T(2)*dtT, st.Ekin, st.dU)
        # A(1/2)
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry, st.vx, st.vy, dtT, st.box2::Definitions.Box2;
                                        unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
        # O(1): OU using pre-generated noise (reuse VV noise draw)
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, spec.params.noise_scale;
                                              corr_time=spec.params.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=nothing,
                                              dt=dtT)
        LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy, st.rf_x, st.rf_y, spec.params, dtT, st.dq)
        # A(1/2)
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry, st.vx, st.vy, dtT, st.box2::Definitions.Box2;
                                        unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)

        # forces at t+dt (2D)
        if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                          st.nbh, st.box2::Definitions.Box2,
                                                          st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                end
            end
            if st.bonds !== nothing
                _apply_bonds2!(st, st.fx, st.fy, compute_energy ? st.Epot : nothing, compute_energy)
            end
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end

        if compute_energy
            _compute_virial!(st, st.fx, st.fy, nothing)
        end

        # Conservative power like VV at end of step (BAOA has no final B)
        LangevinIntegrators.cons_power_2d!(st.vx, st.vy, st.fx, st.fy, st.dU)

        # Update Ekin at end (no extra B)
        LangevinIntegrators.baoab_B_2d!(st.vx, st.vy, st.fx, st.fy, spec.params, T(0), st.Ekin, st.dU)
    else
        # 3D variant
        LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, spec.params, T(2)*dtT, st.Ekin, st.dU)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, dtT, st.box3::Definitions.Box3;
                                        unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, spec.params.noise_scale;
                                              beta_z=st.rf_z,
                                              corr_time=spec.params.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=st.ou_z,
                                              dt=dtT)
        LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz, st.rf_x, st.rf_y, st.rf_z, spec.params, dtT, st.dq)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, dtT, st.box3::Definitions.Box3;
                                        unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)

        # forces at t+dt (3D)
        if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                          st.nbh, st.box3::Definitions.Box3,
                                                          st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                      st.nbh, st.box3::Definitions.Box3,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                           st.nbh, st.box3::Definitions.Box3,
                                                           st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                                       st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                      st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                          st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                end
            end
            if st.bonds !== nothing
                _apply_bonds3!(st, st.fx, st.fy, st.fz, compute_energy ? st.Epot : nothing, compute_energy)
            end
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end

        if compute_energy
            _compute_virial!(st, st.fx, st.fy, st.fz)
        end

        # Conservative power like VV at end of step (BAOA has no final B)
        LangevinIntegrators.cons_power_3d!(st.vx, st.vy, st.vz, st.fx, st.fy, st.fz, st.dU)
        
        LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz, st.fx, st.fy, st.fz, spec.params, T(0), st.Ekin, st.dU)
    end

    st.step += 1
    return nothing
end

function step!(st::SimulationState{T}, spec::GSMSpec{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    # GSM uses VV-middle (identical sequence to BAOAB in this implementation)
    return step!(st, spec.params, dt; compute_energy)
end

function step!(st::SimulationState{T}, spec::BrownianSpec{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, spec.params, dt; compute_energy)
end

function step!(st::SimulationState{T}, spec::EMSpec{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, spec.params, dt; compute_energy)
end


"""
    step!(st, bao::LangevinIntegrators.BAOABParams{T}, dt; compute_energy=true)

BAOAB Langevin integrator: B(1/2) → A(1/2) → O(Δt) → A(1/2) → B(1/2).
Uses forces at t for the first half-kick, then forces at t+dt for the final half-kick.
"""
function step!(st::SimulationState{T}, bao::LangevinIntegrators.BAOABParams{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    dtT = T(dt)
    _require_positive_gamma!(bao.gamma, "BAOAB")
    freeze_active = _freeze_active!(st)
    freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING
    st.last_integrator = UInt8(1)
    D = st.rz === nothing ? 2 : 3

    # Neighbor rebuild check
    do_check = (st.step % st.neigh_interval == 0)
    if do_check
        rebuild_needed = if D == 2
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
        else
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                        skin=st.nbh.skin,
                                        Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                        step=st.step)
        end
        if rebuild_needed
            if D == 2
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
            end
        end
    end

    # Prepare forces at t in f0*
    if st.step > 1
        st.f0x, st.fx = st.fx, st.f0x
        st.f0y, st.fy = st.fy, st.f0y
        if D == 3; st.f0z, st.fz = st.fz, st.f0z; end
    else
        if D == 2
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                           st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                    st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    end
                else
                    if compute_energy
                        NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                              st.nbh, st.box2::Definitions.Box2,
                                                              st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                   st.nbh, st.box2::Definitions.Box2,
                                                                   st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.f0x, st.f0y,
                                                                 st.nbh, st.box2::Definitions.Box2,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                            st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                                 st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    else
                        if st.bonds === nothing
                            NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                        else
                            NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.f0x, st.f0y,
                                                                      st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                        end
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                              st.nbh, st.box2::Definitions.Box2, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                end
            end
            if st.bonds !== nothing
                _apply_bonds2!(st, st.f0x, st.f0y, compute_energy ? st.Epot : nothing, compute_energy)
            end
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.f0x, st.f0y,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if compute_energy
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                       st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                           st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if compute_energy
                        NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                              st.nbh, st.box3::Definitions.Box3,
                                                              st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                   st.nbh, st.box3::Definitions.Box3,
                                                                   st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    else
                        NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                    end
                elseif st.sigma_particle !== nothing
                    if compute_energy
                        NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    else
                        NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                 st.nbh, st.box3::Definitions.Box3,
                                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                    end
                else
                    if compute_energy
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                        st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                             st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                              st.nbh, st.box3::Definitions.Box3, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                end
            end
        end
    end

    # BAOAB sequence
    _ensure_ou_state!(st, bao.corr_time)
    if D == 2
        LangevinIntegrators.baoab_BA_2d!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y, bao, dtT, st.box2::Definitions.Box2;
                                         unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        # Prepare OU noise using the same generator as VV (β = s * N(0,1))
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, bao.noise_scale;
                                              corr_time=bao.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=nothing,
                                              dt=dtT)
        LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy, st.rf_x, st.rf_y, bao, dtT, st.dq)
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry, st.vx, st.vy, dtT, st.box2::Definitions.Box2;
                                        unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end

        # forces at t+dt (write to fx,fy)
        if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                          st.nbh, st.box2::Definitions.Box2,
                                                          st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                      st.nbh, st.box2::Definitions.Box2,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                           st.nbh, st.box2::Definitions.Box2,
                                                           st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                                  st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                                       st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                                      st.nbh, st.box2::Definitions.Box2, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                    end
                end
            end
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
        if compute_energy
            _compute_virial!(st, st.fx, st.fy, nothing)
        end
        LangevinIntegrators.baoab_B_2d!(st.vx, st.vy, st.fx, st.fy, bao, dtT, st.Ekin, st.dU)
    else
        LangevinIntegrators.baoab_BA_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, bao, dtT, st.box3::Definitions.Box3;
                                         unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, bao.noise_scale;
                                              beta_z=st.rf_z,
                                              corr_time=bao.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=st.ou_z,
                                              dt=dtT)
        LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz, st.rf_x, st.rf_y, st.rf_z, bao, dtT, st.dq)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, dtT, st.box3::Definitions.Box3;
                                        unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end

        # forces at t+dt (3D)
        if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                          st.nbh, st.box3::Definitions.Box3,
                                                          st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                      st.nbh, st.box3::Definitions.Box3,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                           st.nbh, st.box3::Definitions.Box3,
                                                           st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            else
                NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            end
        elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
            if compute_energy
                NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                                  st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                                       st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                      st.nbh, st.box3::Definitions.Box3, st.softrep)
                    else
                        NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                          st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                    end
                end
            end
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
        if compute_energy
            _compute_virial!(st, st.fx, st.fy, st.fz)
        end
        LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz, st.fx, st.fy, st.fz, bao, dtT, st.Ekin, st.dU)
    end

    st.step += 1
    return nothing
end

"""
    step!(st, bao::LangevinIntegrators.BAOABParams{T}, dt; compute_energy=true)
"""


"""
    step!(st, bp::BrownianIntegrators.BrownianParams{T}, dt; compute_energy=true)

Brownian dynamics (overdamped) Euler–Maruyama midpoint step with Sekimoto heat.
This uses positions-only dynamics (no kinetic energy). Velocities buffers in the
state are reused as temporary storage for midpoint positions.
"""
function step!(st::SimulationState{T}, bp::BrownianIntegrators.BrownianParams{T}, dt::Real; compute_energy::Bool=true
    ) where {T<:AbstractFloat}
    dtT = T(dt)
    _require_positive_gamma!(bp.gamma, "Brownian midpoint")
    freeze_active = _freeze_active!(st)
    freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING
    st.last_integrator = UInt8(2)
    D = st.rz === nothing ? 2 : 3
    _ensure_ou_state!(st, bp.corr_time)

    do_check = (st.step % st.neigh_interval == 0)
    if do_check
        rebuild_needed = if D == 2
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry; skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=st.step)
        else
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz; skin=st.nbh.skin, Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3], step=st.step)
        end
        if rebuild_needed
            if D == 2
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            end
        end
    end

    if st.step == 0
        if D == 2
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                          st.nbh, st.box2::Definitions.Box2,
                                                          st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.softrep)
            end
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                          st.nbh, st.box3::Definitions.Box3,
                                                          st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                end
            else
                @assert st.softrep !== nothing "softrep params missing"
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.softrep)
            end
            if freeze_spring
                _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end
    end

    if D == 2
        # Draw noise once and compute midpoint positions into vx,vy
        BrownianIntegrators.bd_prepare_noise_2d!(st.rf_x, st.rf_y;
                                                 noise_scale=bp.noise_scale,
                                                 corr_time=bp.corr_time,
                                                 state_x=st.ou_x, state_y=st.ou_y,
                                                 dt=dtT)
        BrownianIntegrators.bd_midpoint_positions_2d!(
            st.rx, st.ry, st.fx, st.fy,
            st.rf_x, st.rf_y,
            st.vx, st.vy,
            bp.gamma, bp.noise_scale,
            dtT, st.box2::Definitions.Box2)
        if freeze_hold
            _apply_freeze_hold!(st, st.vx, st.vy)
        end
        if st.nb_kind == NB_KIND_LJ
            if st.sigma_particle === nothing
                if st.bonds === nothing
                    NonBondedForces.lj_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                end
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.vx, st.vy, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA
            if st.sigma_pair !== nothing
                NonBondedForces.wca_forces_soa_noE_pairs!(st.vx, st.vy, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            elseif st.sigma_particle !== nothing
                NonBondedForces.wca_forces_soa_noE_mixed!(st.vx, st.vy, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                if st.bonds === nothing
                    NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                end
            end
        else
            @assert st.softrep !== nothing "softrep params missing"
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y,
                                                              st.nbh, st.box2::Definitions.Box2, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y,
                                                                  st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
            end
        end
        if st.bonds !== nothing
            _apply_bonds2!(st, st.f0x, st.f0y, nothing, false)
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.vx, st.vy, st.f0x, st.f0y, nothing, false)
        end
        # Finalize step using forces at midpoint (in f0*) and same noise
        BrownianIntegrators.bd_finish_step_2d!(
            st.rx, st.ry, st.f0x, st.f0y,
            st.rf_x, st.rf_y,
            bp.gamma, bp.noise_scale,
            dtT, st.dq, st.dU, st.box2::Definitions.Box2;
            unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
        if st.nb_kind == NB_KIND_LJ
            if st.sigma_particle === nothing
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                if compute_energy
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                          st.nbh, st.box2::Definitions.Box2,
                                                          st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                               st.nbh, st.box2::Definitions.Box2,
                                                               st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            end
        elseif st.nb_kind == NB_KIND_WCA
            if st.sigma_pair !== nothing
                if compute_energy
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                else
                    NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                end
            elseif st.sigma_particle !== nothing
                if compute_energy
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            else
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            end
        else
            @assert st.softrep !== nothing "softrep params missing"
            if compute_energy
                if st.bonds === nothing
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                end
            else
                if st.bonds === nothing
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                end
            end
        end
        if st.bonds !== nothing
            _apply_bonds2!(st, st.fx, st.fy, compute_energy ? st.Epot : nothing, compute_energy)
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
    else
        BrownianIntegrators.bd_prepare_noise_3d!(st.rf_x, st.rf_y, st.rf_z;
                                                 noise_scale=bp.noise_scale,
                                                 corr_time=bp.corr_time,
                                                 state_x=st.ou_x, state_y=st.ou_y, state_z=st.ou_z,
                                                 dt=dtT)
        BrownianIntegrators.bd_midpoint_positions_3d!(
            st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
            st.rf_x, st.rf_y, st.rf_z,
            st.vx, st.vy, st.vz,
            bp.gamma, bp.noise_scale,
            dtT, st.box3::Definitions.Box3)
        if freeze_hold
            _apply_freeze_hold!(st, st.vx, st.vy, st.vz)
        end
        if st.nb_kind == NB_KIND_LJ
            if st.sigma_particle === nothing
                if st.bonds === nothing
                    NonBondedForces.lj_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                end
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA
            if st.sigma_pair !== nothing
                NonBondedForces.wca_forces_soa_noE_pairs!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            elseif st.sigma_particle !== nothing
                NonBondedForces.wca_forces_soa_noE_mixed!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                if st.bonds === nothing
                    NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                end
            end
        else
            @assert st.softrep !== nothing "softrep params missing"
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                              st.nbh, st.box3::Definitions.Box3, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                                  st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
            end
        end
        if st.bonds !== nothing
            _apply_bonds3!(st, st.f0x, st.f0y, st.f0z, nothing, false)
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, nothing, false)
        end
        BrownianIntegrators.bd_finish_step_3d!(
            st.rx, st.ry, st.rz,
            st.f0x, st.f0y, st.f0z,
            st.rf_x, st.rf_y, st.rf_z,
            bp.gamma, bp.noise_scale,
            dtT, st.dq, st.dU, st.box3::Definitions.Box3;
            unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
        if st.nb_kind == NB_KIND_LJ
            if st.sigma_particle === nothing
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                if compute_energy
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                          st.nbh, st.box3::Definitions.Box3,
                                                          st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                               st.nbh, st.box3::Definitions.Box3,
                                                               st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            end
        elseif st.nb_kind == NB_KIND_WCA
            if st.sigma_pair !== nothing
                if compute_energy
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                else
                    NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                end
            elseif st.sigma_particle !== nothing
                if compute_energy
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            else
                if compute_energy
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            end
        else
            @assert st.softrep !== nothing "softrep params missing"
            if compute_energy
                if st.bonds === nothing
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                end
            else
                if st.bonds === nothing
                    NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                end
            end
        end
        if st.bonds !== nothing
            _apply_bonds3!(st, st.fx, st.fy, st.fz, compute_energy ? st.Epot : nothing, compute_energy)
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
    end

    if compute_energy
        _compute_virial!(st, st.fx, st.fy, st.fz)
    end

    st.step += 1
    return nothing
end

"""
    step_bd!(st, dt, bp; compute_energy=true)

Deprecated thin wrapper. Use `step!(st, bp, dt; ...)` instead.
"""
function step_bd!(st::SimulationState{T}, dt::Real, bp::BrownianIntegrators.BrownianParams{T}; compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, bp, T(dt); compute_energy)
end
"""
    step!(st, em::BrownianIntegrators.EMParams{T}, dt; compute_energy=true)

Euler–Maruyama overdamped step with additive noise: Δr = μ f Δt + √(2DΔt) ξ.
Accumulates conservative work w = f · Δr into dq (heat) and dU (conservative power proxy).
"""
function step!(st::SimulationState{T}, em::BrownianIntegrators.EMParams{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    dtT = T(dt)
    _require_positive_gamma!(em.gamma, "Euler-Maruyama")
    freeze_active = _freeze_active!(st)
    freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING
    st.last_integrator = UInt8(2)
    D = st.rz === nothing ? 2 : 3
    _ensure_ou_state!(st, em.corr_time)

    # Neighbor-list maintenance (mirror Brownian EH implementation)
    do_check = (st.step % st.neigh_interval == 0)
    if do_check
        rebuild_needed = if D == 2
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
        else
            NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                        skin=st.nbh.skin,
                                        Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                        step=st.step)
        end
        if rebuild_needed
            if D == 2
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
                _collisions_reinit_on_rebuild!(st)
            end
        end
    end

    # Ensure forces at t
    if st.step == 0
        if D == 2
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing
                if st.bonds === nothing
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
                end
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    if st.bonds === nothing
                        NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    if st.bonds === nothing
                        NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                    else
                        NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                    end
                end
            else
                @assert st.softrep !== nothing
                if st.bonds === nothing
                    NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.softrep)
                else
                    NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
                end
            end
        end
        # Include bonded interactions in the initial forces
        if st.bonds !== nothing
            if D == 2
                _apply_bonds2!(st, st.fx, st.fy, compute_energy ? st.Epot : nothing, compute_energy)
            else
                _apply_bonds3!(st, st.fx, st.fy, st.fz, compute_energy ? st.Epot : nothing, compute_energy)
            end
        end
        if freeze_spring
            if D == 2
                _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            else
                _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                      compute_energy ? st.Epot : nothing, compute_energy)
            end
        end
    end

    # Position update and dq/dU accumulation using midpoint so EPR == UPR
    if D == 2
        # Predictor: draw noise then compute midpoint positions into vx,vy
        BrownianIntegrators.bd_prepare_noise_2d!(st.rf_x, st.rf_y;
                                                 noise_scale=em.noise_scale,
                                                 corr_time=em.corr_time,
                                                 state_x=st.ou_x, state_y=st.ou_y,
                                                 dt=dtT)
        BrownianIntegrators.bd_midpoint_positions_2d!(
            st.rx, st.ry, st.fx, st.fy,
            st.rf_x, st.rf_y,
            st.vx, st.vy,
            em.gamma, em.noise_scale,
            dtT, st.box2::Definitions.Box2)
        if freeze_hold
            _apply_freeze_hold!(st, st.vx, st.vy)
        end
        # Forces at midpoint positions (no energy)
        if st.nb_kind == NB_KIND_LJ
            if st.sigma_particle === nothing
                if st.bonds === nothing
                    NonBondedForces.lj_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                end
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.vx, st.vy, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA
            if st.sigma_pair !== nothing
                NonBondedForces.wca_forces_soa_noE_pairs!(st.vx, st.vy, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            elseif st.sigma_particle !== nothing
                NonBondedForces.wca_forces_soa_noE_mixed!(st.vx, st.vy, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                if st.bonds === nothing
                    NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
                end
            end
        else
            @assert st.softrep !== nothing
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y,
                                                              st.nbh, st.box2::Definitions.Box2, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y,
                                                                  st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
            end
        end
        if st.bonds !== nothing
            _apply_bonds2!(st, st.f0x, st.f0y, nothing, false)
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.vx, st.vy, st.f0x, st.f0y, nothing, false)
        end
        # Finish step: advance positions with midpoint force; accumulate dq and dU equally
        BrownianIntegrators.bd_finish_step_2d!(
            st.rx, st.ry, st.f0x, st.f0y,
            st.rf_x, st.rf_y,
            em.gamma, em.noise_scale,
            dtT, st.dq, st.dU, st.box2::Definitions.Box2;
            unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
        # New forces
        if compute_energy
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                         st.nbh, st.box2::Definitions.Box2,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                end
            else
                @assert st.softrep !== nothing
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.softrep)
            end
            # add bonded interactions to forces at t+Δt
            if st.bonds !== nothing
                _apply_bonds2!(st, st.fx, st.fy, st.Epot, true)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                             st.nbh, st.box2::Definitions.Box2,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                end
            else
                @assert st.softrep !== nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.softrep)
            end
            if st.bonds !== nothing
                _apply_bonds2!(st, st.fx, st.fy, nothing, false)
            end
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
    else
        # Predictor: draw noise then compute midpoint positions into vx,vy,vz
        BrownianIntegrators.bd_prepare_noise_3d!(st.rf_x, st.rf_y, st.rf_z;
                                                 noise_scale=em.noise_scale,
                                                 corr_time=em.corr_time,
                                                 state_x=st.ou_x, state_y=st.ou_y, state_z=st.ou_z,
                                                 dt=dtT)
        BrownianIntegrators.bd_midpoint_positions_3d!(
            st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
            st.rf_x, st.rf_y, st.rf_z,
            st.vx, st.vy, st.vz,
            em.gamma, em.noise_scale,
            dtT, st.box3::Definitions.Box3)
        if freeze_hold
            _apply_freeze_hold!(st, st.vx, st.vy, st.vz)
        end
        # Forces at midpoint positions (no energy)
        if st.nb_kind == NB_KIND_LJ
            if st.sigma_particle === nothing
                if st.bonds === nothing
                    NonBondedForces.lj_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                end
            else
                NonBondedForces.lj_forces_soa_noE_mixed!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            end
        elseif st.nb_kind == NB_KIND_WCA
            if st.sigma_pair !== nothing
                NonBondedForces.wca_forces_soa_noE_pairs!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
            elseif st.sigma_particle !== nothing
                NonBondedForces.wca_forces_soa_noE_mixed!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
            else
                if st.bonds === nothing
                    NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
                end
            end
        else
            @assert st.softrep !== nothing
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
            end
        end
        if st.bonds !== nothing
            _apply_bonds3!(st, st.f0x, st.f0y, st.f0z, nothing, false)
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, nothing, false)
        end
        # Finish step: advance positions with midpoint force; accumulate dq and dU equally
        BrownianIntegrators.bd_finish_step_3d!(
            st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
            st.rf_x, st.rf_y, st.rf_z,
            em.gamma, em.noise_scale,
            dtT, st.dq, st.dU, st.box3::Definitions.Box3;
            unwrapped_x=st.rx_unwrap, unwrapped_y=st.ry_unwrap, unwrapped_z=st.rz_unwrap)
        if freeze_hold
            _apply_freeze_hold_positions!(st)
        end
        _collisions_update_after_positions!(st)
        if compute_energy
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                         st.nbh, st.box3::Definitions.Box3,
                                                         st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                end
            else
                @assert st.softrep !== nothing
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.softrep)
            end
            if st.bonds !== nothing
                _apply_bonds3!(st, st.fx, st.fy, st.fz, st.Epot, true)
            end
        else
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if st.sigma_pair !== nothing
                    NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                elseif st.sigma_particle !== nothing
                    NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                             st.nbh, st.box3::Definitions.Box3,
                                                             st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                end
            else
                @assert st.softrep !== nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.softrep)
            end
            if st.bonds !== nothing
                _apply_bonds3!(st, st.fx, st.fy, st.fz, nothing, false)
            end
        end
        if freeze_spring
            _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                  compute_energy ? st.Epot : nothing, compute_energy)
        end
    end

    st.step += 1
    return nothing
end


end # module
