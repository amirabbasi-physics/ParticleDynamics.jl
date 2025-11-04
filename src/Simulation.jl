module Simulation

using CUDA
using ..Definitions
using ..NeighborLists
using ..NonBondedForces
using ..BondedForces
using ..LangevinIntegrators
using ..BrownianIntegrators
 

const NL_CHECK_STRIDE = 20  # only check NL rebuild every N steps to cut overhead

# Nonbonded kind tags (host-side routing only)
const NB_KIND_LJ      = UInt8(1)
const NB_KIND_WCA     = UInt8(2)
const NB_KIND_SOFTREP = UInt8(3)

export SimulationState, build_simulation, step!, step_graph!, step_fused!, zero_forces!
export IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama

# =========================
#   Simulation state (SoA)
# =========================
mutable struct SimulationState{T<:AbstractFloat}
    # SoA arrays
    rx::CuArray{T,1}; ry::CuArray{T,1}
    rz::Union{Nothing,CuArray{T,1}}
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
    # interval accumulators (GPU) to avoid host reductions each step
    Epot_accum::CuArray{T,1}
    Ekin_accum::CuArray{T,1}

    # misc
    step::Int
    # last integrator used: 1=Langevin, 2=Brownian, 0=unknown
    last_integrator::UInt8
    nb_kind::UInt8
    softrep::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
end

# -------------------------
# Unified integrator specs
# -------------------------
abstract type IntegratorSpec{T<:AbstractFloat} end

struct VVSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.VVParams{T}
end

struct BAOABSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
end

struct BAOASpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
end

struct GSMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
end

struct BrownianSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianIntegrators.BrownianParams{T}
end

struct EMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianIntegrators.EMParams{T}
end

velocityverlet(st::SimulationState{T}) where {T<:AbstractFloat} = VVSpec{T}(st.vv)
baoab(st::SimulationState{T}) where {T<:AbstractFloat} = BAOABSpec{T}(LangevinIntegrators.BAOABParams{T}(st.vv.gamma, st.vv.mass, st.vv.noise_scale))
baoa(st::SimulationState{T}) where {T<:AbstractFloat} = BAOASpec{T}(LangevinIntegrators.BAOABParams{T}(st.vv.gamma, st.vv.mass, st.vv.noise_scale))
gsm(st::SimulationState{T})  where {T<:AbstractFloat} = GSMSpec{T}(LangevinIntegrators.BAOABParams{T}(st.vv.gamma, st.vv.mass, st.vv.noise_scale))
eulerheun(st::SimulationState{T}) where {T<:AbstractFloat} = BrownianSpec{T}(BrownianIntegrators.BrownianParams(st))
eulermaruyama(st::SimulationState{T}) where {T<:AbstractFloat} = EMSpec{T}(BrownianIntegrators.EMParams(st.vv.gamma, st.vv.noise_scale))
"""
BrownianIntegrators.BrownianParams(st)

Build a Brownian parameter container reusing the simulation's current
`gamma` and `noise_scale` buffers. Keeps element type `T` consistent.
"""
function BrownianIntegrators.BrownianParams(st::SimulationState{T}) where {T<:AbstractFloat}
    return BrownianIntegrators.BrownianParams{T}(st.vv.gamma, st.vv.noise_scale)
end

function zero_forces!(st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(st.fx, zero(T)); fill!(st.fy, zero(T))
    st.fz === nothing || fill!(st.fz, zero(T))
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
function build_simulation(;N::Int,
                           box,
                           cutoff::Real=1.0,
                           skin::Real=0.4,
                           cap::Int32=Int32(96),
                           neigh_interval::Int=20,
                           use_neighborlist::Bool=true,
                           epsilon::Real=1,
                           sigma::Real=1,
                           gamma::Union{Array{Real,1},Real}=1,
                           temperature::Union{Array{Real,1},Real}=1,
                           init_temperature::Union{Nothing,Real}=nothing,
                           dt::Real=0.001,
                           mass::Real=1,
                           bonds::Union{Nothing,Vector{Tuple{Int32,Int32}}}=nothing,
                           bonding::Union{Nothing,Definitions.BondPotential}=nothing,
                           nonbonded::Symbol = :lj,
                           softrep_params::Union{Nothing,Definitions.SoftRepulsiveParams{Real}}=nothing,
                           precision::Symbol = :f32)

    # Dimension from box
    D = length(box)
    
    if precision == :f32
        T = Float32
    elseif precision == :f64
        T = Float64
    else
        error("Unknown precision=$(precision). Use :f32 or :f64")
    end

    # Allocate SoA buffers
    rx = CUDA.CuArray{T}(undef, N); ry = CUDA.CuArray{T}(undef, N)
    vx = CUDA.CuArray{T}(undef, N); vy = CUDA.CuArray{T}(undef, N)
    fx = CUDA.CuArray{T}(undef, N); fy = CUDA.CuArray{T}(undef, N)
    rz = nothing; vz = nothing; fz = nothing

    # previous forces
    f0x = CUDA.CuArray{T}(undef, N)
    f0y = CUDA.CuArray{T}(undef, N)
    f0z = nothing

    # per-step random impulse
    rf_x = CUDA.CuArray{T}(undef, N)
    rf_y = CUDA.CuArray{T}(undef, N)
    rf_z = nothing

    if D == 3
        rz  = CUDA.CuArray{T}(undef, N)
        vz  = CUDA.CuArray{T}(undef, N)
        fz  = CUDA.CuArray{T}(undef, N)
        f0z = CUDA.CuArray{T}(undef, N)
        rf_z = CUDA.CuArray{T}(undef, N)
    end

    fill!(rx, zero(T)); fill!(ry, zero(T))
    rz === nothing || fill!(rz, zero(T))

    fill!(fx, zero(T)); fill!(fy, zero(T)); fz === nothing || fill!(fz, zero(T))
    fill!(f0x, zero(T)); fill!(f0y, zero(T)); f0z === nothing || fill!(f0z, zero(T))
    fill!(rf_x, zero(T)); fill!(rf_y, zero(T)); rf_z === nothing || fill!(rf_z, zero(T))

    # Maxwell-Boltzmann initial velocities on GPU
    # Back-compat: init_temperature overrides temperature when provided
    local temp_choice
    if init_temperature !== nothing
        temp_choice = init_temperature
    else
        temp_choice = temperature
    end
    if temp_choice isa Real
        temperature_vec = CUDA.fill(T(temp_choice), N)
    else
        temperature_vec = CuArray(T.(temp_choice))
    end

    if D == 2
        _init_vel2!(vx, vy, temperature_vec)
    else
        _init_vel3!(vx, vy, vz, temperature_vec)
    end

    typeid = CUDA.fill(Int32(1), N)

    # Neighbors (dense cell-list or sentinel all-pairs)
    if use_neighborlist
        if D == 2
            nbh = NeighborLists.build_neighbors_dense!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=T(cutoff), cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_dense!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=T(cutoff), cap, skin=T(skin))
        end
    else
        if D == 2
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=T(cutoff), cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=T(cutoff), cap, skin=T(skin))
        end
    end

    lj = Definitions.LJParams{T}(T(epsilon), T(sigma), T(cutoff))

    if gamma isa Real
        gamma_vec = CUDA.fill(T(gamma), N)
    else
        gamma_vec = CuArray(T.(gamma))
    end

    noise_scale = CuArray(sqrt.(T(2) .* gamma_vec .* temperature_vec .* T(dt)))


    vv = LangevinIntegrators.VVParams{T}(gamma_vec, T(mass), noise_scale)

    Epot = CUDA.CuArray{T}(undef, N); fill!(Epot, zero(T))
    dq   = CUDA.CuArray{T}(undef, N); fill!(dq, zero(T))
    dU   = CUDA.CuArray{T}(undef, N); fill!(dU, zero(T))
    Ekin = CUDA.CuArray{T}(undef, N); fill!(Ekin, zero(T))
    # interval accumulators (GPU)
    Epot_accum = CUDA.CuArray{T}(undef, N); fill!(Epot_accum, zero(T))
    Ekin_accum = CUDA.CuArray{T}(undef, N); fill!(Ekin_accum, zero(T))

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
        srp = softrep_params === nothing ? Definitions.SoftRepulsiveParams{T}(T(epsilon), T(sigma)) : softrep_params
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
    st = SimulationState(rx, ry, rz, vx, vy, vz, fx, fy, fz,
                         f0x, f0y, f0z,
                         rf_x, rf_y, rf_z,
                         typeid,
                         nothing,   # box2
                         nothing,   # box3
                         nbh, neigh_interval, lj,
                         nothing, T(2^(1/6)),
                         nothing, nothing, nothing,
                         bondlist, bond_spec,
                         vv,
                         Epot, dq, dU, Ekin, Epot_accum, Ekin_accum, 0, UInt8(0), nb_tag, srp) 

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

function accumulate_energies!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.Ekin)
    threads = min(256, N)
    blocks  = cld(N, threads)
    k = CUDA.@cuda launch=false _accumulate_energies!(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot)
    k(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot; threads, blocks)
    return nothing
end

# =========================
#   One integrator step
# =========================
function step!(st::SimulationState{T}, dt::Real; compute_energy::Bool=true) where {T<:AbstractFloat}
    # Ensure the time step matches the simulation precision
    dtT = T(dt)
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
        else
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
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
        end
    end

    # Prepare noise ONCE for the step and reuse in both updates
    if D == 2
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=nothing)
        LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                              st.rf_x, st.rf_y, st.vv, dtT, st.box2::Definitions.Box2)
    else
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=st.rf_z)
        LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz,
                                              st.f0x, st.f0y, st.f0z,
                                              st.rf_x, st.rf_y, st.rf_z, st.vv, dtT, st.box3::Definitions.Box3)
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
        LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                               st.fx, st.fy, st.fz,
                                               st.rf_x, st.rf_y, st.rf_z,
                                               st.dq, st.dU, st.Ekin, st.vv, dtT)
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
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
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
                if compute_energy
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                                    st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                         st.nbh, st.box2::Definitions.Box2, st.pair_lj)
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
                if compute_energy
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                    st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3, st.pair_lj)
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

    if D == 2
        CUDA.@captured begin
            LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=nothing)
            LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                                  st.rf_x, st.rf_y, st.vv, dtT, st.box2::Definitions.Box2)
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                       st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                    st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.box2::Definitions.Box2, st.pair_lj)
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
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.f0x, st.f0y, st.fx, st.fy,
                                                   st.rf_x, st.rf_y, st.dq, st.dU, st.Ekin, st.vv, dtT)
        end
    else
        CUDA.@captured begin
            LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=st.rf_z)
            LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz,
                                                  st.f0x, st.f0y, st.f0z,
                                                  st.rf_x, st.rf_y, st.rf_z, st.vv, dtT, st.box3::Definitions.Box3)
            if st.nb_kind == NB_KIND_LJ
                if compute_energy
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                       st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                end
            elseif st.nb_kind == NB_KIND_WCA
                if compute_energy
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                                    st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.box3::Definitions.Box3, st.pair_lj)
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
        end
    end

    if D == 2
        # B(1): full kick using f(t)
        LangevinIntegrators.baoab_B_2d!(st.vx, st.vy, st.f0x, st.f0y, spec.params, T(2)*dtT, st.Ekin, st.dU)
        # A(1/2)
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry, st.vx, st.vy, dtT, st.box2::Definitions.Box2)
        # O(1): OU using pre-generated noise (reuse VV noise draw)
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale)
        LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy, st.rf_x, st.rf_y, spec.params, dtT, st.dq)
        # A(1/2)
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry, st.vx, st.vy, dtT, st.box2::Definitions.Box2)

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
        end

        # Conservative power like VV at end of step (BAOA has no final B)
        LangevinIntegrators.cons_power_2d!(st.vx, st.vy, st.fx, st.fy, st.dU)

        # Update Ekin at end (no extra B)
        LangevinIntegrators.baoab_B_2d!(st.vx, st.vy, st.fx, st.fy, spec.params, T(0), st.Ekin, st.dU)
    else
        # 3D variant
        LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, spec.params, T(2)*dtT, st.Ekin, st.dU)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, dtT, st.box3::Definitions.Box3)
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=st.rf_z)
        LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz, st.rf_x, st.rf_y, st.rf_z, spec.params, dtT, st.dq)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, dtT, st.box3::Definitions.Box3)

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
                if compute_energy
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                                    st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                         st.nbh, st.box3::Definitions.Box3, st.pair_lj)
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
    if D == 2
        LangevinIntegrators.baoab_BA_2d!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y, bao, dtT, st.box2::Definitions.Box2)
        # Prepare OU noise using the same generator as VV (β = s * N(0,1))
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale)
        LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy, st.rf_x, st.rf_y, bao, dtT, st.dq)
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry, st.vx, st.vy, dtT, st.box2::Definitions.Box2)

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
        LangevinIntegrators.baoab_B_2d!(st.vx, st.vy, st.fx, st.fy, bao, dtT, st.Ekin, st.dU)
    else
        LangevinIntegrators.baoab_BA_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, bao, dtT, st.box3::Definitions.Box3)
        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=st.rf_z)
        LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz, st.rf_x, st.rf_y, st.rf_z, bao, dtT, st.dq)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, dtT, st.box3::Definitions.Box3)

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
    st.last_integrator = UInt8(2)
    D = st.rz === nothing ? 2 : 3

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
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
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
                NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                @assert st.softrep !== nothing "softrep params missing"
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.softrep)
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
                NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                @assert st.softrep !== nothing "softrep params missing"
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.softrep)
            end
        end
    end

    if D == 2
        # Draw noise once and compute midpoint positions into vx,vy
        BrownianIntegrators.bd_prepare_noise_2d!(st.rf_x, st.rf_y)
        BrownianIntegrators.bd_midpoint_positions_2d!(
            st.rx, st.ry, st.fx, st.fy,
            st.rf_x, st.rf_y,
            st.vx, st.vy,
            bp.gamma, bp.noise_scale,
            dtT, st.box2::Definitions.Box2)
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
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
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
        # Finalize step using forces at midpoint (in f0*) and same noise
        BrownianIntegrators.bd_finish_step_2d!(
            st.rx, st.ry, st.f0x, st.f0y,
            st.rf_x, st.rf_y,
            bp.gamma, bp.noise_scale,
            dtT, st.dq, st.dU, st.box2::Definitions.Box2)
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
    else
        BrownianIntegrators.bd_prepare_noise_3d!(st.rf_x, st.rf_y, st.rf_z)
        BrownianIntegrators.bd_midpoint_positions_3d!(
            st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
            st.rf_x, st.rf_y, st.rf_z,
            st.vx, st.vy, st.vz,
            bp.gamma, bp.noise_scale,
            dtT, st.box3::Definitions.Box3)
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
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
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
        BrownianIntegrators.bd_finish_step_3d!(
            st.rx, st.ry, st.rz,
            st.f0x, st.f0y, st.f0z,
            st.rf_x, st.rf_y, st.rf_z,
            bp.gamma, bp.noise_scale,
            dtT, st.dq, st.dU, st.box3::Definitions.Box3)
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
    st.last_integrator = UInt8(2)
    D = st.rz === nothing ? 2 : 3

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
            else
                NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box = st.box3, step=st.step)
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
                if st.bonds === nothing
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
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
                if st.bonds === nothing
                    NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
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
    end

    # Position update and dq/dU accumulation using midpoint so EPR == UPR
    if D == 2
        # Predictor: draw noise then compute midpoint positions into vx,vy
        BrownianIntegrators.bd_prepare_noise_2d!(st.rf_x, st.rf_y)
        BrownianIntegrators.bd_midpoint_positions_2d!(
            st.rx, st.ry, st.fx, st.fy,
            st.rf_x, st.rf_y,
            st.vx, st.vy,
            em.gamma, em.noise_scale,
            dtT, st.box2::Definitions.Box2)
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
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.f0x, st.f0y, st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
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
        # Finish step: advance positions with midpoint force; accumulate dq and dU equally
        BrownianIntegrators.bd_finish_step_2d!(
            st.rx, st.ry, st.f0x, st.f0y,
            st.rf_x, st.rf_y,
            em.gamma, em.noise_scale,
            dtT, st.dq, st.dU, st.box2::Definitions.Box2)
        # New forces
        if compute_energy
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
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
                NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                @assert st.softrep !== nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2::Definitions.Box2, st.softrep)
            end
            if st.bonds !== nothing
                _apply_bonds2!(st, st.fx, st.fy, nothing, false)
            end
        end
    else
        # Predictor: draw noise then compute midpoint positions into vx,vy,vz
        BrownianIntegrators.bd_prepare_noise_3d!(st.rf_x, st.rf_y, st.rf_z)
        BrownianIntegrators.bd_midpoint_positions_3d!(
            st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
            st.rf_x, st.rf_y, st.rf_z,
            st.vx, st.vy, st.vz,
            em.gamma, em.noise_scale,
            dtT, st.box3::Definitions.Box3)
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
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa_noE!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_noE_excl!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z, st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
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
        # Finish step: advance positions with midpoint force; accumulate dq and dU equally
        BrownianIntegrators.bd_finish_step_3d!(
            st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
            st.rf_x, st.rf_y, st.rf_z,
            em.gamma, em.noise_scale,
            dtT, st.dq, st.dU, st.box3::Definitions.Box3)
        if compute_energy
            if st.nb_kind == NB_KIND_LJ
                if st.sigma_particle === nothing
                    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
                else
                    NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == NB_KIND_WCA
                NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
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
                NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                @assert st.softrep !== nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3::Definitions.Box3, st.softrep)
            end
            if st.bonds !== nothing
                _apply_bonds3!(st, st.fx, st.fy, st.fz, nothing, false)
            end
        end
    end

    st.step += 1
    return nothing
end


end # module
