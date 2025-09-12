module Simulation

using CUDA
using ..Definitions
using ..NeighborLists
using ..NonBondedForces
using ..Integrators

const NL_CHECK_STRIDE = 20  # only check NL rebuild every N steps to cut overhead

export SimulationState, build_simulation, step!, step_graph!, step_fused!, zero_forces!

# =========================
#   Simulation state (SoA)
# =========================
mutable struct SimulationState
    # SoA arrays
    rx::CuArray{Float32,1}; ry::CuArray{Float32,1}
    rz::Union{Nothing,CuArray{Float32,1}}
    vx::CuArray{Float32,1}; vy::CuArray{Float32,1}
    vz::Union{Nothing,CuArray{Float32,1}}
    fx::CuArray{Float32,1}; fy::CuArray{Float32,1}
    fz::Union{Nothing,CuArray{Float32,1}}

    # previous forces (to avoid per-step allocations)
    f0x::CuArray{Float32,1}; f0y::CuArray{Float32,1}
    f0z::Union{Nothing,CuArray{Float32,1}}

    # per-step random impulse (shared between pos/vel updates)
    rf_x::CuArray{Float32,1}; rf_y::CuArray{Float32,1}
    rf_z::Union{Nothing,CuArray{Float32,1}}

    # per-particle type id
    typeid::CuArray{Int32,1}

    # box (stored directly; no splatting)
    box2::Union{Definitions.Box2,Nothing}
    box3::Union{Definitions.Box3,Nothing}

    # neighbor list
    nbh::NeighborLists.NeighborMatrix
    neigh_interval::Int

    # pair params
    pair_lj::Definitions.LJParams{Float32}

    # integrator params
    vv::Integrators.VVParams{Float32}

    # observables buffers
    Epot::CuArray{Float32,1}
    dq::CuArray{Float32,1}
    Ekin::CuArray{Float32,1}

    # misc
    step::Int
end

function zero_forces!(st::SimulationState)
    fill!(st.fx, 0f0); fill!(st.fy, 0f0)
    st.fz === nothing || fill!(st.fz, 0f0)
    return nothing
end

# ==========================================
#  Top-level, non-capturing init kernels
#  (avoid nested functions / closures)
# ==========================================

function _init_vel2_kernel!(
    vx::CuDeviceVector{Float32},
    vy::CuDeviceVector{Float32},
    scale::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = randn(Float32) * scale
        vy[i] = randn(Float32) * scale
    end
    return
end

function _init_vel3_kernel!(
    vx::CuDeviceVector{Float32},
    vy::CuDeviceVector{Float32},
    vz::CuDeviceVector{Float32},
    scale::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = randn(Float32) * scale
        vy[i] = randn(Float32) * scale
        vz[i] = randn(Float32) * scale
    end
    return
end

# Host launchers
function _init_vel2!(vx::CuArray{Float32,1}, vy::CuArray{Float32,1}, scale::Float32)
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel2_kernel!(vx, vy, scale)
    CUDA.@sync k(vx, vy, scale; threads, blocks)
    return nothing
end

function _init_vel3!(vx::CuArray{Float32,1}, vy::CuArray{Float32,1}, vz::CuArray{Float32,1}, scale::Float32)
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel3_kernel!(vx, vy, vz, scale)
    CUDA.@sync k(vx, vy, vz, scale; threads, blocks)
    return nothing
end

# =========================
#   Build simulation
# =========================
function build_simulation(; D::Int, N::Int,
                           box,
                           cutoff::Float32,
                           skin::Float32=0.4f0,
                           cap::Int32=Int32(96),
                           neigh_interval::Int=20,
                           epsilon::Float32=1f0,
                           sigma::Float32=1f0,
                           gamma::Float32=1f0,
                           mass::Float32=1f0,
                           noise_scale::CuArray{Float32,1},
                           init_temperature::Float32 = 1f0)

    # Allocate SoA buffers
    rx = CUDA.CuArray{Float32}(undef, N); ry = CUDA.CuArray{Float32}(undef, N)
    vx = CUDA.CuArray{Float32}(undef, N); vy = CUDA.CuArray{Float32}(undef, N)
    fx = CUDA.CuArray{Float32}(undef, N); fy = CUDA.CuArray{Float32}(undef, N)
    rz = nothing; vz = nothing; fz = nothing

    # previous forces
    f0x = CUDA.CuArray{Float32}(undef, N)
    f0y = CUDA.CuArray{Float32}(undef, N)
    f0z = nothing

    # per-step random impulse
    rf_x = CUDA.CuArray{Float32}(undef, N)
    rf_y = CUDA.CuArray{Float32}(undef, N)
    rf_z = nothing

    if D == 3
        rz  = CUDA.CuArray{Float32}(undef, N)
        vz  = CUDA.CuArray{Float32}(undef, N)
        fz  = CUDA.CuArray{Float32}(undef, N)
        f0z = CUDA.CuArray{Float32}(undef, N)
        rf_z = CUDA.CuArray{Float32}(undef, N)
    end

    fill!(rx, 0f0); fill!(ry, 0f0)
    rz === nothing || fill!(rz, 0f0)

    fill!(fx, 0f0); fill!(fy, 0f0); fz === nothing || fill!(fz, 0f0)
    fill!(f0x, 0f0); fill!(f0y, 0f0); f0z === nothing || fill!(f0z, 0f0)
    fill!(rf_x, 0f0); fill!(rf_y, 0f0); rf_z === nothing || fill!(rf_z, 0f0)

    # Maxwell-Boltzmann initial velocities on GPU
    scale = sqrt(init_temperature)
    if D == 2
        _init_vel2!(vx, vy, scale)
    else
        _init_vel3!(vx, vy, vz, scale)
    end

    typeid = CUDA.fill(Int32(1), N)

    # Neighbors
    if D == 2
        nbh = NeighborLists.build_neighbors_dense!(rx, ry; box=(box[1], box[2]), cutoff, cap, skin)
    else
        nbh = NeighborLists.build_neighbors_dense!(rx, ry, rz; box=(box[1], box[2], box[3]), cutoff, cap, skin)
    end

    lj = Definitions.LJParams{Float32}(epsilon, sigma, cutoff)
    vv = Integrators.VVParams{Float32}(gamma, mass, noise_scale)

    Epot = CUDA.CuArray{Float32}(undef, N); fill!(Epot, 0f0)
    dq   = CUDA.CuArray{Float32}(undef, N); fill!(dq, 0f0)
    Ekin   = CUDA.CuArray{Float32}(undef, N); fill!(Ekin, 0f0)

    # Construct with boxes set to nothing; assign after
    st = SimulationState(rx, ry, rz, vx, vy, vz, fx, fy, fz,
                         f0x, f0y, f0z,
                         rf_x, rf_y, rf_z,
                         typeid,
                         nothing,   # box2
                         nothing,   # box3
                         nbh, neigh_interval, lj, vv,
                         Epot, dq, Ekin, 0)

    # Assign the appropriate box directly (no extra tuple layer)
    if D == 2
        st.box2 = box  # ::Tuple{Float32,Float32}
        st.box3 = nothing
    else
        st.box2 = nothing
        st.box3 = box  # ::Tuple{Float32,Float32,Float32}
    end

    return st
end

# =========================
#   One integrator step
# =========================
function step!(st::SimulationState, dt::Float32; compute_energy::Bool=true)
    D = st.rz === nothing ? 2 : 3

    # NL rebuild policy using new displacement-based algorithm only
    do_check = (st.step % NL_CHECK_STRIDE == 0)
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
            if compute_energy
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            end
        else
            if compute_energy
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            end
        end
    end

    # Prepare noise ONCE for the step and reuse in both updates
    if D == 2
        Integrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=nothing)
        Integrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                      st.rf_x, st.rf_y, st.vv, dt, st.box2::Definitions.Box2)
    else
        Integrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=st.rf_z)
        Integrators.vv_positions_soa!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz,
                                      st.f0x, st.f0y, st.f0z,
                                      st.rf_x, st.rf_y, st.rf_z, st.vv, dt, st.box3::Definitions.Box3)
    end

    # Forces at t + dt (write into fx,fy[,fz])
    if D == 2
        if compute_energy
            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                           st.nbh, st.box2::Definitions.Box2, st.pair_lj)
        else
            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
        end
        Integrators.vv_velocities_soa!(st.vx, st.vy, st.f0x, st.f0y, st.fx, st.fy,
                                       st.rf_x, st.rf_y, st.dq, st.Ekin, st.vv, dt)
    else
        if compute_energy
            NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                           st.nbh, st.box3::Definitions.Box3, st.pair_lj)
        else
            NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
        end
        Integrators.vv_velocities_soa!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                       st.fx, st.fy, st.fz,
                                       st.rf_x, st.rf_y, st.rf_z,
                                       st.dq, st.Ekin, st.vv, dt)
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
function step_graph!(st::SimulationState, dt::Float32; compute_energy::Bool=true)
    D = st.rz === nothing ? 2 : 3

    # NL rebuild decision outside graph
    do_check = (st.step % NL_CHECK_STRIDE == 0)
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
            if compute_energy
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.f0x, st.f0y,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            end
        else
            if compute_energy
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z, st.Epot,
                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.f0x, st.f0y, st.f0z,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            end
        end
    end

    if D == 2
        CUDA.@captured begin
            Integrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=nothing)
            Integrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                          st.rf_x, st.rf_y, st.vv, dt, st.box2::Definitions.Box2)
            if compute_energy
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            end
            Integrators.vv_velocities_soa!(st.vx, st.vy, st.f0x, st.f0y, st.fx, st.fy,
                                           st.rf_x, st.rf_y, st.dq, st.Ekin, st.vv, dt)
        end
    else
        CUDA.@captured begin
            Integrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=st.rf_z)
            Integrators.vv_positions_soa!(st.rx, st.ry, st.rz, st.vx, st.vy, st.vz,
                                          st.f0x, st.f0y, st.f0z,
                                          st.rf_x, st.rf_y, st.rf_z, st.vv, dt, st.box3::Definitions.Box3)
            if compute_energy
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot,
                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            end
            Integrators.vv_velocities_soa!(st.vx, st.vy, st.vz, st.f0x, st.f0y, st.f0z,
                                           st.fx, st.fy, st.fz,
                                           st.rf_x, st.rf_y, st.rf_z,
                                           st.dq, st.Ekin, st.vv, dt)
        end
    end

    st.step += 1
    return nothing
end

end # module
