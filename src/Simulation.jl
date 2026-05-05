module Simulation

using CUDA
import ..IntegratorInterfaces
import ..IntegratorInterfaces: AbstractIntegratorSpec,
                               validate_integrator_inputs!,
                               ensure_integrator_workspace!,
                               integrator_id,
                               integrator_name,
                               stage_sequence,
                               execute_integrator_stage!,
                               collect_integrator_observables
using ..Definitions
using ..Backends
using ..NeighborLists
using ..NonBondedForces
using ..NonBondedInteractions
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

export SimulationState, build_simulation, step!, step_graph!, zero_forces!, sync_unwrapped!, accumulate_virial!, virial_components, virial_tensor
export run_integrator_step!, collect_step_observables, thermostatted_dof, thermostatted_particle_mask
export reset_bath_exchange_accumulators!
export IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec
export velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nosehooverchain, csvr

include("SimulationState.jl")

# -------------------------
# Unified integrator specs
# -------------------------
const IntegratorSpec = AbstractIntegratorSpec

# Stage 8: Extract integrator specs and constructors
include("simulation/StochasticWorkspace.jl")
include("simulation/IntegratorSpecs.jl")
include("simulation/LangevinSpecConstructors.jl")
include("simulation/BrownianSpecConstructors.jl")
include("simulation/ThermostatSpecConstructors.jl")

Backends.storage_backend(::SimulationState) = Backends.CUDABackend()

@inline function _device_particle_buffer(backend::Backends.AbstractBackend,
                                         ::Type{T},
                                         N::Integer,
                                         value::Union{AbstractVector{<:Real},Real},
                                         name::AbstractString) where {T<:AbstractFloat}
    if value isa Real
        return Backends.fill_vector(backend, T(value), N)
    end
    length(value) == N ||
        throw(ArgumentError("$(name) vector must have length $(N), got $(length(value))."))
    return Backends.from_host(backend, T.(value))
end

@inline function _device_corr_time_buffer(backend::Backends.AbstractBackend,
                                          ::Type{T},
                                          N::Integer,
                                          value::Union{Nothing,AbstractVector{<:Real},Real}) where {T<:AbstractFloat}
    value === nothing && return nothing
    return _device_particle_buffer(backend, T, N, value, "noise_corr_time")
end

@inline function _all_particle_indices(backend::Backends.AbstractBackend, N::Integer)
    return Backends.from_host(backend, Int32.(collect(1:N)))
end

@inline function _mode_vector(::Type{T},
                              value::Union{AbstractVector{<:Real},Real},
                              target::Int,
                              name::AbstractString) where {T<:AbstractFloat}
    if value isa Real
        return fill(T(value), target)
    end
    vals = T.(collect(value))
    length(vals) == target ||
        throw(ArgumentError("$(name) must have length $(target), got $(length(vals))."))
    return vals
end

@inline function _canonical_mode_vectors(::Type{T},
                                         taus::Union{AbstractVector{<:Real},Real},
                                         scales::Union{AbstractVector{<:Real},Real}) where {T<:AbstractFloat}
    tau_vals = taus isa Real ? T[T(taus)] : T.(collect(taus))
    scale_vals = scales isa Real ? T[T(scales)] : T.(collect(scales))
    M = max(length(tau_vals), length(scale_vals))
    tau_vals = _mode_vector(T, tau_vals, M, "OU taus")
    scale_vals = _mode_vector(T, scale_vals, M, "OU scales")
    return tau_vals, scale_vals
end

@inline function _ou_coefficients(backend::Backends.AbstractBackend,
                                  ::Type{T},
                                  dt::T,
                                  tau::AbstractMatrix{T},
                                  scale::AbstractMatrix{T}) where {T<:AbstractFloat}
    coeff_a = Matrix{T}(undef, size(tau))
    coeff_c = Matrix{T}(undef, size(scale))
    @inbounds for j in axes(tau, 2), i in axes(tau, 1)
        τ = tau[i, j]
        s = scale[i, j]
        if τ <= zero(T)
            coeff_a[i, j] = zero(T)
            coeff_c[i, j] = s
        else
            a = exp(-dt / τ)
            coeff_a[i, j] = a
            coeff_c[i, j] = s * sqrt(max(one(T) - a * a, zero(T)))
        end
    end
    return Backends.from_host(backend, coeff_a), Backends.from_host(backend, coeff_c)
end

function _build_single_mode_ou(backend::Backends.AbstractBackend,
                               ::Type{T},
                               noise_scale::CuArray{T,1},
                               corr::CuArray{T,1},
                               dt::Real) where {T<:AbstractFloat}
    corr_host = Array(corr)
    idx_host = findall(!iszero, corr_host)
    isempty(idx_host) && return nothing

    scale_host = Array(noise_scale)
    tau_mat = reshape(T.(corr_host[idx_host]), 1, :)
    scale_mat = reshape(T.(scale_host[idx_host]), 1, :)
    active_idx = Backends.from_host(backend, Int32.(idx_host))
    coeff_a, coeff_c = _ou_coefficients(backend, T, T(dt), tau_mat, scale_mat)
    return Definitions.OUSpectrum{T}(T(dt), active_idx,
                                     Backends.from_host(backend, tau_mat),
                                     Backends.from_host(backend, scale_mat),
                                     coeff_a, coeff_c)
end

function _build_mode_ou(backend::Backends.AbstractBackend,
                        ::Type{T},
                        active_idx::CuArray{Int32,1},
                        taus::Union{AbstractVector{<:Real},Real},
                        scales::Union{AbstractVector{<:Real},Real},
                        dt::Real) where {T<:AbstractFloat}
    K = length(active_idx)
    K == 0 && return nothing
    tau_vals, scale_vals = _canonical_mode_vectors(T, taus, scales)
    tau_mat = repeat(reshape(tau_vals, :, 1), 1, K)
    scale_mat = repeat(reshape(scale_vals, :, 1), 1, K)
    coeff_a, coeff_c = _ou_coefficients(backend, T, T(dt), tau_mat, scale_mat)
    return Definitions.OUSpectrum{T}(T(dt), active_idx,
                                     Backends.from_host(backend, tau_mat),
                                     Backends.from_host(backend, scale_mat),
                                     coeff_a, coeff_c)
end

@inline function _compat_corr_time(backend::Backends.AbstractBackend,
                                   ::Type{T},
                                   N::Integer,
                                   taus::AbstractVector{T},
                                   scales::AbstractVector{T}) where {T<:AbstractFloat}
    length(taus) == 1 && length(scales) == 1 || return nothing
    return Backends.fill_vector(backend, taus[1], N)
end

function _refresh_ou_coefficients!(ou::Definitions.OUSpectrum{T}, dt::T) where {T<:AbstractFloat}
    ou.dt == dt && return ou
    coeff_a, coeff_c = _ou_coefficients(Backends.CUDABackend(), T, dt, Array(ou.tau), Array(ou.scale))
    copyto!(ou.coeff_a, coeff_a)
    copyto!(ou.coeff_c, coeff_c)
    ou.dt = dt
    return ou
end

@inline _is_3d(st::SimulationState) = st.rz !== nothing

@inline function _stage_tracing_enabled()
    flag = lowercase(get(ENV, "NEQSIMGPU_STAGE_TRACE", "0"))
    return flag == "1" || flag == "true" || flag == "yes" || flag == "on"
end

"""
    _trace_integrator_stage!(st, spec, stage_tag; force_evaluated=false,
                             rebuild_applied=false)

Emit an optional per-stage trace line for debugging and diagnostics. Tracing is
controlled by `ENV["NEQSIMGPU_STAGE_TRACE"]` and is disabled by default.
"""
function _trace_integrator_stage!(st::SimulationState,
                                  spec::IntegratorSpec,
                                  stage_tag::Symbol;
                                  force_evaluated::Bool=false,
                                  rebuild_applied::Bool=false)
    _stage_tracing_enabled() || return nothing
    @info "integrator stage" integrator=integrator_name(spec) stage=stage_tag step=st.step force_evaluated rebuild_applied
    return nothing
end

"""
    _ensure_workspace_buffers!(workspace, st; require_ou)

Ensure integrator-local stochastic buffers are allocated with the correct shape
for the current simulation state.
"""
function _ensure_workspace_buffers!(workspace::StochasticWorkspace{T},
                                    st::SimulationState{T};
                                    ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat}
    N = length(st.rx)

    if length(workspace.rf_x) != N
        workspace.rf_x = CUDA.CuArray{T}(undef, N)
        fill!(workspace.rf_x, zero(T))
    end
    if length(workspace.rf_y) != N
        workspace.rf_y = CUDA.CuArray{T}(undef, N)
        fill!(workspace.rf_y, zero(T))
    end

    if _is_3d(st)
        if workspace.rf_z === nothing || length(workspace.rf_z) != N
            workspace.rf_z = CUDA.CuArray{T}(undef, N)
            fill!(workspace.rf_z, zero(T))
        end
    else
        workspace.rf_z = nothing
    end

    if ou !== nothing
        M, K = size(ou.coeff_a)
        if workspace.ou_x === nothing || size(workspace.ou_x) != (M, K)
            workspace.ou_x = CUDA.zeros(T, M, K)
        end
        if workspace.ou_y === nothing || size(workspace.ou_y) != (M, K)
            workspace.ou_y = CUDA.zeros(T, M, K)
        end
        if _is_3d(st)
            if workspace.ou_z === nothing || size(workspace.ou_z) != (M, K)
                workspace.ou_z = CUDA.zeros(T, M, K)
            end
        else
            workspace.ou_z = nothing
        end
    else
        workspace.ou_x = nothing
        workspace.ou_y = nothing
        workspace.ou_z = nothing
    end

    return nothing
end

include("simulation/Freeze.jl")

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

include("SimulationBuild.jl")

# TODO: move these includes under observables/ once the Simulation module split is complete.
include("simulation/Virial.jl")
include("simulation/EnergyAccumulation.jl")
include("simulation/SteppingEngine.jl")
include("simulation/ForceEvaluation.jl")
include("simulation/NeighborRebuild.jl")

"""
    _prepare_langevin_noise!(spec, st, dt)

Draw Langevin stochastic impulses into the spec workspace.
"""
function _prepare_langevin_noise!(params::Union{LangevinIntegrators.VVParams{T},LangevinIntegrators.BAOABParams{T}},
                                  workspace::StochasticWorkspace{T},
                                  st::SimulationState{T},
                                  dt::T) where {T<:AbstractFloat}
    params.ou !== nothing && _refresh_ou_coefficients!(params.ou, dt)
    if _is_3d(st)
        LangevinIntegrators.vv_prepare_noise!(workspace.rf_x, workspace.rf_y, params.noise_scale;
                                              beta_z=workspace.rf_z,
                                              ou=params.ou,
                                              state_x=workspace.ou_x,
                                              state_y=workspace.ou_y,
                                              state_z=workspace.ou_z)
    else
        LangevinIntegrators.vv_prepare_noise!(workspace.rf_x, workspace.rf_y, params.noise_scale;
                                              beta_z=nothing,
                                              ou=params.ou,
                                              state_x=workspace.ou_x,
                                              state_y=workspace.ou_y,
                                              state_z=nothing)
    end
    return nothing
end

"""
    _prepare_brownian_noise!(params, workspace, st, dt)

Draw Brownian noise into the spec workspace.
"""
function _prepare_brownian_noise!(params::Union{BrownianIntegrators.BrownianParams{T},BrownianIntegrators.EMParams{T}},
                                  workspace::StochasticWorkspace{T},
                                  st::SimulationState{T},
                                  dt::T) where {T<:AbstractFloat}
    params.ou !== nothing && _refresh_ou_coefficients!(params.ou, dt)
    if _is_3d(st)
        BrownianIntegrators.bd_prepare_noise_3d!(workspace.rf_x, workspace.rf_y, workspace.rf_z;
                                                 noise_scale=params.noise_scale,
                                                 ou=params.ou,
                                                 state_x=workspace.ou_x,
                                                 state_y=workspace.ou_y,
                                                 state_z=workspace.ou_z)
    else
        BrownianIntegrators.bd_prepare_noise_2d!(workspace.rf_x, workspace.rf_y;
                                                 noise_scale=params.noise_scale,
                                                 ou=params.ou,
                                                 state_x=workspace.ou_x,
                                                 state_y=workspace.ou_y)
    end
    return nothing
end

"""
    _refresh_kinetic_buffer!(st, mass)

Refresh the per-particle kinetic-energy buffer `st.Ekin` from the current
velocity field.
"""
function _refresh_kinetic_buffer!(st::SimulationState{T}, mass::T) where {T<:AbstractFloat}
    if _is_3d(st)
        @. st.Ekin = T(0.5) * mass * (st.vx * st.vx + st.vy * st.vy + st.vz * st.vz)
    else
        @. st.Ekin = T(0.5) * mass * (st.vx * st.vx + st.vy * st.vy)
    end
    return nothing
end

"""
    _refresh_kinetic_energy!(st, mass) -> T

Compatibility helper that refreshes `st.Ekin` and returns its total on host.
"""
function _refresh_kinetic_energy!(st::SimulationState{T}, mass::T) where {T<:AbstractFloat}
    _refresh_kinetic_buffer!(st, mass)
    return T(CUDA.sum(st.Ekin))
end

function _sum_into_scalar_kernel!(out::CuDeviceVector{T},
                                  src::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(src)
        @inbounds CUDA.@atomic out[1] += src[i]
    end
    return nothing
end

"""
    _reduce_sum_to_device!(out, src)

Reduce `src` into the one-element device buffer `out` without host transfers.
"""
function _reduce_sum_to_device!(out::CuArray{T,1},
                                src::CuArray{T,1}) where {T<:AbstractFloat}
    fill!(out, zero(T))
    N = length(src)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _sum_into_scalar_kernel!(out, src)
    k(out, src; threads, blocks)
    return nothing
end

function _nhc_active_bath_counts_kernel!(bath_counts::CuDeviceVector{Int32},
                                         particle_bath_id::CuDeviceVector{Int32})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(particle_bath_id)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(bath_counts)
                CUDA.@atomic bath_counts[b] += Int32(1)
            end
        end
    end
    return nothing
end

function _nhc_active_bath_counts_with_freeze_kernel!(bath_counts::CuDeviceVector{Int32},
                                                     particle_bath_id::CuDeviceVector{Int32},
                                                     freeze_mask::CuDeviceVector{UInt8})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(particle_bath_id)
        @inbounds begin
            if freeze_mask[i] == UInt8(0)
                b = Int(particle_bath_id[i])
                if 1 <= b <= length(bath_counts)
                    CUDA.@atomic bath_counts[b] += Int32(1)
                end
            end
        end
    end
    return nothing
end

function _nhc_finalize_dof_kernel!(dof_per_bath::CuDeviceVector{T},
                                   bath_counts::CuDeviceVector{Int32},
                                   dim::T) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if b <= length(dof_per_bath)
        @inbounds dof_per_bath[b] = dim * T(bath_counts[b])
    end
    return nothing
end

function _update_bath_dof!(bath_counts::CuArray{Int32,1},
                           dof_per_bath::CuArray{T,1},
                           particle_bath_id::CuArray{Int32,1},
                           st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(bath_counts, Int32(0))
    fill!(dof_per_bath, zero(T))

    N = length(particle_bath_id)
    if N > 0
        threads = min(256, N)
        blocks = cld(N, threads)
        if st.freeze_mode != FREEZE_NONE && st.freeze_mask !== nothing
            k = CUDA.@cuda launch=false _nhc_active_bath_counts_with_freeze_kernel!(bath_counts, particle_bath_id, st.freeze_mask::CuArray{UInt8,1})
            k(bath_counts, particle_bath_id, st.freeze_mask::CuArray{UInt8,1}; threads, blocks)
        else
            k = CUDA.@cuda launch=false _nhc_active_bath_counts_kernel!(bath_counts, particle_bath_id)
            k(bath_counts, particle_bath_id; threads, blocks)
        end
    end

    B = length(dof_per_bath)
    if B > 0
        threads = min(256, B)
        blocks = cld(B, threads)
        dim = T(_is_3d(st) ? 3 : 2)
        k = CUDA.@cuda launch=false _nhc_finalize_dof_kernel!(dof_per_bath, bath_counts, dim)
        k(dof_per_bath, bath_counts, dim; threads, blocks)
    end
    return nothing
end

function _nhc_update_dof_per_bath!(spec::NHCSpec{T},
                                   st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    _update_bath_dof!(ws.bath_counts, ws.dof_per_bath, ws.particle_bath_id, st)
    ws.dof_dirty = false
    return nothing
end

function _csvr_update_dof_per_bath!(spec::CSVRSpec{T},
                                    st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    _update_bath_dof!(ws.bath_counts, ws.dof_per_bath, ws.particle_bath_id, st)
    ws.dof_dirty = false
    return nothing
end

function _nhc_reduce_kinetic_by_bath_kernel!(kinetic_total_per_bath::CuDeviceVector{T},
                                             Ekin::CuDeviceVector{T},
                                             particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Ekin)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += Ekin[i]
            end
        end
    end
    return nothing
end

function _nhc_reduce_kinetic_by_bath!(kinetic_total_per_bath::CuArray{T,1},
                                      Ekin::CuArray{T,1},
                                      particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    fill!(kinetic_total_per_bath, zero(T))
    N = length(Ekin)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _nhc_reduce_kinetic_by_bath_kernel!(kinetic_total_per_bath, Ekin, particle_bath_id)
    k(kinetic_total_per_bath, Ekin, particle_bath_id; threads, blocks)
    return nothing
end

"""
    _ensure_nhc_kinetic_initialized!(spec, st)

Initialize per-bath kinetic totals and `st.Ekin` once before the first NHC
thermostat stage if no prior kick/thermostat stage has populated them.
"""
function _ensure_nhc_kinetic_initialized!(spec::NHCSpec{T},
                                          st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    if !ws.kinetic_initialized
        _refresh_kinetic_buffer!(st, spec.params.mass)
        _nhc_reduce_kinetic_by_bath!(ws.kinetic_total_per_bath, st.Ekin, ws.particle_bath_id)
        ws.kinetic_initialized = true
    end
    return nothing
end

function _ensure_csvr_kinetic_initialized!(spec::CSVRSpec{T},
                                           st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    if !ws.kinetic_initialized
        _refresh_kinetic_buffer!(st, spec.params.mass)
        _nhc_reduce_kinetic_by_bath!(ws.kinetic_total_per_bath, st.Ekin, ws.particle_bath_id)
        ws.kinetic_initialized = true
    end
    return nothing
end

function _nhc_apply_stage_scale2_by_bath_kernel!(vx::CuDeviceVector{T},
                                                 vy::CuDeviceVector{T},
                                                 Ekin::CuDeviceVector{T},
                                                 stage_scale_per_bath::CuDeviceVector{T},
                                                 particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(stage_scale_per_bath)
                s = stage_scale_per_bath[b]
                s2 = s * s
                vx[i] *= s
                vy[i] *= s
                Ekin[i] *= s2
            end
        end
    end
    return nothing
end

function _nhc_apply_stage_scale3_by_bath_kernel!(vx::CuDeviceVector{T},
                                                 vy::CuDeviceVector{T},
                                                 vz::CuDeviceVector{T},
                                                 Ekin::CuDeviceVector{T},
                                                 stage_scale_per_bath::CuDeviceVector{T},
                                                 particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(stage_scale_per_bath)
                s = stage_scale_per_bath[b]
                s2 = s * s
                vx[i] *= s
                vy[i] *= s
                vz[i] *= s
                Ekin[i] *= s2
            end
        end
    end
    return nothing
end

"""
    _nhc_apply_stage_scale!(st, stage_scale_per_bath, particle_bath_id)

Apply the net thermostat scale per bath for one NHC stage to velocities and to
`st.Ekin` in a single GPU pass.
"""
function _nhc_apply_stage_scale!(st::SimulationState{T},
                                 stage_scale_per_bath::CuArray{T,1},
                                 particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    N = length(st.vx)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    if _is_3d(st)
        k = CUDA.@cuda launch=false _nhc_apply_stage_scale3_by_bath_kernel!(st.vx, st.vy, st.vz::CuArray{T,1}, st.Ekin, stage_scale_per_bath, particle_bath_id)
        k(st.vx, st.vy, st.vz::CuArray{T,1}, st.Ekin, stage_scale_per_bath, particle_bath_id; threads, blocks)
    else
        k = CUDA.@cuda launch=false _nhc_apply_stage_scale2_by_bath_kernel!(st.vx, st.vy, st.Ekin, stage_scale_per_bath, particle_bath_id)
        k(st.vx, st.vy, st.Ekin, stage_scale_per_bath, particle_bath_id; threads, blocks)
    end
    return nothing
end

@inline function _csvr_gamma_sample(shape::T) where {T<:AbstractFloat}
    shape > zero(T) || return zero(T)
    if shape < one(T)
        u = max(rand(T), eps(T))
        return _csvr_gamma_sample(shape + one(T)) * u^(inv(shape))
    end

    d = shape - T(1) / T(3)
    c = inv(sqrt(T(9) * d))
    while true
        x = randn(T)
        v = one(T) + c * x
        v <= zero(T) && continue
        v3 = v * v * v
        u = rand(T)
        x2 = x * x
        if u < one(T) - T(0.0331) * x2 * x2
            return d * v3
        end
        if log(max(u, eps(T))) < T(0.5) * x2 + d * (one(T) - v3 + log(v3))
            return d * v3
        end
    end
end

@inline function _csvr_chisq(::Type{T}, dof::Int) where {T<:AbstractFloat}
    dof <= 0 && return zero(T)
    return T(2) * _csvr_gamma_sample(T(dof) / T(2))
end

function _csvr_thermostat_stage_kernel!(kinetic_total_per_bath::CuDeviceVector{T},
                                        cumulative_energy_exchange_per_bath::CuDeviceVector{T},
                                        last_velocity_scale_per_bath::CuDeviceVector{T},
                                        dof_per_bath::CuDeviceVector{T},
                                        target_temperature::CuDeviceVector{T},
                                        tau::CuDeviceVector{T},
                                        stage_dt::T) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            Kold = kinetic_total_per_bath[b]
            Ttarget = target_temperature[b]
            τ = tau[b]
            if !(Kold > eps(T)) || !(Ttarget > zero(T)) || !(τ > zero(T))
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            ndof = max(1, Int(floor(dof + T(0.5))))
            Kbar = T(0.5) * dof * Ttarget
            c1 = exp(-stage_dt / τ)
            c2 = ((one(T) - c1) * Kbar) / (Kold * T(ndof))
            r1 = randn(T)
            r2 = _csvr_chisq(T, ndof - 1)
            cross = sqrt(max(c1 * c2, zero(T)))
            scale2 = c1 + c2 * (r1 * r1 + r2) + T(2) * r1 * cross
            scale2 = max(scale2, zero(T))
            scale = sqrt(scale2)
            Knew = Kold * scale2

            kinetic_total_per_bath[b] = Knew
            cumulative_energy_exchange_per_bath[b] += Kold - Knew
            last_velocity_scale_per_bath[b] = scale
        end
    end
    return nothing
end

function _run_csvr_thermostat_stage!(spec::CSVRSpec{T},
                                     stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    B = length(ws.kinetic_total_per_bath)
    B == 0 && return nothing
    threads = min(256, B)
    blocks = cld(B, threads)
    k = CUDA.@cuda launch=false _csvr_thermostat_stage_kernel!(ws.kinetic_total_per_bath,
                                                               ws.cumulative_energy_exchange_per_bath,
                                                               ws.last_velocity_scale_per_bath,
                                                               ws.dof_per_bath,
                                                               ws.target_temperature,
                                                               ws.tau,
                                                               stage_dt)
    k(ws.kinetic_total_per_bath,
      ws.cumulative_energy_exchange_per_bath,
      ws.last_velocity_scale_per_bath,
      ws.dof_per_bath,
      ws.target_temperature,
      ws.tau,
      stage_dt;
      threads,
      blocks)
    return nothing
end

function _apply_csvr_thermostat_stage!(spec::CSVRSpec{T},
                                       st::SimulationState{T},
                                       stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    _csvr_update_dof_per_bath!(spec, st)
    _ensure_csvr_kinetic_initialized!(spec, st)
    _run_csvr_thermostat_stage!(spec, stage_dt)
    _nhc_apply_stage_scale!(st, ws.last_velocity_scale_per_bath, ws.particle_bath_id)
    return nothing
end

function _nhc_half_kick2_by_bath_kernel!(vx::CuDeviceVector{T},
                                         vy::CuDeviceVector{T},
                                         fx::CuDeviceVector{T},
                                         fy::CuDeviceVector{T},
                                         Ekin::CuDeviceVector{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         particle_bath_id::CuDeviceVector{Int32},
                                         coef::T,
                                         mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            vx_new = vx[i] + coef * fx[i]
            vy_new = vy[i] + coef * fy[i]
            ek = T(0.5) * mass * (vx_new * vx_new + vy_new * vy_new)
            vx[i] = vx_new
            vy[i] = vy_new
            Ekin[i] = ek
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += ek
            end
        end
    end
    return nothing
end

function _nhc_half_kick3_by_bath_kernel!(vx::CuDeviceVector{T},
                                         vy::CuDeviceVector{T},
                                         vz::CuDeviceVector{T},
                                         fx::CuDeviceVector{T},
                                         fy::CuDeviceVector{T},
                                         fz::CuDeviceVector{T},
                                         Ekin::CuDeviceVector{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         particle_bath_id::CuDeviceVector{Int32},
                                         coef::T,
                                         mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            vx_new = vx[i] + coef * fx[i]
            vy_new = vy[i] + coef * fy[i]
            vz_new = vz[i] + coef * fz[i]
            ek = T(0.5) * mass * (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new)
            vx[i] = vx_new
            vy[i] = vy_new
            vz[i] = vz_new
            Ekin[i] = ek
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += ek
            end
        end
    end
    return nothing
end

"""
    _nhc_apply_half_kick!(st, fx, fy, fz, dt, mass, kinetic_total_per_bath, particle_bath_id)

Apply a deterministic half-force kick `v <- v + (dt / (2m)) f`, refresh
`st.Ekin`, and reduce per-bath kinetic energies into `kinetic_total_per_bath`.
"""
function _nhc_apply_half_kick!(st::SimulationState{T},
                               fx::CuArray{T,1},
                               fy::CuArray{T,1},
                               fz::Union{Nothing,CuArray{T,1}},
                               dt::T,
                               mass::T,
                               kinetic_total_per_bath::CuArray{T,1},
                               particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    fill!(kinetic_total_per_bath, zero(T))
    N = length(st.vx)
    N == 0 && return nothing
    coef = dt / (T(2) * mass)
    threads = min(256, N)
    blocks = cld(N, threads)
    if _is_3d(st)
        k = CUDA.@cuda launch=false _nhc_half_kick3_by_bath_kernel!(st.vx, st.vy, st.vz::CuArray{T,1},
                                                                     fx, fy, fz::CuArray{T,1},
                                                                     st.Ekin, kinetic_total_per_bath,
                                                                     particle_bath_id,
                                                                     coef, mass)
        k(st.vx, st.vy, st.vz::CuArray{T,1},
          fx, fy, fz::CuArray{T,1},
          st.Ekin, kinetic_total_per_bath,
          particle_bath_id,
          coef, mass; threads, blocks)
    else
        k = CUDA.@cuda launch=false _nhc_half_kick2_by_bath_kernel!(st.vx, st.vy,
                                                                     fx, fy,
                                                                     st.Ekin, kinetic_total_per_bath,
                                                                     particle_bath_id,
                                                                     coef, mass)
        k(st.vx, st.vy,
          fx, fy,
          st.Ekin, kinetic_total_per_bath,
          particle_bath_id,
          coef, mass; threads, blocks)
    end
    return nothing
end

"""
    _nhc_drift_positions!(st, dt)

Advance positions by one full deterministic drift under periodic boundaries.
Implemented through the existing BAOAB A-kernel using `2dt` so the effective
drift is exactly `dt`.
"""
function _nhc_drift_positions!(st::SimulationState{T}, dt::T) where {T<:AbstractFloat}
    drift_dt = T(2) * dt
    if _is_3d(st)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                        st.vx, st.vy, st.vz,
                                        drift_dt, st.box3::Definitions.Box3;
                                        unwrapped_x=st.rx_unwrap,
                                        unwrapped_y=st.ry_unwrap,
                                        unwrapped_z=st.rz_unwrap)
    else
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                        st.vx, st.vy,
                                        drift_dt, st.box2::Definitions.Box2;
                                        unwrapped_x=st.rx_unwrap,
                                        unwrapped_y=st.ry_unwrap)
    end
    return nothing
end

@inline function _nhc_gromacs_sy_weight(::Type{T}, idx::Int32) where {T<:AbstractFloat}
    if idx == Int32(3)
        return T(-0.186929716880426)
    elseif idx >= Int32(1) && idx <= Int32(5)
        return T(0.2967324292201065)
    end
    return zero(T)
end

function _nhc_chain_stage_legacy_kernel!(xi::CuDeviceMatrix{T},
                                         eta::CuDeviceMatrix{T},
                                         chain_force::CuDeviceMatrix{T},
                                         chain_masses::CuDeviceMatrix{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         thermostat_kinetic_per_bath::CuDeviceVector{T},
                                         thermostat_potential_per_bath::CuDeviceVector{T},
                                         last_velocity_scale_per_bath::CuDeviceVector{T},
                                         dof_per_bath::CuDeviceVector{T},
                                         target_temperature::CuDeviceVector{T},
                                         stage_dt::T,
                                         substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            h = stage_dt / T(ns)
            half_h = h / T(2)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for _ in 1:ns
                if M == 1
                    g1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]
                    xi[1, b] += half_h * g1
                else
                    chain_force[1, b] = (T(2) * K - dof * Ttarget) / chain_masses[1, b] - xi[1, b] * xi[2, b]
                    for j in 2:(M - 1)
                        chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b] - xi[j, b] * xi[j + 1, b]
                    end
                    chain_force[M, b] = (chain_masses[M - 1, b] * xi[M - 1, b] * xi[M - 1, b] - Ttarget) / chain_masses[M, b]
                    for j in 1:M
                        xi[j, b] += half_h * chain_force[j, b]
                    end
                end

                scale = exp(-h * xi[1, b])
                total_scale *= scale
                K *= scale * scale

                if M == 1
                    g1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]
                    xi[1, b] += half_h * g1
                else
                    chain_force[1, b] = (T(2) * K - dof * Ttarget) / chain_masses[1, b] - xi[1, b] * xi[2, b]
                    for j in 2:(M - 1)
                        chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b] - xi[j, b] * xi[j + 1, b]
                    end
                    chain_force[M, b] = (chain_masses[M - 1, b] * xi[M - 1, b] * xi[M - 1, b] - Ttarget) / chain_masses[M, b]
                    for j in 1:M
                        xi[j, b] += half_h * chain_force[j, b]
                    end
                end

                for j in 1:M
                    eta[j, b] += h * xi[j, b]
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _nhc_chain_stage_gromacs_kernel!(xi::CuDeviceMatrix{T},
                                          eta::CuDeviceMatrix{T},
                                          chain_masses::CuDeviceMatrix{T},
                                          kinetic_total_per_bath::CuDeviceVector{T},
                                          thermostat_kinetic_per_bath::CuDeviceVector{T},
                                          thermostat_potential_per_bath::CuDeviceVector{T},
                                          last_velocity_scale_per_bath::CuDeviceVector{T},
                                          dof_per_bath::CuDeviceVector{T},
                                          target_temperature::CuDeviceVector{T},
                                          stage_dt::T,
                                          substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for _ in 1:ns
                for sy_idx in Int32(1):Int32(5)
                    time_step = stage_dt * _nhc_gromacs_sy_weight(T, sy_idx) / T(ns)

                    for j in M:-1:1
                        kinetic2 = if j == 1
                            T(2) * K
                        else
                            chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b]
                        end
                        num_dof = j == 1 ? dof : one(T)
                        xi_accel = (kinetic2 - num_dof * Ttarget) / chain_masses[j, b]
                        local_scale = if j < M
                            exp(-T(0.25) * time_step * xi[j + 1, b])
                        else
                            one(T)
                        end
                        xi[j, b] = local_scale * (xi[j, b] * local_scale + T(0.5) * time_step * xi_accel)
                    end

                    system_scale = exp(-time_step * xi[1, b])
                    total_scale *= system_scale
                    K *= system_scale * system_scale

                    for j in 1:M
                        eta[j, b] += time_step * xi[j, b]

                        kinetic2 = if j == 1
                            T(2) * K
                        else
                            chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b]
                        end
                        num_dof = j == 1 ? dof : one(T)
                        xi_accel = (kinetic2 - num_dof * Ttarget) / chain_masses[j, b]
                        local_scale = if j < M
                            exp(-T(0.25) * time_step * xi[j + 1, b])
                        else
                            one(T)
                        end
                        xi[j, b] = local_scale * (xi[j, b] * local_scale + T(0.5) * time_step * xi_accel)
                    end
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _nhc_chain_stage_lammps_kernel!(xi::CuDeviceMatrix{T},
                                         eta::CuDeviceMatrix{T},
                                         chain_force::CuDeviceMatrix{T},
                                         chain_masses::CuDeviceMatrix{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         thermostat_kinetic_per_bath::CuDeviceVector{T},
                                         thermostat_potential_per_bath::CuDeviceVector{T},
                                         last_velocity_scale_per_bath::CuDeviceVector{T},
                                         dof_per_bath::CuDeviceVector{T},
                                         target_temperature::CuDeviceVector{T},
                                         stage_dt::T,
                                         substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            h = stage_dt / T(ns)
            quarter_h = h / T(4)
            eighth_h = h / T(8)
            half_h = h / T(2)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for j in 2:M
                chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b]
            end

            for _ in 1:ns
                accel1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]

                for j in M:-1:2
                    expfac = j < M ? exp(-eighth_h * xi[j + 1, b]) : one(T)
                    xi[j, b] *= expfac
                    xi[j, b] += chain_force[j, b] * quarter_h
                    xi[j, b] *= expfac
                end

                expfac1 = M > 1 ? exp(-eighth_h * xi[2, b]) : one(T)
                xi[1, b] *= expfac1
                xi[1, b] += accel1 * quarter_h
                xi[1, b] *= expfac1

                scale = exp(-half_h * xi[1, b])
                total_scale *= scale
                K *= scale * scale

                accel1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]

                for j in 1:M
                    eta[j, b] += half_h * xi[j, b]
                end

                xi[1, b] *= expfac1
                xi[1, b] += accel1 * quarter_h
                xi[1, b] *= expfac1

                for j in 2:M
                    expfac = j < M ? exp(-eighth_h * xi[j + 1, b]) : one(T)
                    xi[j, b] *= expfac
                    chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b]
                    xi[j, b] += chain_force[j, b] * quarter_h
                    xi[j, b] *= expfac
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _run_nhc_chain_stage!(spec::NHCSpec{T},
                               stage_dt::T) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    B = length(ws.kinetic_total_per_bath)
    B == 0 && return nothing
    threads = min(256, B)
    blocks = cld(B, threads)
    if p.propagator == NHC_PROPAGATOR_LEGACY
        k = CUDA.@cuda launch=false _nhc_chain_stage_legacy_kernel!(ws.xi,
                                                                    ws.eta,
                                                                    ws.chain_force,
                                                                    ws.chain_masses,
                                                                    ws.kinetic_total_per_bath,
                                                                    ws.thermostat_kinetic_per_bath,
                                                                    ws.thermostat_potential_per_bath,
                                                                    ws.last_velocity_scale_per_bath,
                                                                    ws.dof_per_bath,
                                                                    ws.target_temperature,
                                                                    stage_dt,
                                                                    Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_force,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    elseif p.propagator == NHC_PROPAGATOR_GROMACS
        k = CUDA.@cuda launch=false _nhc_chain_stage_gromacs_kernel!(ws.xi,
                                                                     ws.eta,
                                                                     ws.chain_masses,
                                                                     ws.kinetic_total_per_bath,
                                                                     ws.thermostat_kinetic_per_bath,
                                                                     ws.thermostat_potential_per_bath,
                                                                     ws.last_velocity_scale_per_bath,
                                                                     ws.dof_per_bath,
                                                                     ws.target_temperature,
                                                                     stage_dt,
                                                                     Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    elseif p.propagator == NHC_PROPAGATOR_LAMMPS
        k = CUDA.@cuda launch=false _nhc_chain_stage_lammps_kernel!(ws.xi,
                                                                    ws.eta,
                                                                    ws.chain_force,
                                                                    ws.chain_masses,
                                                                    ws.kinetic_total_per_bath,
                                                                    ws.thermostat_kinetic_per_bath,
                                                                    ws.thermostat_potential_per_bath,
                                                                    ws.last_velocity_scale_per_bath,
                                                                    ws.dof_per_bath,
                                                                    ws.target_temperature,
                                                                    stage_dt,
                                                                    Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_force,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    else
        throw(ArgumentError("Unsupported NHC propagator id $(p.propagator)."))
    end
    return nothing
end

"""
    _apply_nhc_thermostat_stage!(spec, st, stage_dt)

Propagate the NHC chain and apply deterministic velocity scaling over
`stage_dt`, internally split into `spec.params.substeps`.
"""
function _apply_nhc_thermostat_stage!(spec::NHCSpec{T},
                                      st::SimulationState{T},
                                      stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    _nhc_update_dof_per_bath!(spec, st)
    _ensure_nhc_kinetic_initialized!(spec, st)
    copyto!(ws.kinetic_stage_start_per_bath, ws.kinetic_total_per_bath)
    _run_nhc_chain_stage!(spec, stage_dt)
    @. ws.cumulative_energy_exchange_per_bath += ws.kinetic_stage_start_per_bath - ws.kinetic_total_per_bath
    _nhc_apply_stage_scale!(st, ws.last_velocity_scale_per_bath, ws.particle_bath_id)
    return nothing
end

function validate_integrator_inputs!(spec::Union{BAOABSpec,BAOASpec,GSMSpec}, st, dt)
    _require_positive_gamma!(spec.params.gamma, string(integrator_name(spec)))
    return nothing
end

function validate_integrator_inputs!(spec::BrownianSpec, st, dt)
    _require_positive_gamma!(spec.params.gamma, "Brownian midpoint")
    return nothing
end

function validate_integrator_inputs!(spec::EMSpec, st, dt)
    _require_positive_gamma!(spec.params.gamma, "Euler-Maruyama")
    return nothing
end

function validate_integrator_inputs!(spec::NHCSpec{T}, st, dt) where {T<:AbstractFloat}
    p = spec.params
    p.mass > zero(T) || throw(ArgumentError("NHC requires mass > 0."))
    p.substeps >= 1 || throw(ArgumentError("NHC requires substeps >= 1."))
    p.chain_length >= 1 || throw(ArgumentError("NHC requires chain_length >= 1."))
    (p.propagator == NHC_PROPAGATOR_LEGACY ||
     p.propagator == NHC_PROPAGATOR_GROMACS ||
     p.propagator == NHC_PROPAGATOR_LAMMPS) ||
        throw(ArgumentError("NHC requires a supported propagator id, got $(p.propagator)."))
    nbaths = length(p.target_temperature)
    nbaths >= 1 || throw(ArgumentError("NHC requires at least one bath."))
    length(p.tau) == nbaths ||
        throw(ArgumentError("NHC requires length(tau) == length(target_temperature)."))
    size(p.chain_masses, 1) == p.chain_length ||
        throw(ArgumentError("NHC chain_masses first dimension must equal chain_length."))
    size(p.chain_masses, 2) == nbaths ||
        throw(ArgumentError("NHC chain_masses second dimension must equal number of baths."))
    @inbounds for b in 1:nbaths
        p.target_temperature[b] > zero(T) ||
            throw(ArgumentError("NHC requires target_temperature[$(b)] > 0."))
        p.tau[b] > zero(T) || throw(ArgumentError("NHC requires tau[$(b)] > 0."))
    end
    @inbounds for j in 1:p.chain_length, b in 1:nbaths
        p.chain_masses[j, b] > zero(T) ||
            throw(ArgumentError("NHC chain mass Q[$(j), bath=$(b)] must be > 0."))
    end
    return nothing
end

function validate_integrator_inputs!(spec::CSVRSpec{T}, st, dt) where {T<:AbstractFloat}
    p = spec.params
    p.mass > zero(T) || throw(ArgumentError("CSVR requires mass > 0."))
    nbaths = length(p.target_temperature)
    nbaths >= 1 || throw(ArgumentError("CSVR requires at least one bath."))
    length(p.tau) == nbaths ||
        throw(ArgumentError("CSVR requires length(tau) == length(target_temperature)."))
    @inbounds for b in 1:nbaths
        p.target_temperature[b] > zero(T) ||
            throw(ArgumentError("CSVR requires target_temperature[$(b)] > 0."))
        p.tau[b] > zero(T) || throw(ArgumentError("CSVR requires tau[$(b)] > 0."))
    end
    return nothing
end

function ensure_integrator_workspace!(spec::VVSpec{T}, st::SimulationState{T}) where {T<:AbstractFloat}
    _ensure_workspace_buffers!(spec.workspace, st; ou=spec.params.ou)
    return nothing
end

function ensure_integrator_workspace!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    _ensure_workspace_buffers!(spec.workspace, st; ou=spec.params.ou)
    return nothing
end

function ensure_integrator_workspace!(spec::Union{BrownianSpec{T},EMSpec{T}},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    _ensure_workspace_buffers!(spec.workspace, st; ou=spec.params.ou)
    return nothing
end

function ensure_integrator_workspace!(spec::NHCSpec{T},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    M = p.chain_length
    B = length(p.target_temperature)
    N = length(st.rx)

    if size(ws.xi) != (M, B)
        ws.xi = CUDA.zeros(T, M, B)
        ws.eta = CUDA.zeros(T, M, B)
        ws.chain_force = CUDA.zeros(T, M, B)
        ws.chain_masses = CUDA.zeros(T, M, B)
        ws.chain_masses_signature = UInt64(0)
        ws.kinetic_initialized = false
    end

    if length(ws.target_temperature) != B
        ws.target_temperature = CUDA.zeros(T, B)
    end
    copyto!(ws.target_temperature, p.target_temperature)

    if length(ws.particle_bath_id) != N
        ws.particle_bath_id = CUDA.fill(Int32(1), N)
        ws.kinetic_initialized = false
        ws.dof_dirty = true
    end
    if length(ws.bath_counts) != B
        ws.bath_counts = CUDA.zeros(Int32, B)
        ws.dof_dirty = true
    end
    if length(ws.dof_per_bath) != B
        ws.dof_per_bath = CUDA.zeros(T, B)
        ws.dof_dirty = true
    end
    if length(ws.kinetic_total_per_bath) != B
        ws.kinetic_total_per_bath = CUDA.zeros(T, B)
        ws.kinetic_initialized = false
    end
    if length(ws.kinetic_stage_start_per_bath) != B
        ws.kinetic_stage_start_per_bath = CUDA.zeros(T, B)
        ws.kinetic_initialized = false
    end
    if length(ws.cumulative_energy_exchange_per_bath) != B
        ws.cumulative_energy_exchange_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.thermostat_kinetic_per_bath) != B
        ws.thermostat_kinetic_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.thermostat_potential_per_bath) != B
        ws.thermostat_potential_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.last_velocity_scale_per_bath) != B
        ws.last_velocity_scale_per_bath = CUDA.fill(one(T), B)
    end

    sig = _nhc_chain_masses_signature(p.chain_masses)
    if ws.chain_masses_signature != sig
        copyto!(ws.chain_masses, p.chain_masses)
        ws.chain_masses_signature = sig
        ws.kinetic_initialized = false
    end

    if st.step == 0
        ws.kinetic_initialized = false
    end
    return nothing
end

function ensure_integrator_workspace!(spec::CSVRSpec{T},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    B = length(p.target_temperature)
    N = length(st.rx)

    if length(ws.target_temperature) != B
        ws.target_temperature = CUDA.zeros(T, B)
    end
    copyto!(ws.target_temperature, p.target_temperature)

    if length(ws.tau) != B
        ws.tau = CUDA.zeros(T, B)
    end
    copyto!(ws.tau, p.tau)

    if length(ws.particle_bath_id) != N
        ws.particle_bath_id = CUDA.fill(Int32(1), N)
        ws.kinetic_initialized = false
        ws.dof_dirty = true
    end
    if length(ws.bath_counts) != B
        ws.bath_counts = CUDA.zeros(Int32, B)
        ws.dof_dirty = true
    end
    if length(ws.dof_per_bath) != B
        ws.dof_per_bath = CUDA.zeros(T, B)
        ws.dof_dirty = true
    end
    if length(ws.kinetic_total_per_bath) != B
        ws.kinetic_total_per_bath = CUDA.zeros(T, B)
        ws.kinetic_initialized = false
    end
    if length(ws.cumulative_energy_exchange_per_bath) != B
        ws.cumulative_energy_exchange_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.last_velocity_scale_per_bath) != B
        ws.last_velocity_scale_per_bath = CUDA.fill(one(T), B)
    end

    if st.step == 0
        ws.kinetic_initialized = false
    end
    return nothing
end

function execute_integrator_stage!(spec::VVSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    params = spec.params
    ws = spec.workspace

    if stage_tag === :kick1
        _prepare_langevin_noise!(params, ws, st, dt)
    elseif stage_tag === :drift
        if _is_3d(st)
            LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.rz,
                                                  st.vx, st.vy, st.vz,
                                                  st.f0x, st.f0y, st.f0z,
                                                  ws.rf_x, ws.rf_y, ws.rf_z,
                                                  params, dt, st.box3::Definitions.Box3;
                                                  unwrapped_x=st.rx_unwrap,
                                                  unwrapped_y=st.ry_unwrap,
                                                  unwrapped_z=st.rz_unwrap)
        else
            LangevinIntegrators.vv_positions_soa!(st.rx, st.ry,
                                                  st.vx, st.vy,
                                                  st.f0x, st.f0y,
                                                  ws.rf_x, ws.rf_y,
                                                  params, dt, st.box2::Definitions.Box2;
                                                  unwrapped_x=st.rx_unwrap,
                                                  unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, :after_drift; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :kick2
        if _is_3d(st)
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.vz,
                                                   st.f0x, st.f0y, st.f0z,
                                                   st.fx, st.fy, st.fz,
                                                   ws.rf_x, ws.rf_y, ws.rf_z,
                                                   st.dq, st.dU, st.Ekin,
                                                   params, dt)
        else
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy,
                                                   st.f0x, st.f0y,
                                                   st.fx, st.fy,
                                                   ws.rf_x, ws.rf_y,
                                                   st.dq, st.dU, st.Ekin,
                                                   params, dt)
        end
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

function _execute_baoab_family_stage!(params::LangevinIntegrators.BAOABParams{T},
                                      ws::StochasticWorkspace{T},
                                      st::SimulationState{T},
                                      stage_tag::Symbol,
                                      dt::T;
                                      compute_energy::Bool,
                                      freeze_hold::Bool,
                                      freeze_spring::Bool) where {T<:AbstractFloat}
    if stage_tag === :B1
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.f0x, st.f0y, st.f0z,
                                            params, dt, st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.f0x, st.f0y,
                                            params, dt, st.Ekin, st.dU)
        end
    elseif stage_tag === :A1 || stage_tag === :A2
        if _is_3d(st)
            LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                            st.vx, st.vy, st.vz,
                                            dt, st.box3::Definitions.Box3;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap,
                                            unwrapped_z=st.rz_unwrap)
        else
            LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                            st.vx, st.vy,
                                            dt, st.box2::Definitions.Box2;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, stage_tag === :A1 ? :after_drift : :after_final_position;
                                   freeze_hold=freeze_hold)
    elseif stage_tag === :O
        _prepare_langevin_noise!(params, ws, st, dt)
        if _is_3d(st)
            LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz,
                                             ws.rf_x, ws.rf_y, ws.rf_z,
                                             params, dt, st.dq)
        else
            LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy,
                                             ws.rf_x, ws.rf_y,
                                             params, dt, st.dq)
        end
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :B2
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.fx, st.fy, st.fz,
                                            params, dt, st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.fx, st.fy,
                                            params, dt, st.Ekin, st.dU)
        end
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for BAOAB-family integrator."))
    end
    return nothing
end

function execute_integrator_stage!(spec::Union{BAOABSpec{T},GSMSpec{T}},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    return _execute_baoab_family_stage!(spec.params, spec.workspace, st, stage_tag, dt;
                                        compute_energy=compute_energy,
                                        freeze_hold=freeze_hold,
                                        freeze_spring=freeze_spring)
end

function execute_integrator_stage!(spec::BAOASpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    params = spec.params
    ws = spec.workspace

    if stage_tag === :B1
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.f0x, st.f0y, st.f0z,
                                            params, T(2) * dt, st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.f0x, st.f0y,
                                            params, T(2) * dt, st.Ekin, st.dU)
        end
    elseif stage_tag === :A1 || stage_tag === :A2
        if _is_3d(st)
            LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                            st.vx, st.vy, st.vz,
                                            dt, st.box3::Definitions.Box3;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap,
                                            unwrapped_z=st.rz_unwrap)
        else
            LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                            st.vx, st.vy,
                                            dt, st.box2::Definitions.Box2;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, stage_tag === :A1 ? :after_drift : :after_final_position;
                                   freeze_hold=freeze_hold)
    elseif stage_tag === :O
        _prepare_langevin_noise!(params, ws, st, dt)
        if _is_3d(st)
            LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz,
                                             ws.rf_x, ws.rf_y, ws.rf_z,
                                             params, dt, st.dq)
        else
            LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy,
                                             ws.rf_x, ws.rf_y,
                                             params, dt, st.dq)
        end
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :power
        if _is_3d(st)
            LangevinIntegrators.cons_power_3d!(st.vx, st.vy, st.vz,
                                               st.fx, st.fy, st.fz,
                                               st.dU)
        else
            LangevinIntegrators.cons_power_2d!(st.vx, st.vy,
                                               st.fx, st.fy,
                                               st.dU)
        end
    elseif stage_tag === :kinetic_refresh
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.fx, st.fy, st.fz,
                                            params, T(0), st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.fx, st.fy,
                                            params, T(0), st.Ekin, st.dU)
        end
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

function _execute_brownian_midpoint_stage!(params::Union{BrownianIntegrators.BrownianParams{T},BrownianIntegrators.EMParams{T}},
                                           ws::StochasticWorkspace{T},
                                           st::SimulationState{T},
                                           stage_tag::Symbol,
                                           dt::T;
                                           compute_energy::Bool,
                                           freeze_hold::Bool,
                                           freeze_spring::Bool) where {T<:AbstractFloat}
    if stage_tag === :midpoint_predict
        _prepare_brownian_noise!(params, ws, st, dt)
        if _is_3d(st)
            BrownianIntegrators.bd_midpoint_positions_3d!(st.rx, st.ry, st.rz,
                                                          st.fx, st.fy, st.fz,
                                                          ws.rf_x, ws.rf_y, ws.rf_z,
                                                          st.vx, st.vy, st.vz,
                                                          params.gamma, params.noise_scale,
                                                          dt, st.box3::Definitions.Box3)
            if freeze_hold
                _apply_freeze_hold!(st, st.vx, st.vy, st.vz)
            end
        else
            BrownianIntegrators.bd_midpoint_positions_2d!(st.rx, st.ry,
                                                          st.fx, st.fy,
                                                          ws.rf_x, ws.rf_y,
                                                          st.vx, st.vy,
                                                          params.gamma, params.noise_scale,
                                                          dt, st.box2::Definitions.Box2)
            if freeze_hold
                _apply_freeze_hold!(st, st.vx, st.vy)
            end
        end
    elseif stage_tag === :midpoint_force
        evaluate_midpoint_forces_into_f0!(st; freeze_spring=freeze_spring)
    elseif stage_tag === :final_position
        if _is_3d(st)
            BrownianIntegrators.bd_finish_step_3d!(st.rx, st.ry, st.rz,
                                                   st.f0x, st.f0y, st.f0z,
                                                   ws.rf_x, ws.rf_y, ws.rf_z,
                                                   params.gamma, params.noise_scale,
                                                   dt, st.dq, st.dU,
                                                   st.box3::Definitions.Box3;
                                                   unwrapped_x=st.rx_unwrap,
                                                   unwrapped_y=st.ry_unwrap,
                                                   unwrapped_z=st.rz_unwrap)
        else
            BrownianIntegrators.bd_finish_step_2d!(st.rx, st.ry,
                                                   st.f0x, st.f0y,
                                                   ws.rf_x, ws.rf_y,
                                                   params.gamma, params.noise_scale,
                                                   dt, st.dq, st.dU,
                                                   st.box2::Definitions.Box2;
                                                   unwrapped_x=st.rx_unwrap,
                                                   unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, :after_final_position; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for Brownian-family integrator."))
    end

    return nothing
end

function execute_integrator_stage!(spec::BrownianSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    return _execute_brownian_midpoint_stage!(spec.params, spec.workspace, st, stage_tag, dt;
                                             compute_energy=compute_energy,
                                             freeze_hold=freeze_hold,
                                             freeze_spring=freeze_spring)
end

function execute_integrator_stage!(spec::EMSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    return _execute_brownian_midpoint_stage!(spec.params, spec.workspace, st, stage_tag, dt;
                                             compute_energy=compute_energy,
                                             freeze_hold=freeze_hold,
                                             freeze_spring=freeze_spring)
end

function execute_integrator_stage!(spec::NHCSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace

    if stage_tag === :thermostat_pre
        _apply_nhc_thermostat_stage!(spec, st, dt / T(2))
    elseif stage_tag === :kick1
        _nhc_apply_half_kick!(st, st.f0x, st.f0y, st.f0z, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :drift
        _nhc_drift_positions!(st, dt)
        apply_post_position_hooks!(st, :after_drift; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :kick2
        _nhc_apply_half_kick!(st, st.fx, st.fy, st.fz, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :thermostat_post
        _apply_nhc_thermostat_stage!(spec, st, dt / T(2))
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

function execute_integrator_stage!(spec::CSVRSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace

    if stage_tag === :kick1
        _nhc_apply_half_kick!(st, st.f0x, st.f0y, st.f0z, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :drift
        _nhc_drift_positions!(st, dt)
        apply_post_position_hooks!(st, :after_drift; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :kick2
        _nhc_apply_half_kick!(st, st.fx, st.fy, st.fz, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :thermostat
        _apply_csvr_thermostat_stage!(spec, st, dt)
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

@inline _device_scalar(x::CuArray{T,1}) where {T<:AbstractFloat} = T(Array(x)[1])
include("simulation/Observables.jl")

end # module
