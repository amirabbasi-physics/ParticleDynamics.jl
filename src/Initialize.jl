"""
Velocity initialization utilities used by the examples and manual workflows.

`Initialize` exposes CUDA kernels that draw Maxwell–Boltzmann velocities directly
in GPU memory and helpers that remove center-of-mass drift after every draw.
"""
module Initialize

using CUDA
using StaticArrays
using ..Definitions

export init_velocities_maxwell!, center_of_mass_velocity!, subtract_com_velocity!

"""
    init_velocities_maxwell!(vx, vy; temperature=1)
    init_velocities_maxwell!(vx, vy, vz; temperature=1)

Fill the SoA velocity buffers with independent Maxwell–Boltzmann draws at the
requested temperature. The RNG lives inside the CUDA kernel so no host staging
is necessary.

# Arguments
- `vx, vy[, vz]`: CuArrays storing particle velocities.
- `temperature`: `kT` used for the Maxwell–Boltzmann width. Matches the
  `temperature` keyword in `build_simulation`.

# Examples
    st = build_simulation(N=256, box=(80f0, 80f0),
                          cutoff=Float32(2^(1/6)), skin=0.4f0,
                          cap=Int32(64), neigh_interval=10,
                          epsilon=10f0, sigma=1f0,
                          gamma=50f0, temperature=1f0,
                          nonbonded=:wca, precision=:f32)
    init_velocities_maxwell!(st.vx, st.vy; temperature=1f0)
    vcom = center_of_mass_velocity!(st.vx, st.vy)
    subtract_com_velocity!(st.vx, st.vy, vcom...)
"""
function init_velocities_maxwell!(vx::CuArray{T,1},
                                  vy::CuArray{T,1};
                                  temperature::T = one(T)) where {T<:AbstractFloat}
    function kern(vx, vy, scale)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(vx); if i > N; return; end
        @inbounds begin
            vx[i] = CUDA.randn(T) * scale
            vy[i] = CUDA.randn(T) * scale
        end
        return
    end
    N = length(vx); threads = min(256,N); blocks = cld(N,threads)
    scale = sqrt(temperature)
    k = CUDA.@cuda launch=false kern(vx, vy, scale)
    CUDA.@sync k(vx, vy, scale; threads, blocks)
    return nothing
end

# 3D variant
function init_velocities_maxwell!(vx::CuArray{T,1},
                                  vy::CuArray{T,1},
                                  vz::CuArray{T,1};
                                  temperature::T = one(T)) where {T<:AbstractFloat}
    function kern(vx, vy, vz, scale)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(vx); if i > N; return; end
        @inbounds begin
            vx[i] = CUDA.randn(T) * scale
            vy[i] = CUDA.randn(T) * scale
            vz[i] = CUDA.randn(T) * scale
        end
        return
    end
    N = length(vx); threads = min(256,N); blocks = cld(N,threads)
    scale = sqrt(temperature)
    k = CUDA.@cuda launch=false kern(vx, vy, vz, scale)
    CUDA.@sync k(vx, vy, vz, scale; threads, blocks)
    return nothing
end

"""
    center_of_mass_velocity!(vx, vy[, vz]) -> tuple

Transfer the velocity arrays back to the host, compute the average velocity,
and return it as a tuple. Used by the warmup blocks in
`examples/TwoT_2D_LD_VV.jl` to ensure there is no center-of-mass drift.
"""
function center_of_mass_velocity!(vx::CuArray{T,1}, vy::CuArray{T,1}) where {T<:AbstractFloat}
    Vx = sum(Array(vx)) / length(vx)
    Vy = sum(Array(vy)) / length(vy)
    return (T(Vx), T(Vy))
end

function center_of_mass_velocity!(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1}) where {T<:AbstractFloat}
    Vx = sum(Array(vx)) / length(vx)
    Vy = sum(Array(vy)) / length(vy)
    Vz = sum(Array(vz)) / length(vz)
    return (T(Vx), T(Vy), T(Vz))
end

"""
    subtract_com_velocity!(vx, vy[, vz], Vx, Vy[, Vz])

Shift every velocity component by the center-of-mass velocity that was computed
via [`center_of_mass_velocity!`](@ref). This is particularly important for the
large-WCA examples where random initiation sometimes produces a finite drift.
"""
function subtract_com_velocity!(vx::CuArray{T,1}, vy::CuArray{T,1}, Vx::T, Vy::T) where {T<:AbstractFloat}
    @. vx = vx - Vx
    @. vy = vy - Vy
    return nothing
end

function subtract_com_velocity!(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1},
                                Vx::T, Vy::T, Vz::T) where {T<:AbstractFloat}
    @. vx = vx - Vx
    @. vy = vy - Vy
    @. vz = vz - Vz
    return nothing
end

end # module
