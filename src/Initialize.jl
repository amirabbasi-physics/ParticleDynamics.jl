module Initialize

using CUDA
using StaticArrays
using ..Definitions

export init_velocities_maxwell!, center_of_mass_velocity!, subtract_com_velocity!

# Maxwell–Boltzmann velocities on GPU (SoA). RNG inside kernel as requested.
function init_velocities_maxwell!(vx::CuArray{Definitions.FloatX,1},
                                  vy::CuArray{Definitions.FloatX,1};
                                  temperature::Definitions.FloatX = 1f0)
    function kern(vx, vy, scale)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(vx); if i > N; return; end
        @inbounds begin
            vx[i] = CUDA.randn(Definitions.FloatX) * scale
            vy[i] = CUDA.randn(Definitions.FloatX) * scale
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
function init_velocities_maxwell!(vx::CuArray{Definitions.FloatX,1},
                                  vy::CuArray{Definitions.FloatX,1},
                                  vz::CuArray{Definitions.FloatX,1};
                                  temperature::Definitions.FloatX = 1f0)
    function kern(vx, vy, vz, scale)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(vx); if i > N; return; end
        @inbounds begin
            vx[i] = CUDA.randn(Definitions.FloatX) * scale
            vy[i] = CUDA.randn(Definitions.FloatX) * scale
            vz[i] = CUDA.randn(Definitions.FloatX) * scale
        end
        return
    end
    N = length(vx); threads = min(256,N); blocks = cld(N,threads)
    scale = sqrt(temperature)
    k = CUDA.@cuda launch=false kern(vx, vy, vz, scale)
    CUDA.@sync k(vx, vy, vz, scale; threads, blocks)
    return nothing
end

# center-of-mass velocity (host)
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