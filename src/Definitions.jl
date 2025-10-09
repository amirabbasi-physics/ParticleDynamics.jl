module Definitions

using CUDA
using StaticArrays

export IntX, Dim2, Dim3, Box2, Box3,
       LJParams, wrap_pbc2!, wrap_pbc3!, clamp_cap,
       HarmonicBondParams, FENEParams,
       SoftRepulsiveParams, LJMixParams,
       BondPotential, HarmonicBond, FENEBond,
       StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime
export harmonic_bond, fene_bond
const IntX   = Int32

const Dim2 = 2
const Dim3 = 3

const Box2{T} = NTuple{2,T}
const Box3{T} = NTuple{3,T}

struct LJParams{T}
    ϵ::T
    σ::T
    rcut::T
end

# Parameters for LJ with type-based size mixing (σ_ij = 0.5(σ_i+σ_j))
struct LJMixParams{T}
    ϵ::T              # global epsilon (can be extended later to per-type)
    rcut_factor::T    # factor applied to σ_ij to get cutoff (e.g., 2^(1/6))
end

# Bonded interaction parameters
struct HarmonicBondParams{T}
    k::T     # spring constant
    r0::T    # equilibrium distance
end

struct FENEParams{T}
    k::T     # spring constant
    R0::T    # maximum extension parameter
end

# Nonbonded soft repulsive harmonic parameters
struct SoftRepulsiveParams{T}
    ϵ::T
    σ::T
end

# Unified bonded interaction container
abstract type BondPotential{T<:AbstractFloat} end

struct HarmonicBond{T<:AbstractFloat} <: BondPotential{T}
    params::HarmonicBondParams{T}
end

struct FENEBond{T<:AbstractFloat} <: BondPotential{T}
    params::FENEParams{T}
end

@inline clamp_cap(idx::IntX, cap::IntX) = ifelse(idx <= cap, idx, IntX(0))

# ---------------- PBC wrappers (SoA) ----------------

@inline function _wrap(x::T, L::T)::T where {T<:AbstractFloat}
    half = T(0.5)
    y = x + L*half
    y -= floor(y / L) * L
    return y - L*half
end

function wrap_pbc2!(rx::CuArray{T,1}, ry::CuArray{T,1}, box::Box2{T}) where {T<:AbstractFloat}
    function kern(rx, ry, Lx, Ly)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(rx); if i > N; return; end
        @inbounds begin
            rx[i] = _wrap(rx[i], Lx)
            ry[i] = _wrap(ry[i], Ly)
        end
        return
    end
    N = length(rx); threads = min(256,N); blocks = cld(N,threads)
    k = CUDA.@cuda launch=false kern(rx, ry, box[1], box[2])
    CUDA.@sync k(rx, ry, box[1], box[2]; threads, blocks)
    return nothing
end

function wrap_pbc3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}, box::Box3{T}) where {T<:AbstractFloat}
    function kern(rx, ry, rz, Lx, Ly, Lz)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(rx); if i > N; return; end
        @inbounds begin
            rx[i] = _wrap(rx[i], Lx)
            ry[i] = _wrap(ry[i], Ly)
            rz[i] = _wrap(rz[i], Lz)
        end
        return
    end
    N = length(rx); threads = min(256,N); blocks = cld(N,threads)
    k = CUDA.@cuda launch=false kern(rx, ry, rz, box[1], box[2], box[3])
    CUDA.@sync k(rx, ry, rz, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

# Convenience constructors for bonds
harmonic_bond(; k::Real = 100.0, r0::Real = 1.0) = HarmonicBond(HarmonicBondParams(float(k), float(r0)))
fene_bond(;k::Real = 300.0, r0::Real = 1.5) = FENEBond(FENEParams(float(k), float(r0)))

# ---------------- Physical property calculations ----------------

# Stokes friction coefficient for a sphere
# γ = 6 * π * η * R (where η is viscosity, R is radius)
function StokesFrictionCoefficient(viscosity::Real, radius::Real)
    return 6 * π * viscosity * radius
end

# Mass of a sphere given density and radius
function SphereMass(density::Real, radius::Real) 
    return (4/3) * π * radius^3 * density
end

# Inertial time (mass/friction)
function InertialTime(mass::Real, frictioncoefficient::Real) 
    return mass / frictioncoefficient
end

# Diffusive time to move its own size (4*R^2 / (2*d*D), D=kT/γ)
# where d is dimension (2 or 3)
# Thus DiffusiveTime = 4*R^2 * γ / (2*d*kT)
# Note: this is 2x the time to diffuse its own radius
function DiffusiveTime(radius::Real, frictioncoefficient::Real, dimension::Real)
    temperature = 1.0  # in reduced units kT=1, Pay attention that this is the same for paticles with different temperatures.
    return 4*(radius^2 * frictioncoefficient) / (2 * dimension * temperature)
end

function gamma_reduced(diffusive_time::Real, inertial_time::Real)
    return sqrt(diffusive_time / inertial_time)
end

end # module
