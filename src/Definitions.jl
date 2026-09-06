"""
Domain and potential parameter definitions shared by ParticleDynamics modules.

`Definitions` exposes the light-weight structs that describe the simulation
domain (periodic boxes), nonbonded parameters (Lennard-Jones, soft repulsive,
per-type mixing), and bonded potentials (harmonic, FENE). The same types are
used consistently across `Simulation`, `NonBondedForces`, and the examples.
"""
module Definitions

using CUDA
using StaticArrays

export IntX, Dim2, Dim3, Box2, Box3,
       LJParams, wrap_pbc2!, wrap_pbc3!, clamp_cap,
       HarmonicBondParams, FENEParams,
       SoftRepulsiveParams, LJMixParams, OUSpectrum,
       BondPotential, HarmonicBond, FENEBond,
       StokesFrictionCoefficient, SphereMass, InertialTime, DiffusiveTime
export harmonic_bond, fene_bond
function _require_stochastic_dt!(params, dt)
    T = typeof(params.dt)
    dtT = T(dt)
    isfinite(dtT) && dtT > zero(T) || throw(ArgumentError("dt must be finite and positive"))
    dtT == params.dt || throw(ArgumentError(
        "Stochastic coefficients were built for dt=$(params.dt), but dt=$dtT was requested. Construct a new integrator spec with the intended dt."))
    return nothing
end

const IntX   = Int32

const Dim2 = 2
const Dim3 = 3

const Box2{T} = NTuple{2,T}
const Box3{T} = NTuple{3,T}

"""
    LJParams{T}(ϵ, σ, rcut)

Lennard-Jones parameters used by the nonbonded force kernels.

# Arguments
- `ϵ`: Depth of the LJ well (energy scale).
- `σ`: Particle diameter (sets the zero of the LJ potential).
- `rcut`: Pairwise cutoff distance. For the WCA branch this is set to
  `2^(1/6)σ` as in `examples/2D_allpairs_quicktest.jl` and the 3D scripts.

# Examples
    params = LJParams(Float32(10), Float32(1), Float32(2^(1/6)))
    st = build_simulation(N=256, box=(80f0, 80f0), cutoff=params.rcut,
                          skin=0.4f0, cap=Int32(64),
                          neigh_interval=10, use_neighborlist=false,
                          epsilon=params.ϵ, sigma=params.σ,
                          gamma=50f0, temperature=1f0,
                          nonbonded=:wca, precision=:f32)
"""
struct LJParams{T}
    ϵ::T
    σ::T
    rcut::T
end

"""
    LJMixParams{T}(ϵ, rcut_factor)

Per-type Lennard-Jones size mixing with Lorentz combination
`σᵢⱼ = (σᵢ + σⱼ)/2`. `rcut_factor` multiplies each `σᵢⱼ` when computing the
per-pair cutoff. The 3D stencil examples (`examples/3D_stencil_two_sizes*.jl`)
use `rcut_factor = 2^(1/6)` so that every contact behaves like a WCA pair.
"""
struct LJMixParams{T}
    ϵ::T
    rcut_factor::T
end

"""
    HarmonicBondParams{T}(k, r0)

Parameters for harmonic bonds `U(r) = ½ k (r - r₀)²`. The polymer examples
(`examples/2D_polymer_bonded*.jl`) use `k = 300` and `r0 = 1.0` to keep bead
spacings near a single diameter.
"""
struct HarmonicBondParams{T}
    k::T
    r0::T
end

"""
    FENEParams{T}(k, R0)

Finite-extensible nonlinear elastic parameters, with potential
`U(r) = -½kR₀² log(1 - (r/R₀)²)`. `examples/2D_polymer_bonded_BP.jl`
demonstrates `k = 300`, `R0 = 1.5`. Force evaluation requires finite `k ≥ 0`,
a finite positive representable `R0²`, and finite bond lengths `r < R0`.
Overstretched bonds throw `DomainError`; force and energy are never clamped.
"""
struct FENEParams{T}
    k::T
    R0::T
end

"""
    SoftRepulsiveParams{T}(ϵ, σ)

Parameters for the truncated harmonic repulsion used in the two-temperature
soft-repulsive scripts (e.g. `examples/TwoT_2D_LD_VV.jl` with
`σ = 1.0`, `ϵ = 1e9`). The cutoff equals `σ`.
"""
struct SoftRepulsiveParams{T}
    ϵ::T
    σ::T
end

include("random/OUSpectrum.jl")

"""
    BondPotential

Abstract supertype for bonded potentials so `SimulationState` can store either
`HarmonicBond` or `FENEBond` and dispatch bond kernels accordingly.
"""
abstract type BondPotential{T<:AbstractFloat} end

"""
    HarmonicBond(params)

Concrete wrapper storing `HarmonicBondParams`. Construct via [`harmonic_bond`](@ref)
so the element type matches the simulation precision.
"""
struct HarmonicBond{T<:AbstractFloat} <: BondPotential{T}
    params::HarmonicBondParams{T}
end

"""
    FENEBond(params)

Concrete wrapper storing `FENEParams`. Use [`fene_bond`](@ref) to create
instances whose parameter type (`Float32`/`Float64`) matches the simulation.
"""
struct FENEBond{T<:AbstractFloat} <: BondPotential{T}
    params::FENEParams{T}
end

"""
    clamp_cap(idx, cap) -> Int32

Utility used while filling neighbor matrices: returns `idx` if the number of
neighbors is within the user-provided `cap`, otherwise returns `0` to signal
overflow. Ensures memory safety when caps are too small.
"""
@inline clamp_cap(idx::IntX, cap::IntX) = ifelse(idx <= cap, idx, IntX(0))

# ---------------- PBC wrappers (SoA) ----------------

@inline function _wrap(x::T, L::T)::T where {T<:AbstractFloat}
    half = T(0.5)
    y = x + L*half
    y -= floor(y / L) * L
    return y - L*half
end

"""
    wrap_pbc2!(rx, ry, box)

Wrap 2D positions back into the periodic box `[-L/2, L/2)` along both axes.
Called after integrator updates to prevent coordinates from drifting too far
before the next neighbor rebuild.
"""
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

"""
    wrap_pbc3!(rx, ry, rz, box)

3D variant of [`wrap_pbc2!`](@ref) used by the 3D examples (e.g.
`examples/3D_filters.jl`) to keep coordinates centered during long runs.
"""
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
"""
    harmonic_bond(; k=100.0, r0=1.0)

Convenience constructor that promotes the supplied spring constant and
equilibrium distance to the simulation precision. The polymer demos use
`harmonic_bond(k=300, r0=1.0)` for bead–spring chains.
"""
harmonic_bond(; k::Real = 100.0, r0::Real = 1.0) = HarmonicBond(HarmonicBondParams(float(k), float(r0)))
"""
    fene_bond(; k=300.0, r0=1.5)

Convenience constructor for `FENEBond`. The 2D/3D FENE polymer examples draw
directly from this helper with the default parameters.
"""
fene_bond(;k::Real = 300.0, r0::Real = 1.5) = FENEBond(FENEParams(float(k), float(r0)))

# ---------------- Physical property calculations ----------------

"""
    StokesFrictionCoefficient(η, R)

Return the Stokes drag `γ = 6π η R` for a spherical particle of radius `R`
embedded in fluid with viscosity `η`. Handy when translating experimental
values into the `gamma` input of `build_simulation`.
"""
function StokesFrictionCoefficient(viscosity::Real, radius::Real)
    return 6 * π * viscosity * radius
end

"""
    SphereMass(ρ, R)

Mass of a solid sphere with density `ρ` and radius `R`:
`m = (4/3)π R³ ρ`.
"""
function SphereMass(density::Real, radius::Real) 
    return (4/3) * π * radius^3 * density
end

"""
    InertialTime(m, γ)

Return the inertial relaxation time `τᵢ = m / γ`. Useful for picking `dt`
relative to the physical time scales probed in the examples.
"""
function InertialTime(mass::Real, frictioncoefficient::Real) 
    return mass / frictioncoefficient
end

"""
    DiffusiveTime(R, γ, d)

Estimate the time required for a particle to diffuse its own diameter under
isothermal conditions. Uses the relation `τ_D = 4 R² γ / (2 d kT)` with
`kT = 1` in the reduced units adopted throughout the repository.
"""
function DiffusiveTime(radius::Real, frictioncoefficient::Real, dimension::Real)
    temperature = 1.0
    return 4*(radius^2 * frictioncoefficient) / (2 * dimension * temperature)
end

"""
    gamma_reduced(diffusive_time, inertial_time)

Non-dimensional damping ratio `γ̃ = √(τ_D / τᵢ)` which compares diffusive and
inertial time scales. Scripts such as `examples/TwoT_2D_LD_VV.jl` monitor this
value when tuning `gamma`.
"""
function gamma_reduced(diffusive_time::Real, inertial_time::Real)
    return sqrt(diffusive_time / inertial_time)
end

end # module
