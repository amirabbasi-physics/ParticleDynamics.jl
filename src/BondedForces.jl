"""
Bonded interaction kernels (harmonic and FENE) used by the polymer examples.
"""
module BondedForces

using CUDA
using ..Definitions

export BondList, build_bondlist,
       harmonic_forces_soa!, harmonic_forces_soa_noE!,
       fene_forces_soa!, fene_forces_soa_noE!

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

@inline function mic_fast(dx::T, halfL::T, L::T) where {T<:AbstractFloat}
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

# Bond rows are symmetric in the CSR list, so each physical bond is visited
# twice. Store half the dyadic contribution per row so the particle-summed
# virial matches the single-count configurational virial.
@inline function _half_virial2(dx::T, dy::T, fx::T, fy::T) where {T<:AbstractFloat}
    half = T(0.5)
    return half * dx * fx, half * dy * fy, half * dx * fy
end

@inline function _half_virial3(dx::T, dy::T, dz::T, fx::T, fy::T, fz::T) where {T<:AbstractFloat}
    half = T(0.5)
    return half * dx * fx, half * dy * fy, half * dz * fz,
           half * dx * fy, half * dx * fz, half * dy * fz
end

@inline _bond_threads(N::Int) = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)

include("bonded/BondTypes.jl")
include("bonded/HarmonicBondCUDA.jl")
include("bonded/FENEBondCUDA.jl")

end # module BondedForces
