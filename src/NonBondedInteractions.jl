module NonBondedInteractions

using CUDA
using CUDA: CuArray
using ..Definitions
using ..NeighborLists
using ..BondedForces
import ..NonBondedForces

export AbstractPotential, LennardJonesPotential, WCAPotential, SoftRepulsivePotential,
       AbstractCoefficientStyle, UniformLJCoefficients, MixedSigmaCoefficients,
       PairMatrixCoefficients, UniformSoftRepCoefficients,
       AbstractExclusionStyle, NoExclusions, BondExclusions,
       AbstractAccumulationMode, ForceOnly, ForceEnergyVirial,
       NonBondedInteraction, compute_nonbonded!

abstract type AbstractPotential end
struct LennardJonesPotential <: AbstractPotential end
struct WCAPotential <: AbstractPotential end
struct SoftRepulsivePotential <: AbstractPotential end

abstract type AbstractCoefficientStyle end

struct UniformLJCoefficients{T<:AbstractFloat} <: AbstractCoefficientStyle
    params::Definitions.LJParams{T}
end

struct MixedSigmaCoefficients{T<:AbstractFloat} <: AbstractCoefficientStyle
    epsilon::T
    sigma_particle::CuArray{T,1}
    rcut_factor::T
end

struct PairMatrixCoefficients{T<:AbstractFloat} <: AbstractCoefficientStyle
    typeid::CuArray{Int32,1}
    sigma_pair::CuArray{T,2}
    epsilon_pair::CuArray{T,2}
    rcut_pair::CuArray{T,2}
end

struct UniformSoftRepCoefficients{T<:AbstractFloat} <: AbstractCoefficientStyle
    params::Definitions.SoftRepulsiveParams{T}
end

abstract type AbstractExclusionStyle end
struct NoExclusions <: AbstractExclusionStyle end

struct BondExclusions <: AbstractExclusionStyle
    bonds::BondedForces.BondList
end

abstract type AbstractAccumulationMode end
struct ForceOnly <: AbstractAccumulationMode end
struct ForceEnergyVirial <: AbstractAccumulationMode end

struct NonBondedInteraction{P<:AbstractPotential,C<:AbstractCoefficientStyle,X<:AbstractExclusionStyle}
    potential::P
    coefficients::C
    exclusions::X
end

_mixed_exclusion_bonds(::NoExclusions) = nothing
_mixed_exclusion_bonds(exclusions::BondExclusions) = exclusions.bonds

# 2D dispatch

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa_noE!(rx, ry, fx, fy, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa_noE_excl!(rx, ry, fx, fy, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa!(rx, ry, fx, fy, Epot, V, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa_excl!(rx, ry, fx, fy, Epot, V, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa_noE!(rx, ry, fx, fy, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa_noE_excl!(rx, ry, fx, fy, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa!(rx, ry, fx, fy, Epot, V, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa_excl!(rx, ry, fx, fy, Epot, V, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa_noE!(rx, ry, fx, fy, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa_noE_excl!(rx, ry, fx, fy, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa!(rx, ry, fx, fy, Epot, V, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa_excl!(rx, ry, fx, fy, Epot, V, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,MixedSigmaCoefficients{T},X},
                            ::ForceOnly) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_noE_mixed!(rx, ry, fx, fy, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,MixedSigmaCoefficients{T},X},
                            ::ForceEnergyVirial) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_mixed!(rx, ry, fx, fy, Epot, V, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,MixedSigmaCoefficients{T},X},
                            ::ForceOnly) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_noE_mixed!(rx, ry, fx, fy, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,MixedSigmaCoefficients{T},X},
                            ::ForceEnergyVirial) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_mixed!(rx, ry, fx, fy, Epot, V, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_noE_pairs!(rx, ry, fx, fy, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_noE_excl!(rx, ry, fx, fy, nbh, interaction.exclusions.bonds,
                                            box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_pairs!(rx, ry, fx, fy, Epot, V, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_excl!(rx, ry, fx, fy, Epot, V, nbh, interaction.exclusions.bonds,
                                        box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_noE_pairs!(rx, ry, fx, fy, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_noE_excl!(rx, ry, fx, fy, nbh, interaction.exclusions.bonds,
                                             box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_pairs!(rx, ry, fx, fy, Epot, V, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box2{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_excl!(rx, ry, fx, fy, Epot, V, nbh, interaction.exclusions.bonds,
                                         box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

# 3D dispatch

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa_noE!(rx, ry, rz, fx, fy, fz, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa_noE_excl!(rx, ry, rz, fx, fy, fz, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.lj_forces_soa_excl!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa_noE!(rx, ry, rz, fx, fy, fz, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa_noE_excl!(rx, ry, rz, fx, fy, fz, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,UniformLJCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.wca_forces_soa_excl!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa_noE!(rx, ry, rz, fx, fy, fz, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa_noE_excl!(rx, ry, rz, fx, fy, fz, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{SoftRepulsivePotential,UniformSoftRepCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    NonBondedForces.harmonic_rep_forces_soa_excl!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, interaction.exclusions.bonds, box, interaction.coefficients.params)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,MixedSigmaCoefficients{T},X},
                            ::ForceOnly) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_noE_mixed!(rx, ry, rz, fx, fy, fz, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,MixedSigmaCoefficients{T},X},
                            ::ForceEnergyVirial) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_mixed!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,MixedSigmaCoefficients{T},X},
                            ::ForceOnly) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_noE_mixed!(rx, ry, rz, fx, fy, fz, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,MixedSigmaCoefficients{T},X},
                            ::ForceEnergyVirial) where {T<:AbstractFloat,X<:AbstractExclusionStyle}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_mixed!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, c.epsilon, c.sigma_particle, c.rcut_factor; bonds=_mixed_exclusion_bonds(interaction.exclusions))
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_noE_pairs!(rx, ry, rz, fx, fy, fz, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_noE_excl!(rx, ry, rz, fx, fy, fz, nbh, interaction.exclusions.bonds,
                                            box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_pairs!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{LennardJonesPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.lj_forces_soa_excl!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, interaction.exclusions.bonds,
                                        box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_noE_pairs!(rx, ry, rz, fx, fy, fz, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceOnly) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_noE_excl!(rx, ry, rz, fx, fy, fz, nbh, interaction.exclusions.bonds,
                                             box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},NoExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_pairs!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

function compute_nonbonded!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                            nbh::NeighborLists.AbstractNeighborMatrix,
                            box::Definitions.Box3{T},
                            interaction::NonBondedInteraction{WCAPotential,PairMatrixCoefficients{T},BondExclusions},
                            ::ForceEnergyVirial) where {T<:AbstractFloat}
    c = interaction.coefficients
    NonBondedForces.wca_forces_soa_excl!(rx, ry, rz, fx, fy, fz, Epot, V, nbh, interaction.exclusions.bonds,
                                         box, c.typeid, c.sigma_pair, c.epsilon_pair, c.rcut_pair)
    return nothing
end

end
