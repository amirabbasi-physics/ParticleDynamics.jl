# Feature: external force providers (SimulationCore `ExternalPotential.jl`).
# A provider replaces every force term, so the engine launches no pair or bonded
# force kernels. Each step is split into graph segments that end where the staged
# path evaluates forces (split graph per step). Between
# segments the host calls `external_forces!(provider, st, compute_energy)`, which
# overwrites `st.fx…` (the run's force buffers) and, with energies, fills `st.Epot`
# and zeroes `st.virial_nonbonded`. Euler–Heun calls it at the midpoint (x_m in
# `st.rx`, energy off) and at the final positions. As staged, the virial is not
# combined. User restraints survive on top of the provider:
# the after-force hooks (freeze tether) open the segment that follows each host
# call. Typed bonded terms cannot be combined with a provider (attach errors).

struct ExternalPotentialFeature <: EngineFeature end
make_feature(::Type{ExternalPotentialFeature},st,spec)=st.external_potential===nothing ? nothing :
    ExternalPotentialFeature()
host_forces(::ExternalPotentialFeature)=true
function host_force!(::ExternalPotentialFeature,st,energy::Bool)
    SC.external_forces!(st.external_potential,st,energy)
    return nothing
end

"""Provider states the engine runs: any pair description and list layout (the
provider replaces the forces; the engine keeps a neighbor list only for features
that read it), no sorting tags, no bonded or electrostatic terms (the attach
functions reject them), no rigid water or cluster backend yet. Collision counting
also needs the dense list layout for its staged release path."""
function external_supported(st)
    return (st.rz!==nothing ? st.box3!==nothing : st.box2!==nothing) && st.tag===nothing &&
        st.bonds===nothing && st.bonded===nothing && st.electrostatics===nothing &&
        st.rigid_water===nothing && st.cluster===nothing &&
        (!st.coll_enabled || (st.nbh isa NL.NeighborMatrix && st.nonbonded_backend===:ell)) &&
        _max_skin(st)>0
end
