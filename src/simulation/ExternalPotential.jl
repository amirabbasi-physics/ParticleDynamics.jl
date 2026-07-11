# -------------------------
# External potential providers
# -------------------------
# Opt-in hook that routes all force evaluation of a `SimulationState` through
# a user-supplied provider (e.g. a machine-learned interatomic potential).
# The provider fully replaces the internal nonbonded + bonded force terms.

"""
    AbstractExternalPotential

Supertype for external force providers. A provider overloads
[`external_forces!`](@ref) and is activated with
[`attach_external_potential!`](@ref). While attached, it replaces every
internal force term (nonbonded and bonded) of the state.
"""
abstract type AbstractExternalPotential end

"""
    external_forces!(pot, st::SimulationState, compute_energy::Bool)

Provider contract, called from `evaluate_forces_into_f!` once per force
evaluation:

- Overwrite `st.fx`, `st.fy` (and `st.fz` in 3D) completely; do not
  accumulate onto existing values.
- When `compute_energy` is true, fill `st.Epot` with the total potential
  energy smeared uniformly (`E/N` per slot — the sum is exact; per-particle
  values are not meaningful) and zero `st.virial_nonbonded`. External
  potentials do not provide a virial; pressure observables are unsupported
  while a provider is attached.
"""
function external_forces! end

"""
    attach_external_potential!(st, pot::AbstractExternalPotential) -> st

Route all subsequent force evaluations of `st` through `pot`. Requirements:

- bond-free state (external potentials replace bonded terms too), and
- `spatial_reorder=false` at build time: reordering permutes storage order,
  which would silently break the provider's particle↔index correspondence.

Use [`detach_external_potential!`](@ref) to restore the internal force path.
"""
function attach_external_potential!(st::SimulationState, pot::AbstractExternalPotential)
    st.bonds === nothing ||
        error("external potentials replace all force terms; states with bonds are unsupported")
    st.tag === nothing ||
        error("external potentials require spatial_reorder=false (storage order must stay fixed)")
    st.external_potential = pot
    return st
end

"""
    detach_external_potential!(st) -> st

Remove the external provider and restore the internal nonbonded/bonded path.
"""
detach_external_potential!(st::SimulationState) = (st.external_potential = nothing; st)
