# =========================
#   Conservative configuration evaluation
# =========================
#
# Side-effect-free evaluation of the declared potential at the current
# coordinates and box, shared by MD, minimization and pressure control. It
# never advances clocks, draws thermostat noise, touches heat/work or
# collision accounting, or calls writers; ordinary stepping performs its own
# accounting around force evaluations.

"""
    EvaluationRequest(; forces=true, energy=false, virial=false)

Quantities requested from [`evaluate_configuration!`](@ref). Energy and virial
are produced by the same kernels, so requesting either refreshes both.
"""
struct EvaluationRequest
    forces::Bool
    energy::Bool
    virial::Bool
end

EvaluationRequest(; forces::Bool=true, energy::Bool=false, virial::Bool=false) =
    EvaluationRequest(forces, energy, virial)

"""
    evaluate_configuration!(st, request=EvaluationRequest(); freeze_spring=st.force_freeze_spring) -> st

Make the active force slot, and the energy/virial buffers when requested,
describe the current coordinates. Buffers that are already current are
reused. An energy or virial request currently also recomputes forces, since
the energy kernels produce them anyway.
"""
function evaluate_configuration!(st::SimulationState,
                                 request::EvaluationRequest=EvaluationRequest();
                                 freeze_spring::Bool=st.force_freeze_spring)
    current = st.force_valid && st.force_freeze_spring == freeze_spring
    if request.energy || request.virial
        (current && st.energy_valid) && return st
        evaluate_current_configuration!(st, true; freeze_spring=freeze_spring)
    elseif request.forces
        current && return st
        evaluate_current_configuration!(st, false; freeze_spring=freeze_spring)
    end
    return st
end

"""
    attach_bonded_terms!(st, terms) -> st

Attach typed bonded families (`BondedForces.BondedTerms`) that are added after
the legacy uniform bond kernels. Interaction tables hold build-time particle
indices, so spatial reordering is disabled (and rejected once it happened).
"""
function attach_bonded_terms!(st::SimulationState, terms::BondedForces.BondedTerms)
    terms.dim == (_is_3d(st) ? 3 : 2) || throw(ArgumentError("bonded terms dimension does not match the state"))
    st.external_potential === nothing ||
        error("the attached external potential replaces all force terms; detach it first (detach_external_potential!)")
    for fam in terms.families
        last_atom = isempty(fam.table.offsets) ? 0 : length(fam.table.offsets) - 1
        last_atom == length(st.rx) || throw(ArgumentError("bonded table $(fam.name) was built for $(last_atom) particles, state has $(length(st.rx))"))
        eltype(fam.lfx) == eltype(st.rx) || throw(ArgumentError("bonded table $(fam.name) precision does not match the state"))
    end
    # Tables hold build-time particle ids; sorted states map them through `tag`/`slot_of`.
    st.bonded = isempty(terms) ? nothing : terms
    invalidate_forces!(st)
    return st
end

function detach_bonded_terms!(st::SimulationState)
    st.bonded = nothing
    invalidate_forces!(st)
    return st
end

"""
    total_potential_energy(st) -> Float64

Whole-system potential energy at the current coordinates, including global
terms that are distributed into `Epot` (the electrostatic mesh energy). The
reduction runs on the device in Float64 even for Float32 states. Stale energy
buffers are refreshed first.
"""
function total_potential_energy(st::SimulationState)
    evaluate_configuration!(st, EvaluationRequest(energy=true))
    return CUDA.mapreduce(Float64, +, st.Epot)
end

"""
    total_virial_trace(st) -> Float64

Trace of the whole-system configurational virial `tr(W)`, `W = Σ r ⊗ F`,
including global terms and the velocity-dependent rigid-water constraint
tensor when attached. Constraint forces are not added to the kick buffers. Device
reduction in Float64; stale buffers are refreshed first.
"""
function total_virial_trace(st::SimulationState)
    evaluate_configuration!(st, EvaluationRequest(virial=true))
    w = CUDA.mapreduce(Float64, +, st.virial)
    if has_global_virial(st)
        D = _is_3d(st) ? 3 : 2
        g = Array(st.virial_global)
        w += Base.sum(Float64, view(g, 1:D))
    end
    return w + constraint_virial_trace(st)
end

"""
    total_configurational_virial(st) -> NamedTuple

Whole-system configurational virial tensor (Float64 components in
[`virial_components`](@ref) order), including global terms.
"""
function total_configurational_virial(st::SimulationState)
    evaluate_configuration!(st, EvaluationRequest(virial=true))
    ncomp = _virial_ncomponents(st)
    comps = ntuple(c -> CUDA.mapreduce(Float64, +, view(st.virial_tensor, :, c)), ncomp)
    if has_global_virial(st)
        g = Array(st.virial_global)
        comps = ntuple(c -> comps[c] + Float64(g[c]), ncomp)
    end
    if st.rigid_water !== nothing
        rigid = _constraint_virial_components(st)
        comps = ntuple(c -> comps[c] + rigid[c], ncomp)
    end
    return NamedTuple{virial_components(st)}(comps)
end

"""
    bonded_energy_components(st) -> Vector{Pair{Symbol,Float64}}

Energy of each typed bonded family at the current coordinates (e.g.
`:angles_HarmonicAngleKind => 3.2`). Legacy uniform bonds are part of `Epot`
only and are not listed.
"""
function bonded_energy_components(st::SimulationState)
    st.bonded === nothing && return Pair{Symbol,Float64}[]
    evaluate_configuration!(st, EvaluationRequest(energy=true))
    return BondedForces.family_energies(st.bonded)
end
