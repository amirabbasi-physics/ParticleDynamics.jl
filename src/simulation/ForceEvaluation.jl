# -------------------------
# Bond helpers (2D / 3D)
# -------------------------
function _apply_bonds2!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1},
                        E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool,
                        V::Union{Nothing,CuArray{T,2}}=nothing) where {T<:AbstractFloat}
    if (st.bonds === nothing) || (st.bonding === nothing)
        V === nothing || fill!(V, zero(T))
        return
    end
    if st.bonding isa Definitions.HarmonicBond{T}
        p = (st.bonding::Definitions.HarmonicBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, fx, fy, E, st.bonds, st.box2::Definitions.Box2{T}, p)
            else
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, fx, fy, E, V, st.bonds, st.box2::Definitions.Box2{T}, p)
            end
        else
            BondedForces.harmonic_forces_soa_noE!(st.rx, st.ry, fx, fy, st.bonds, st.box2::Definitions.Box2{T}, p)
        end
    elseif st.bonding isa Definitions.FENEBond{T}
        p = (st.bonding::Definitions.FENEBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.fene_forces_soa!(st.rx, st.ry, fx, fy, E, st.bonds, st.box2::Definitions.Box2{T}, p)
            else
                BondedForces.fene_forces_soa!(st.rx, st.ry, fx, fy, E, V, st.bonds, st.box2::Definitions.Box2{T}, p)
            end
        else
            BondedForces.fene_forces_soa_noE!(st.rx, st.ry, fx, fy, st.bonds, st.box2::Definitions.Box2{T}, p)
        end
    end
    return
end

function _apply_bonds3!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool,
                        V::Union{Nothing,CuArray{T,2}}=nothing) where {T<:AbstractFloat}
    if (st.bonds === nothing) || (st.bonding === nothing)
        V === nothing || fill!(V, zero(T))
        return
    end
    if st.bonding isa Definitions.HarmonicBond{T}
        p = (st.bonding::Definitions.HarmonicBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, st.bonds, st.box3::Definitions.Box3{T}, p)
            else
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, V, st.bonds, st.box3::Definitions.Box3{T}, p)
            end
        else
            BondedForces.harmonic_forces_soa_noE!(st.rx, st.ry, st.rz, fx, fy, fz, st.bonds, st.box3::Definitions.Box3{T}, p)
        end
    elseif st.bonding isa Definitions.FENEBond{T}
        p = (st.bonding::Definitions.FENEBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.fene_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, st.bonds, st.box3::Definitions.Box3{T}, p)
            else
                BondedForces.fene_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, V, st.bonds, st.box3::Definitions.Box3{T}, p)
            end
        else
            BondedForces.fene_forces_soa_noE!(st.rx, st.ry, st.rz, fx, fy, fz, st.bonds, st.box3::Definitions.Box3{T}, p)
        end
    end
    return
end

function _compute_final_nonbonded2!(st::SimulationState{T}, compute_energy::Bool) where {T<:AbstractFloat}
    interaction = _nonbonded_interaction(st)
    if compute_energy
        NonBondedInteractions.compute_nonbonded!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.box2::Definitions.Box2{T},
                                                 interaction, NonBondedInteractions.ForceEnergyVirial())
    else
        NonBondedInteractions.compute_nonbonded!(st.rx, st.ry, st.fx, st.fy,
                                                 st.nbh, st.box2::Definitions.Box2{T},
                                                 interaction, NonBondedInteractions.ForceOnly())
    end
    return nothing
end

function _compute_final_nonbonded3!(st::SimulationState{T}, compute_energy::Bool) where {T<:AbstractFloat}
    interaction = _nonbonded_interaction(st)
    if compute_energy
        NonBondedInteractions.compute_nonbonded!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.box3::Definitions.Box3{T},
                                                 interaction, NonBondedInteractions.ForceEnergyVirial())
    else
        NonBondedInteractions.compute_nonbonded!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                 st.nbh, st.box3::Definitions.Box3{T},
                                                 interaction, NonBondedInteractions.ForceOnly())
    end
    return nothing
end

@inline _nonbonded_exclusions(st::SimulationState) =
    (st.bonds === nothing || !st.exclude_bonded) ? NonBondedInteractions.NoExclusions() :
                                                   NonBondedInteractions.BondExclusions(st.bonds)

function _nonbonded_interaction(st::SimulationState{T}) where {T<:AbstractFloat}
    if st.nb_kind == NB_KIND_LJ
        if st.sigma_pair !== nothing
            @assert st.epsilon_pair !== nothing && st.rcut_pair !== nothing "pair-matrix LJ coefficients are incomplete"
            return NonBondedInteractions.NonBondedInteraction(
                NonBondedInteractions.LennardJonesPotential(),
                NonBondedInteractions.PairMatrixCoefficients(st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair),
                _nonbonded_exclusions(st),
            )
        elseif st.sigma_particle !== nothing
            return NonBondedInteractions.NonBondedInteraction(
                NonBondedInteractions.LennardJonesPotential(),
                NonBondedInteractions.MixedSigmaCoefficients(st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor),
                NonBondedInteractions.NoExclusions(),
            )
        end
        return NonBondedInteractions.NonBondedInteraction(
            NonBondedInteractions.LennardJonesPotential(),
            NonBondedInteractions.UniformLJCoefficients(st.pair_lj),
            _nonbonded_exclusions(st),
        )
    elseif st.nb_kind == NB_KIND_WCA
        if st.sigma_pair !== nothing
            @assert st.epsilon_pair !== nothing && st.rcut_pair !== nothing "pair-matrix WCA coefficients are incomplete"
            return NonBondedInteractions.NonBondedInteraction(
                NonBondedInteractions.WCAPotential(),
                NonBondedInteractions.PairMatrixCoefficients(st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair),
                _nonbonded_exclusions(st),
            )
        elseif st.sigma_particle !== nothing
            return NonBondedInteractions.NonBondedInteraction(
                NonBondedInteractions.WCAPotential(),
                NonBondedInteractions.MixedSigmaCoefficients(st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor),
                NonBondedInteractions.NoExclusions(),
            )
        end
        return NonBondedInteractions.NonBondedInteraction(
            NonBondedInteractions.WCAPotential(),
            NonBondedInteractions.UniformLJCoefficients(st.pair_lj),
            _nonbonded_exclusions(st),
        )
    end

    @assert st.softrep !== nothing "softrep params missing"
    return NonBondedInteractions.NonBondedInteraction(
        NonBondedInteractions.SoftRepulsivePotential(),
        NonBondedInteractions.UniformSoftRepCoefficients(st.softrep),
        _nonbonded_exclusions(st),
    )
end

function _finalize_force_eval2!(st::SimulationState{T}, compute_energy::Bool, freeze_spring::Bool) where {T<:AbstractFloat}
    _apply_bonds2!(st, st.fx, st.fy, compute_energy ? st.Epot : nothing, compute_energy,
                   compute_energy ? st.virial_bonded : nothing)
    if freeze_spring
        _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                              compute_energy ? st.Epot : nothing, compute_energy)
    end
    if compute_energy
        _combine_virial!(st)
    end
    return nothing
end

function _finalize_force_eval3!(st::SimulationState{T}, compute_energy::Bool, freeze_spring::Bool) where {T<:AbstractFloat}
    _apply_bonds3!(st, st.fx, st.fy, st.fz, compute_energy ? st.Epot : nothing, compute_energy,
                   compute_energy ? st.virial_bonded : nothing)
    if freeze_spring
        _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                              compute_energy ? st.Epot : nothing, compute_energy)
    end
    if compute_energy
        _combine_virial!(st)
    end
    return nothing
end

"""
    evaluate_forces_into_f!(st, compute_energy; freeze_spring=false)

Evaluate nonbonded + bonded + optional freeze-spring contributions into the
active force slot (`f`).
"""
function evaluate_forces_into_f!(st::SimulationState{T},
                                 compute_energy::Bool;
                                 freeze_spring::Bool=false) where {T<:AbstractFloat}
    if st.external_potential !== nothing
        external_forces!(st.external_potential, st, compute_energy)
        return nothing
    end
    # build_simulation allocates an unbuilt list until real coordinates exist.
    # A failed capacity check also leaves the list invalid and must never be
    # followed by an unchecked force read.
    if st.nbh isa NeighborLists.CellListNeighborMatrix && !st.nbh.valid
        if _is_3d(st)
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
        else
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
        end
        _collisions_reinit_on_rebuild!(st)
    end
    if _is_3d(st)
        _compute_final_nonbonded3!(st, compute_energy)
        _finalize_force_eval3!(st, compute_energy, freeze_spring)
    else
        _compute_final_nonbonded2!(st, compute_energy)
        _finalize_force_eval2!(st, compute_energy, freeze_spring)
    end
    return nothing
end

"""
    evaluate_forces_into_f0!(st, compute_energy; freeze_spring=false)

Evaluate forces into the reference force slot (`f0`) without changing the
external slot ownership.
"""
function evaluate_forces_into_f0!(st::SimulationState{T},
                                  compute_energy::Bool;
                                  freeze_spring::Bool=false) where {T<:AbstractFloat}
    _swap_force_slots!(st)
    try
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    finally
        _swap_force_slots!(st)
    end
    return nothing
end

"""
    evaluate_midpoint_forces_into_f0!(st; freeze_spring=false)

Evaluate forces at midpoint coordinates (stored in `vx,vy[,vz]`) into the
reference force slot (`f0`). This helper is used by Brownian midpoint / EM.
"""
function evaluate_midpoint_forces_into_f0!(st::SimulationState{T};
                                           freeze_spring::Bool=false) where {T<:AbstractFloat}
    _swap_midpoint_position_slots!(st)
    _swap_force_slots!(st)
    try
        evaluate_forces_into_f!(st, false; freeze_spring=freeze_spring)
    finally
        _swap_force_slots!(st)
        _swap_midpoint_position_slots!(st)
    end
    return nothing
end
