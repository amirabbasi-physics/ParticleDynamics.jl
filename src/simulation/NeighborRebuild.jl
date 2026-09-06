"""
    plan_neighbor_rebuild!(st, dt) -> Bool

Determine whether step-boundary neighbor maintenance is due. Force-time
coverage is checked separately by `ensure_force_neighbors!`.
"""
function plan_neighbor_rebuild!(st::SimulationState{T}, dt::T) where {T<:AbstractFloat}
    do_check = (st.step % st.neigh_interval == 0)
    do_check || return false
    if _is_3d(st)
        return NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                            skin=st.nbh.skin,
                                            Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                            step=st.step)
    end
    return NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
end

"""
    _spatial_reorder_active(st, spec) -> Bool

Spatial reordering permutes every per-particle array into cell-sorted order at
rebuild time so that force kernels read neighbor data from nearby memory. It
requires that nothing outside `st` holds per-particle data keyed by storage
index: deterministic NVE carries no such integrator state, and the gate
excludes bonds, freeze controls, and collision counting. `st.tag` maps storage
slots back to build-time particle ids for I/O.
"""
_spatial_reorder_active(st::SimulationState, spec::IntegratorSpec) = false

function _spatial_reorder_active(st::SimulationState, spec::NVESpec)
    return st.tag !== nothing &&
           (st.last_reorder_step < 0 ||
            st.step - st.last_reorder_step >= st.reorder_interval) &&
           st.nbh isa NeighborLists.NeighborMatrix &&
           st.bonds === nothing &&
           st.freeze_mask === nothing &&
           !st.coll_enabled
end

# Gather every per-particle array into the cell-sorted order `perm`
# (slot k receives the particle previously stored at `perm[k]`).
function _permute_particle_state!(st::SimulationState{T},
                                  perm::CuArray{Int32,1}) where {T<:AbstractFloat}
    st.rx = st.rx[perm]; st.ry = st.ry[perm]
    st.rz === nothing || (st.rz = (st.rz::CuArray{T,1})[perm])
    if st.rx_unwrap !== nothing
        st.rx_unwrap = (st.rx_unwrap::CuArray{T,1})[perm]
        st.ry_unwrap = (st.ry_unwrap::CuArray{T,1})[perm]
        st.rz_unwrap === nothing || (st.rz_unwrap = (st.rz_unwrap::CuArray{T,1})[perm])
    end
    st.vx = st.vx[perm]; st.vy = st.vy[perm]
    st.vz === nothing || (st.vz = (st.vz::CuArray{T,1})[perm])
    st.fx = st.fx[perm]; st.fy = st.fy[perm]
    st.fz === nothing || (st.fz = (st.fz::CuArray{T,1})[perm])
    st.f0x = st.f0x[perm]; st.f0y = st.f0y[perm]
    st.f0z === nothing || (st.f0z = (st.f0z::CuArray{T,1})[perm])
    st.typeid = st.typeid[perm]
    st.sigma_particle === nothing || (st.sigma_particle = (st.sigma_particle::CuArray{T,1})[perm])
    st.mass_particle === nothing || (st.mass_particle = (st.mass_particle::CuArray{T,1})[perm])
    st.inv_mass_particle === nothing || (st.inv_mass_particle = (st.inv_mass_particle::CuArray{T,1})[perm])
    st.Epot = st.Epot[perm]; st.dq = st.dq[perm]; st.dU = st.dU[perm]
    st.Ekin = st.Ekin[perm]; st.virial = st.virial[perm]
    st.virial_nonbonded = st.virial_nonbonded[perm, :]
    st.virial_bonded = st.virial_bonded[perm, :]
    st.virial_tensor = st.virial_tensor[perm, :]
    st.Epot_accum = st.Epot_accum[perm]
    st.Ekin_accum = st.Ekin_accum[perm]
    st.virial_accum = st.virial_accum[perm]
    st.virial_tensor_accum = st.virial_tensor_accum[perm, :]
    st.tag = (st.tag::CuArray{Int32,1})[perm]

    # After permuting the storage, the cell-sorted order *is* the storage
    # order: fix up the binning bookkeeping accordingly.
    nbh = st.nbh
    nbh.cell_of_particle = nbh.cell_of_particle[perm]
    nbh.particle_ids_sorted .= Int32(1):Int32(length(perm))
    return nothing
end

# Rebuild with spatial reordering: bin, permute the particle storage into
# cell-sorted order, then build rows against the permuted coordinates.
function _rebuild_with_reorder!(st::SimulationState{T}) where {T<:AbstractFloat}
    nbh = st.nbh::NeighborLists.NeighborMatrix{T}
    if _is_3d(st)
        box = st.box3::Definitions.Box3{T}
        rz = st.rz::CuArray{T,1}
        NeighborLists._bin_particles!(nbh, st.rx, st.ry, rz, box)
        _permute_particle_state!(st, nbh.particle_ids_sorted)
        NeighborLists._finish_rebuild!(nbh, st.rx, st.ry, st.rz::CuArray{T,1}, box, st.step)
    else
        box = st.box2::Definitions.Box2{T}
        NeighborLists._bin_particles!(nbh, st.rx, st.ry, box)
        _permute_particle_state!(st, nbh.particle_ids_sorted)
        NeighborLists._finish_rebuild!(nbh, st.rx, st.ry, box, st.step)
    end
    st.last_reorder_step = st.step
    return nothing
end

"""
    apply_neighbor_rebuild_if_needed!(st, spec, rebuild_needed)

Apply neighbor-list rebuild and integrator-independent post-rebuild hooks,
spatially reordering the particle storage when the configuration allows it.
"""
function apply_neighbor_rebuild_if_needed!(st::SimulationState,
                                           spec::IntegratorSpec,
                                           rebuild_needed::Bool)
    rebuild_needed || return nothing
    if _spatial_reorder_active(st, spec)
        _rebuild_with_reorder!(st)
    elseif _is_3d(st)
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    else
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
    end
    _collisions_reinit_on_rebuild!(st; preserve_history=true)
    return nothing
end

"""
    apply_post_position_hooks!(st, stage_tag; freeze_hold=false)

Apply integrator-independent post-position hooks after a position-changing
stage.
"""
function apply_post_position_hooks!(st::SimulationState,
                                    stage_tag::Symbol;
                                    freeze_hold::Bool=false)
    if freeze_hold
        _apply_freeze_hold_positions!(st)
    end
    invalidate_forces!(st)
    if st.coll_enabled
        ensure_force_neighbors!(st)
        _collisions_update_after_positions!(st)
    end
    return nothing
end

"""
Check list coverage at the actual force/collision coordinates, including
midpoints. Never reorder here: saved integration-stage buffers retain their
particle indexing. `neigh_interval` only gates step-boundary maintenance.
"""
function ensure_force_neighbors!(st::SimulationState{T}) where {T}
    st.nbh isa NeighborLists.AllPairsNeighborMatrix && return nothing
    needed = if _is_3d(st)
        NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
            skin=st.nbh.skin, Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3], step=st.step)
    else
        NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
            skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=st.step)
    end
    if needed
        if _is_3d(st)
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
        else
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
        end
        _collisions_reinit_on_rebuild!(st; preserve_history=true)
    end
    return nothing
end
