"""
    plan_neighbor_rebuild!(st, dt) -> Bool

Determine whether neighbor lists should be rebuilt on this step.
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
    apply_neighbor_rebuild_if_needed!(st, rebuild_needed)

Apply neighbor-list rebuild and integrator-independent post-rebuild hooks.
"""
function apply_neighbor_rebuild_if_needed!(st::SimulationState,
                                           rebuild_needed::Bool)
    rebuild_needed || return nothing
    if _is_3d(st)
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    else
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
    end
    _collisions_reinit_on_rebuild!(st)
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
    _collisions_update_after_positions!(st)
    return nothing
end
