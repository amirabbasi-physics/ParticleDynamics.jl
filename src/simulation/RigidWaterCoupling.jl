# Fixed-cell, constrained velocity-Verlet integration. The pressure tensor is
# evaluated separately from conservative forces, using current velocities.
struct RigidWaterState{T}
    workspace::RigidWaterConstraints.WaterWorkspace{T}
    virial::CuMatrix{T}
    virial_accum::CuMatrix{T}
    in_water::CuVector{Bool}   # per particle: member of a rigid water (molecular cell scaling)
end

"""
    attach_rigid_water!(st, triples; oh, hh, tolerance, maxiter=50)

Attach disjoint `(O,H1,H2)` triples to a 3D state before stepping. Coordinates
must already satisfy the supplied lengths; velocities are projected once.
Masses come from the state. Spatial reordering is disabled. This low-level
API supports `nve` only; it does not install SPC/E parameters or nonbonded
exclusions. The caller must configure those separately. No COM DOF are
subtracted implicitly. Changing masses after attachment is unsupported.
"""
function attach_rigid_water!(st::SimulationState{T}, triples; oh::Real, hh::Real,
                             tolerance::Real=(T === Float32 ? 2e-6 : 1e-10),
                             maxiter::Integer=50) where {T}
    _is_3d(st) || throw(ArgumentError("Rigid water requires a 3D state"))
    st.rigid_water === nothing || throw(ArgumentError("Rigid water is already attached"))
    st.step == 0 || throw(ArgumentError("Attach rigid water before stepping"))
    st.last_reorder_step < 0 || throw(ArgumentError("Attach rigid water before spatial reordering"))
    st.freeze_mode == FREEZE_NONE && !st.coll_enabled ||
        throw(ArgumentError("Rigid water does not support freeze or collision controls"))
    masses = st.mass_particle === nothing ? fill(st.mass, length(st.rx)) : Array(st.mass_particle)
    ws = RigidWaterConstraints.WaterWorkspace(T, triples, masses; oh, hh, tolerance, maxiter)
    box = RigidWaterConstraints._water_box(ws, st.box3)
    r = (Array(st.rx), Array(st.ry), Array(st.rz))
    # A one-time host geometry check avoids silently changing initial energy.
    # Allow the representational error of absolute coordinates, separately
    # from the local solver tolerance (especially relevant for Float32).
    storage_tol = 16eps(T) * maximum(box) / sqrt(minimum(ws.lengths2))
    for ids in Array(ws.atoms)
        raw = RigidWaterConstraints._load_water(r..., ids)
        x = RigidWaterConstraints.local_water(raw, raw[1], box)
        RigidWaterConstraints.distance_error(x, ws.lengths2) <= max(ws.tolerance, storage_tol) ||
            throw(ArgumentError("Initial rigid-water coordinates do not satisfy the requested lengths"))
    end
    in_water = falses(length(st.rx))
    for t in triples, j in t
        in_water[j] = true
    end
    rigid = RigidWaterState(ws, CUDA.zeros(T, length(ws.atoms), 6), CUDA.zeros(T, length(ws.atoms), 6), CuVector{Bool}(in_water))
    RigidWaterConstraints.constrain_velocities!(ws, st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, box)
    st.rigid_water = rigid
    st.tag = nothing
    invalidate_forces!(st)
    return st
end

"""
    kinetic_dof(st; remove_com=false)

Whole-system kinetic DOF, including three constraints per rigid water.
Set `remove_com=true` only when total COM momentum was actually removed.
This helper does not infer COM removal from the initial velocity generator.
"""
function kinetic_dof(st::SimulationState; remove_com::Bool=false)
    st.freeze_mode == FREEZE_NONE || throw(ArgumentError("kinetic_dof requires all particles mobile"))
    d = _is_3d(st) ? 3 : 2
    nc = st.rigid_water === nothing ? 0 : 3length(st.rigid_water.workspace.atoms)
    return max(0, d * length(st.rx) - nc - (remove_com ? d : 0))
end

function _validate_rigid_integrator!(st, spec)
    st.rigid_water === nothing && return nothing
    spec isa NVESpec || throw(ArgumentError("Rigid water currently supports only fixed-cell NVE"))
    st.freeze_mode == FREEZE_NONE && !st.coll_enabled ||
        throw(ArgumentError("Rigid water does not support freeze or collision controls"))
    return nothing
end

function _save_rigid_reference!(st)
    st.rigid_water === nothing && return nothing
    RigidWaterConstraints.save_reference!(st.rigid_water.workspace, st.rx, st.ry, st.rz)
    return nothing
end



function _constraint_virial_buffer(st; accumulated::Bool=false)
    rigid = st.rigid_water
    accumulated && return rigid.virial_accum
    # The constraint force depends on velocities as well as coordinates.
    # Recompute on demand, even when conservative forces remain valid.
    evaluate_configuration!(st)
    RigidWaterConstraints.constraint_virial!(rigid.virial, rigid.workspace,
        (st.rx, st.ry, st.rz), (st.vx, st.vy, st.vz), (st.fx, st.fy, st.fz), st.box3)
    return rigid.virial
end

function constraint_virial_trace(st; accumulated::Bool=false)
    st.rigid_water === nothing && return 0.0
    v = _constraint_virial_buffer(st; accumulated)
    return sum(c -> CUDA.mapreduce(Float64, +, view(v, :, c)), 1:3)
end

# Molecular virial (rigid water): W_mol = W_atom − Σ_i (r_i − R_M)·F_i, with F the
# non-constraint forces and R_M the water's centre of mass; the molecule is made whole
# around its oxygen. Constraint forces drop out (they sum to zero in each molecule).
function _intramolecular_virial_kernel!(out, rx, ry, rz, fx, fy, fz, atoms, inverse_masses, box::NTuple{3,T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return
    @inbounds begin
        ids = atoms[i]; w = inverse_masses[i]
        raw = RigidWaterConstraints._load_water(rx, ry, rz, ids)
        x = RigidWaterConstraints.local_water(raw, raw[1], box)
        m1 = one(T) / w[1]; m2 = one(T) / w[2]; m3 = one(T) / w[3]
        c = (m1 * x[1] + m2 * x[2] + m3 * x[3]) / (m1 + m2 + m3)
        acc = 0.0
        for k in 1:3
            j = ids[k]; d = x[k] - c
            acc += Float64(d[1]) * Float64(fx[j]) + Float64(d[2]) * Float64(fy[j]) + Float64(d[3]) * Float64(fz[j])
        end
        out[i] = acc
    end
    return
end

"""
    molecular_virial_trace(st) -> Float64

Trace of the molecular virial `Σ_M R_M·F_M` at the current coordinates: the
configurational (pair, mesh and bonded) virial without the constraint part, minus
the intramolecular term `Σ_i (r_i − R_M)·F_i` of every rigid water. Particles outside
waters count as their own molecules. Without rigid water it is the configurational
virial.
"""
function molecular_virial_trace(st)
    w = total_virial_trace(st)          # refreshes forces and virials at the current coordinates
    st.rigid_water === nothing && return w
    w -= constraint_virial_trace(st)
    ws = st.rigid_water.workspace; n = length(ws.atoms)
    out = CUDA.zeros(Float64, n)
    CUDA.@cuda threads=128 blocks=cld(n, 128) _intramolecular_virial_kernel!(out, st.rx, st.ry, st.rz,
        st.fx, st.fy, st.fz, ws.atoms, ws.inverse_masses, st.box3)
    return w - Base.sum(out)
end

function _constraint_virial_components(st; accumulated::Bool=false)
    st.rigid_water === nothing && return ntuple(_ -> 0.0, 6)
    v = _constraint_virial_buffer(st; accumulated)
    return ntuple(c -> CUDA.mapreduce(Float64, +, view(v, :, c)), 6)
end
