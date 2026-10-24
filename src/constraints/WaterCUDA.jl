mutable struct WaterWorkspace{T}
    atoms::CuVector{NTuple{3,Int32}}
    inverse_masses::CuVector{NTuple{3,T}}
    reference::CuVector{NTuple{3,SVector{3,T}}}
    status::CuVector{Int32}
    lengths2::NTuple{3,T}
    tolerance::T
    maxiter::Int32
    nparticles::Int
    reference_ready::Bool
end

function WaterWorkspace(::Type{T}, triples, masses; oh::Real, hh::Real,
                        tolerance::Real=(T === Float32 ? 2e-6 : 1e-10), maxiter::Integer=50) where {T<:AbstractFloat}
    all(m -> isfinite(m) && m > 0, masses) || throw(ArgumentError("Water masses must be finite and positive"))
    T in (Float32, Float64) || throw(ArgumentError("Water kernels require Float32 or Float64"))
    isfinite(oh) && isfinite(hh) && 0 < hh < 2oh || throw(ArgumentError("Require oh > 0 and 0 < hh < 2oh"))
    isfinite(T(tolerance)) && T(tolerance) > 0 && 0 < maxiter <= typemax(Int32) || throw(ArgumentError("Invalid constraint tolerance/iteration limit"))
    lengths2 = (T(oh)^2, T(oh)^2, T(hh)^2)
    all(x -> isfinite(x) && x > 0, lengths2) || throw(ArgumentError("Constraint lengths are not representable in $T"))
    atoms = NTuple{3,Int32}[Tuple(t) for t in triples]
    isempty(atoms) && throw(ArgumentError("Empty water topology"))
    ids = collect(Iterators.flatten(atoms))
    all(i -> 1 <= i <= length(masses), ids) || throw(ArgumentError("Water atom index out of range"))
    length(unique(ids)) == length(ids) || throw(ArgumentError("Rigid waters must not share atoms"))
    w = [ntuple(k -> inv(T(masses[t[k]])), 3) for t in atoms]
    all(v -> all(x -> isfinite(x) && x > 0, v), w) || throw(ArgumentError("Inverse masses are not representable in $T"))
    return WaterWorkspace(CuArray(atoms), CuArray(w), CuVector{NTuple{3,SVector{3,T}}}(undef, length(atoms)),
                          CUDA.zeros(Int32, length(atoms)), lengths2, T(tolerance), Int32(maxiter), length(masses), false)
end

function _validate_buffers(ws::WaterWorkspace{T}, arrays) where {T}
    all(a -> length(a) == ws.nparticles && eltype(a) === T, arrays) ||
        throw(ArgumentError("Constraint buffers must match the workspace particle count and precision"))
    return nothing
end

function _water_box(ws::WaterWorkspace{T}, box) where {T}
    length(box) == 3 || throw(ArgumentError("Rigid water requires a 3D orthorhombic box"))
    L = T.(Tuple(box))
    all(x -> isfinite(x) && x > 2sqrt(maximum(ws.lengths2)), L) ||
        throw(ArgumentError("Box must be finite and exceed twice each constrained distance"))
    return L
end

@inline function _load_water(rx::AbstractVector{T}, ry::AbstractVector{T},
                             rz::AbstractVector{T}, ids::NTuple{3,Int32}) where {T}
    return (SVector{3,T}(rx[ids[1]], ry[ids[1]], rz[ids[1]]),
            SVector{3,T}(rx[ids[2]], ry[ids[2]], rz[ids[2]]),
            SVector{3,T}(rx[ids[3]], ry[ids[3]], rz[ids[3]]))
end

function _save_water_kernel!(ref, atoms, rx, ry, rz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return
    @inbounds ref[i] = _load_water(rx, ry, rz, atoms[i])
    return
end

function save_reference!(ws::WaterWorkspace, rx, ry, rz)
    _validate_buffers(ws, (rx,ry,rz))
    @cuda threads=128 blocks=cld(length(ws.atoms), 128) _save_water_kernel!(ws.reference, ws.atoms, rx, ry, rz)
    ws.reference_ready = true
    return ws
end

function _shake_water_kernel!(atoms, weights, ref, status, rx, ry, rz, vx, vy, vz, rxu, ryu, rzu,
                              box, lengths2, tol, maxiter, h, ::Val{UNWRAP}) where {UNWRAP}
    box = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return
    @inbounds begin
        ids = atoms[i]
        origin = ref[i][1]
        old = local_water(ref[i], origin, box)
        trial = local_water(_load_water(rx, ry, rz, ids), origin, box)
        x, ok, _ = shake_water(old, trial, weights[i], lengths2, tol, maxiter)
        status[i] = ok ? Int32(0) : Int32(1)
        ok || return
        for k in 1:3
            j = ids[k]
            dx = x[k] - trial[k]
            wrapped = _image(origin + x[k], box)
            # Explicit indexing avoids device-side iteration over SVector.
            rx[j] = wrapped[1]
            ry[j] = wrapped[2]
            rz[j] = wrapped[3]
            vx[j] += dx[1] / h; vy[j] += dx[2] / h; vz[j] += dx[3] / h
            if UNWRAP
                rxu[j] += dx[1]; ryu[j] += dx[2]; rzu[j] += dx[3]
            end
        end
    end
    return
end

function check_status(ws::WaterWorkspace)
    maximum(ws.status) == 0 || throw(DomainError(:rigid_water, "Rigid water constraints failed; configuration is not valid"))
    return nothing
end

"""
    constrain_positions!(ws, rx, ry, rz, vx, vy, vz, box, h; unwrapped=nothing)

After saving reference positions and drifting by h, apply SHAKE and add the
position correction divided by h to velocities. Does not replace the final
RATTLE velocity projection. The caller owns force/energy invalidation.
The solve tolerance applies in local coordinates. Storing absolute Float32
coordinates adds box-dependent rounding; the integration acceptance tests
must measure the distances again after storage, especially in large boxes.
"""
function constrain_positions!(ws::WaterWorkspace{T}, rx, ry, rz, vx, vy, vz, box, h;
                              unwrapped=nothing) where {T}
    T(h) > 0 && isfinite(T(h)) || throw(ArgumentError("Constraint drift interval must be positive and representable"))
    ws.reference_ready || throw(ArgumentError("Save constraint reference coordinates before each drift"))
    _validate_buffers(ws, (rx,ry,rz,vx,vy,vz))
    L = _water_box(ws, box)
    u = unwrapped === nothing ? (rx, ry, rz) : unwrapped
    length(u) == 3 || throw(ArgumentError("Expected three unwrapped coordinate buffers"))
    _validate_buffers(ws, u)
    @cuda threads=128 blocks=cld(length(ws.atoms), 128) _shake_water_kernel!(ws.atoms, ws.inverse_masses,
        ws.reference, ws.status, rx, ry, rz, vx, vy, vz, u..., L, ws.lengths2,
        ws.tolerance, ws.maxiter, T(h), Val(unwrapped !== nothing))
    ws.reference_ready = false
    check_status(ws)
    return nothing
end

function _rattle_water_kernel!(atoms, weights, status, rx, ry, rz, vx, vy, vz, box)
    box = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return
    @inbounds begin
        ids = atoms[i]
        raw = _load_water(rx, ry, rz, ids)
        x = local_water(raw, raw[1], box)
        v, ok = rattle_water(x, _load_water(vx, vy, vz, ids), weights[i])
        status[i] = ok ? Int32(0) : Int32(2)
        ok || return
        for k in 1:3
            j = ids[k]
            vx[j] = v[k][1]
            vy[j] = v[k][2]
            vz[j] = v[k][3]
        end
    end
    return
end

function constrain_velocities!(ws::WaterWorkspace{T}, rx, ry, rz, vx, vy, vz, box) where {T}
    _validate_buffers(ws, (rx,ry,rz,vx,vy,vz))
    L = _water_box(ws, box)
    @cuda threads=128 blocks=cld(length(ws.atoms), 128) _rattle_water_kernel!(ws.atoms, ws.inverse_masses,
        ws.status, rx, ry, rz, vx, vy, vz, L)
    check_status(ws)
    return nothing
end

function _constraint_virial_kernel!(out, atoms, weights, status, rx, ry, rz, vx, vy, vz, fx, fy, fz, box)
    box = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return
    @inbounds begin
        ids = atoms[i]
        raw = _load_water(rx, ry, rz, ids)
        x = local_water(raw, raw[1], box)
        f, ok = constraint_force_water(x, _load_water(vx, vy, vz, ids), _load_water(fx, fy, fz, ids), weights[i])
        status[i] = ok ? Int32(0) : Int32(3)
        tensor = constraint_tensor_water(x, f)
        out[i,1] = tensor[1]
        out[i,2] = tensor[2]
        out[i,3] = tensor[3]
        out[i,4] = tensor[4]
        out[i,5] = tensor[5]
        out[i,6] = tensor[6]
    end
    return
end

function constraint_virial!(out, ws::WaterWorkspace, r, v, f, box)
    _validate_buffers(ws, (r..., v..., f...))
    size(out) == (length(ws.atoms), 6) || throw(ArgumentError("Invalid constraint virial buffer"))
    L = _water_box(ws, box)
    @cuda threads=128 blocks=cld(length(ws.atoms),128) _constraint_virial_kernel!(out, ws.atoms,
        ws.inverse_masses, ws.status, r..., v..., f..., L)
    check_status(ws)
    return out
end
