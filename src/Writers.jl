module Writers

using CUDA
using Printf
using StaticArrays
using GSDFiles
using DelimitedFiles

export InMemoryLogger, CSVWriter, XYZWriter, ObservableCSVWriter
export write_xyz!, write_observables_csv!
export gsd_open, gsd_close, write_gsd_frame!, read_last_gsd

# =======================================================================
# Simple in-memory logger (kept for API completeness)
# =======================================================================
abstract type Writer end

mutable struct InMemoryLogger <: Writer
    every::Int
    data::Dict{String, Vector}
    steps::Vector{Int}
end

function InMemoryLogger(data_keys; every::Int=1)
    data = Dict{String, Vector}()
    for key in data_keys
        data[string(key)] = Vector{Any}()
    end
    return InMemoryLogger(every, data, Int[])
end

function write!(w::InMemoryLogger, _simulation, step::Int, _dt::Real)
    if step % w.every != 0
        return
    end
    push!(w.steps, step)
    return nothing
end

# =======================================================================
# CSV observables (host)
# =======================================================================

"""
Append simple observables to CSV (host side).
Columns: step,Etot,Kavg,Qtot
"""
function write_observables_csv!(path::AbstractString, step::Int;
                                Epot::CuArray{T,1},
                                Ekin::CuArray{T,1},
                                dq::CuArray{T,1}) where {T<:AbstractFloat}
    Etot = sum(Array(Epot))
    Kavg = sum(Array(Ekin))
    Qtot = sum(Array(dq))
    hdr = !isfile(path)
    open(path, "a") do io
        if hdr
            @printf(io, "step,Etot,Kavg,Qtot\n")
        end
        @printf(io, "%d,%.7e,%.7e,%.7e\n", step, Etot, Kavg, Qtot)
    end
    return nothing
end

# =======================================================================
# Particle CSV writer (kept for completeness; expects SoA SimulationState)
# =======================================================================

mutable struct CSVWriter <: Writer
    path::String
    every::Int
    io::Union{Nothing,IO}
    wrote_header::Bool
end

function CSVWriter(path::AbstractString; every::Int=1)
    p = endswith(path, ".csv") ? String(path) : string(path, ".csv")
    return CSVWriter(p, every, nothing, false)
end

function _ensure_csv_open!(w::CSVWriter)
    if w.io === nothing
        mkpath(dirname(w.path))
        w.io = open(w.path, isfile(w.path) ? "a" : "w")
        if !w.wrote_header
            println(w.io, "step,id,x,y,z,vx,vy,vz,typeid")
            w.wrote_header = true
        end
    end
    return nothing
end

function Base.finalize(w::CSVWriter)
    if w.io !== nothing
        try close(w.io) catch end
        w.io = nothing
    end
end

"""
Write a particle table (SoA SimulationState).
"""
function write!(w::CSVWriter, st, step::Int, _dt::Real)
    if step % w.every != 0
        return
    end
    _ensure_csv_open!(w)

    N = length(st.rx)
    X = Array(st.rx); Y = Array(st.ry)
    Z = st.rz === nothing ? fill(zero(eltype(st.rx)), N) : Array(st.rz)

    VX = Array(st.vx); VY = Array(st.vy)
    VZ = st.vz === nothing ? fill(zero(eltype(st.vx)), N) : Array(st.vz)

    TID = Array(st.typeid)

    for i in 1:N
        @printf(w.io, "%d,%d,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%d\n",
            step, i, X[i], Y[i], Z[i], VX[i], VY[i], VZ[i], TID[i])
    end
    return nothing
end

# =======================================================================
# XYZ writer (SoA)
# =======================================================================

mutable struct XYZWriter <: Writer
    path::String
    every::Int
    io::Union{Nothing,IO}
end

function XYZWriter(path::AbstractString; every::Int=1)
    p = endswith(path, ".xyz") ? String(path) : string(path, ".xyz")
    return XYZWriter(p, every, nothing)
end

function _ensure_xyz_open!(w::XYZWriter)
    if w.io === nothing
        mkpath(dirname(w.path))
        w.io = open(w.path, "w")
    end
    return nothing
end

function Base.finalize(w::XYZWriter)
    if w.io !== nothing
        try close(w.io) catch end
        w.io = nothing
    end
end

"""
Write a single XYZ frame from a SoA SimulationState.
Uses z=0 for 2D.
"""
function write!(w::XYZWriter, st, step::Int, _dt::Real)
    if step % w.every != 0
        return
    end
    _ensure_xyz_open!(w)
    N = length(st.rx)
    X = Array(st.rx); Y = Array(st.ry)
    Z = st.rz === nothing ? fill(zero(eltype(st.rx)), N) : Array(st.rz)

    println(w.io, N)
    println(w.io, "step=$step")
    for i in 1:N
        println(w.io, "A $(X[i]) $(Y[i]) $(Z[i])")
    end
    return nothing
end

"""
Direct helper: write an XYZ snapshot without a writer object.
"""
function write_xyz!(path::AbstractString; rx::CuArray{T,1},
                    ry::CuArray{T,1},
                    rz::Union{Nothing,CuArray{T,1}}=nothing,
                    atomsym::AbstractString="A") where {T<:AbstractFloat}
    X = Array(rx); Y = Array(ry)
    Z = rz === nothing ? fill(zero(T), length(X)) : Array(rz)
    N = length(X)
    open(path, "a") do io
        @printf(io, "%d\n", N)
        @printf(io, "Generated by NonEqSimGPU SoA\n")
        @inbounds for i in 1:N
            @printf(io, "%s %.7e %.7e %.7e\n", atomsym, X[i], Y[i], Z[i])
        end
    end
    return nothing
end

# =======================================================================
# GSD support (GSDFiles, HOOMD schema)
# =======================================================================

"""
Open a GSD file for writing.
"""
function gsd_open(path::AbstractString; application="NonEqSimGPU", schema="hoomd", schema_version=(1,4))
    mkpath(dirname(path))
    w = GSDFiles.GSDWriter(path; application, schema, schema_version)
    h = GSDFiles.open_gsd(w)              # <- returns GSDFilesHandle
    return h
end


"""
Close a previously opened GSD handle.
"""
gsd_close(h) = GSDFiles.close_gsd(h)

# -- internal helpers -----------------------------------------------------

@inline function _pack_box2(box::Tuple{T,T}) where {T<:AbstractFloat}
    # HOOMD box: (Lx, Ly, Lz, xy, xz, yz)
    return SVector{6,T}(box[1], box[2], zero(T), zero(T), zero(T), zero(T))
end

@inline function _pack_box3(box::Tuple{T,T,T}) where {T<:AbstractFloat}
    return SVector{6,T}(box[1], box[2], box[3], zero(T), zero(T), zero(T))
end

@inline function _soa_to_posmat(rx::CuArray{T,1}, ry::CuArray{T,1}) where {T<:AbstractFloat}
    # Asynchronous GPU->CPU transfer (non-blocking)
    X = Vector{T}(undef, length(rx))
    Y = Vector{T}(undef, length(ry))
    copyto!(X, rx)  # Async copy
    copyto!(Y, ry)  # Async copy
    Z = fill(zero(T), length(X))
    CUDA.synchronize()  # Single sync point for both transfers
    return hcat(X, Y, Z)
end

@inline function _soa_to_posmat(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}) where {T<:AbstractFloat}
    # Asynchronous GPU->CPU transfer (non-blocking)
    X = Vector{T}(undef, length(rx))
    Y = Vector{T}(undef, length(ry))
    Z = Vector{T}(undef, length(rz))
    copyto!(X, rx)  # Async copy
    copyto!(Y, ry)  # Async copy  
    copyto!(Z, rz)  # Async copy
    CUDA.synchronize()  # Single sync point for all transfers
    return hcat(X, Y, Z)
end

@inline function _soa_to_velmat(vx::CuArray{T,1}, vy::CuArray{T,1}) where {T<:AbstractFloat}
    # Asynchronous GPU->CPU transfer (non-blocking)
    VX = Vector{T}(undef, length(vx))
    VY = Vector{T}(undef, length(vy))
    copyto!(VX, vx)  # Async copy
    copyto!(VY, vy)  # Async copy
    VZ = fill(zero(T), length(VX))
    CUDA.synchronize()  # Single sync point for both transfers
    return hcat(VX, VY, VZ)
end

@inline function _soa_to_velmat(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1}) where {T<:AbstractFloat}
    # Asynchronous GPU->CPU transfer (non-blocking)
    VX = Vector{T}(undef, length(vx))
    VY = Vector{T}(undef, length(vy))
    VZ = Vector{T}(undef, length(vz))
    copyto!(VX, vx)  # Async copy
    copyto!(VY, vy)  # Async copy  
    copyto!(VZ, vz)  # Async copy
    CUDA.synchronize()  # Single sync point for all transfers
    return hcat(VX, VY, VZ)
end


@inline function _soa_to_frcmat(fx::CuArray{T,1}, fy::CuArray{T,1}) where {T<:AbstractFloat}
    # Asynchronous GPU->CPU transfer (non-blocking)
    X = Vector{T}(undef, length(fx))
    Y = Vector{T}(undef, length(fy))
    copyto!(X, fx)  # Async copy
    copyto!(Y, fy)  # Async copy
    Z = fill(zero(T), length(X))
    CUDA.synchronize()  # Single sync point for both transfers
    return hcat(X, Y, Z)
end

@inline function _soa_to_frcmat(fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}) where {T<:AbstractFloat}
    # Asynchronous GPU->CPU transfer (non-blocking)
    X = Vector{T}(undef, length(fx))
    Y = Vector{T}(undef, length(fy))
    Z = Vector{T}(undef, length(fz))
    copyto!(X, fx)  # Async copy
    copyto!(Y, fy)  # Async copy
    copyto!(Z, fz)  # Async copy
    CUDA.synchronize()  # Single sync point for all transfers
    return hcat(X, Y, Z)
end

# -- public API -----------------------------------------------------------

"""
Write one frame to an already-open GSD file (HOOMD schema).

Usage (2D):
    h = Writers.gsd_open("traj.gsd")
    Writers.write_gsd_frame!(h, state; diameter=1.0, types_names=["A"], step=state.step)
    Writers.gsd_close(h)

Usage (3D) is identical; z-components are written when present.
"""
function write_gsd_frame!(h, st; diameter=1.0, types_names=["A"], step::Int=0, write_forces::Bool=false)
    # Element type used for numeric conversions
    T = eltype(st.rx)
    N = length(st.rx)

    # positions and velocities (N×3 T)
    posM = st.rz === nothing ? _soa_to_posmat(st.rx, st.ry) :
                               _soa_to_posmat(st.rx, st.ry, st.rz)
    # velocities are undefined for Brownian dynamics; skip when last_integrator==2
    write_velocities = !(hasproperty(st, :last_integrator) && st.last_integrator == UInt8(2))
    if write_velocities
        velM = st.vz === nothing ? _soa_to_velmat(st.vx, st.vy) :
                                   _soa_to_velmat(st.vx, st.vy, st.vz)
    end

    if write_forces
        frcM = st.fz === nothing ? _soa_to_frcmat(st.fx, st.fy) :
                                   _soa_to_frcmat(st.fx, st.fy, st.fz)
    end

    # dimensionality & box
    if st.box3 === nothing
        D = UInt8(2)
        box6 = _pack_box2(st.box2::Tuple{T,T})
    else
        D = UInt8(3)
        box6 = _pack_box3(st.box3::Tuple{T,T,T})
    end

    # types (async transfer)
    tid_host = Vector{Int32}(undef, length(st.typeid))
    copyto!(tid_host, st.typeid)  # Async copy
    CUDA.synchronize()  # Wait for typeid transfer
    tid_0based = UInt32.(tid_host .- 1)  # HOOMD expects 0-based

    # write frame
    GSDFiles.write_configuration_step!(h, UInt64(step))
    GSDFiles.write_configuration_dimensions!(h, D)
    GSDFiles.write_configuration_box!(h, T.(box6))
    GSDFiles.write_particles_N!(h, N)
    GSDFiles.write_particles_types!(h, Vector{String}(types_names))
    GSDFiles.write_particles_typeid!(h, tid_0based)
    local diam::Vector{T}
    if diameter isa Number
        diam = fill(T(diameter), N)
    elseif diameter isa AbstractVector
        @assert length(diameter) == N "diameter vector length $(length(diameter)) must equal N=$(N)"
        diam = T.(collect(diameter))
    else
        error("Unsupported diameter type: $(typeof(diameter))")
    end
    GSDFiles.write_particles_diameter!(h, diam)
    GSDFiles.write_particles_position!(h, T.(posM))
    if write_velocities
        GSDFiles.write_particles_velocity!(h, T.(velM))
    end

    # Forces: default is false for both Brownian and Langevin; enable only if user asks
    if write_forces
        GSDFiles.write_particles_force!(h, T.(frcM))
    end

    # Optional: bonded interactions (HOOMD bonds group)
    if hasproperty(st, :bonds) && (st.bonds !== nothing)
        # Download CSR adjacency to host
        idx    = Vector{Int32}(undef, length(st.bonds.index));    copyto!(idx, st.bonds.index)
        counts = Vector{Int32}(undef, length(st.bonds.counts));   copyto!(counts, st.bonds.counts)
        flat   = Vector{Int32}(undef, length(st.bonds.flat));     copyto!(flat, st.bonds.flat)
        CUDA.synchronize()

        # Build unique bond list as pairs (i,j) with j>i, convert to 0-based
        pairs = Vector{NTuple{2,UInt32}}()
        pairs_size = 0
        for i in 1:length(idx)
            base = Int(idx[i])
            nb   = Int(counts[i])
            for t in 0:(nb-1)
                j = Int(flat[base + t + 1])
                if j > i
                    push!(pairs, (UInt32(i-1), UInt32(j-1)))
                    pairs_size += 1
                end
            end
        end
        if pairs_size > 0
            # bonds: one type named "bond"; typeid all zeros
            GSDFiles.write_bonds_N!(h, pairs_size)
            GSDFiles.write_bonds_types!(h, ["bond"])  # single type
            GSDFiles.write_bonds_typeid!(h, fill(UInt32(0), pairs_size))
            # group matrix Nb×2
            grp = Array{UInt32}(undef, pairs_size, 2)
            @inbounds for k in 1:pairs_size
                grp[k,1] = pairs[k][1]
                grp[k,2] = pairs[k][2]
            end
            GSDFiles.write_bonds_group!(h, grp)
        else
            # No bonds; ensure bonds/N=0 for clarity (optional)
            GSDFiles.write_bonds_N!(h, 0)
        end
    end
    GSDFiles.end_frame!(h)
    return h
end

"""
Read the last valid frame from a GSD file and return SoA arrays.

Returns:
    step::Int,
    rx::Vector{T}, ry::Vector{T}, rz::Union{Nothing,Vector{T}},
    vx::Vector{T}, vy::Vector{T}, vz::Union{Nothing,Vector{T}},
    typeid_1based::Vector{Int32},
    types_names::Vector{String},
    box::Union{Tuple{T,T},Tuple{T,T,T}},
    forceM::Union{Nothing,Matrix{T}}   # N×3 when present
"""
function read_last_gsd(file_path::AbstractString; step::Union{Nothing,Integer}=nothing)
    r = GSDFiles.open_read(file_path)
    try
        # Determine the last frame index directly from the raw index table
        if isempty(r.index)
            error("No frames in GSD: $file_path")
        end
        last_idx = Int(maximum(e -> e.frame, r.index)) + 1

        # Now attempt to parse from last_idx backwards in case the tail is partial
        last_error = nothing
        candidates = Int[]
        if step === nothing
            candidates = collect(last_idx:-1:1)
        else
            # try to find frame with matching configuration/step
            for i in 1:last_idx
                fid_i = UInt64(i-1)
                ents_i = GSDFiles._entries_for_frame(r, fid_i)
                stp_e = GSDFiles._maybe_one(r, ents_i, "configuration/step")
                if stp_e !== nothing
                    stp_val = Int(GSDFiles._read_scalar(r.io, stp_e, UInt64))
                    if stp_val == step
                        push!(candidates, i)
                    end
                end
            end
            # If not found, treat step as frame index when valid
            if isempty(candidates) && 1 <= step <= last_idx
                push!(candidates, Int(step))
            end
        end
        for idx in candidates
            try
                # Gather entries for this frame directly from the index
                fid = UInt64(idx - 1)
                ents = GSDFiles._entries_for_frame(r, fid)

                # Helper: find latest entry for a given name up to this frame
                function _latest_entry(name::AbstractString)
                    id = GSDFiles._name_id(r, name)
                    id === nothing && return nothing
                    candidates = filter(e -> (e.id == id && e.frame <= fid), r.index)
                    isempty(candidates) && return nothing
                    # pick by maximum frame id
                    best = candidates[argmax(getfield.(candidates, :frame))]
                    return best
                end

                # configuration/*
                step_e = _latest_entry("configuration/step")
                dim_e  = _latest_entry("configuration/dimensions")
                box_e  = _latest_entry("configuration/box")
                if step_e === nothing || dim_e === nothing || box_e === nothing
                    throw(ArgumentError("configuration chunks missing"))
                end
                step = Int(GSDFiles._read_scalar(r.io, step_e, UInt64))
                D    = Int(GSDFiles._read_scalar(r.io, dim_e, UInt8))
                box6 = GSDFiles._read_vec(r.io, box_e, Float32)

                # particles/N
                N_e = _latest_entry("particles/N")
                N_e === nothing && throw(ArgumentError("particles/N missing"))
                N = Int(GSDFiles._read_scalar(r.io, N_e, UInt32))

                # particles/position (required)
                pos_e = begin
                    x = GSDFiles._maybe_one_of(r, ents, ["particles/position", "particles/positions"])
                    x === nothing ? _latest_entry("particles/position") : x
                end
                pos_e === nothing && throw(ArgumentError("particles/position missing"))
                posM = GSDFiles._read_mat_f32(r.io, pos_e)
                T = eltype(posM)
                rx = T.(posM[:,1]); ry = T.(posM[:,2])
                rz = D == 3 ? T.(posM[:,3]) : nothing

                # particles/velocity (optional)
                vel_e = GSDFiles._maybe_one_of(r, ents, ["particles/velocity", "particles/velocities"])
                local vx, vy, vz
                if vel_e === nothing
                    vx = fill(zero(T), N); vy = fill(zero(T), N)
                    vz = D == 3 ? fill(zero(T), N) : nothing
                    @warn "Velocities missing in GSD frame; using zeros" file=file_path frame_index=idx
                else
                    velM = GSDFiles._read_mat_f32(r.io, vel_e)
                    vx = T.(velM[:,1]); vy = T.(velM[:,2])
                    vz = D == 3 ? T.(velM[:,3]) : nothing
                end

                # particles/types + typeid (optional)
                types_e = _latest_entry("particles/types")
                local types::Vector{String}
                types = types_e === nothing ? String[] : GSDFiles._decode_types(r.io, types_e)
                tid_e = begin
                    x = GSDFiles._maybe_one_of(r, ents, ["particles/typeid", "particles/typeids"])  # per-frame if available
                    x === nothing ? _latest_entry("particles/typeid") : x
                end
                local typeid1::Vector{Int32}
                if tid_e === nothing
                    typeid1 = fill(Int32(1), N)
                    isempty(types) && (types = ["A"])  # default single type
                else
                    tid0 = GSDFiles._read_vec(r.io, tid_e, UInt32)
                    typeid1 = Int32.(tid0 .+ 1)
                    isempty(types) && (types = ["A"])  # ensure at least one name
                end

                # box tuple
                box  = D == 2 ? (T(box6[1]), T(box6[2])) :
                                (T(box6[1]), T(box6[2]), T(box6[3]))

                # Build force matrix if present
                if frc_e === nothing
                    local_forceM = nothing
                else
                    fM = GSDFiles._read_mat_f32(r.io, frc_e)
                    local_forceM = T.(fM)
                end

                # Warn if we had to fall back from the last probed frame
                if idx != last_idx
                    @warn "Last frame appears incomplete; using previous valid frame" file=file_path chosen_frame=idx last_probe=last_idx
                end
                return step, rx, ry, rz, vx, vy, vz, typeid1, types, box, local_forceM
            catch err
                last_error = err
                # try previous frame
            end
        end
        # If we reach here, none of the frames could be read fully
        if last_error !== nothing
            throw(last_error)
        else
            error("Failed to read any valid frame from $file_path")
        end
    finally
        GSDFiles.close(r)
    end
end

end # module
