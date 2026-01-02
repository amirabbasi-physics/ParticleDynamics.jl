module Writers

using CUDA
using Printf
using StaticArrays
using GSDFiles
using DelimitedFiles

export InMemoryLogger, CSVWriter, XYZWriter, ObservableCSVWriter
export write_xyz!, write_observables_csv!
export gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!, GSDFrameData, GSDTopology

# -----------------------------------------------------------------------------
# Rich GSD frame containers
# -----------------------------------------------------------------------------
"""
Structured toplogical information decoded from a GSD frame.

Each field is a `NamedTuple` describing the corresponding collection and
contains both the original 0-based data (`group0`, `typeid0`) and the
converted 1-based indices (`group`, `typeid`) that can be passed directly to
NonEqSimGPU utilities such as `BondedForces.build_bondlist`.
"""
struct GSDTopology
    bonds::NamedTuple
    angles::NamedTuple
    dihedrals::NamedTuple
    impropers::NamedTuple
    constraints::NamedTuple
    special_pairs::NamedTuple
end

"""
Container returned by `read_gsd_frame!`.

Behaves like the legacy tuple `(step, rx, ry, rz, vx, vy, vz, typeid, types, box, force)`
when destructured/iterated, while exposing richer metadata via fields:

- `step`, `N`, `D`, `box`: configuration metadata.
- `rx`, `ry`, `rz`, `vx`, `vy`, `vz`: particle state vectors (host arrays).
- `typeid`, `types`: 1-based ids and type names.
- `forceM`: optional force matrix (N×3, same precision as positions).
- `particle_properties`: dictionary with optional particle properties
  (mass, charge, diameter, body, orientation, etc) converted to sensible
  Julia arrays; custom `particles/property/*` chunks appear under the
  `:property` key.
- `per_type_properties`: dictionary for `particles/type_*` chunks
  (e.g. `:shapes` for per-type shape JSON blobs).
- `topology`: a `GSDTopology` with bonds/angles/… converted to 1-based indices.
- `configuration`: hierarchical `NamedTuple` combining metadata, particle data,
  topology and any extra configuration chunks for convenience.
"""
struct GSDFrameData{T<:AbstractFloat}
    step::Int
    N::Int
    D::Int
    rx::Vector{T}
    ry::Vector{T}
    rz::Union{Nothing,Vector{T}}
    vx::Vector{T}
    vy::Vector{T}
    vz::Union{Nothing,Vector{T}}
    typeid::Vector{Int32}
    types::Vector{String}
    box::Union{Tuple{T,T},Tuple{T,T,T}}
    forceM::Union{Nothing,Matrix{T}}
    particle_properties::Dict{Symbol,Any}
    per_type_properties::Dict{Symbol,Any}
    topology::GSDTopology
    configuration::NamedTuple
end

# -- Legacy tuple compatibility ------------------------------------------------
Base.length(::GSDFrameData) = 11
Base.eltype(::Type{GSDFrameData}) = Any

function Base.iterate(f::GSDFrameData{T}) where {T}
    return (f.step, 2)
end

function Base.iterate(f::GSDFrameData{T}, state::Int) where {T}
    if state == 2
        return (f.rx, 3)
    elseif state == 3
        return (f.ry, 4)
    elseif state == 4
        return (f.rz, 5)
    elseif state == 5
        return (f.vx, 6)
    elseif state == 6
        return (f.vy, 7)
    elseif state == 7
        return (f.vz, 8)
    elseif state == 8
        return (f.typeid, 9)
    elseif state == 9
        return (f.types, 10)
    elseif state == 10
        return (f.box, 11)
    elseif state == 11
        return (f.forceM, 12)
    else
        return nothing
    end
end

Base.Tuple(f::GSDFrameData) = (f.step, f.rx, f.ry, f.rz, f.vx, f.vy, f.vz,
                               f.typeid, f.types, f.box, f.forceM)

# =======================================================================
# Simple in-memory logger (kept for API completeness)
# =======================================================================
abstract type Writer end

"""
Lightweight logger that records selected observables in host memory every
`every` steps. Useful for quick ad-hoc diagnostics during testing.
"""
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
    write_observables_csv!(path, step; Epot, Ekin, dq)

Append a single line with `step, Etot, Kavg, Qtot` (mirrors
`examples/2D_example.jl` and the 3D variants).
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

"""
Stream particle tables to CSV (one row per particle). Mirrors the logging used
in the earlier `examples/` scripts.
"""
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

"""
Minimal XYZ trajectory writer. Each call to `write!` appends one frame; z is
set to zero for 2D states, matching the usage in `examples/2D_example.jl`.
"""
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
    write_xyz!(path; rx, ry[, rz], atomsym=\"A\")

Write a single XYZ frame without constructing an `XYZWriter`. Mirrors the
ad-hoc dumping performed in `examples/2D_example.jl`.
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
    gsd_open(path; application=\"NonEqSimGPU\", schema=\"hoomd\")

Open a GSD trajectory for appending frames. Used extensively in `examples/`.
Call [`gsd_close`](@ref) when finished or use the do-block form.
"""
function gsd_open(path::AbstractString; application="NonEqSimGPU", schema="hoomd", schema_version=(1,4))
    mkpath(dirname(path))
    w = GSDFiles.GSDWriter(path; application, schema, schema_version)
    h = GSDFiles.open_gsd(w)              # <- returns GSDFilesHandle
    return h
end


"""
gsd_open(f::Function, path; kwargs...)

Convenience do-block form that guarantees the GSD handle is closed,
even if an exception occurs during writing. Example:

    Writers.gsd_open("traj.gsd") do h
        for step in 1:ns
            Writers.write_gsd_frame!(h, st; step)
        end
    end
"""
function gsd_open(f::Function, path::AbstractString; application="NonEqSimGPU", schema="hoomd", schema_version=(1,4))
    h = gsd_open(path; application, schema, schema_version)
    try
        return f(h)
    finally
        try
            gsd_close(h)
        catch err
            @warn "Failed to close GSD handle" error=err
        end
    end
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
function write_gsd_frame!(h, st; diameter=1.0, types_names=["A"], step::Int=0, write_forces::Bool=false, sync_on_write::Bool=false)
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
    if sync_on_write
        # Make file readable by OVITO mid-run by writing an index and updating header
        if isdefined(GSDFiles, :sync!)
            try
                GSDFiles.sync!(h)
            catch err
                @warn "GSD sync failed; file may be unreadable mid-run" error=err
            end
        else
            @warn "GSDFiles.sync! not available; file may be unreadable mid-run"
        end
    end
    return h
end

"""
Read a valid frame from a GSD file and return a `GSDFrameData`.

The returned object can still be destructured into the legacy tuple
`step, rx, ry, rz, vx, vy, vz, typeid, types, box, force` for backwards
compatibility, while also exposing rich metadata (`frame.topology`,
`frame.particle_properties`, `frame.configuration`, …) that can be fed
directly into NonEqSimGPU initialisation routines.

Arguments:
- `file_path`: path to the GSD file.
- `step`: optional timestep or frame index to target. When `nothing`, the
  most recent complete frame is returned. When provided, the routine first
  searches for a frame whose `configuration/step` matches and, if none exist,
  treats the value as a 1-based frame index.

Throws:
- `ArgumentError` if mandatory chunks are missing or no readable frame exists.
"""
function read_gsd_frame!(file_path::AbstractString; step::Union{Nothing,Integer}=nothing)
    r = GSDFiles.open_read(file_path)
    try
        if isempty(r.index)
            error("No frames in GSD: $file_path")
        end

        TYPE_MAP = Dict{UInt8,DataType}(
            GSDFiles._R_UINT8   => UInt8,
            UInt8(2)            => UInt16,
            GSDFiles._R_UINT32  => UInt32,
            GSDFiles._R_UINT64  => UInt64,
            GSDFiles._R_INT8    => Int8,
            UInt8(6)            => Int16,
            UInt8(7)            => Int32,
            UInt8(8)            => Int64,
            GSDFiles._R_FLOAT32 => Float32,
            UInt8(10)           => Float64,
            UInt8(11)           => UInt8,
        )

        function read_chunk(entry::GSDFiles.IndexEntry)
            dtype = entry.type
            Traw = get(TYPE_MAP, dtype) do
                name = r.names[Int(entry.id)+1]
                error("Unsupported dtype code $(dtype) for chunk $name in $file_path")
            end
            seek(r.io, entry.location)
            N = Int(entry.N)
            M = Int(entry.M)
            total = max(1, M) * N
            data = read!(r.io, Array{Traw}(undef, total))
            return M <= 1 ? data : reshape(data, (M, N))'
        end

        _convert(::Type{Tc}, arr::AbstractVector) where {Tc} = Tc.(arr)
        _convert(::Type{Tc}, arr::AbstractMatrix) where {Tc} = Tc.(arr)

        function as_scalar(x)
            if x isa AbstractVector && length(x) == 1
                return x[1]
            elseif x isa AbstractMatrix && size(x,1) == 1 && size(x,2) == 1
                return x[1,1]
            else
                return x
            end
        end

        function convert_topology(raw, ::Val{C}) where {C}
            group0 = raw.group
            group1 = Array{Int32}(undef, size(group0))
            if size(group0, 1) == 0
                fill!(group1, Int32(0))
            else
                @inbounds for i in 1:size(group0,1), j in 1:C
                    group1[i,j] = Int32(group0[i,j]) + Int32(1)
                end
            end
            typeid0 = raw.typeid
            typeid = Vector{Int32}(undef, length(typeid0))
            @inbounds for i in eachindex(typeid0)
                typeid[i] = Int32(typeid0[i]) + Int32(1)
            end
            tuples = NTuple{C,Int32}[]
            if size(group1, 1) > 0
                sizehint!(tuples, size(group1, 1))
                @inbounds for i in 1:size(group1,1)
                    push!(tuples, ntuple(j -> group1[i,j], C))
                end
            end
            return (;
                N      = raw.N,
                types  = copy(raw.types),
                typeid0 = copy(typeid0),
                typeid = typeid,
                group0 = copy(group0),
                group  = group1,
                tuples = tuples,
            )
        end

        function decode_string_blob(bytes::Vector{UInt8})
            out = String[]
            buf = IOBuffer()
            for b in bytes
                if b == 0x00
                    push!(out, String(take!(buf)))
                else
                    write(buf, b)
                end
            end
            while !isempty(out) && out[end] == ""
                pop!(out)
            end
            return out
        end

        last_idx = Int(maximum(e -> e.frame, r.index)) + 1
        last_error = nothing
        candidates = Int[]
        if step === nothing
            candidates = collect(last_idx:-1:1)
        else
            for i in 1:last_idx
                fid_i = UInt64(i - 1)
                ents_i = GSDFiles._entries_for_frame(r, fid_i)
                stp_e = GSDFiles._maybe_one(r, ents_i, "configuration/step")
                if stp_e !== nothing
                    stp_val = Int(GSDFiles._read_scalar(r.io, stp_e, UInt64))
                    stp_val == step && push!(candidates, i)
                end
            end
            if isempty(candidates) && 1 <= step <= last_idx
                push!(candidates, Int(step))
            end
        end
        isempty(candidates) && error("Requested step=$(step) not found in $file_path")

        for idx in candidates
            try
                fid = UInt64(idx - 1)
                ents = GSDFiles._entries_for_frame(r, fid)

                function latest(name::AbstractString)
                    id = GSDFiles._name_id(r, name)
                    id === nothing && return nothing
                    matches = filter(e -> (e.id == id && e.frame <= fid), r.index)
                    isempty(matches) && return nothing
                    return matches[argmax(getfield.(matches, :frame))]
                end

                step_e = latest("configuration/step")
                dim_e  = latest("configuration/dimensions")
                box_e  = latest("configuration/box")
                if step_e === nothing || dim_e === nothing || box_e === nothing
                    throw(ArgumentError("configuration chunks missing"))
                end
                frame_step = Int(GSDFiles._read_scalar(r.io, step_e, UInt64))
                D = Int(GSDFiles._read_scalar(r.io, dim_e, UInt8))
                D == 2 || D == 3 || throw(ArgumentError("Unsupported configuration/dimensions=$D"))
                box6_raw = GSDFiles._read_vec(r.io, box_e, Float32)

                N_e = latest("particles/N")
                N_e === nothing && throw(ArgumentError("particles/N missing"))
                N = Int(GSDFiles._read_scalar(r.io, N_e, UInt32))

                pos_e = begin
                    cand = GSDFiles._maybe_one_of(r, ents, ["particles/position", "particles/positions"])
                    cand === nothing ? latest("particles/position") : cand
                end
                pos_e === nothing && throw(ArgumentError("particles/position missing"))
                pos_raw = read_chunk(pos_e)
                pos_raw isa AbstractMatrix || throw(ArgumentError("particles/position expected matrix data"))
                el = eltype(pos_raw)
                el <: AbstractFloat || throw(ArgumentError("particles/position must be floating-point"))
                T = el
                posM = Matrix{T}(pos_raw)

                rx = posM[:,1]
                ry = posM[:,2]
                rz = D == 3 ? posM[:,3] : nothing

                vel_e = GSDFiles._maybe_one_of(r, ents, ["particles/velocity", "particles/velocities"])
                local vx::Vector{T}; local vy::Vector{T}; local vz
                if vel_e === nothing
                    vx = fill(zero(T), N)
                    vy = fill(zero(T), N)
                    vz = D == 3 ? fill(zero(T), N) : nothing
                    @warn "Velocities missing in GSD frame; using zeros" file=file_path frame_index=idx
                else
                    vel_raw = read_chunk(vel_e)
                    vel_raw isa AbstractMatrix || throw(ArgumentError("particles/velocity expected matrix data"))
                    velM = Matrix{T}(vel_raw)
                    vx = velM[:,1]
                    vy = velM[:,2]
                    vz = D == 3 ? velM[:,3] : nothing
                end

                types_e = latest("particles/types")
                types = types_e === nothing ? String[] : GSDFiles._decode_types(r.io, types_e)
                tid_e = begin
                    cand = GSDFiles._maybe_one_of(r, ents, ["particles/typeid", "particles/typeids"])
                    cand === nothing ? latest("particles/typeid") : cand
                end
                local typeid1::Vector{Int32}
                if tid_e === nothing
                    typeid1 = fill(Int32(1), N)
                    isempty(types) && (types = ["A"])
                else
                    tid0 = GSDFiles._read_vec(r.io, tid_e, UInt32)
                    length(tid0) == N || @warn "particles/typeid length $(length(tid0)) != N=$N" file=file_path frame_index=idx
                    typeid1 = Int32.(tid0 .+ UInt32(1))
                    isempty(types) && (types = ["A"])
                end

                frc_e = GSDFiles._maybe_one_of(r, ents, [
                    "particles/force",
                    "particles/forces",
                    "particles/net_force",
                    "particles/property/force",
                ])
                local_forceM = nothing
                property_skip = Set{String}()
                if frc_e !== nothing
                    force_raw = read_chunk(frc_e)
                    name_frc = r.names[Int(frc_e.id)+1]
                    if force_raw isa AbstractMatrix
                        local_forceM = Matrix{T}(force_raw)
                    elseif force_raw isa AbstractVector
                        if length(force_raw) == 3N
                            reshaped = reshape(force_raw, (3, N))'
                            local_forceM = Matrix{T}(reshaped)
                        else
                            @warn "Unexpected force chunk shape; skipping forceM" file=file_path frame_index=idx chunk=name_frc
                        end
                    else
                        @warn "Unsupported force chunk type; skipping forceM" file=file_path frame_index=idx chunk=name_frc
                    end
                    startswith(name_frc, "particles/property/") && push!(property_skip, name_frc)
                end

                particle_props = Dict{Symbol,Any}()

                function maybe_read_vec!(dict::Dict{Symbol,Any}, key::Symbol, names::Vector{String}, ::Type{Tc}) where {Tc}
                    entry = GSDFiles._maybe_one_of(r, ents, names)
                    entry === nothing && return nothing
                    raw = read_chunk(entry)
                    raw isa AbstractVector || throw(ArgumentError("Chunk $(r.names[Int(entry.id)+1]) expected to be a vector"))
                    dict[key] = _convert(Tc, raw)
                    return dict[key]
                end

                function maybe_read_mat!(dict::Dict{Symbol,Any}, key::Symbol, names::Vector{String}, ::Type{Tc}) where {Tc}
                    entry = GSDFiles._maybe_one_of(r, ents, names)
                    entry === nothing && return nothing
                    raw = read_chunk(entry)
                    raw isa AbstractMatrix || throw(ArgumentError("Chunk $(r.names[Int(entry.id)+1]) expected to be a matrix"))
                    dict[key] = _convert(Tc, raw)
                    return dict[key]
                end

                maybe_read_vec!(particle_props, :diameter, ["particles/diameter"], T)
                maybe_read_vec!(particle_props, :mass, ["particles/mass"], T)
                maybe_read_vec!(particle_props, :charge, ["particles/charge"], T)
                maybe_read_vec!(particle_props, :body, ["particles/body"], Int32)
                maybe_read_mat!(particle_props, :image, ["particles/image"], Int32)
                maybe_read_mat!(particle_props, :orientation, ["particles/orientation"], T)
                maybe_read_mat!(particle_props, :angmom, ["particles/angmom"], T)
                maybe_read_mat!(particle_props, :moment_inertia, ["particles/moment_inertia"], T)
                maybe_read_mat!(particle_props, :acceleration, ["particles/acceleration"], T)
                maybe_read_mat!(particle_props, :momentum, ["particles/momentum"], T)
                maybe_read_mat!(particle_props, :torque, ["particles/torque"], T)

                if local_forceM !== nothing
                    particle_props[:force] = local_forceM
                end

                property_data = Dict{Symbol,Any}()
                for entry in ents
                    name = r.names[Int(entry.id)+1]
                    startswith(name, "particles/property/") || continue
                    name in property_skip && continue
                    suffix = name[length("particles/property/")+1:end]
                    isempty(suffix) && continue
                    sym = Symbol(replace(suffix, "/" => "_"))
                    raw = read_chunk(entry)
                    property_data[sym] = as_scalar(raw)
                end
                if !isempty(property_data)
                    particle_props[:property] = property_data
                end

                per_type = Dict{Symbol,Any}()
                type_shapes_e = GSDFiles._maybe_one(r, ents, "particles/type_shapes")
                if type_shapes_e !== nothing
                    raw = read_chunk(type_shapes_e)
                    if raw isa Vector{UInt8}
                        per_type[:shapes] = decode_string_blob(raw)
                    else
                        per_type[:shapes] = raw
                    end
                end
                for entry in ents
                    name = r.names[Int(entry.id)+1]
                    startswith(name, "particles/type_") || continue
                    name in ("particles/types", "particles/type_shapes") && continue
                    suffix = name[length("particles/type_")+1:end]
                    sym = Symbol(replace(suffix, "/" => "_"))
                    raw = read_chunk(entry)
                    per_type[sym] = as_scalar(raw)
                end

                config_extras = Dict{Symbol,Any}()
                for entry in ents
                    name = r.names[Int(entry.id)+1]
                    startswith(name, "configuration/") || continue
                    name in ("configuration/step", "configuration/dimensions", "configuration/box") && continue
                    suffix = name[length("configuration/")+1:end]
                    sym = Symbol(replace(suffix, "/" => "_"))
                    raw = read_chunk(entry)
                    config_extras[sym] = as_scalar(raw)
                end

                bonds_section     = convert_topology(GSDFiles.read_bonds(r, idx), Val(2))
                angles_section    = convert_topology(GSDFiles.read_angles(r, idx), Val(3))
                dihedrals_section = convert_topology(GSDFiles.read_dihedrals(r, idx), Val(4))
                impropers_section = convert_topology(GSDFiles.read_impropers(r, idx), Val(4))

                function read_constraints_section(::Type{T}) where {T}
                    types_e = GSDFiles._maybe_one(r, ents, "constraints/types")
                    types = types_e === nothing ? String[] : GSDFiles._decode_types(r.io, types_e)
                    N_e = GSDFiles._maybe_one(r, ents, "constraints/N")
                    grp_e = GSDFiles._maybe_one_of(r, ents, ["constraints/group", "constraints/constraint"])
                    tid_e = GSDFiles._maybe_one_of(r, ents, ["constraints/typeid", "constraints/typeids"])
                    val_e = GSDFiles._maybe_one(r, ents, "constraints/value")
                    if N_e === nothing || grp_e === nothing
                        group0 = reshape(UInt32[], 0, 2)
                        group1 = reshape(Int32[], 0, 2)
                        typeid0 = tid_e === nothing ? UInt32[] : GSDFiles._read_vec(r.io, tid_e, UInt32)
                        typeid = Int32.(typeid0 .+ UInt32(1))
                        return (;
                            N = 0,
                            types = types,
                            typeid0 = typeid0,
                            typeid = typeid,
                            group0 = group0,
                            group = group1,
                            value = nothing,
                        )
                    end
                    Nval = Int(GSDFiles._read_scalar(r.io, N_e, UInt32))
                    group0 = GSDFiles._read_mat_u32(r.io, grp_e)
                    cols = size(group0, 2)
                    group1 = Array{Int32}(undef, size(group0))
                    if size(group0,1) == 0
                        fill!(group1, Int32(0))
                    else
                        @inbounds for i in 1:size(group0,1), j in 1:cols
                            group1[i,j] = Int32(group0[i,j]) + Int32(1)
                        end
                    end
                    typeid0 = tid_e === nothing ? fill(UInt32(0), Nval) : GSDFiles._read_vec(r.io, tid_e, UInt32)
                    typeid = Int32.(typeid0 .+ UInt32(1))
                    values = nothing
                    if val_e !== nothing
                        raw_val = read_chunk(val_e)
                        if raw_val isa AbstractVector
                            values = T.(raw_val)
                        elseif raw_val isa AbstractMatrix
                            values = T.(vec(raw_val))
                        elseif raw_val isa Number
                            values = T[T(raw_val)]
                        end
                    end
                    return (;
                        N = Nval,
                        types = types,
                        typeid0 = typeid0,
                        typeid = typeid,
                        group0 = group0,
                        group = group1,
                        value = values,
                    )
                end

                function read_pairs_section()
                    types_e = GSDFiles._maybe_one(r, ents, "pairs/types")
                    types = types_e === nothing ? String[] : GSDFiles._decode_types(r.io, types_e)
                    N_e = GSDFiles._maybe_one(r, ents, "pairs/N")
                    grp_e = GSDFiles._maybe_one_of(r, ents, ["pairs/pair", "pairs/group"])
                    tid_e = GSDFiles._maybe_one_of(r, ents, ["pairs/typeid", "pairs/typeids"])
                    if N_e === nothing || grp_e === nothing
                        raw = (N = 0, types = types, typeid = UInt32[], group = reshape(UInt32[], 0, 2))
                        return convert_topology(raw, Val(2))
                    end
                    Nval = Int(GSDFiles._read_scalar(r.io, N_e, UInt32))
                    group0 = GSDFiles._read_mat_u32(r.io, grp_e)
                    typeid0 = tid_e === nothing ? fill(UInt32(0), Nval) : GSDFiles._read_vec(r.io, tid_e, UInt32)
                    raw = (N = Nval, types = types, typeid = typeid0, group = group0)
                    return convert_topology(raw, Val(2))
                end

                constraints_section = read_constraints_section(T)
                pairs_section = read_pairs_section()
                topology_obj = GSDTopology(bonds_section, angles_section, dihedrals_section,
                                           impropers_section, constraints_section, pairs_section)

                box_tuple = D == 2 ? (T(box6_raw[1]), T(box6_raw[2])) :
                                     (T(box6_raw[1]), T(box6_raw[2]), T(box6_raw[3]))
                box6_tuple = ntuple(i -> T(box6_raw[i]), 6)
                metadata = (;
                    step = frame_step,
                    N = N,
                    dimension = D,
                    box = box_tuple,
                    box6 = box6_tuple,
                    extras = config_extras,
                )
                particles_nt = (;
                    rx = rx,
                    ry = ry,
                    rz = rz,
                    vx = vx,
                    vy = vy,
                    vz = vz,
                    typeid = typeid1,
                    types = types,
                    force = local_forceM,
                    properties = particle_props,
                    per_type = per_type,
                )
                configuration = (;
                    metadata = metadata,
                    particles = particles_nt,
                    topology = topology_obj,
                )

                if idx != last_idx && step === nothing
                    @warn "Last frame appears incomplete; using previous valid frame" file=file_path chosen_frame=idx last_probe=last_idx
                end

                return GSDFrameData(frame_step, N, D, rx, ry, rz, vx, vy, vz, typeid1,
                                    types, box_tuple, local_forceM,
                                    particle_props, per_type, topology_obj, configuration)
            catch err
                last_error = err
            end
        end

        last_error !== nothing && throw(last_error)
        error("Failed to read any valid frame from $file_path")
    finally
        GSDFiles.close(r)
    end
end

end # module
