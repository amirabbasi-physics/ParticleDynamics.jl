# -----------------------------------------------------------------------------
# Rich GSD frame containers
# -----------------------------------------------------------------------------
"""
Structured toplogical information decoded from a GSD frame.

Each field is a `NamedTuple` describing the corresponding collection and
contains both the original 0-based data (`group0`, `typeid0`) and the
converted 1-based indices (`group`, `typeid`) that can be passed directly to
ParticleDynamics utilities such as `BondedForces.build_bondlist`.
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
  Julia arrays. When present, custom force/virial chunks are also exposed as
  `:force` and `:virial`; other `particles/property/*` chunks appear under the
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
# GSD support (GSDFiles, HOOMD schema)
# =======================================================================

"""
    gsd_open(path; application="ParticleDynamics", schema="hoomd", append=false)

Open a GSD trajectory for appending frames. Used extensively in `examples/`.
By default this starts a fresh file. Set `append=true` to continue writing
frames into an existing file without truncating it. Call [`gsd_close`](@ref)
when finished or use the do-block form.
"""
function _read_full_gsd_header(io::IO)
    seek(io, 0)
    magic = read(io, UInt64)
    magic == GSDFiles.GSD_MAGIC || throw(ArgumentError("Invalid GSD magic in append mode"))
    index_location = read(io, UInt64)
    index_allocated_entries = read(io, UInt64)
    namelist_location = read(io, UInt64)
    namelist_allocated_entries = read(io, UInt64)
    schema_version = read(io, UInt32)
    gsd_version = read(io, UInt32)
    application = ntuple(_ -> read(io, UInt8), 64)
    schema = ntuple(_ -> read(io, UInt8), 64)
    reserved = ntuple(_ -> read(io, UInt8), 80)
    return GSDFiles.Header(
        magic,
        index_location,
        index_allocated_entries,
        namelist_location,
        namelist_allocated_entries,
        schema_version,
        gsd_version,
        application,
        schema,
        reserved,
    )
end

function _append_handle(path::AbstractString)
    reader = GSDFiles.open_read(path)
    try
        names = copy(reader.names)
        index = copy(reader.index)
        nframes = GSDFiles.nframes(reader)
        name_to_id = Dict{String,UInt16}(name => UInt16(i - 1) for (i, name) in enumerate(names))
        names_blob = UInt8[]
        for name in names
            append!(names_blob, codeunits(name))
            push!(names_blob, 0x00)
        end

        io = open(path, "r+")
        try
            header = _read_full_gsd_header(io)
            writer = GSDFiles.GSDWriter(io, header, index, name_to_id, names_blob, UInt64(nframes))
            return GSDFiles.open_gsd(writer)
        catch
            close(io)
            rethrow()
        end
    finally
        GSDFiles.close(reader)
    end
end

function gsd_open(path::AbstractString; application="ParticleDynamics", schema="hoomd", schema_version=(1,4), append::Bool=false)
    mkpath(dirname(path))
    if append && isfile(path) && filesize(path) > 0
        return _append_handle(path)
    end
    w = GSDFiles.GSDWriter(path; application, schema, schema_version)
    h = GSDFiles.open_gsd(w)
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
function gsd_open(f::Function, path::AbstractString; application="ParticleDynamics", schema="hoomd", schema_version=(1,4), append::Bool=false)
    h = gsd_open(path; application, schema, schema_version, append)
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

# Undo the spatial-reorder permutation so trajectory frames keep a stable
# particle identity: element k of the state array belongs to original
# particle `tag[k]`.
_unpermuted(a, ::Nothing) = a
function _unpermuted(a::CuArray{<:Any,1}, tag::CuArray{Int32,1})
    out = similar(a)
    out[tag] = a
    return out
end
function _unpermuted(A::CuArray{<:Any,2}, tag::CuArray{Int32,1})
    out = similar(A)
    out[tag, :] = A
    return out
end

@inline function _pack_box2(box::Tuple{T,T}) where {T<:AbstractFloat}
    return SVector{6,T}(box[1], box[2], zero(T), zero(T), zero(T), zero(T))
end

@inline function _pack_box3(box::Tuple{T,T,T}) where {T<:AbstractFloat}
    return SVector{6,T}(box[1], box[2], box[3], zero(T), zero(T), zero(T))
end

@inline function _soa_to_posmat(rx::CuArray{T,1}, ry::CuArray{T,1}) where {T<:AbstractFloat}
    X = Vector{T}(undef, length(rx))
    Y = Vector{T}(undef, length(ry))
    copyto!(X, rx)
    copyto!(Y, ry)
    Z = fill(zero(T), length(X))
    CUDA.synchronize()
    return hcat(X, Y, Z)
end

@inline function _soa_to_posmat(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}) where {T<:AbstractFloat}
    X = Vector{T}(undef, length(rx))
    Y = Vector{T}(undef, length(ry))
    Z = Vector{T}(undef, length(rz))
    copyto!(X, rx)
    copyto!(Y, ry)
    copyto!(Z, rz)
    CUDA.synchronize()
    return hcat(X, Y, Z)
end

@inline function _soa_to_velmat(vx::CuArray{T,1}, vy::CuArray{T,1}) where {T<:AbstractFloat}
    VX = Vector{T}(undef, length(vx))
    VY = Vector{T}(undef, length(vy))
    copyto!(VX, vx)
    copyto!(VY, vy)
    VZ = fill(zero(T), length(VX))
    CUDA.synchronize()
    return hcat(VX, VY, VZ)
end

@inline function _soa_to_velmat(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1}) where {T<:AbstractFloat}
    VX = Vector{T}(undef, length(vx))
    VY = Vector{T}(undef, length(vy))
    VZ = Vector{T}(undef, length(vz))
    copyto!(VX, vx)
    copyto!(VY, vy)
    copyto!(VZ, vz)
    CUDA.synchronize()
    return hcat(VX, VY, VZ)
end


@inline function _soa_to_frcmat(fx::CuArray{T,1}, fy::CuArray{T,1}) where {T<:AbstractFloat}
    X = Vector{T}(undef, length(fx))
    Y = Vector{T}(undef, length(fy))
    copyto!(X, fx)
    copyto!(Y, fy)
    Z = fill(zero(T), length(X))
    CUDA.synchronize()
    return hcat(X, Y, Z)
end

@inline function _soa_to_frcmat(fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}) where {T<:AbstractFloat}
    X = Vector{T}(undef, length(fx))
    Y = Vector{T}(undef, length(fy))
    Z = Vector{T}(undef, length(fz))
    copyto!(X, fx)
    copyto!(Y, fy)
    copyto!(Z, fz)
    CUDA.synchronize()
    return hcat(X, Y, Z)
end

@inline function _soa_to_tensormat(V::CuArray{T,2}) where {T<:AbstractFloat}
    M = Matrix{T}(undef, size(V)...)
    copyto!(M, V)
    CUDA.synchronize()
    return M
end

function _write_particles_virial!(h, virial::AbstractMatrix{<:Real})
    N, M = size(virial)
    @assert M == 3 || M == 6 "particles/virial must be N×3 (2D) or N×6 (3D)"
    A = Array{Float32}(undef, N, M)
    @inbounds for i in 1:N, j in 1:M
        A[i, j] = Float32(virial[i, j])
    end
    data = GSDFiles.rowmajor(A)
    GSDFiles.write_chunk_raw!(h.user, "particles/virial";
                              type_code=GSDFiles._R_FLOAT32, N=N, M=M, data)
    GSDFiles.write_chunk_raw!(h.user, "particles/property/virial";
                              type_code=GSDFiles._R_FLOAT32, N=N, M=M, data)
    return nothing
end

# -- public API -----------------------------------------------------------

"""
Write one frame to an already-open GSD file (HOOMD schema).

Usage (2D):
    h = Writers.gsd_open("traj.gsd")
    Writers.write_gsd_frame!(h, state; diameter=1.0, types_names=["A"], step=state.step)
    Writers.gsd_close(h)

Usage (3D) is identical; z-components are written when present. Set
`write_unwrapped=true` to store unwrapped positions in the custom
`particles/position_unwrapped` chunk. Set `write_virial=true` to store the
current total per-particle configurational virial tensor from `st.virial_tensor`
using the package's native component order:
- 2D: `(xx, yy, xy)`
- 3D: `(xx, yy, zz, xy, xz, yz)`

Virial buffers are refreshed by force evaluations with `compute_energy=true`;
the writer dumps whatever is currently stored in `st.virial_tensor`.
"""
function write_gsd_frame!(h, st; diameter=1.0, types_names=["A"], step::Int=0,
                          write_forces::Bool=false, write_unwrapped::Bool=false,
                          write_virial::Bool=false, sync_on_write::Bool=false)
    T = eltype(st.rx)
    N = length(st.rx)
    tag = hasproperty(st, :tag) ? st.tag : nothing
    u(a) = _unpermuted(a, tag)

    posM = st.rz === nothing ? _soa_to_posmat(u(st.rx), u(st.ry)) :
                               _soa_to_posmat(u(st.rx), u(st.ry), u(st.rz))
    write_velocities = !(hasproperty(st, :last_integrator) && st.last_integrator == UInt8(2))
    if write_velocities
        velM = st.vz === nothing ? _soa_to_velmat(u(st.vx), u(st.vy)) :
                                   _soa_to_velmat(u(st.vx), u(st.vy), u(st.vz))
    end

    if write_forces
        frcM = st.fz === nothing ? _soa_to_frcmat(u(st.fx), u(st.fy)) :
                                   _soa_to_frcmat(u(st.fx), u(st.fy), u(st.fz))
    end

    if write_virial
        virialM = _soa_to_tensormat(u(st.virial_tensor))
    end

    if write_unwrapped
        if st.rz === nothing
            if st.rx_unwrap === nothing || st.ry_unwrap === nothing
                error("write_unwrapped=true requires unwrapped_positions=true in build_simulation")
            end
            posM_unwrap = _soa_to_posmat(u(st.rx_unwrap), u(st.ry_unwrap))
        else
            if st.rx_unwrap === nothing || st.ry_unwrap === nothing || st.rz_unwrap === nothing
                error("write_unwrapped=true requires unwrapped_positions=true in build_simulation")
            end
            posM_unwrap = _soa_to_posmat(u(st.rx_unwrap), u(st.ry_unwrap), u(st.rz_unwrap))
        end
    end

    if st.box3 === nothing
        D = UInt8(2)
        box6 = _pack_box2(st.box2::Tuple{T,T})
    else
        D = UInt8(3)
        box6 = _pack_box3(st.box3::Tuple{T,T,T})
    end

    tid_host = Vector{Int32}(undef, length(st.typeid))
    copyto!(tid_host, u(st.typeid))
    CUDA.synchronize()
    tid_0based = UInt32.(tid_host .- 1)

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

    if write_forces
        GSDFiles.write_particles_force!(h, T.(frcM))
    end

    if write_virial
        _write_particles_virial!(h, virialM)
    end

    if write_unwrapped
        dtype = T === Float32 ? :float32 : :float64
        data = dtype == :float32 ? GSDFiles.rowmajor(Float32.(posM_unwrap)) :
                                   GSDFiles.rowmajor(Float64.(posM_unwrap))
        GSDFiles.write_chunk!(h.user, "particles/position_unwrapped"; dtype, N=N, M=3, data)
    end

    if st.rx_unwrap !== nothing && st.ry_unwrap !== nothing
        imageM = Matrix{Int32}(undef, N, 3)
        rx_host = Array(st.rx)
        ry_host = Array(st.ry)
        rxu_host = Array(st.rx_unwrap)
        ryu_host = Array(st.ry_unwrap)
        write_images = true
        max_image_abs = 0.0
        if st.box3 === nothing
            st.box2 === nothing && error("SimulationState is missing box metadata.")
            Lx, Ly = st.box2
            @inbounds for i in 1:N
                imgx = round(Int64, Float64((rxu_host[i] - rx_host[i]) / Lx))
                imgy = round(Int64, Float64((ryu_host[i] - ry_host[i]) / Ly))
                max_image_abs = max(max_image_abs, abs(Float64(imgx)), abs(Float64(imgy)))
                if imgx < typemin(Int32) || imgx > typemax(Int32) ||
                   imgy < typemin(Int32) || imgy > typemax(Int32)
                    write_images = false
                    break
                end
                imageM[i, 1] = Int32(imgx)
                imageM[i, 2] = Int32(imgy)
                imageM[i, 3] = Int32(0)
            end
        else
            st.rz !== nothing || error("3D SimulationState is missing z coordinates.")
            st.rz_unwrap !== nothing || error("3D SimulationState is missing z unwrapped coordinates.")
            rz_host = Array(st.rz)
            rzu_host = Array(st.rz_unwrap)
            Lx, Ly, Lz = st.box3
            @inbounds for i in 1:N
                imgx = round(Int64, Float64((rxu_host[i] - rx_host[i]) / Lx))
                imgy = round(Int64, Float64((ryu_host[i] - ry_host[i]) / Ly))
                imgz = round(Int64, Float64((rzu_host[i] - rz_host[i]) / Lz))
                max_image_abs = max(max_image_abs,
                                    abs(Float64(imgx)),
                                    abs(Float64(imgy)),
                                    abs(Float64(imgz)))
                if imgx < typemin(Int32) || imgx > typemax(Int32) ||
                   imgy < typemin(Int32) || imgy > typemax(Int32) ||
                   imgz < typemin(Int32) || imgz > typemax(Int32)
                    write_images = false
                    break
                end
                imageM[i, 1] = Int32(imgx)
                imageM[i, 2] = Int32(imgy)
                imageM[i, 3] = Int32(imgz)
            end
        end
        if write_images
            GSDFiles.write_chunk_raw!(h.user, "particles/image";
                                      type_code=UInt8(GSDFiles.GSD_TYPE_INT32 + 0x01),
                                      N=N, M=3, data=GSDFiles.rowmajor(imageM))
        else
            @warn "Skipping particles/image for this GSD frame because the image count exceeded Int32 range. This usually indicates unstable dynamics or an unequilibrated initial state." step=step max_abs_image=max_image_abs
        end
    end

    if hasproperty(st, :bonds) && (st.bonds !== nothing)
        idx    = Vector{Int32}(undef, length(st.bonds.index));    copyto!(idx, st.bonds.index)
        counts = Vector{Int32}(undef, length(st.bonds.counts));   copyto!(counts, st.bonds.counts)
        flat   = Vector{Int32}(undef, length(st.bonds.flat));     copyto!(flat, st.bonds.flat)
        CUDA.synchronize()

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
            GSDFiles.write_bonds_N!(h, pairs_size)
            GSDFiles.write_bonds_types!(h, ["bond"])
            GSDFiles.write_bonds_typeid!(h, fill(UInt32(0), pairs_size))
            grp = Array{UInt32}(undef, pairs_size, 2)
            @inbounds for k in 1:pairs_size
                grp[k,1] = pairs[k][1]
                grp[k,2] = pairs[k][2]
            end
            GSDFiles.write_bonds_group!(h, grp)
        else
            GSDFiles.write_bonds_N!(h, 0)
        end
    end
    GSDFiles.end_frame!(h)
    if sync_on_write
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
