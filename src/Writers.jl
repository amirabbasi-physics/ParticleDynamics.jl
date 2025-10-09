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

# -- public API -----------------------------------------------------------

"""
Write one frame to an already-open GSD file (HOOMD schema).

Usage (2D):
    h = Writers.gsd_open("traj.gsd")
    Writers.write_gsd_frame!(h, state; diameter=1.0, types_names=["A"], step=state.step)
    Writers.gsd_close(h)

Usage (3D) is identical; z-components are written when present.
"""
function write_gsd_frame!(h, st; diameter=1.0, types_names=["A"], step::Int=0, write_forces::Union{Nothing,Bool}=nothing)
    # Element type used for numeric conversions
    T = eltype(st.rx)
    N = length(st.rx)

    # positions and velocities (N×3 T)
    posM = st.rz === nothing ? _soa_to_posmat(st.rx, st.ry) :
                               _soa_to_posmat(st.rx, st.ry, st.rz)
    # velocities are undefined for Brownian dynamics; skip when last_integrator==2
    write_vel = !(hasproperty(st, :last_integrator) && st.last_integrator == UInt8(2))
    if write_vel
        velM = st.vz === nothing ? _soa_to_velmat(st.vx, st.vy) :
                                   _soa_to_velmat(st.vx, st.vy, st.vz)
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
    if write_vel
        GSDFiles.write_particles_velocity!(h, T.(velM))
    end

    # Forces: default is false for both Brownian and Langevin; enable only if user asks
    local do_forces::Bool
    do_forces = (write_forces === true)
    if do_forces
        # Build Nx3 T forces
        FX = Vector{T}(undef, N); FY = Vector{T}(undef, N)
        copyto!(FX, st.fx); copyto!(FY, st.fy)
        FZ = st.fz === nothing ? fill(zero(T), N) : (tmp=Vector{T}(undef,N); copyto!(tmp, st.fz); tmp)
        CUDA.synchronize()
        F = Array{T}(undef, N, 3)
        @inbounds for i in 1:N
            F[i,1] = FX[i]; F[i,2] = FY[i]; F[i,3] = FZ[i]
        end
        # Flatten row-major for GSD chunk
        row = Vector{T}(undef, N*3)
        k = 1
        @inbounds for i in 1:N
            row[k] = F[i,1]; row[k+1] = F[i,2]; row[k+2] = F[i,3]
            k += 3
        end
        GSDFiles.write_chunk_raw!(h.user, "particles/force";
                                  type_code=GSDFiles.GSD_TYPE_FLOAT,
                                  N=N, M=3, data=row)
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
Read the **last** frame from a GSD file and return SoA arrays.

Returns:
    step::Int,
    rx::Vector{T}, ry::Vector{T}, rz::Union{Nothing,Vector{T}},
    vx::Vector{T}, vy::Vector{T}, vz::Union{Nothing,Vector{T}},
    typeid_1based::Vector{Int32},
    types_names::Vector{String},
    box::Union{Tuple{T,T},Tuple{T,T,T}}
"""
function read_last_gsd(file_path::AbstractString)
    r = GSDFiles.GSDReader(file_path)
    GSDFiles.open_gsd(r)
    try
        nf = GSDFiles.num_frames(r)
        nf == 0 && error("No frames in GSD: $file_path")
        fid = nf - 1  # 0-based index of last frame

        step = Int(GSDFiles.read_configuration_step(r, fid))
        D    = Int(GSDFiles.read_configuration_dimensions(r, fid))
        N    = Int(GSDFiles.read_particles_N(r, fid))

        posM = GSDFiles.read_particles_position(r, fid)  # N×3
        velM = GSDFiles.read_particles_velocity(r, fid)  # N×3
        T = eltype(posM)

        rx = T.(posM[:,1]); ry = T.(posM[:,2])
        rz = D == 3 ? T.(posM[:,3]) : nothing

        vx = T.(velM[:,1]); vy = T.(velM[:,2])
        vz = D == 3 ? T.(velM[:,3]) : nothing

        typeid0 = Vector{UInt32}(GSDFiles.read_particles_typeid(r, fid))
        types   = Vector{String}(GSDFiles.read_particles_types(r, fid))
        typeid1 = Int32.(typeid0 .+ 1)

        box6 = GSDFiles.read_configuration_box(r, fid)
        box  = D == 2 ? (T(box6[1]), T(box6[2])) :
                        (T(box6[1]), T(box6[2]), T(box6[3]))

        return step, rx, ry, rz, vx, vy, vz, typeid1, types, box
    finally
        GSDFiles.close_gsd(r)
    end
end

end # module
