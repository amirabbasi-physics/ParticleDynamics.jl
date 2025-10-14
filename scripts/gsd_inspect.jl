#!/usr/bin/env julia

using Printf

const GSD_MAGIC = 0x65DF65DF65DF65DF
const HDR_BYTES = 256
const NAME_SEG_BYTES = 64

# OVITO/GSD v2 type codes we may see on disk
const TYPE_UINT32 = UInt8(0x03)
const TYPE_UINT64 = UInt8(0x04)
const TYPE_INT8   = UInt8(0x05)
const TYPE_FLOAT  = UInt8(0x09) # OVITO float32
const TYPE_FLOAT_ALT = UInt8(0x08) # GSD float32 (alternate)

function read_header(io)
    seek(io, 0)
    magic                    = read(io, UInt64)
    index_location           = read(io, UInt64)
    index_allocated_entries  = read(io, UInt64)
    namelist_location        = read(io, UInt64)
    namelist_alloc_entries   = read(io, UInt64)
    schema_version           = read(io, UInt32)
    gsd_version              = read(io, UInt32)
    application              = read!(io, Vector{UInt8}(undef, 64))
    schema                   = read!(io, Vector{UInt8}(undef, 64))
    reserved                 = read!(io, Vector{UInt8}(undef, 80))
    return (;
        magic,
        index_location,
        index_allocated_entries,
        namelist_location,
        namelist_alloc_entries,
        schema_version,
        gsd_version,
        application,
        schema,
        reserved
    )
end

function read_names(io, hdr)
    seek(io, Int(hdr.namelist_location))
    blob = read!(io, Vector{UInt8}(undef, Int(hdr.namelist_alloc_entries) * NAME_SEG_BYTES))
    names = String[]
    start = 1
    for i in 1:length(blob)
        if blob[i] == 0x00
            if i >= start
                push!(names, String(copy(blob[start:i-1])))
            else
                push!(names, "")
            end
            start = i+1
        end
    end
    return names
end

function read_index(io, hdr)
    seek(io, Int(hdr.index_location))
    entries = NamedTuple[]
    while true
        frame     = read(io, UInt64)
        N         = read(io, UInt64)
        location  = read(io, Int64)
        M         = read(io, UInt32)
        id        = read(io, UInt16)
        typ       = read(io, UInt8)
        flags     = read(io, UInt8)
        if location == 0
            break
        end
        push!(entries, (frame=frame, N=N, location=location, M=M, id=id, typ=typ, flags=flags))
    end
    return entries
end

function find_name_id(names::Vector{String}, target::String)
    for (i,n) in enumerate(names)
        if n == target
            return UInt16(i-1) # ids are 0-based
        end
    end
    return nothing
end

function read_chunk_float32(io, e)
    # Accept either OVITO (0x09) or GSD (0x08) float codes
    if e.typ != TYPE_FLOAT && e.typ != TYPE_FLOAT_ALT
        error(@sprintf("Chunk type 0x%02x is not float32", Int(e.typ)))
    end
    seek(io, Int(e.location))
    count = Int(e.N) * Int(e.M)
    buf = Vector{Float32}(undef, count)
    read!(io, buf)
    return buf
end

function main()
    if length(ARGS) == 0
        println("Usage: julia scripts/gsd_inspect.jl <file.gsd> [frame=1]")
        return
    end
    path = ARGS[1]
    frame = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    io = open(path, "r")
    hdr = read_header(io)
    @assert hdr.magic == GSD_MAGIC "Not a GSD file (bad magic)"
    names = read_names(io, hdr)
    idx   = read_index(io, hdr)
    nf    = maximum(e.frame for e in idx; init=0) + 1
    println("File: ", path)
    println("Frames: ", nf)
    println("Known chunk names (", length(names), "):")
    for (i,n) in enumerate(names)
        n == "" && continue
        @printf("  id=%3d name=%s\n", i-1, n)
    end
    fid = UInt64(frame-1)
    ents = filter(e -> e.frame == fid, idx)
    alt_names = [
        # Preferred pluralized name used by current writer
        "particles/forces",
        # Backward/compatibility aliases some tools may use
        "particles/force",
        "particles/property/force",
        "particles/net_force",
    ]
    found = false
    for nm in alt_names
        id = find_name_id(names, nm)
        if id === nothing
            continue
        end
        matches = filter(e -> e.id == id, ents)
        if !isempty(matches)
            e = only(matches)
            found = true
            println("Found force chunk: ", nm, "  N=", e.N, " M=", e.M,
                    @sprintf(" type=0x%02x", Int(e.typ)))
            # Read and print simple stats
            vec = read_chunk_float32(io, e)
            N = Int(e.N); M = Int(e.M)
            @assert M == 3 "force chunk must have 3 components"
            # compute mean norm over first min(N, 10) particles
            s = 0.0
            k = min(N, 10)
            for i in 1:k
                fx,fy,fz = vec[3*(i-1)+1:3*(i-1)+3]
                s += sqrt(fx*fx + fy*fy + fz*fz)
            end
            @printf("  sample(mean |F| over first %d) = %.6g\n", k, s/k)
        end
    end
    if !found
        println("No force chunk found in frame ", frame, ". Checked names: ", join(alt_names, ", "))
        # Hint: search which frames do contain a force chunk
        frames_with_force = Int[]
        for nm in alt_names
            id = find_name_id(names, nm)
            id === nothing && continue
            for e in filter(e -> e.id == id, idx)
                push!(frames_with_force, Int(e.frame)+1)
            end
        end
        if !isempty(frames_with_force)
            unique!(frames_with_force)
            sort!(frames_with_force)
            println("Force chunk present in frames: ", frames_with_force)
        end
    end
    close(io)
end

main()
