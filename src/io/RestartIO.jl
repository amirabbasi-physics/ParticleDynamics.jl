"""
Read a valid frame from a GSD file and return a `GSDFrameData`.

The returned object can still be destructured into the legacy tuple
`step, rx, ry, rz, vx, vy, vz, typeid, types, box, force` for backwards
compatibility, while also exposing rich metadata (`frame.topology`,
`frame.particle_properties`, `frame.configuration`, …) that can be fed
directly into ParticleDynamics initialisation routines.

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

                vir_e = GSDFiles._maybe_one_of(r, ents, [
                    "particles/virial",
                    "particles/property/virial",
                ])
                local_virialM = nothing
                if vir_e !== nothing
                    virial_raw = read_chunk(vir_e)
                    name_vir = r.names[Int(vir_e.id)+1]
                    if virial_raw isa AbstractMatrix
                        local_virialM = Matrix{T}(virial_raw)
                    elseif virial_raw isa AbstractVector
                        if length(virial_raw) == 3N
                            reshaped = reshape(virial_raw, (3, N))'
                            local_virialM = Matrix{T}(reshaped)
                        elseif length(virial_raw) == 6N
                            reshaped = reshape(virial_raw, (6, N))'
                            local_virialM = Matrix{T}(reshaped)
                        else
                            @warn "Unexpected virial chunk shape; skipping virial" file=file_path frame_index=idx chunk=name_vir
                        end
                    else
                        @warn "Unsupported virial chunk type; skipping virial" file=file_path frame_index=idx chunk=name_vir
                    end
                    startswith(name_vir, "particles/property/") && push!(property_skip, name_vir)
                end
                if local_virialM !== nothing
                    particle_props[:virial] = local_virialM
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
