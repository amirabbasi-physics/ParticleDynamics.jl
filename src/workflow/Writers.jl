using Printf: @sprintf
using ..SimulationCore
using ..Writers: gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!

abstract type Writer end

@kwdef struct TableWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    observables::Vector{Any} = Any[]
    mode::Symbol = :replace
    delimiter::String = ","
    format::Symbol = :scientific
    append::Bool = false
end

function TableWriter(filename::AbstractString;
                     every=nothing,
                     schedule=nothing,
                     observables=Any[],
                     mode::Symbol=:replace,
                     delimiter::AbstractString=",",
                     format::Symbol=:scientific,
                     append::Bool=false)
    return TableWriter(filename=String(filename),
                       every=every,
                       schedule=schedule,
                       observables=Any[observables...],
                       mode=mode,
                       delimiter=String(delimiter),
                       format=format,
                       append=append)
end

@kwdef struct GSDWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    group = nothing
    write_start::Bool = true
    mode::Symbol = :replace
    append::Bool = false
    types = :automatic
    diameter = :automatic
    write_unwrapped::Bool = false
    sync_on_write::Bool = true
    write_forces::Bool = false
    observables::Vector{Any} = Any[]
end

function GSDWriter(filename::AbstractString;
                   every=nothing,
                   schedule=nothing,
                   group=Group(:all, AllSelection()),
                   write_start::Bool=true,
                   mode::Symbol=:replace,
                   append::Bool=false,
                   types=:automatic,
                   diameter=:automatic,
                   write_unwrapped::Bool=false,
                   sync_on_write::Bool=true,
                   write_forces::Bool=false,
                   observables=Any[])
    return GSDWriter(filename=String(filename),
                     every=every,
                     schedule=schedule,
                     group=group,
                     write_start=write_start,
                     mode=mode,
                     append=append,
                     types=types,
                     diameter=diameter,
                     write_unwrapped=write_unwrapped,
                     sync_on_write=sync_on_write,
                     write_forces=write_forces,
                     observables=Any[observables...])
end

mutable struct PreparedWriter
    writer::Writer
    schedule
    requests::Vector{NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}}
    handle
    header::Vector{String}
    header_written::Bool
    start_written::Bool
    needs_energy::Bool
    needs_interval_reset::Bool
end

function _normalize_writer_requests(sim, writer::TableWriter)
    raw = isempty(writer.observables) ? Any[sim.observables...] : writer.observables
    isempty(raw) && throw(ArgumentError("TableWriter($(writer.filename)) requires at least one observable request."))
    requests = NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}[]
    for entry in raw
        if entry isa Pair
            obs = entry.first
            fields = _normalize_observable_fields(obs, entry.second)
        elseif entry isa Observable
            obs = entry
            fields = _normalize_observable_fields(obs, nothing)
        else
            throw(ArgumentError("Unsupported writer observable request $(typeof(entry)). Use `observable` or `observable => [:field, ...]`."))
        end
        obs isa Observable || throw(ArgumentError("Writer requests must reference workflow Observable objects."))
        push!(requests, (observable=obs, fields=fields))
    end
    return requests
end

function _normalize_writer_requests(sim, writer::GSDWriter)
    requests = NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}[]
    for entry in writer.observables
        if entry isa Pair
            obs = entry.first
            fields = _normalize_observable_fields(obs, entry.second)
        elseif entry isa Observable
            obs = entry
            fields = _normalize_observable_fields(obs, nothing)
        else
            throw(ArgumentError("Unsupported GSDWriter observable request $(typeof(entry))."))
        end
        push!(requests, (observable=obs, fields=fields))
    end
    return requests
end

_writer_append_mode(writer::Writer) = getfield(writer, :append) || getfield(writer, :mode) == :append

function _validate_writer(writer::TableWriter)
    writer.mode in (:replace, :append) || throw(ArgumentError("TableWriter mode must be :replace or :append."))
    writer.format in (:scientific, :plain) || throw(ArgumentError("TableWriter format must be :scientific or :plain."))
    return writer
end

function _validate_writer(writer::GSDWriter)
    writer.mode in (:replace, :append) || throw(ArgumentError("GSDWriter mode must be :replace or :append."))
    return writer
end

function prepare_writers!(sim)
    contexts = PreparedWriter[]
    for writer in sim.writers
        _validate_writer(writer)
        requests = _normalize_writer_requests(sim, writer)
        for req in requests
            _ensure_observable_context!(sim, req.observable)
        end
        schedule = normalize_schedule(every=getfield(writer, :every), schedule=getfield(writer, :schedule), default=Every(1))
        needs_energy = any(req -> observable_requires_energy(req.observable, req.fields), requests)
        needs_interval_reset = any(req -> observable_has_interval_fields(req.observable, req.fields), requests)
        push!(contexts, PreparedWriter(writer, schedule, requests, nothing, String[], false, false, needs_energy, needs_interval_reset))
    end
    sim.metadata[:workflow_writer_contexts] = contexts
    return sim
end

function _writer_contexts(sim)
    return get(sim.metadata, :workflow_writer_contexts, PreparedWriter[])
end

function _open_writer!(ctx::PreparedWriter)
    writer = ctx.writer
    if writer isa TableWriter
        ctx.handle === nothing || return ctx
        path = writer.filename
        mkpath(dirname(path))
        if !_writer_append_mode(writer) && isfile(path)
            rm(path; force=true)
        end
        ctx.handle = open(path, _writer_append_mode(writer) ? "a" : "w")
        ctx.header_written = _writer_append_mode(writer) && isfile(path) && filesize(path) > 0
    elseif writer isa GSDWriter
        ctx.handle === nothing || return ctx
        path = writer.filename
        mkpath(dirname(path))
        if !_writer_append_mode(writer) && isfile(path)
            rm(path; force=true)
        end
        ctx.handle = gsd_open(path; append=_writer_append_mode(writer))
    end
    return ctx
end

function _close_writer!(ctx::PreparedWriter)
    if ctx.handle === nothing
        return ctx
    end
    writer = ctx.writer
    if writer isa TableWriter
        close(ctx.handle)
    elseif writer isa GSDWriter
        gsd_close(ctx.handle)
    end
    ctx.handle = nothing
    return ctx
end

function close_writers!(sim)
    for ctx in _writer_contexts(sim)
        _close_writer!(ctx)
    end
    return sim
end

function _writer_types_names(sim, writer::GSDWriter)
    if writer.types === :automatic
        return isempty(sim.system.types) ? ["A"] : String.(sim.system.types)
    elseif writer.types isa AbstractVector
        return String.(collect(writer.types))
    else
        throw(ArgumentError("GSDWriter.types must be :automatic or an AbstractVector."))
    end
end

function _writer_diameter(sim, writer::GSDWriter)
    if writer.diameter !== :automatic
        return writer.diameter
    end
    if haskey(sim.system.metadata, :diameters)
        return sim.system.metadata[:diameters]
    elseif haskey(sim.system.metadata, :diameter)
        return sim.system.metadata[:diameter]
    else
        return one(eltype(sim.state.rx))
    end
end

function _validate_gsd_group(sim, writer::GSDWriter)
    ref = writer.group
    ref === nothing && return nothing
    if ref isa Group
        ref.selection isa AllSelection || throw(ArgumentError("GSDWriter currently supports only all-particle writes."))
    elseif ref isa Symbol
        group = sim.groups[ref]
        group.selection isa AllSelection || throw(ArgumentError("GSDWriter currently supports only all-particle writes."))
    else
        throw(ArgumentError("GSDWriter.group must be a Group or Symbol."))
    end
    return nothing
end

function _format_table_value(writer::TableWriter, value)
    if value isa AbstractFloat
        return writer.format == :scientific ? @sprintf("%.7e", value) : string(value)
    elseif value isa Real
        return string(value)
    elseif value isa Symbol
        return String(value)
    elseif value === nothing
        return ""
    else
        return string(value)
    end
end

function _flatten_table_value(prefix::String, value)
    if value isa NamedTuple
        headers = String[]
        values = Any[]
        for name in keys(value)
            subheaders, subvalues = _flatten_table_value(string(prefix, ".", name), getproperty(value, name))
            append!(headers, subheaders)
            append!(values, subvalues)
        end
        return headers, values
    elseif value isa AbstractDict
        headers = String[]
        values = Any[]
        for key in sort!(collect(keys(value)); by=string)
            push!(headers, string(prefix, ".", key))
            push!(values, value[key])
        end
        return headers, values
    elseif value isa AbstractVector
        headers = [string(prefix, "[", i, "]") for i in eachindex(value)]
        values = collect(value)
        return headers, values
    else
        return [prefix], [value]
    end
end

function _table_headers_and_values(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer::TableWriter
    headers = String["step"]
    values = Any[step]
    for req in ctx.requests
        sample = sample_observable(sim, req.observable; fields=req.fields)
        for field in req.fields
            prefix = string(observable_name(req.observable), ".", field)
            subheaders, subvalues = _flatten_table_value(prefix, getproperty(sample, field))
            append!(headers, subheaders)
            append!(values, subvalues)
        end
    end
    return headers, String[_format_table_value(writer, value) for value in values]
end

function _write_table!(sim, ctx::PreparedWriter, step::Int)
    _open_writer!(ctx)
    writer = ctx.writer::TableWriter
    headers, values = _table_headers_and_values(sim, ctx, step)
    if !ctx.header_written
        println(ctx.handle, join(headers, writer.delimiter))
        ctx.header = headers
        ctx.header_written = true
    end
    println(ctx.handle, join(values, writer.delimiter))
    flush(ctx.handle)
    return ctx
end

function _write_gsd!(sim, ctx::PreparedWriter, step::Int)
    _open_writer!(ctx)
    writer = ctx.writer::GSDWriter
    _validate_gsd_group(sim, writer)
    write_gsd_frame!(
        ctx.handle,
        sim.state;
        diameter=_writer_diameter(sim, writer),
        types_names=_writer_types_names(sim, writer),
        step=step,
        write_forces=writer.write_forces,
        write_unwrapped=writer.write_unwrapped,
        sync_on_write=writer.sync_on_write,
    )
    return ctx
end

function _write_writer!(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer
    if writer isa TableWriter
        return _write_table!(sim, ctx, step)
    else
        return _write_gsd!(sim, ctx, step)
    end
end

function write_initial_frames!(sim)
    st = sim.state
    st === nothing && return sim
    for ctx in _writer_contexts(sim)
        writer = ctx.writer
        if writer isa GSDWriter && writer.write_start && !ctx.start_written
            _write_writer!(sim, ctx, st.step)
            ctx.start_written = true
        end
    end
    return sim
end

function write_scheduled_outputs!(sim, step::Int)
    interval_consumed = false
    for ctx in _writer_contexts(sim)
        schedule_matches(ctx.schedule, step) || continue
        _write_writer!(sim, ctx, step)
        interval_consumed |= ctx.needs_interval_reset
    end
    return interval_consumed
end

function active_writer_requires_energy(sim, step::Int)
    for ctx in _writer_contexts(sim)
        schedule_matches(ctx.schedule, step) || continue
        ctx.needs_energy && return true
    end
    return false
end
