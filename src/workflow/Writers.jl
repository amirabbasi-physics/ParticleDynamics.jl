using Printf: @sprintf
using ..SimulationCore
using ..Writers: gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!

"""
    Writer

Abstract workflow output writer.
"""
abstract type Writer end

"""
    TableWriter(filename; every=nothing, schedule=nothing, observables=[], mode=:replace, delimiter=",", format=:scientific, append=false, style=:auto)

Scheduled text-table writer for workflow observables.
"""
@kwdef struct TableWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    observables::Vector{Any} = Any[]
    mode::Symbol = :replace
    delimiter::String = ","
    format::Symbol = :scientific
    append::Bool = false
    style::Symbol = :auto
end

function TableWriter(filename::AbstractString;
                     every=nothing,
                     schedule=nothing,
                     observables=Any[],
                     mode::Symbol=:replace,
                     delimiter::AbstractString=",",
                     format::Symbol=:scientific,
                     append::Bool=false,
                     style::Symbol=:auto)
    return TableWriter(filename=String(filename),
                       every=every,
                       schedule=schedule,
                       observables=Any[observables...],
                       mode=mode,
                       delimiter=String(delimiter),
                       format=format,
                       append=append,
                       style=style)
end

"""
    GSDWriter(filename; every=nothing, schedule=nothing, group=Group(:all, AllSelection()), write_start=true, mode=:replace, append=false, types=:automatic, diameter=:automatic, write_unwrapped=false, sync_on_write=true, write_forces=false, write_virial=false, observables=[])

Scheduled GSD trajectory writer for workflow simulations.
"""
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
    write_virial::Bool = false
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
                   write_virial::Bool=false,
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
                     write_virial=write_virial,
                     observables=Any[observables...])
end

const IN_MEMORY_FIELDS = (:positions, :unwrapped_positions, :velocities, :forces, :energy, :virial)

"""
    InMemoryWriter(; every=nothing, schedule=nothing, fields=[:positions], write_start=false)

Scheduled writer that keeps frames in host memory instead of a file, for
analysis and debugging inside the same Julia session. `fields` selects any of
`$(join(map(repr, IN_MEMORY_FIELDS), ", "))`:

- per-particle data (`:positions`, `:unwrapped_positions`, `:velocities`,
  `:forces`) is stored in particle-id order, i.e. the order of the
  `ParticleSystem`, independent of the engine's spatial sorting;
- `:energy` stores the total kinetic and potential energy;
- `:virial` stores the total virial tensor.

Velocities are not recorded for Brownian dynamics (overdamped motion has none).
Frames accumulate across stages and `run!` calls until `empty!(writer)`; read
them with [`recorded_data`](@ref).
"""
mutable struct InMemoryWriter <: Writer
    every
    schedule
    fields::Vector{Symbol}
    write_start::Bool
    steps::Vector{Int}
    data::Dict{Symbol,Any}   # field => flat host Vector{T}, frame-major blocks
    n::Int                   # particles per frame (0 until the first frame)
    dim::Int
    scratch::Any             # host staging Vector{T} for device copies
end

function InMemoryWriter(; every=nothing, schedule=nothing, fields=[:positions], write_start::Bool=false)
    fields = unique!(Symbol[fields...])
    isempty(fields) && throw(ArgumentError("InMemoryWriter needs at least one field."))
    for f in fields
        f in IN_MEMORY_FIELDS ||
            throw(ArgumentError("Unknown InMemoryWriter field :$f; use any of $(IN_MEMORY_FIELDS)."))
    end
    return InMemoryWriter(every, schedule, fields, write_start, Int[], Dict{Symbol,Any}(), 0, 0, nothing)
end

function Base.empty!(writer::InMemoryWriter)
    empty!(writer.steps)
    empty!(writer.data)
    writer.n = 0
    writer.dim = 0
    return writer
end

"""
    recorded_data(writer::InMemoryWriter) -> NamedTuple

Frames recorded so far. `step` holds the step of each frame. Per-particle
fields are `N × D × nframes` arrays (`data.positions[:, :, k]` is frame `k`,
rows in particle-id order), `energy` gives `kinetic_energy` and
`potential_energy` vectors, and `virial` is a `D × D × nframes` array. The
arrays are copies; the writer keeps recording.
"""
function recorded_data(writer::InMemoryWriter)
    nframes = length(writer.steps)
    names = Symbol[:step]
    values = Any[copy(writer.steps)]
    for f in writer.fields
        buf = get(writer.data, f, nothing)
        if f === :energy
            e = buf === nothing ? Float64[] : buf
            push!(names, :kinetic_energy, :potential_energy)
            push!(values, e[1:2:end], e[2:2:end])
        elseif f === :virial
            push!(names, :virial)
            push!(values, buf === nothing ? zeros(0, 0, 0) : reshape(copy(buf), writer.dim, writer.dim, nframes))
        else
            push!(names, f)
            push!(values, buf === nothing ? zeros(0, 0, 0) : reshape(copy(buf), writer.n, writer.dim, nframes))
        end
    end
    return NamedTuple{Tuple(names)}(Tuple(values))
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
    opened::Bool
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

_normalize_writer_requests(sim, ::InMemoryWriter) =
    NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}[]

_writer_append_mode(writer::Writer) = getfield(writer, :append) || getfield(writer, :mode) == :append

function _validate_writer(writer::TableWriter)
    writer.mode in (:replace, :append) || throw(ArgumentError("TableWriter mode must be :replace or :append."))
    writer.format in (:scientific, :plain) || throw(ArgumentError("TableWriter format must be :scientific or :plain."))
    writer.style in (:auto, :csv, :log) || throw(ArgumentError("TableWriter style must be :auto, :csv, or :log."))
    return writer
end

function _validate_writer(writer::GSDWriter)
    writer.mode in (:replace, :append) || throw(ArgumentError("GSDWriter mode must be :replace or :append."))
    return writer
end

_validate_writer(writer::InMemoryWriter) = writer

# Energies (and, on sampled steps, virials) the writer reads must be computed on its steps.
_writer_needs_energy(writer::GSDWriter) = writer.write_virial
_writer_needs_energy(writer::InMemoryWriter) = :energy in writer.fields || :virial in writer.fields
_writer_needs_energy(::Writer) = false

function prepare_writers!(sim)
    previous = _writer_contexts(sim)
    contexts = PreparedWriter[]
    for writer in sim.writers
        _validate_writer(writer)
        requests = _normalize_writer_requests(sim, writer)
        for req in requests
            _ensure_observable_context!(sim, req.observable)
        end
        schedule = normalize_schedule(every=getfield(writer, :every), schedule=getfield(writer, :schedule), default=Every(1))
        needs_energy = any(req -> observable_requires_energy(req.observable, req.fields), requests) ||
                       _writer_needs_energy(writer)
        needs_interval_reset = any(req -> observable_has_interval_fields(req.observable, req.fields), requests)
        ctx = PreparedWriter(writer, schedule, requests, nothing, String[], false, false, needs_energy, needs_interval_reset, false)
        old_index = findfirst(old -> old.writer === writer, previous)
        if old_index !== nothing
            old = previous[old_index]
            ctx.handle = old.handle
            ctx.header = old.header
            ctx.header_written = old.header_written
            ctx.start_written = old.start_written
            ctx.opened = old.opened
        end
        push!(contexts, ctx)
    end
    for old in previous
        any(ctx -> ctx.writer === old.writer, contexts) || _close_writer!(old)
    end
    sim.metadata[:workflow_writer_contexts] = contexts
    return sim
end

function _writer_contexts(sim)
    return get(sim.metadata, :workflow_writer_contexts, PreparedWriter[])
end

function _open_writer!(ctx::PreparedWriter)
    writer = ctx.writer
    # Replacement applies once per prepared writer session, not once per stage.
    append = ctx.opened || _writer_append_mode(writer)
    if writer isa TableWriter
        ctx.handle === nothing || return ctx
        path = writer.filename
        mkpath(dirname(path))
        header_exists = append && isfile(path) && filesize(path) > 0
        ctx.handle = open(path, append ? "a" : "w")
        ctx.header_written = header_exists
    elseif writer isa GSDWriter
        ctx.handle === nothing || return ctx
        path = writer.filename
        mkpath(dirname(path))
        if !append && isfile(path)
            rm(path; force=true)
        end
        ctx.handle = gsd_open(path; append=append)
    end
    ctx.opened = true
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
        return nothing
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

_table_style(writer::TableWriter) =
    writer.style == :auto ? (endswith(lowercase(writer.filename), ".log") ? :log : :csv) : writer.style

_titleize_label(label::AbstractString) = replace(uppercasefirst(replace(String(label), "_" => " ")), " Msd" => " MSD", " Vacf" => " VACF")

function _workflow_group_name(obs)
    hasproperty(obs, :group) || return nothing
    group = getfield(obs, :group)
    if group isa Group
        return group.name
    elseif group isa Symbol
        return group
    end
    return nothing
end

function _group_suffix(obs)
    name = _workflow_group_name(obs)
    name === nothing && return ""
    name == :all && return ""
    return " (" * String(name) * ")"
end

function _log_field_label(obs::ThermodynamicObservable, field::Symbol)
    suffix = _group_suffix(obs)
    if field == :temperature
        return "Temperature" * suffix
    elseif field == :kinetic_energy
        return "E_kin" * suffix
    elseif field == :potential_energy
        return "E_pot" * suffix
    elseif field == :total_energy
        return "E_tot" * suffix
    elseif field == :virial
        return "virial" * suffix
    elseif field == :kinetic_energy_accumulated
        return "E_kin accum" * suffix
    elseif field == :potential_energy_accumulated
        return "E_pot accum" * suffix
    elseif field == :virial_accumulated
        return "virial accum" * suffix
    end
    return _titleize_label(field) * suffix
end

_log_field_label(::BathExchangeObservable, field::Symbol) =
    field == :heat ? "Bath Energy" :
    field == :entropy ? "Bath Entropy" :
    field == :entropy_production_rate ? "Entropy Production Rate" :
    field == :temperature_error ? "Temperature Error" :
    field == :extended_hamiltonian ? "Extended Hamiltonian" :
    field == :thermostat_kinetic ? "Thermostat Kinetic" :
    field == :thermostat_potential ? "Thermostat Potential" :
    _titleize_label(field)

function _log_field_label(obs::VirialObservable, field::Symbol)
    suffix = _group_suffix(obs)
    if field == :virial
        return "virial" * suffix
    elseif field == :virial_accumulated
        return "virial accum" * suffix
    end
    return _titleize_label(field) * suffix
end

function _log_collision_pair_label(pairkey)
    text = replace(String(pairkey), "_" => "/")
    return text * " collision rate"
end

_log_field_label(::CollisionObservable, field::Symbol) =
    field in (:counts, :rate) ? "collision rate" :
    field in (:pair_counts, :pair_rates) ? "pair collision rate" :
    _titleize_label(field)

function _log_field_label(obs::Union{MSDObservable,VACFObservable}, field::Symbol)
    suffix = _group_suffix(obs)
    return (
        field == :msd ? "MSD" * suffix :
        field == :vacf ? "VACF" * suffix :
        field == :elapsed_time ? "elapsed time" * suffix :
        field == :elapsed_steps ? "elapsed steps" * suffix :
        _titleize_label(field) * suffix
    )
end

function _log_format_value(value)
    if value isa AbstractFloat
        return @sprintf("%.5e", value)
    elseif value isa Integer
        return @sprintf("%.5e", float(value))
    elseif value isa Real
        return @sprintf("%.5e", Float64(value))
    elseif value isa Symbol
        return String(value)
    elseif value === nothing
        return ""
    else
        return string(value)
    end
end

function _center_text(text::AbstractString, width::Int)
    pad = max(0, width - length(text))
    left = pad ÷ 2
    right = pad - left
    return repeat(" ", left) * text * repeat(" ", right)
end

function _log_widths(headers::Vector{String})
    return [max(length(header), 13) + 2 for header in headers]
end

function _log_headers_and_values(sim, ctx::PreparedWriter, step::Int)
    headers = String["Time"]
    values = Any[step]
    for req in ctx.requests
        _reject_subgroup_global_virial(sim, req.observable, req.fields)
        _refresh_energies_for!(sim, req.observable, req.fields)
        data = _observable_data(sim, req.observable)
        for field in req.fields
            if req.observable isa CollisionObservable && field in (:counts, :rate)
                push!(headers, _log_field_label(req.observable, field))
                push!(values, getproperty(data, :rate))
            elseif req.observable isa CollisionObservable && field in (:pair_counts, :pair_rates)
                pair_rates = getproperty(data, :pair_rates)
                for key in _collision_pair_labels(sim)
                    push!(headers, _log_collision_pair_label(key))
                    push!(values, get(pair_rates, key, 0.0))
                end
            else
                push!(headers, _log_field_label(req.observable, field))
                push!(values, getproperty(data, field))
            end
        end
    end
    return headers, String[_log_format_value(value) for value in values]
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

function _csv_headers_and_values(sim, ctx::PreparedWriter, step::Int)
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

function _table_headers_and_values(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer::TableWriter
    if _table_style(writer) == :log
        return _log_headers_and_values(sim, ctx, step)
    end
    return _csv_headers_and_values(sim, ctx, step)
end

function _write_table!(sim, ctx::PreparedWriter, step::Int)
    _open_writer!(ctx)
    writer = ctx.writer::TableWriter
    headers, values = _table_headers_and_values(sim, ctx, step)
    if !ctx.header_written
        if _table_style(writer) == :log
            widths = _log_widths(headers)
            println(ctx.handle, join((_center_text(header, widths[i]) for (i, header) in pairs(headers)), " | "))
        else
            println(ctx.handle, join(headers, writer.delimiter))
        end
        ctx.header = headers
        ctx.header_written = true
    end
    if _table_style(writer) == :log
        widths = _log_widths(headers)
        println(ctx.handle, join((_center_text(values[i], widths[i]) for i in eachindex(values)), " | "))
    else
        println(ctx.handle, join(values, writer.delimiter))
    end
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
        write_virial=writer.write_virial,
        write_unwrapped=writer.write_unwrapped,
        sync_on_write=writer.sync_on_write,
        logs=_writer_logs(sim),
    )
    return ctx
end

# Restart records written with every GSD frame.
function _writer_logs(sim)
    ctx = get(sim.metadata, :barostat_context, nothing)
    ctx === nothing && return ()
    return ("log/barostat/state" => barostat_state_record(sim),)
end

_brownian_run(sim) = sim.lowlevel_integrator isa Union{SimulationCore.BrownianSpec,SimulationCore.EMSpec}

# Device arrays `cols` (one per dimension) appended as one N × D block in particle-id order.
function _record_particle_block!(buf::Vector{T}, cols, tag_host, scratch::Vector{T}) where {T}
    n = length(scratch)
    offset = length(buf)
    resize!(buf, offset + n * length(cols))
    for (d, col) in enumerate(cols)
        copyto!(scratch, col)                    # waits for the queued device work
        base = offset + (d - 1) * n
        if tag_host === nothing
            copyto!(buf, base + 1, scratch, 1, n)
        else
            @inbounds for k in 1:n               # slot k holds particle tag[k]
                buf[base + tag_host[k]] = scratch[k]
            end
        end
    end
    return buf
end

function _write_in_memory!(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer::InMemoryWriter
    st = sim.state::SimulationState
    T = eltype(st.rx)
    n = length(st.rx)
    dim = st.rz === nothing ? 2 : 3
    if writer.n == 0
        writer.n = n
        writer.dim = dim
    elseif (writer.n, writer.dim) != (n, dim)
        throw(ArgumentError("InMemoryWriter recorded $(writer.n) particles in $(writer.dim)D; the state has $n in $(dim)D. Call empty!(writer) first."))
    end
    if !(writer.scratch isa Vector{T}) || length(writer.scratch) != n
        writer.scratch = Vector{T}(undef, n)
    end
    tag_host = st.tag === nothing ? nothing : Array(st.tag)
    buffer(f) = get!(() -> T[], writer.data, f)::Vector{T}
    for f in writer.fields
        if f === :positions
            _record_particle_block!(buffer(f), (st.rx, st.ry, st.rz)[1:dim], tag_host, writer.scratch)
        elseif f === :unwrapped_positions
            st.rx_unwrap === nothing &&
                throw(ArgumentError("InMemoryWriter :unwrapped_positions requires unwrapped_positions=true in the simulation build."))
            _record_particle_block!(buffer(f), (st.rx_unwrap, st.ry_unwrap, st.rz_unwrap)[1:dim], tag_host, writer.scratch)
        elseif f === :velocities
            _brownian_run(sim) &&
                throw(ArgumentError("InMemoryWriter :velocities is not defined for Brownian dynamics; drop the field."))
            _record_particle_block!(buffer(f), (st.vx, st.vy, st.vz)[1:dim], tag_host, writer.scratch)
        elseif f === :forces
            _record_particle_block!(buffer(f), (st.fx, st.fy, st.fz)[1:dim], tag_host, writer.scratch)
        elseif f === :energy
            st.energy_valid || SimulationCore.ensure_energies!(st)
            push!(buffer(f), T(Base.sum(st.Ekin)), T(Base.sum(st.Epot)))
        elseif f === :virial
            v = SimulationCore.virial_tensor(st)
            block = dim == 2 ? T[v.xx, v.xy, v.xy, v.yy] :
                T[v.xx, v.xy, v.xz, v.xy, v.yy, v.yz, v.xz, v.yz, v.zz]
            append!(buffer(f), block)
        end
    end
    push!(writer.steps, step)
    return ctx
end

function _write_writer!(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer
    if writer isa TableWriter
        return _write_table!(sim, ctx, step)
    elseif writer isa InMemoryWriter
        return _write_in_memory!(sim, ctx, step)
    else
        return _write_gsd!(sim, ctx, step)
    end
end

_start_frame(writer::Union{GSDWriter,InMemoryWriter}) = writer.write_start
_start_frame(::Writer) = false
_start_needs_forces(writer::GSDWriter) = writer.write_forces || writer.write_virial
_start_needs_forces(writer::InMemoryWriter) = :forces in writer.fields || :virial in writer.fields
_start_needs_virial(writer::GSDWriter) = writer.write_virial
_start_needs_virial(writer::InMemoryWriter) = :virial in writer.fields

function write_initial_frames!(sim)
    st = sim.state
    st === nothing && return sim
    for ctx in _writer_contexts(sim)
        writer = ctx.writer
        if _start_frame(writer) && !ctx.start_written
            if _start_needs_forces(writer)
                spring = SimulationCore._freeze_active!(st) && st.freeze_mode == SimulationCore.FREEZE_SPRING
                SimulationCore.evaluate_current_configuration!(st, _start_needs_virial(writer); freeze_spring=spring)
            end
            _write_writer!(sim, ctx, st.step)
            ctx.start_written = true
        end
    end
    return sim
end

function write_scheduled_outputs!(sim, step::Int)
    wrote_any = false
    interval_consumed = false
    for ctx in _writer_contexts(sim)
        schedule_matches(ctx.schedule, step) || continue
        _write_writer!(sim, ctx, step)
        wrote_any = true
        interval_consumed |= ctx.needs_interval_reset
    end
    return (wrote_any=wrote_any, interval_consumed=interval_consumed)
end

function active_writer_requires_energy(sim, step::Int)
    for ctx in _writer_contexts(sim)
        schedule_matches(ctx.schedule, step) || continue
        ctx.needs_energy && return true
    end
    return false
end
