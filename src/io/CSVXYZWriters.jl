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

"""
    write_observables_csv!(path, st, spec)

Write observables collected through `SimulationCore.collect_step_observables`,
including a compatibility `Qtot` column.
"""
function write_observables_csv!(path::AbstractString,
                                st,
                                spec)
    obs = SimulationCore.collect_step_observables(st, spec)
    hdr = !isfile(path)
    headers = String[
        "step", "integrator", "Etot", "Epot_total", "Ekin_total",
        "virial_total", "dq_total", "dU_total", "Qtot",
    ]
    values = String[
        string(obs.step),
        String(obs.integrator),
        @sprintf("%.7e", obs.Etot),
        @sprintf("%.7e", obs.Epot_total),
        @sprintf("%.7e", obs.Ekin_total),
        @sprintf("%.7e", obs.virial_total),
        @sprintf("%.7e", obs.dq_total),
        @sprintf("%.7e", obs.dU_total),
        @sprintf("%.7e", obs.Qtot),
    ]

    if hasproperty(obs, :extended_hamiltonian)
        push!(headers, "extended_hamiltonian")
        push!(values, @sprintf("%.7e", obs.extended_hamiltonian))
    end
    if hasproperty(obs, :thermostat_kinetic)
        push!(headers, "thermostat_kinetic")
        push!(values, @sprintf("%.7e", obs.thermostat_kinetic))
    end
    if hasproperty(obs, :thermostat_potential)
        push!(headers, "thermostat_potential")
        push!(values, @sprintf("%.7e", obs.thermostat_potential))
    end
    if hasproperty(obs, :thermostat_temperature_error)
        push!(headers, "thermostat_temperature_error")
        push!(values, @sprintf("%.7e", obs.thermostat_temperature_error))
    end
    if hasproperty(obs, :thermostat_energy)
        push!(headers, "thermostat_energy")
        push!(values, @sprintf("%.7e", obs.thermostat_energy))
    end
    if hasproperty(obs, :bath_heat_total)
        push!(headers, "bath_heat_total")
        push!(values, @sprintf("%.7e", obs.bath_heat_total))
    end
    if hasproperty(obs, :bath_entropy_total)
        push!(headers, "bath_entropy_total")
        push!(values, @sprintf("%.7e", obs.bath_entropy_total))
    end
    if hasproperty(obs, :target_temperature)
        push!(headers, "target_temperature")
        push!(values, @sprintf("%.7e", obs.target_temperature))
    end
    if hasproperty(obs, :thermostat_timescale)
        push!(headers, "thermostat_timescale")
        push!(values, @sprintf("%.7e", obs.thermostat_timescale))
    end
    if hasproperty(obs, :chain_length)
        push!(headers, "chain_length")
        push!(values, string(obs.chain_length))
    end
    if hasproperty(obs, :chain_substeps)
        push!(headers, "chain_substeps")
        push!(values, string(obs.chain_substeps))
    end
    if hasproperty(obs, :nhc_propagator)
        push!(headers, "nhc_propagator")
        push!(values, string(obs.nhc_propagator))
    end

    open(path, "a") do io
        if hdr
            println(io, join(headers, ","))
        end
        println(io, join(values, ","))
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
    write_xyz!(path; rx, ry[, rz], atomsym="A")

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
        @printf(io, "Generated by ParticleDynamics SoA\n")
        @inbounds for i in 1:N
            @printf(io, "%s %.7e %.7e %.7e\n", atomsym, X[i], Y[i], Z[i])
        end
    end
    return nothing
end
