using DelimitedFiles
using Printf
using GSDFiles
using FileIO
using StaticArrays

# Export all writer functions and structs
export InMemoryLogger, CSVWriter, XYZWriter, ObservableCSVWriter
export write_xyz, write_log, write_gsd, read_last_gsd
export gsd_open, gsd_close, write_gsd_frame!
export write!, finalize

# ---------------------------------------------------------------
# Base Writer interface adapted for NonEqSimGPU
# ---------------------------------------------------------------
abstract type Writer end

# Default write! method - should be overridden by specific writers
function write!(w::Writer, simulation, step::Int, dt::Real)
    error("write! method not implemented for $(typeof(w))")
end

# ---------------------------------------------------------------
# InMemoryLogger: stores data in memory
# ---------------------------------------------------------------
mutable struct InMemoryLogger <: Writer
    every::Int
    data::Dict{String, Vector}
    steps::Vector{Int}
end

function InMemoryLogger(data_keys; every::Int=1)
    data = Dict{String, Vector}()
    for key in data_keys
        data[string(key)] = []
    end
    return InMemoryLogger(every, data, [])
end

function write!(w::InMemoryLogger, simulation, step::Int, dt::Real)
    if step % w.every != 0
        return
    end
    push!(w.steps, step)
    # Users can extend this by overriding and adding specific data
    return nothing
end

# ---------------------------------------------------------------
# CSVWriter: writes particle data to CSV
# ---------------------------------------------------------------
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
            println(w.io, "step,id,x,y,z,vx,vy,vz,type")
            w.wrote_header = true
        end
    end
    return nothing
end

function Base.finalize(w::CSVWriter)
    if w.io !== nothing
        try
            close(w.io)
        catch
        end
        w.io = nothing
    end
end

function write!(w::CSVWriter, simulation, step::Int, dt::Real)
    if step % w.every != 0
        return
    end
    _ensure_csv_open!(w)

    particles = simulation.particles
    N = length(particles)
    
    for i in 1:N
        p = particles[i]
        pos = p.r
        vel = p.v
        
        x = pos[1]
        y = length(pos) >= 2 ? pos[2] : 0.0
        z = length(pos) >= 3 ? pos[3] : 0.0
        
        vx = vel[1]  
        vy = length(vel) >= 2 ? vel[2] : 0.0
        vz = length(vel) >= 3 ? vel[3] : 0.0
        
        part_type = p.part_type
        
        println(w.io, "$step,$i,$x,$y,$z,$vx,$vy,$vz,$part_type")
    end
    return nothing
end

# ---------------------------------------------------------------
# ObservableCSVWriter: writes observables to CSV
# ---------------------------------------------------------------
mutable struct ObservableCSVWriter <: Writer
    path::String
    every::Int
    observables::Vector{String}
    io::Union{Nothing,IO}
    wrote_header::Bool
end

function ObservableCSVWriter(path, observables; every::Int=1)
    p = endswith(path, ".csv") ? path : string(path, ".csv")
    return ObservableCSVWriter(p, every, observables, nothing, false)
end

function _ensure_obs_open!(w::ObservableCSVWriter)
    if w.io === nothing
        mkpath(dirname(w.path))
        w.io = open(w.path, "w")
        header = "step"
        for obs in w.observables
            header *= "," * obs
        end
        println(w.io, header)
        w.wrote_header = true
    end
end

function write!(w::ObservableCSVWriter, simulation, step::Int, dt::Real; data::Dict=Dict())
    if step % w.every != 0
        return
    end
    _ensure_obs_open!(w)
    line = string(step)
    for obs in w.observables
        value = get(data, obs, 0.0)
        line *= "," * string(value)
    end
    println(w.io, line)
end

function Base.finalize(w::ObservableCSVWriter)
    if w.io !== nothing
        close(w.io)
        w.io = nothing
    end
end

# ---------------------------------------------------------------
# XYZWriter: writes XYZ format files
# ---------------------------------------------------------------
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
        try
            close(w.io)
        catch
        end
        w.io = nothing
    end
end

function write!(w::XYZWriter, simulation, step::Int, dt::Real)
    if step % w.every != 0
        return
    end
    _ensure_xyz_open!(w)

    particles = simulation.particles
    N = length(particles)
    println(w.io, N)
    println(w.io, "step=$step")

    for particle in particles
        pos = particle.r
        x = pos[1]
        y = length(pos) >= 2 ? pos[2] : 0.0
        z = length(pos) >= 3 ? pos[3] : 0.0
        element = particle.part_type[1]  # Use first character as element
        println(w.io, "$element $x $y $z")
    end
    return nothing
end

# ---------------------------------------------------------------
# Legacy functions from original IO.jl - adapted for compatibility
# ---------------------------------------------------------------

"""
    write_xyz(
        ofname::String,
        Npart::Int,
        alpha_lst::Vector{T},
        σ::T,
        L::T,
        step::Int,
        dim::Int,
        part_type::Vector{String},
        r::Vector{SVector{N,T}},
        v::Vector{SVector{N,T}},
        dQ::Vector{T}
    ) where {N,T}

Write the state of a simulation to an XYZ file with additional properties.
"""
function write_xyz(
    ofname::String,
    Npart::Int,
    alpha_lst::Vector{T},
    σ::T,
    L::T,
    step::Int,
    dim::Int,
    part_type::Vector{String},
    r::Vector{SVector{N,T}},
    v::Vector{SVector{N,T}},
    dQ::Vector{T}) where {N,T}

    out_file = ofname*".xyz"
    sdot = dQ ./alpha_lst
    snapshot = [vcat(part_type[i],σ/2,r[i],v[i],sdot[i]) for i = 1:Npart]
    if step == 0
       open(out_file,"w") do file
            println(file, Npart)
            if dim == 2
                println(file,"""Lattice="$L $L 0.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:2:Velocity:R:2:Entropy:R:1" """)
                writedlm(file,snapshot)
            elseif dim == 3
                println(file,"""Lattice="$L $L $L 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:3:Velocity:R:3:Entropy:R:1" """)
                writedlm(file,snapshot)
            end
        end
    else
        open(out_file,"a+") do file
            println(file,Npart)
            if dim == 2
                println(file,"""Lattice="$L $L 0.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:2:Velocity:R:2:Entropy:R:1" """)
                writedlm(file,snapshot)
            elseif dim == 3
                println(file,"""Lattice="$L $L $L 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:3:Velocity:R:3:Entropy:R:1" """)
                writedlm(file,snapshot)
            end
        end
    end
end

"""
    write_gsd(step::Int, simulation, part_ids::Vector{Int}, positions::Vector{SVector{N,T}}, velocities::Vector{SVector{N,T}}) where {N,T}

Write the state of a simulation to a .gsd file for visualization or restart in HOOMD-blue format.
"""
function write_gsd(step::Int, simulation, part_ids::Vector{Int}, positions::Vector{SVector{N,T}}, velocities::Vector{SVector{N,T}}) where {N,T}
    output_file = simulation.output_file * ".gsd"
    
    # Convert to 3D if needed
    box = simulation.box
    if length(box) == 2
        positions_3d = [SVector{3,T}(pos[1], pos[2], zero(T)) for pos in positions]
        velocities_3d = [SVector{3,T}(vel[1], vel[2], zero(T)) for vel in velocities]
        box_6 = SVector{6,eltype(box)}(box[1], box[2], 0, 0, 0, 0)
        D = 2
    else
        positions_3d = positions
        velocities_3d = velocities
        box_6 = SVector{6,eltype(box)}(box[1], box[2], box[3], 0, 0, 0)
        D = 3
    end
    
    # Open or create GSD file
    gsd_writer = if step == 0 || !isfile(output_file)
        GSDFiles.GSDWriter(output_file; application="NonEqSimGPU", schema="hoomd", schema_version=(1,4))
    else
        GSDFiles.GSDWriter(output_file; application="NonEqSimGPU", schema="hoomd", schema_version=(1,4))
    end
    
    gsd_handle = GSDFiles.open_gsd(gsd_writer)
    
    try
        # Convert positions and velocities to matrices
        pos_matrix = reduce(hcat, [collect(p) for p in positions_3d])'
        vel_matrix = reduce(hcat, [collect(v) for v in velocities_3d])'
        
        num_particles = length(positions)
        
        # Write frame data
        GSDFiles.write_configuration_step!(gsd_handle, UInt64(step))
        GSDFiles.write_configuration_dimensions!(gsd_handle, UInt8(D))
        GSDFiles.write_configuration_box!(gsd_handle, Float32.(box_6))
        GSDFiles.write_particles_N!(gsd_handle, num_particles)
        GSDFiles.write_particles_types!(gsd_handle, simulation.part_types)
        GSDFiles.write_particles_typeid!(gsd_handle, UInt32.(part_ids))  # Already 0-based
        GSDFiles.write_particles_position!(gsd_handle, Float32.(pos_matrix))
        GSDFiles.write_particles_velocity!(gsd_handle, Float32.(vel_matrix))
        
        GSDFiles.end_frame!(gsd_handle)
    finally
        GSDFiles.close_gsd(gsd_handle)
    end
end

function read_last_gsd(file_path::String, dim::D) where D
    # Open GSD file for reading
    gsd_handle = GSDFiles.GSDReader(file_path)
    GSDFiles.open_gsd(gsd_handle)
    
    try
        # Get the number of frames
        num_frames = GSDFiles.num_frames(gsd_handle)
        if num_frames == 0
            error("No frames found in GSD file")
        end
        
        # Read the last frame (0-indexed)
        frame_idx = num_frames - 1
        
        # Read frame data
        step = Int(GSDFiles.read_configuration_step(gsd_handle, frame_idx))
        N_particles = GSDFiles.read_particles_N(gsd_handle, frame_idx)
        
        # Read positions and velocities
        pos_matrix = GSDFiles.read_particles_position(gsd_handle, frame_idx)
        vel_matrix = GSDFiles.read_particles_velocity(gsd_handle, frame_idx)
        
        # Convert to SVector format, taking only the required dimensions
        positions = [SVector{dim}(pos_matrix[i, 1:dim]...) for i = 1:N_particles]
        velocities = [SVector{dim}(vel_matrix[i, 1:dim]...) for i = 1:N_particles]
        
        # Read type information
        part_ids = GSDFiles.read_particles_typeid(gsd_handle, frame_idx) .+ 1  # Convert to 1-based
        part_types = GSDFiles.read_particles_types(gsd_handle, frame_idx)
        
        # Read box
        box_6 = GSDFiles.read_configuration_box(gsd_handle, frame_idx)
        if dim == 2
            box = SVector{2}(box_6[1], box_6[2])
        else
            box = SVector{3}(box_6[1], box_6[2], box_6[3])
        end
        
        # Count cold particles (assuming type id 0 corresponds to cold particles)
        num_cold = count(t == 1 for t in part_ids)  # 1-based indexing
        
        return step, positions, velocities, Vector{Int}(part_ids), part_types, box, num_cold
        
    finally
        GSDFiles.close_gsd(gsd_handle)
    end
end

"""
    write_log(step::Int, simulation, ...)

Write simulation log data. Multiple method signatures for different data types.
"""
function write_log(
    step::Int,
    simulation,
    Eₖ::Float64,
    Eₚ::Float64,
    sdot::Float64,
    virial::Float64,
    udot::Float64,
    Eₖ_α::Float64,
    colls::Vector{Float64}) 

    coll_cold_cold, coll_hot_cold, coll_hot_hot = colls[1], colls[2], colls[3]

    output_file = simulation.output_file*".log"

    sdotpp = sdot/length(simulation.particles)
    udotpp = udot/length(simulation.particles)
    Ekin = Eₖ
    Epot = Eₚ

    sdotpp_ave = Eₖ_α/length(simulation.particles)

    step_str = @sprintf("%+.5e", step)
    Ekin_str = @sprintf("%+.5e", Ekin)
    Epot_str = @sprintf("%+.5e", Epot)
    Etot_str = @sprintf("%+.5e", Ekin + Epot)
    virial_str = @sprintf("%+.5e", virial)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)

    udot_str = @sprintf("%+.5e", udot)
    udotpp_str = @sprintf("%+.5e", udotpp)

    sdotpp_ave_str = @sprintf("%+.5e", sdotpp_ave)
    coll_cold_cold_str = @sprintf("%+.5e", coll_cold_cold)
    coll_hot_cold_str = @sprintf("%+.5e", coll_hot_cold)
    coll_hot_hot_str = @sprintf("%+.5e", coll_hot_hot)
    data = join([step_str, Ekin_str, Epot_str, Etot_str, virial_str,  sdot_str, udot_str, sdotpp_str, udotpp_str, sdotpp_ave_str, coll_cold_cold_str, coll_hot_cold_str, coll_hot_hot_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"    Time       |      E_kin     |       E_pot     |      E_tot        |    virial      |      EPR       |       UPR       |    EPR / part   |    UPR / part    | EPR / part Ave  | cold/cold coll  |  hot/cold coll  |  hot/hot coll  ")
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str, "| E_kin = ", Ekin_str,"| E_pot = ", Epot_str,"| E_tot = ", Etot_str, "| virial = ", virial_str, " | EPR = " ,sdot_str," | UPR = " ,udot_str, " | EPR per particle = " , sdotpp_str," | UPR per particle = " ,udotpp_str, "  |  EPR per particle averaged = " ,sdotpp_ave_str,"  |  cold/cold coll rate = " ,coll_cold_cold_str, "  |  hot/cold coll rate = " ,coll_hot_cold_str, "  |  hot/hot coll rate = " ,coll_hot_hot_str)
            println(file,data)
        end
    end
end

# Additional overloaded write_log methods for different parameter combinations
function write_log(
    step::Int,
    simulation,
    Eₖ::Float64,
    Eₚ::Float64,
    sdot::Float64,
    udot::Float64,
    Eₖ_α::Float64,
    colls::Array{3,Float64})  

    coll_cold_hot   = colls / simulation.dt
    output_file = simulation.output_file*".log"

    sdotpp = sdot/length(simulation.particles)
    udotpp = udot/length(simulation.particles)
    Ekin = Eₖ
    Epot = Eₚ

    sdotpp_ave = Eₖ_α/length(simulation.particles)

    step_str = @sprintf("%+.5e", step)
    Ekin_str = @sprintf("%+.5e", Ekin)
    Epot_str = @sprintf("%+.5e", Epot)
    Etot_str = @sprintf("%+.5e", Ekin + Epot)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)

    udot_str = @sprintf("%+.5e", udot)
    udotpp_str = @sprintf("%+.5e", udotpp)

    sdotpp_ave_str = @sprintf("%+.5e", sdotpp_ave)
    coll_cold_hot_str = @sprintf("%+.5e", coll_cold_hot)
    data = join([step_str, Ekin_str, Epot_str, Etot_str, sdot_str, udot_str, sdotpp_str, udotpp_str, sdotpp_ave_str, coll_cold_hot_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"     Time     |     E_kin     |     E_pot     |     E_tot     |      EPR      |      UPR      |  EPR / part  |   UPR / part  | EPR / part Ave | cold/hot coll rate ")
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str, " | E_kin = ", Ekin_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str," | UPR = " ,udot_str, " | EPR per particle = " , sdotpp_str," | UPR per particle = " ,udotpp_str, "  |  EPR per particle averaged = " ,sdotpp_ave_str,"  |  cold/hot coll rate = " ,coll_cold_hot_str)
            println(file,data)
        end
    end
end

function write_log(
    step::Int,
    simulation,
    Eₚ::Float64,
    sdot::Float64,
    colls::Float64) 

    coll_cold_hot   = colls / simulation.dt
    output_file = simulation.output_file*".log"

    sdotpp = sdot/length(simulation.particles)
    Epot = Eₚ

    step_str = @sprintf("%+.5e", step)
    Epot_str = @sprintf("%+.5e", Epot)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)
    coll_cold_hot_str = @sprintf("%+.5e", coll_cold_hot)
    data = join([step_str, Epot_str, sdot_str, sdotpp_str, coll_cold_hot_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"     Time     |     E_pot     |      EPR      |  EPR / part  | cold/hot coll rate ")
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str, "  |  cold/hot coll rate = " ,coll_cold_hot_str)
            println(file,data)
        end
    end
end

function write_log(
    step::Int,
    simulation,
    Eₚ::Float64,
    sdot::Float64) 

    output_file = simulation.output_file*".log"

    sdotpp = sdot/length(simulation.particles)
    Epot = Eₚ

    step_str = @sprintf("%+.5e", step)
    Epot_str = @sprintf("%+.5e", Epot)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)
    data = join([step_str, Epot_str, sdot_str, sdotpp_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"     Time     |     E_pot     |      EPR      |  EPR / part  ")
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str)
            println(file,data)
        end
    end
end

# ---------------------------------------------------------------
# GSD convenience functions using GSDFiles.jl
# ---------------------------------------------------------------

function gsd_open(path::AbstractString; application="NonEqSimGPU", schema="hoomd", schema_version=(1,4))
    w = GSDFiles.GSDWriter(path; application, schema, schema_version)
    GSDFiles.open_gsd(w)
    return w
end

gsd_close(h) = GSDFiles.close_gsd(h)

function write_gsd_frame!(h, simulation; diameter::Real=1.0, step::Int=0)
    particles = simulation.particles
    N = length(particles)
    
    # Extract positions and velocities
    positions = [particle.r for particle in particles]
    velocities = [particle.v for particle in particles]
    
    # Determine dimensionality and ensure 3D for GSD
    if length(positions[1]) == 2
        positions_3d = [SVector{3}(pos[1], pos[2], 0.0) for pos in positions]
        velocities_3d = [SVector{3}(vel[1], vel[2], 0.0) for vel in velocities]
        D = 2
    else
        positions_3d = positions
        velocities_3d = velocities
        D = 3
    end
    
    # Convert to matrices
    pos_matrix = reduce(hcat, [collect(p) for p in positions_3d])'
    vel_matrix = reduce(hcat, [collect(v) for v in velocities_3d])'
    
    # Extract type information
    part_types = unique([p.part_type for p in particles])
    type_dict = Dict(type => i-1 for (i, type) in enumerate(part_types))  # 0-based
    typeids = UInt32.([type_dict[p.part_type] for p in particles])
    
    # Set box
    box = simulation.box
    if D == 2
        box_6 = SVector{6}(box[1], box[2], 0.0, 0.0, 0.0, 0.0)
    else
        box_6 = SVector{6}(box[1], box[2], box[3], 0.0, 0.0, 0.0)
    end
    
    # Write frame data
    GSDFiles.write_configuration_step!(h, UInt64(step))
    GSDFiles.write_configuration_dimensions!(h, UInt8(D))
    GSDFiles.write_configuration_box!(h, Float32.(box_6))
    GSDFiles.write_particles_N!(h, N)
    GSDFiles.write_particles_types!(h, part_types)
    GSDFiles.write_particles_typeid!(h, typeids)
    GSDFiles.write_particles_diameter!(h, fill(Float32(diameter), N))
    GSDFiles.write_particles_position!(h, Float32.(pos_matrix))
    GSDFiles.write_particles_velocity!(h, Float32.(vel_matrix))
    
    GSDFiles.end_frame!(h)
    return h
end
