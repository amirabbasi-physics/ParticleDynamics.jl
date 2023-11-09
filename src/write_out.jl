using DelimitedFiles
using PyCall
using FileIO
export write_xyz
export write_log
export write_gsd


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

# Arguments
- `ofname::String`: Base name of the output file to which the XYZ data will be written.
- `Npart::Int`: Number of particles in the simulation.
- `alpha_lst::Vector{T}`: Vector of alpha values associated with each particle.
- `σ::T`: A common value related to particle size or interaction strength (e.g., particle diameter).
- `L::T`: The length of the simulation box (assumed to be cubic or square).
- `step::Int`: The current time step of the simulation.
- `dim::Int`: Dimensionality of the simulation (2 or 3 dimensions).
- `part_type::Vector{String}`: Vector of strings indicating the type of each particle.
- `r::Vector{SVector{N,T}}`: Vector of `StaticVector`s representing the positions of each particle.
- `v::Vector{SVector{N,T}}`: Vector of `StaticVector`s representing the velocities of each particle.
- `dQ::Vector{T}`: Vector of entropy production or other scalar quantities associated with each particle.

# Description
This function writes the current state of a particle-based simulation to an XYZ file, a common format for representing atomistic simulations. The output includes particle type, radius (σ/2), position, velocity, and a scalar value representing entropy or another property (calculated as `dQ ./ alpha_lst`).

The function decides whether to create a new file or append to an existing one based on the step argument. For the first step (`step == 0`), it creates a new file, while for subsequent steps, it appends to the file.

The XYZ file starts with the number of particles followed by a comment line specifying the lattice (simulation box size) and the properties associated with each particle. The properties include:
- Particle Type (string)
- Radius (real)
- Position (2D or 3D vector)
- Velocity (2D or 3D vector)
- Entropy or other scalar property (real)

The lattice dimensions and particle properties are formatted according to the specified dimensionality of the simulation.

# Example
```julia
# Define the parameters for the simulation snapshot
ofname = "simulation"
Npart = 100
alpha_lst = fill(1.0, Npart)
σ = 1.0
L = 10.0
step = 0
dim = 3
part_type = fill("A", Npart)
r = [SVector(rand(), rand(), rand()) for _ = 1:Npart]
v = [SVector(rand(), rand(), rand()) for _ = 1:Npart]
dQ = rand(Npart)

# Write the XYZ file
write_xyz(ofname, Npart, alpha_lst, σ, L, step, dim, part_type, r, v, dQ)
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
#ENV["PYTHON"]="/local_scratch/abbaa90/miniconda3/envs/hoomd3-venv/bin/python"
#Pkg.build("PyCall")

"""
    write_gsd(step::Int, simulation, part_ids::Vector{Int}, positions::Vector{SVector{N,T}}, velocities::Vector{SVector{N,T}}) where {N,T}

Write the state of a simulation to a .gsd file for visualization or restart in HOOMD-blue format.

# Arguments
- `step::Int`: The current time step of the simulation to be recorded in the file.
- `simulation`: A structure containing the current state and configuration of the simulation.
- `part_ids::Vector{Int}`: A vector containing the unique identifiers for the particles.
- `positions::Vector{SVector{N,T}}`: A vector of `StaticVector`s representing the positions of each particle in the simulation.
- `velocities::Vector{SVector{N,T}}`: A vector of `StaticVector`s representing the velocities of each particle in the simulation.

# Description
This function writes the current state of a simulation to a file in the GSD format, which is used by the HOOMD-blue simulation package for storing particle system configurations. The function determines whether to append to an existing file or create a new one based on whether the file exists and the current step is not zero.

The function handles both 2D and 3D simulations. For 2D simulations, a zero z-component is added to positions and velocities to ensure compatibility with GSD's 3D format.

# How it works
1. Import the `gsd.hoomd` module using PyCall.
2. Determine the file name and mode (write or append).
3. Open the GSD file with the appropriate mode.
4. Adjust positions and velocities for 2D simulations by adding a zero z-component.
5. Create a snapshot of the current simulation state.
6. Set the simulation step, number of particles, particle types, particle type identifiers, positions, and velocities in the snapshot.
7. Adjust the box dimensions for the snapshot to include zero values for dimensions not present in the simulation.
8. Append the snapshot to the GSD file.
9. Close the file.

# Example
```julia
# Assuming `simulation` is a predefined simulation object with appropriate properties
write_gsd(100, simulation, [1, 2, 3], [SVector(0.0, 1.0), SVector(1.0, 0.0)], [SVector(0.1, 0.2), SVector(0.2, 0.1)])
"""

function write_gsd(step::Int,simulation, part_ids::Vector{Int},positions::Vector{SVector{N,T}}, velocities::Vector{SVector{N,T}}) where {N,T}
    gsdhoomd = pyimport("gsd.hoomd")
    output_file = simulation.output_file*".gsd"

    if isfile(output_file) && step !=0
        mode = "rb+"
    else 
        mode = "wb"
    end
    f = gsdhoomd.open(name = output_file, mode=mode)
    
    box = simulation.box
    
    if length(box) == 2
        positions = [vcat(positions[i],zero(T)) for i = 1:length(positions)]
        velocities = [vcat(velocities[i],zero(T)) for i = 1:length(velocities)]       
    end

    s = gsdhoomd.Snapshot()
   
    s.configuration.step = step
    s.particles.N = length(positions)
    s.particles.types = simulation.part_types
    s.particles.typeid = part_ids
    s.particles.position = positions
    s.particles.velocity = velocities
    if length(box) == 2
        s.configuration.box = vcat(box, zeros(eltype(box), 4))
    elseif length(box) == 3
        s.configuration.box = vcat(box, zeros(eltype(box), 3))
    end


    f.append(s)
    f.close()
end


function write_log(
    step::Int,
    simulation,
    Eₖ::Float64,
    Eₚ::Float64,
    sdot::Float64,
    Eₖ_α::Float64,
    colls::Float64) 

    coll_cold_hot   = colls / simulation.dt

    output_file = simulation.output_file*".log"

    sdotpp = sdot/length(simulation.particles)
    Ekin = Eₖ
    Epot = Eₚ

    sdotpp_ave = Eₖ_α/length(simulation.particles)


    step_str = @sprintf("%+.5e", step)
    Ekin_str = @sprintf("%+.5e", Ekin)
    Epot_str = @sprintf("%+.5e", Epot)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)
    sdotpp_ave_str = @sprintf("%+.5e", sdotpp_ave)
    coll_cold_hot_str = @sprintf("%+.5e", coll_cold_hot)
    data = join([step_str, Ekin_str, Epot_str, sdot_str, sdotpp_str, sdotpp_ave_str, coll_cold_hot_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"     Time     |     E_kin     |     E_pot     |      EPR      |  EPR / part  | EPR / part Ave | cold/hot coll rate ")
        #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str, " | E_kin = ", Ekin_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str, "  |  EPR per particle averaged = " ,sdotpp_ave_str,"  |  cold/hot coll rate = " ,coll_cold_hot_str)
            #println(coll_tot - coll_hot_hot -coll_cold_hot - coll_cold_cold)
            println(file,data)
        end
    end
end

function write_log(
    step::Int,
    simulation,
    Eₖ::Float64,
    Eₚ::Float64,
    sdot::Float64,
    udot::Float64,
    Eₖ_α::Float64,
    colls::Float64) 

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
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)

    udot_str = @sprintf("%+.5e", udot)
    udotpp_str = @sprintf("%+.5e", udotpp)

    sdotpp_ave_str = @sprintf("%+.5e", sdotpp_ave)
    coll_cold_hot_str = @sprintf("%+.5e", coll_cold_hot)
    data = join([step_str, Ekin_str, Epot_str, sdot_str, udot_str, sdotpp_str, udotpp_str, sdotpp_ave_str, coll_cold_hot_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"     Time     |     E_kin     |     E_pot     |      EPR      |      UPR      |  EPR / part  |   UPR / part  | EPR / part Ave | cold/hot coll rate ")
        #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str, " | E_kin = ", Ekin_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str," | UPR = " ,udot_str, " | EPR per particle = " , sdotpp_str," | UPR per particle = " ,udotpp_str, "  |  EPR per particle averaged = " ,sdotpp_ave_str,"  |  cold/hot coll rate = " ,coll_cold_hot_str)
            #println(coll_tot - coll_hot_hot -coll_cold_hot - coll_cold_cold)
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
        # writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str, "  |  cold/hot coll rate = " ,coll_cold_hot_str)
            # println(coll_tot - coll_hot_hot -coll_cold_hot - coll_cold_cold)
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
        #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str)
            #println(coll_tot - coll_hot_hot -coll_cold_hot - coll_cold_cold)
            println(file,data)
        end
    end
end
