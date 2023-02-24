using DelimitedFiles
using PyCall
using FileIO
export write_xyz
export write_log
export write_gsd

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

# Append snapshot to GSD file
function write_gsd(step::Int,simulation, part_id::Vector{Int},positions::Vector{SVector{N,T}}, velocities::Vector{SVector{N,T}}) where {N,T}

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
    s.particles.typeid = part_id
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
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T}) where T

    output_file = simulation.output_file*".log"

    α_list = [simulation.particles[i].α for i = 1:length(simulation.particles)]
    
    sdot = sum(dQ ./ α_list )
    sdotpp = sdot/length(simulation.particles)
    Ekin = sum(Eₖ)
    Epot = sum(Eₚ)
    data = hcat(step, Ekin, Epot, sdot, sdotpp)
    if step == 0
       open(output_file,"w") do file
            println(file,"Time          E_kin           E_pot           EPR             EPR per particle")
            #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",data[1], " | E_kin = ", data[2], " | E_pot = ", data[3], " | EPR = " ,data[4], " | EPR per particle = " ,data[5] )
            writedlm(file,data, '\t')
        end
    end
end


function write_log(
    step::Int,
    simulation,
    num_pl::Int,
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T},
    coll::Array{T}) where T

    cold_cold = coll[1:num_pl , 1:num_pl]
    cold_hot = coll[(num_pl+1):end , 1:num_pl]
    hot_hot = coll[(num_pl+1):end , num_pl+1:end]

    coll_cold_cold  = T(0.5*sum(cold_cold)./c2)
    coll_cold_hot   = T(sum(cold_hot)./c2)
    coll_hot_hot    = T(0.5*sum(hot_hot)./c2)

    α_list = [simulation.particles[i].α for i = 1:length(simulation.particles)]

    output_file = simulation.output_file*".log"
    sdot = sum(dQ ./ α_list)
    sdotpp = sdot/length(simulation.particles)
    Ekin = sum(Eₖ)
    Epot = sum(Eₚ)
    println("simulation.dump_freq not considered!")
    data = hcat(step, Ekin, Epot, sdot, sdotpp, coll_cold_cold, coll_cold_hot, coll_hot_hot)
    if step == 0
       open(output_file,"w") do file
            println(file,"Time      |       E_kin       |       E_pot      |    EPR per part    |   cold/cold coll rate    |   cold/hot coll rate     |   hot/hot coll rate  ")
            #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",data[1], " | E_kin = ", data[2]," | E_pot = ", data[3], " | EPR = " ,data[4], " | EPR per particle = " ,data[5], "  |  cold/cold coll rate = " ,data[6],"  |  cold/hot coll rate = " ,data[7],"  |  hot/hot coll rate = " ,data[8])
            #println(coll_tot - coll_hot_hot -coll_cold_hot - coll_cold_cold)
            writedlm(file,data, '\t')
        end
    end
end

