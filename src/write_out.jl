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

    c1 = [sqrt(simulation.particles[i].τD/simulation.particles[i].τm) for i=1:length(simulation.particles)]
    α_list = [simulation.particles[i].α for i = 1:length(simulation.particles)]

    output_file = simulation.output_file*".log"
    sdot = sum(dQ ./ α_list)
    sdotpp = sdot/length(simulation.particles)
    Ekin = sum(Eₖ)
    Epot = sum(Eₚ)

    

    sdotpp_ave = sum(c1 .* ( 2 .* Eₖ ./ α_list .- length(simulation.box)))/length(simulation.particles)


    step_str = @sprintf("%+.5e", step)
    Ekin_str = @sprintf("%+.5e", Ekin)
    Epot_str = @sprintf("%+.5e", Epot)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)
    sdotpp_ave_str = @sprintf("%+.5e", sdotpp_ave)
    data = join([step_str, Ekin_str, Epot_str, sdot_str, sdotpp_str, sdotpp_ave_str], "\t")

    if step == 0
       open(output_file,"w") do file
        println(file,"     Time     |     E_kin     |     E_pot     |     EPR    | EPR per part | EPR per part ave ")
        #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str, " | E_kin = ", Ekin_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str, "  |  EPR per particle averaged = " ,sdotpp_ave_str)
            println(file,data)
        end
    end
end



function write_log(
    step::Int,
    simulation,
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T},
    colls::Vector{T}) where T

    coll_cold_hot   = sum(colls) / simulation.dt

    α_list = [simulation.particles[i].α for i = 1:length(simulation.particles)]

    output_file = simulation.output_file*".log"
    sdot = sum(dQ ./ α_list)
    sdotpp = sdot/(simulation.save_interval*length(simulation.particles))
    Ekin = sum(Eₖ)./simulation.save_interval
    Epot = sum(Eₚ)./simulation.save_interval

    c1 = [sqrt(simulation.particles[i].τD/simulation.particles[i].τm) for i=1:length(simulation.particles)]

    sdotpp_ave = sum(2 .* c1 .* ( Eₖ ./ α_list .- 1))/length(simulation.particles)


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




"""


function write_log(
    step::Int,
    simulation,
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T}) where T

    c1 = [sqrt(simulation.particles[i].τD/simulation.particles[i].τm) for i=1:length(simulation.particles)]
    α_list = [simulation.particles[i].α for i = 1:length(simulation.particles)]

    output_file = simulation.output_file*".log"
    sdot = sum(dQ ./ α_list)
    sdotpp = sdot/length(simulation.particles)
    Ekin = sum(Eₖ)
    Epot = sum(Eₚ)

    
    c1_64 = Float64.(c1)
    Eₖ_64 = Float64.(Eₖ)
    α_list_64 = Float64.(α_list)

    sdotpp_ave = sum(sort(c1_64 .* ( 2 .* Eₖ_64 ./ α_list_64 .- length(simulation.box))))/length(simulation.particles)


    step_str = @sprintf("%+.5e", step)
    Ekin_str = @sprintf("%+.5e", Ekin)
    Epot_str = @sprintf("%+.5e", Epot)
    sdot_str = @sprintf("%+.5e", sdot)
    sdotpp_str = @sprintf("%+.5e", sdotpp)
    sdotpp_ave_str = @sprintf("%+.5e", sdotpp_ave)
    data = join([step_str, Ekin_str, Epot_str, sdot_str, sdotpp_str, sdotpp_ave_str], "\t")

    if step == 0
       open(output_file,"w") do file
        # Print simulation details
        println(file, "Simulation Details:")
        println(file, "Descriptor: ", simulation.descriptor)
        println(file, "Box: ", simulation.box)
        println(file, "Part types: ", simulation.part_types)
        println(file, "ϵ: ", simulation.ϵ)
        println(file, "σ: ", simulation.σ)
        println(file, "Neighbour cut-off: ", simulation.neigh_cut_off)
        println(file, "Neighbour update: ", simulation.neigh_update)
        println(file, "Number of cold particles: ", simulation.num_cold)
        println(file, "dt: ", simulation.dt)
        println(file, "Integrator: ", simulation.integrator)
        println(file, "Number of steps: ", simulation.num_steps)
        println(file, "Save interval: ", simulation.save_interval)
        println(file,"     Time     |     E_kin     |     E_pot     |     EPR    | EPR per part | EPR per part ave ")
        #writedlm(file,data)
        end
    else
        open(output_file,"a+") do file
            println("Time = ",step_str, " | E_kin = ", Ekin_str," | E_pot = ", Epot_str, " | EPR = " ,sdot_str, " | EPR per particle = " ,sdotpp_str, "  |  EPR per particle averaged = " ,sdotpp_ave_str)
            println(file,data)
        end
    end
end



function write_log(
    step::Int,
    simulation,
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T},
    colls::Vector{T}) where T

    coll_cold_hot   = sum(colls) / simulation.dt

    α_list = [simulation.particles[i].α for i = 1:length(simulation.particles)]

    output_file = simulation.output_file*".log"
    sdot = sum(dQ ./ α_list)
    sdotpp = sdot/length(simulation.particles)
    Ekin = sum(Eₖ)
    Epot = sum(Eₚ)

    c1 = [sqrt(simulation.particles[i].τD/simulation.particles[i].τm) for i=1:length(simulation.particles)]

    c1_64 = Float64.(c1)
    Eₖ_64 = Float64.(Eₖ)
    α_list_64 = Float64.(α_list)

    
    ssdotpp_ave = sum(sort(c1_64 .* ( 2 .* Eₖ_64 ./ α_list_64 .- length(simulation.box))))/length(simulation.particles)


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
                # Print simulation details
        println(file, "Simulation Details:")
        println(file, "Descriptor: ", simulation.descriptor)
        println(file, "Box: ", simulation.box)
        println(file, "Part types: ", simulation.part_types)
        println(file, "ϵ: ", simulation.ϵ)
        println(file, "σ: ", simulation.σ)
        println(file, "Neighbour cut-off: ", simulation.neigh_cut_off)
        println(file, "Neighbour update: ", simulation.neigh_update)
        println(file, "Number of cold particles: ", simulation.num_cold)
        println(file, "dt: ", simulation.dt)
        println(file, "Integrator: ", simulation.integrator)
        println(file, "Number of steps: ", simulation.num_steps)
        println(file, "Save interval: ", simulation.save_interval)
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

"""