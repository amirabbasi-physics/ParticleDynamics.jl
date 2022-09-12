using DelimitedFiles

export write_xyz
export write_log

function write_xyz(
    ofname::String,
    Npart::Int,
    c2::T,
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
            println(file,Npart)
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


function write_log(
    ofname::String,
    step::Int,
    c2::T,
    α::Vector{T},
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T}) where T

    out_file = ofname*".log"
    sdot = sum(sort(dQ ./ α, by = abs))
    sdotpp = sdot/length(α)
    Ekin = sum(sort(Eₖ,by=abs))

    data = hcat(step, Ekin, sdot, sdotpp)
    if step == 0
       open(out_file,"w") do file
            println(file,"Time  E_kin  E_pot EPR")
            #writedlm(file,data)
        end
    else
        open(out_file,"a+") do file
            println("Time = ",data[1], " | E_kin = ", data[2], " | EPR = " ,data[3], " | EPR per particle = " ,data[4] )
            writedlm(file,data, '\t')
        end
    end
end

function write_log(
    ofname::String,
    step::Int,
    c2::T,
    α::Vector{T},
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T},
    coll::Array{T}) where T

    out_file = ofname*".log"
    sdot = sum(sort(dQ ./ α, by = abs))
    sdotpp = sdot/length(α)
    Ekin = sum(sort(Eₖ,by=abs))

    data = hcat(step, Ekin, sdot, sdotpp)
    if step == 0
       open(out_file,"w") do file
            println(file,"Time  E_kin  E_pot EPR")
            #writedlm(file,data)
        end
    else
        open(out_file,"a+") do file
            println("Time = ",data[1], " | E_kin = ", data[2], " | EPR = " ,data[3], " | EPR per particle = " ,data[4] )
            writedlm(file,data, '\t')
        end
    end
end
