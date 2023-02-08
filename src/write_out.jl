using DelimitedFiles

export write_xyz
export write_log

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
    num_pl::Int,
    Eₖ::Vector{T},
    Eₚ::Vector{T},
    dQ::Vector{T},
    coll::Array{T}) where T

    cold_cold = coll[1:num_pl , 1:num_pl]
    cold_hot = coll[(num_pl+1):end , 1:num_pl]

    hot_hot = coll[(num_pl+1):end , num_pl+1:end]


    coll_cold_cold  = 0.5*sum(cold_cold)./c2
    coll_cold_hot   = sum(cold_hot)./c2
    coll_hot_hot    = 0.5*sum(hot_hot)./c2
    coll_tot        = 0.5*sum(coll)./c2


    out_file = ofname*".log"
    sdot = sum(sort(dQ ./ α, by = abs))
    sdotpp = sdot/length(α)
    Ekin = sum(sort(Eₖ,by=abs))

    data = hcat(step, Ekin, sdot, sdotpp, Float32(coll_cold_cold), Float32(coll_cold_hot), Float32(coll_hot_hot), Float32(coll_tot))
    if step == 0
       open(out_file,"w") do file
            println(file,"Time      |   E_kin  |    EPR  |    EPR per part    |   cold/cold coll rate    |   cold/hot coll rate     |   hot/hot coll rate  |   total coll rate")
            #writedlm(file,data)
        end
    else
        open(out_file,"a+") do file
            println("Time = ",data[1], " | E_kin = ", data[2], " | EPR = " ,data[3], " | EPR per particle = " ,data[4], "  |  cold/cold coll rate = " ,data[5],"  |  cold/hot coll rate = " ,data[6],"  |  hot/hot coll rate = " ,data[7], "  |  total coll rate = " ,data[8])
            #println(coll_tot - coll_hot_hot -coll_cold_hot - coll_cold_cold)
            writedlm(file,data, '\t')
        end
    end
end

