using DelimitedFiles

export write_xyz
export write_log

function write_xyz(ofname::String, Npart::Int, σ::T, L::T, step::Int, dim::Int, part_type::Vector{String},r::Vector{SVector{N,T}},
    v::Vector{SVector{N,T}}, f::Vector{SVector{N,T}}) where {N,T}
    out_file = ofname*".xyz"
    snapshot = [vcat(part_type[i],σ/2,r[i],v[i],f[i]) for i = 1:Npart]
    if step == 0
       open(out_file,"w") do file
            println(file,Npart)
            if dim == 2
                println(file,"""Lattice="$L $L 1.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:2:Velocity:R:2:Force:R:2:Entropy:R:1" """)
                writedlm(file,snapshot)
            elseif dim == 3
                println(file,"""Lattice="$L $L $L 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:3:Velocity:R:3:Force:R:3:Entropy:R:1" """)
                writedlm(file,snapshot)
            end
        end
    else
        open(out_file,"a+") do file
            println(file,Npart)
            if dim == 2
                println(file,"""Lattice="$L $L 1.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:2:Velocity:R:2:Force:R:2:Entropy:R:1" """)
                writedlm(file,snapshot)
            elseif dim == 3
                println(file,"""Lattice="$L $L $L 0.0 0.0 0.0 0.0 0.0 0.0" Properties="Particle Type:S:1:Radius:R:1:Position:R:3:Velocity:R:3:Force:R:3:Entropy:R:1" """)
                writedlm(file,snapshot)
            end
        end
    end
end


function write_log(ofname::String, Npart::Int, step::Int, c2::T, τD::T, τm::T, τₕ::T, alpha_lst::Vector{T}, Eₖ::Vector{T}, Eₚ::Vector{T}, dQ::Vector{T}) where T
    out_file = ofname*".log"
    sdot = -Float32(sum(dQ ./ alpha_lst)/(Npart*c2))
    Ekin = sum(Eₖ)*(τm/(2.0f0*τD*Npart))
    Epot = sum(Eₚ)*(τD/(2.0f0*τₕ*Npart))

    data = hcat(step, Ekin, Epot, sdot)
    if step == 0
       open(out_file,"w") do file
            println(file,"Time  E_kin  E_pot  EPR")
            #writedlm(file,data)
        end
    else
        open(out_file,"a+") do file
            println("Time = ",data[1], " | E_kin = ", data[2], " | E_pot = " ,data[3], " | EPR = ", data[4])
            writedlm(file,data)
        end
    end
end
