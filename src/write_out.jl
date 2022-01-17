using DelimitedFiles

export write_xyz

function write_xyz(ofname, nPart,step , dim, particles)
    out_file = ofname*".xyz"
    if isnothing(particles[1].τΓ)
        snapshot = [vcat(particles[i].part_type,particles[i].σ/2,particles[i].r,particles[i].v) for i in 1:nPart]
    else
        snapshot = [vcat(particles[i].part_type,particles[i].σ/2,particles[i].r,particles[i].v,particles[i].r_pseu,particles[i].v_pseu) for i in 1:nPart]
    end

    open(out_file,"a+") do file
        println(file,nPart)
        if dim == 2
            println(file,"ITEM: ATOMS radius x y v_x v_y")
            writedlm(file,snapshot)
        elseif dim == 3
            println(file,"ITEM: ATOMS radius x y z v_x v_y v_z")
            writedlm(file,snapshot)
        end
    end
end

function write_log(ofname, nPart,step , dim, particles)
    out_file = ofname*".log"
    if isnothing(particles[1].τΓ)
        snapshot = [vcat(particles[i].part_type,particles[i].σ/2,particles[i].r,particles[i].v) for i in 1:nPart]
    else
        snapshot = [vcat(particles[i].part_type,particles[i].σ/2,particles[i].r,particles[i].v,particles[i].r_pseu,particles[i].v_pseu) for i in 1:nPart]
    end

    open(out_file,"a+") do file
        println(file,nPart)
        if dim == 2
            println(file,"ITEM: ATOMS radius x y v_x v_y")
            writedlm(file,snapshot)
        elseif dim == 3
            println(file,"ITEM: ATOMS radius x y z v_x v_y v_z")
            writedlm(file,snapshot)
        end
    end
end
