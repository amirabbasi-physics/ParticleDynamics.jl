#!/usr/bin/env julia

using CUDA

const ROOT = normpath(joinpath(@__DIR__, ".."))
const EXAMPLES_DIR = joinpath(ROOT, "examples")

const SMOKE_ENV = Dict(
    "NEQSIM_NPARTICLES" => "100",
    "NEQSIM_N" => "100",
    "NEQSIM_MAX_STEPS" => "2",
    "NEQSIM_STEPS" => "2",
    "NEQSIM_LOG_INTERVAL" => "1",
    "NEQSIM_WARMUP_STEPS" => "0",
    "NEQSIM_INIT_STEPS" => "0",
    "NEQSIM_RELAX_STEPS" => "0",
    "NEQSIM_FREEZE_STEPS" => "0",
    "NEQSIM_BURN_STEPS" => "0",
    "NEQSIM_SAMPLE_STRIDE" => "1",
    "NEQSIM_MAX_SECONDS" => "20",
    "NEQSIM_APPEND_GSD" => "false",
    "NEQSIM_APPEND_LOG" => "false",
)

function run_example(path::String; args::Vector{String}=String[], env::Dict{String,String}=Dict{String,String}())
    merged_env = merge(SMOKE_ENV, env)
    cmd = Cmd([
        Base.julia_cmd().exec...,
        "--project=$(ROOT)",
        path,
        args...,
    ])
    println("\n==> Running smoke example: ", relpath(path, ROOT))
    run(addenv(cmd, merged_env...))
end

function example_args(path::String)
    name = basename(path)
    if name == "SingleT_2D_MD_CSVR.jl"
        return ["0.7", "100.0"]
    elseif name == "SingleT_2D_LD_VV_hex_sweep.jl"
        return ["0.7", "100.0"]
    end
    return String[]
end

function example_env(path::String)
    name = basename(path)
    if name == "restart_from_gsd.jl"
        return Dict(
            "NEQSIM_GSD_IN" => joinpath(EXAMPLES_DIR, "traj3d_quick.gsd"),
            "NEQSIM_GSD_OUT" => joinpath(EXAMPLES_DIR, "traj3d_quick_restart.gsd"),
            "NEQSIM_LOG_OUT" => joinpath(EXAMPLES_DIR, "traj3d_quick_restart.log"),
        )
    elseif name == "TwoT_2D_LD_Circle.jl"
        return Dict("NEQSIM_NPARTICLES" => "400")
    elseif name == "TwoT_2D_LD_slab.jl"
        return Dict("NEQSIM_NPARTICLES" => "512")
    elseif name == "TwoT_2D_MD_NHC_LJ_slab.jl"
        return Dict("NEQSIM_NPARTICLES" => "768")
    elseif name == "TwoT_2D_MD_NHC_softrepulsive_slab.jl"
        return Dict("NEQSIM_NPARTICLES" => "1024")
    elseif name == "TwoT_LJ2D_MD_CSVR_slab.jl"
        return Dict("NEQSIM_NPARTICLES" => "512")
    elseif name == "TwoT_SR2D_MD_CSVR_slab.jl"
        return Dict("NEQSIM_NPARTICLES" => "512")
    end
    return Dict{String,String}()
end

function example_paths()
    paths = filter(path -> endswith(path, ".jl") && basename(path) != "_example_utils.jl",
                   readdir(EXAMPLES_DIR; join=true))
    sort!(paths)

    restart_idx = findfirst(path -> basename(path) == "restart_from_gsd.jl", paths)
    if restart_idx !== nothing
        restart = paths[restart_idx]
        deleteat!(paths, restart_idx)
        push!(paths, restart)
    end
    return paths
end

if !CUDA.functional()
    println("CUDA.functional() == false; skipping examples smoke.")
    exit(0)
end

failures = String[]

for path in example_paths()
    try
        run_example(path; args=example_args(path), env=example_env(path))
    catch err
        push!(failures, "$(relpath(path, ROOT)): $(err)")
    end
end

if isempty(failures)
    println("\nExamples smoke completed successfully.")
    exit(0)
else
    println("\nExamples smoke failures:")
    for f in failures
        println(" - ", f)
    end
    exit(1)
end
