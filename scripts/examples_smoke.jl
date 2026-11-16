#!/usr/bin/env julia

using CUDA

const ROOT = normpath(joinpath(@__DIR__, ".."))
const EXAMPLES_DIR = joinpath(ROOT, "examples")

const SMOKE_ENV = Dict(
    "SIM_NPARTICLES" => "100",
    "SIM_MAX_STEPS" => "2",
    "SIM_LOG_INTERVAL" => "1",
    "SIM_WARMUP_STEPS" => "0",
    "SIM_INIT_STEPS" => "0",
    "SIM_RELAX_STEPS" => "0",
    "SIM_MAX_SECONDS" => "20",
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
    basename(path) == "SingleT_2D_MD_CSVR.jl" && return ["0.7", "100.0"]
    return String[]
end

# Per-example overrides of SMOKE_ENV. The 3D Lennard-Jones example needs a box of
# at least three list radii (9σ), i.e. more than ~580 particles at its density.
function example_env(path::String)
    basename(path) == "3D_LJ_NVE.jl" && return Dict("SIM_NPARTICLES" => "1000")
    return Dict{String,String}()
end

function example_paths()
    paths = filter(path -> endswith(path, ".jl") && !startswith(basename(path), "_"),
                   readdir(EXAMPLES_DIR; join=true))
    return sort!(paths)
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
