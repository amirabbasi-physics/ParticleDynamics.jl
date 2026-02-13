#!/usr/bin/env julia

using CUDA

function run_example(path::String; env::Dict{String,String}=Dict{String,String}())
    cmd = `$(Base.julia_cmd()) --project $path`
    println("\n==> Running smoke example: ", path)
    run(addenv(cmd, env...))
end

if !CUDA.functional()
    println("CUDA.functional() == false; skipping examples smoke.")
    exit(0)
end

# Always run a fast, stable subset.
base_examples = [
    "examples/2D_allpairs_quicktest.jl",
    "examples/2D_softrep_nl_check.jl",
    "examples/3D_quicktest.jl",
]

# Optional heavier smoke script with explicit runtime limits.
heavy_examples = [
    ("examples/TwoT_2D_BD_EH.jl", Dict("NEQSIM_MAX_STEPS" => "8", "NEQSIM_LOG_INTERVAL" => "4", "NEQSIM_MAX_SECONDS" => "20")),
]

failures = String[]

for path in base_examples
    try
        run_example(path)
    catch err
        push!(failures, "$(path): $(err)")
    end
end

if get(ENV, "NEQSIM_SMOKE_HEAVY", "0") == "1"
    for (path, env) in heavy_examples
        try
            run_example(path; env=env)
        catch err
            push!(failures, "$(path): $(err)")
        end
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
