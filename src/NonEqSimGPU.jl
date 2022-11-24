module NonEqSimGPU

"""
Flush output so that jobs can be monitored on cluster.
"""
@inline println(args...) = println(stdout, args...)
@inline function println(io::IO, args...)
    Base.println(io, args...)
    flush(io)
end



using CUDA
using StaticArrays
using DelimitedFiles
using BenchmarkTools
using Test
using CSV
using DataFrames
using LinearAlgebra
import Random: randperm

include("definitions.jl")
include("simulation.jl")
include("integrators.jl")
include("analysis.jl")
include("sim_run.jl")
include("boundary_conditions.jl")
include("interactions.jl")
include("write_out.jl")



println("##########################################################")
println("                  NonEqSimGPU is Launched!                ")
println("##########################################################")


end
