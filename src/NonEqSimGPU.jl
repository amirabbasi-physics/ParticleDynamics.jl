module NonEqSim

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
using LinearAlgebra
import Random: randperm

include("definitions.jl")
include("simulation.jl")
include("integrators.jl")
include("cell_list.jl")
include("boundary_conditions.jl")
include("interactions.jl")
include("write_out.jl")



println("##########################################################")
println("                  NonEqSim is Loaded!                     ")
println("                      Amir Abbasi                         ")
println("##########################################################")


end
