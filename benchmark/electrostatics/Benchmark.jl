# Entry point bundling the cross-engine electrostatics benchmark modules.
#   include("benchmark/electrostatics/Benchmark.jl")
# defines `Fixtures`, `Engines` and `PDHarness` in the including module.
include(joinpath(@__DIR__, "Fixtures.jl"))
include(joinpath(@__DIR__, "Engines.jl"))
include(joinpath(@__DIR__, "PDHarness.jl"))
