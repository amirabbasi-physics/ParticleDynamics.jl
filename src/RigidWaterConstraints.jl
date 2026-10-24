"""
Low-level three-site rigid-water constraint primitives.

These kernels support the low-level fixed-cell constrained NVE path through
`SimulationCore.attach_rigid_water!`. They are not an NPT integrator:
thermostats, molecular volume moves and restart serialization remain pending.
All distances and masses use the caller's consistent unit convention.
"""
module RigidWaterConstraints

using CUDA, StaticArrays
import ..Definitions
using LinearAlgebra: dot, cross

include("constraints/WaterGeometry.jl")
include("constraints/WaterCUDA.jl")

end
