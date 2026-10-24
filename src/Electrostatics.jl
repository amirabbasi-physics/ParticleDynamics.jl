"""
Native GPU long-range electrostatics.

`Electrostatics` implements particle-mesh Coulomb solvers that share one mesh
engine: smooth particle-mesh Ewald (SPME, analytic differentiation), P3M with
optimal influence functions (AD or ik differentiation), and Ewald summation
with prolates (ESP, Liang et al. 2026). The radial split, the assignment
window and the influence function are separate policy types, but only the
validated combinations listed in [`ElectrostaticsSpec`](@ref) can be built.

Production force evaluation runs entirely on the GPU: Julia CUDA kernels for
real space, exceptions, spreading, the spectral solve and gathering, plus
cuFFT R2C/C2R transforms with preallocated plans. Host code only validates
input, generates coefficients (prolate functions, window corrections) and
plans transforms at setup. The convention sheet (units, normalisation, signs)
is in the electrostatics manual page.
"""
module Electrostatics

using CUDA
using LinearAlgebra
using ..Definitions
using ..NeighborLists

export ElectrostaticsSpec, GaussianSplit, ProlateSplit, BSplineWindow, ProlateWindow,
       ElectrostaticsSolver, ProlateFunction, prolate_shape_for_tolerance

include("electrostatics/Quadrature.jl")
include("electrostatics/Prolates.jl")
include("electrostatics/Splits.jl")
include("electrostatics/Windows.jl")
include("electrostatics/Spec.jl")
include("electrostatics/Influence.jl")
include("electrostatics/RealSpace.jl")
include("electrostatics/MeshKernels.jl")
include("electrostatics/Solver.jl")

end # module Electrostatics
