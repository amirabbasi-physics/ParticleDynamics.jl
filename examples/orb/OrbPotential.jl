# Orb foundation-model force provider for ParticleDynamics.jl.
#
# Implements the `AbstractExternalPotential` contract, mirroring
# examples/mace/MACEPotential.jl: positions are staged host-side, forces and
# energy come from orb-models (GPU) through its ASE calculator, and are written
# back into the state's force buffers. Sharing the staging path with the MACE
# provider is deliberate — a head-to-head between the two models then differs
# only in the model call.
#
# Units are Å, eV, amu, as for MACE (derived time unit Å·√(amu/eV) ≈ 10.18 fs,
# so dt = 1 fs = 0.098226). Engine positions live in [-L/2, L/2); ASE accepts
# them as-is with pbc=true.
#
# Two families of Orb models matter here and behave differently under MD:
#
#   * `conservative` models predict an energy and obtain forces as its
#     analytic gradient, so NVE energy is conserved.
#   * `direct` models predict forces as an independent output. They are ~2x
#     faster but the force field is not a gradient of any scalar, so NVE
#     energy is *not* conserved. Use them only for sampling, never for NVE.
#
# Orb runs float32 natively (`precision="float32-high"` enables TF32 matmuls).
# float64 is supported but ~30x slower on consumer GPUs; see the head-to-head
# script for the matched-precision discussion.
#
# Load PythonCall and this file from an environment with orb-models available
# (see examples/orb/requirements.txt).

using PythonCall

struct OrbPotential <: ParticleDynamics.AbstractExternalPotential
    atoms::Py                 # ase.Atoms with ORBCalculator attached
    np::Py                    # numpy module handle
    hx::Vector{Float64}       # host staging (D2H targets)
    hy::Vector{Float64}
    hz::Vector{Float64}
    pos::Matrix{Float64}      # N×3 position staging for ASE
    N::Int
    model::String             # loader name, for reporting
end

"""
    OrbPotential(numbers, box_lengths; model="orb_v3_conservative_inf_omat",
                 device="cuda", precision="float32-high", charge=nothing,
                 spin=nothing)

Build an Orb provider for `N = length(numbers)` atoms with atomic numbers
`numbers` in an orthorhombic periodic box `box_lengths = (Lx, Ly, Lz)` (Å).

`model` names a loader in `orb_models.forcefield.pretrained`, e.g.
`"orb_v3_conservative_inf_omat"` (periodic/materials training),
`"orb_v3_conservative_omol"` (molecular training) or the corresponding
`direct` variants. `precision` is one of `"float32-high"`, `"float32-highest"`
or `"float64"`.

The `omol` models are charge/spin conditioned and require `charge` and `spin`
(total charge and spin multiplicity of the cell contents); passing them for
other models is harmless.

Note that loading an Orb model calls `torch.set_default_dtype` process-wide.
When several providers coexist in one process, build them at a common
precision or rebuild between measurements.
"""
function OrbPotential(numbers::AbstractVector{<:Integer}, box_lengths;
                      model::String="orb_v3_conservative_inf_omat",
                      device::String="cuda", precision::String="float32-high",
                      charge=nothing, spin=nothing)
    np = pyimport("numpy")
    ase = pyimport("ase")
    pretrained = pyimport("orb_models.forcefield.pretrained")
    calcmod = pyimport("orb_models.forcefield.inference.calculator")
    N = length(numbers)
    cell = np.diag(np.asarray(Float64[box_lengths...]))
    atoms = ase.Atoms(; numbers=np.asarray(collect(Int, numbers)),
                      positions=np.zeros((N, 3)), cell=cell, pbc=true)
    # Charge/spin conditioning for the OMol-trained models.
    charge === nothing || (atoms.info["charge"] = charge)
    spin === nothing || (atoms.info["spin"] = spin)
    loaded = pygetattr(pretrained, model)(; device=device, precision=precision)
    atoms.calc = calcmod.ORBCalculator(loaded[0], loaded[1]; device=device)
    return OrbPotential(atoms, np, zeros(N), zeros(N), zeros(N),
                        zeros(N, 3), N, model)
end

function ParticleDynamics.external_forces!(p::OrbPotential,
                                           st::ParticleDynamics.SimulationState{Float64},
                                           compute_energy::Bool)
    copyto!(p.hx, st.rx); copyto!(p.hy, st.ry); copyto!(p.hz, st.rz)
    @inbounds for i in 1:p.N
        p.pos[i, 1] = p.hx[i]
        p.pos[i, 2] = p.hy[i]
        p.pos[i, 3] = p.hz[i]
    end
    p.atoms.set_positions(p.np.asarray(p.pos))
    # Orb returns float32 for its native precisions; widen on the numpy side
    # so the engine's Float64 buffers are filled from a matching dtype.
    F = pyconvert(Matrix{Float64},
                  p.np.asarray(p.atoms.get_forces(), dtype=p.np.float64))
    copyto!(st.fx, F[:, 1])
    copyto!(st.fy, F[:, 2])
    copyto!(st.fz, F[:, 3])
    if compute_energy
        E = pyconvert(Float64, p.atoms.get_potential_energy())
        fill!(st.Epot, E / p.N)
        fill!(st.virial_nonbonded, 0.0)
    end
    return nothing
end
