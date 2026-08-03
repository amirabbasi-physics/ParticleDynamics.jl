# MACE foundation-model force provider for ParticleDynamics.jl.
#
# Implements the `AbstractExternalPotential` contract: positions are staged
# host-side, forces/energy come from mace-torch (GPU) through an ASE
# calculator, and are written back into the state's force buffers. At demo
# sizes the model inference dominates; the host round-trip is irrelevant and
# deliberately not optimized.
#
# Units are MACE-native: Å, eV, amu (derived time unit Å·√(amu/eV) ≈ 10.18 fs,
# so dt = 1 fs = 0.098226). Engine positions live in [-L/2, L/2); ASE accepts
# them as-is with pbc=true.
#
# Load PythonCall and this file from an environment with mace-torch available
# (see examples/mace/requirements.txt).

using PythonCall

struct MACEPotential <: ParticleDynamics.AbstractExternalPotential
    atoms::Py                 # ase.Atoms with calculator attached
    np::Py                    # numpy module handle
    hx::Vector{Float64}       # host staging (D2H targets)
    hy::Vector{Float64}
    hz::Vector{Float64}
    pos::Matrix{Float64}      # N×3 position staging for ASE
    N::Int
end

"""
    MACEPotential(numbers, box_lengths; variant=:mp, model="small",
                  device="cuda", dtype="float64")

Build a MACE provider for `N = length(numbers)` atoms with atomic numbers
`numbers` in an orthorhombic periodic box `box_lengths = (Lx, Ly, Lz)` (Å).
`variant=:mp` loads MACE-MP-0 (materials), `variant=:off` loads MACE-OFF
(organic molecules/liquids). `dtype` selects the model precision
(`"float64"` or `"float32"`); the engine-side buffers stay Float64 either way.
"""
function MACEPotential(numbers::AbstractVector{<:Integer}, box_lengths;
                       variant::Symbol=:mp, model::String="small",
                       device::String="cuda", dtype::String="float64")
    np = pyimport("numpy")
    ase = pyimport("ase")
    mc = pyimport("mace.calculators")
    N = length(numbers)
    cell = np.diag(np.asarray(Float64[box_lengths...]))
    atoms = ase.Atoms(; numbers=np.asarray(collect(Int, numbers)),
                      positions=np.zeros((N, 3)), cell=cell, pbc=true)
    calc = variant === :mp ?
        mc.mace_mp(; model=model, device=device, default_dtype=dtype) :
        mc.mace_off(; model=model, device=device, default_dtype=dtype)
    atoms.calc = calc
    return MACEPotential(atoms, np, zeros(N), zeros(N), zeros(N), zeros(N, 3), N)
end

function ParticleDynamics.external_forces!(p::MACEPotential,
                                           st::ParticleDynamics.SimulationState{Float64},
                                           compute_energy::Bool)
    copyto!(p.hx, st.rx); copyto!(p.hy, st.ry); copyto!(p.hz, st.rz)
    @inbounds for i in 1:p.N
        p.pos[i, 1] = p.hx[i]
        p.pos[i, 2] = p.hy[i]
        p.pos[i, 3] = p.hz[i]
    end
    p.atoms.set_positions(p.np.asarray(p.pos))
    F = pyconvert(Matrix{Float64}, p.atoms.get_forces())
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
