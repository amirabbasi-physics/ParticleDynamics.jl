"""
ParticleDynamics side of the cross-engine electrostatics benchmark:
frozen-coordinate states built from `Fixtures.Fixture`, force extraction and
CUDA-event stage timings.
"""
module PDHarness

using CUDA
using LinearAlgebra
using Printf
using ParticleDynamics
using ..Fixtures

const ES = ParticleDynamics.Electrostatics
const SC = ParticleDynamics.SimulationCore

export pd_state, pd_state_lj, pd_forces, pd_energy, pd_times, pd_nonbonded_time, relerr, error_report,
       pme_spec, pppm_spec, esp_spec

relerr(F, Fref) = sqrt(sum(abs2, F .- Fref) / sum(abs2, Fref))

"""
    error_report(F, Fref) -> NamedTuple

The paper's relative RMS force error Δ plus absolute RMS and maximum errors
(in the fixture's force unit).
"""
function error_report(F, Fref)
    d = sqrt.(vec(sum(abs2, F .- Fref; dims=1)))
    return (rel=relerr(F, Fref), abs_rms=sqrt(sum(abs2, d) / length(d)), abs_max=maximum(d),
            ref_rms=sqrt(sum(abs2, Fref) / size(Fref, 2)))
end

pme_spec(f::Fixture; rc, alpha, mesh, order=4, modifier=:none) =
    ES.ElectrostaticsSpec(coulomb_prefactor=f.prefactor, real_cutoff=rc, split=ES.GaussianSplit(alpha),
                          window=ES.BSplineWindow(order), mesh=mesh, influence=:spme, real_space_modifier=modifier)
pppm_spec(f::Fixture; rc, alpha, mesh, order=5, differentiation=:ik) =
    ES.ElectrostaticsSpec(coulomb_prefactor=f.prefactor, real_cutoff=rc, split=ES.GaussianSplit(alpha),
                          window=ES.BSplineWindow(order), mesh=mesh,
                          influence=differentiation === :ik ? :p3m_ik : :p3m_ad)
function esp_spec(f::Fixture; rc, tolerance, mesh=nothing, order=nothing)
    c = ES.prolate_shape_for_tolerance(tolerance)
    P = order === nothing ? ParticleDynamics.Workflow.esp_order_for_tolerance(tolerance) : order
    m = mesh === nothing ? ntuple(d -> max(ES.fft_friendly(ceil(Int, f.box[d] * c / (pi * rc))), P), 3) : mesh
    return ES.ElectrostaticsSpec(coulomb_prefactor=f.prefactor, real_cutoff=rc,
                                 split=ES.ProlateSplit(rc, c),
                                 window=ES.ProlateWindow(P, ES.prolate_shape_for_tolerance(tolerance / 2)),
                                 mesh=m, influence=:esp)
end

"""
    pd_state(f, spec; T=Float32, skin=0.1*r_c, backend=:ell) -> SimulationState

Charged state with no vdW term, the fixture's exclusions as exceptions, and
a dense cell list (all-pairs traversal when the box holds fewer than three
cells per dimension). `backend=:cluster` switches to the cluster-pair
engine with an exact prune radius (`prune_buffer=0`: the list holds only
tiles with pairs inside the cutoff, as GROMACS's unbuffered `rlist = r_c`
runs do). The first force evaluation (list build, compilation) is done here.
"""
function pd_state(f::Fixture, spec; T::Type=Float32, skin::Real=0.1 * spec.real_cutoff, backend::Symbol=:ell)
    N = length(f)
    rc = spec.real_cutoff
    density = N / prod(f.box)
    cap = Int32(ceil(1.3 * 4pi / 3 * (rc + skin)^3 * density + 32))
    pairs = all(L -> floor(L / (rc + skin)) >= 3, f.box)
    st = build_simulation(N=N, box=f.box, nonbonded=:none, cutoff=rc, skin=skin, cap=cap,
                          precision=T === Float64 ? :f64 : :f32, charges=f.q, spatial_reorder=false,
                          use_neighborlist=pairs, neigh_interval=10_000)
    copyto!(st.rx, T.(f.R[1, :])); copyto!(st.ry, T.(f.R[2, :])); copyto!(st.rz, T.(f.R[3, :]))
    attach_electrostatics!(st, spec; exceptions=f.exceptions)
    backend === :cluster && set_nonbonded_backend!(st, :cluster; prune_buffer=0.0)
    SC.evaluate_forces_into_f!(st, true)
    return st
end

"""
    pd_state_lj(f, spec; types, sigma, epsilon, rc_lj, T=Float32) -> SimulationState

Like `pd_state` (including `backend`), plus a per-type Lennard-Jones pair table (`sigma`, `epsilon`
are `ntypes×ntypes` matrices; pairs with ε = 0 are skipped) with its own
cutoff `rc_lj`; the shared neighbor list covers `max(rc_lj, r_c)`.
"""
function pd_state_lj(f::Fixture, spec; types::AbstractVector{<:Integer}, sigma::AbstractMatrix,
                     epsilon::AbstractMatrix, rc_lj::Real, T::Type=Float32, skin::Real=0.1 * spec.real_cutoff,
                     backend::Symbol=:ell)
    N = length(f)
    rc = spec.real_cutoff
    rlist = max(rc, rc_lj)
    cap = Int32(ceil(1.3 * 4pi / 3 * (rlist + skin)^3 * N / prod(f.box) + 32))
    pairs = all(L -> floor(L / (rlist + skin)) >= 3, f.box)
    st = build_simulation(N=N, box=f.box, nonbonded=:lj, cutoff=rc_lj, epsilon=maximum(epsilon),
                          sigma=maximum(sigma), neighbor_cutoff=rlist, skin=skin, cap=cap,
                          precision=T === Float64 ? :f64 : :f32, charges=f.q, spatial_reorder=false,
                          neigh_interval=10_000, use_neighborlist=pairs)
    copyto!(st.rx, T.(f.R[1, :])); copyto!(st.ry, T.(f.R[2, :])); copyto!(st.rz, T.(f.R[3, :]))
    copyto!(st.typeid, Int32.(types))
    rcut = [epsilon[a, b] > 0 ? T(rc_lj) : zero(T) for a in axes(epsilon, 1), b in axes(epsilon, 2)]
    st.sigma_pair = CuArray(T.(sigma)); st.epsilon_pair = CuArray(T.(epsilon)); st.rcut_pair = CuArray(rcut)
    attach_electrostatics!(st, spec; exceptions=f.exceptions)
    backend === :cluster && set_nonbonded_backend!(st, :cluster; prune_buffer=0.0)
    SC.evaluate_forces_into_f!(st, true)
    return st
end

"""
    pd_nonbonded_time(st; compute_energy=false, reps=30, batches=5) -> (median, min, max) ms

One complete nonbonded force evaluation of the state (vdW writer plus
electrostatics) through the engine's force entry point, after warm-up.
"""
function pd_nonbonded_time(st; compute_energy::Bool=false, reps::Int=30, batches::Int=5)
    call() = (SC.invalidate_forces!(st); SC.evaluate_forces_into_f!(st, compute_energy))
    return _timed(call; reps, batches)
end

pd_forces(st) = vcat(Float64.(Array(st.fx))', Float64.(Array(st.fy))', Float64.(Array(st.fz))')

"""
    pd_energy(st) -> (E, W)

Total electrostatic energy and the full virial tensor (xx, yy, zz, xy, xz, yz)
at the current coordinates.
"""
function pd_energy(st)
    SC.evaluate_forces_into_f!(st, true)
    W = virial_tensor(st)
    return sum(Float64, Array(st.Epot)), [W.xx, W.yy, W.zz, W.xy, W.xz, W.yz]
end

function _timed(f; reps=30, batches=5)
    reps > 0 && batches > 0 || throw(ArgumentError("Timing requires positive reps and batches."))
    f(); f(); CUDA.synchronize()
    meds = Float64[]
    for _ in 1:batches
        ts = sort!([1e3 * CUDA.@elapsed(f()) for _ in 1:reps])
        push!(meds, ts[cld(reps, 2)])
    end
    sort!(meds)
    return (median=meds[cld(batches, 2)], min=meds[1], max=meds[end])
end

"""
    pd_times(st) -> NamedTuple

Median (over 5 batches of 30 CUDA-event timings, after warm-up) of the full
electrostatic force call without and with energies/virial, and of each stage
(real space incl. exceptions, spread, FFTs, spectral solve, gather), plus the
peak device memory in use.
"""
function pd_times(st)
    s = st.electrostatics
    ws = s.ws
    T = eltype(st.rx)
    K1, K2, K3 = Int32.(ws.mesh)
    L1, L2, L3 = T.(s.box)
    N = length(st.rx)
    pt, pb = ES._launch1d(N, 128)
    nm = Int32(length(ws.G))
    mt, mb = ES._launch1d(nm, 256)
    order = ES._mesh_order(s, st.nbh)
    args = (st.rx, st.ry, st.rz, st.charge, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded, st.virial_global, st.nbh)
    force = _timed(() -> ES.compute!(s, args..., false))
    force_energy = _timed(() -> ES.compute!(s, args..., true))
    real = _timed(() -> ES.compute_real_space!(s, st.rx, st.ry, st.rz, st.charge, st.fx, st.fy, st.fz,
                                               st.Epot, st.virial_nonbonded, st.nbh, false)).median
    spread = _timed(() -> begin
        fill!(ws.grid, zero(T))
        @cuda threads=pt blocks=pb ES._spread_kernel!(ws.grid_flat, st.rx, st.ry, st.rz, st.charge, order,
                                                      (L1, L2, L3), K1, K2, K3, s.stencil)
    end).median
    fft = _timed(() -> mul!(ws.spec, ws.plan_fwd, ws.grid)).median
    solve = _timed(() -> @cuda threads=mt blocks=mb ES._solve_kernel!(ws.spec_flat, ws.G, ws.Vc, ws.acc,
                                                                      K1, K1 ÷ Int32(2) + Int32(1), nm, Val(false))).median
    if ES.is_ik(s)
        fft += 3 * _timed(() -> mul!(ws.field[1], ws.plan_bwd, ws.scratch)).median
    else
        fft += _timed(() -> mul!(ws.grid, ws.plan_bwd, ws.spec)).median
    end
    gather = force.median - real - spread - fft - solve
    return (force=force.median, force_spread=(force.min, force.max), force_energy=force_energy.median,
            real=real, spread=spread, fft=fft, solve=solve, gather_other=gather,
            mesh=ws.mesh, memory_gib=(CUDA.total_memory() - CUDA.free_memory()) / 2^30)
end

end # module PDHarness
