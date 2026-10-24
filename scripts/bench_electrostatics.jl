#!/usr/bin/env julia
#
# Accuracy-constrained benchmark of the GPU long-range electrostatics solvers.
#
# System: a perturbed rocksalt arrangement of ±1 e ions in Å/eV units
# (lattice spacing 2.82 Å, random displacements of 0.25 Å, Coulomb prefactor
# 14.399645 eV·Å/e²), i.e. a dense molten-salt-like charge distribution without
# overlapping pairs. Coulomb only (no vdW), as in the Coulomb timings of Liang
# et al. (2026).
#
# For each method and each target relative force error Δ, the accuracy knob is
# tightened until the measured error against a converged reference meets Δ:
# the mesh spacing for SPME/P3M (fixed α, order 5), and the prolate tolerance
# ε for ESP (with the paper's rules: mesh spacing π r_c/c(ε), Table-2 order).
# ESP's error saturates near ε under mesh refinement (the prolate window's
# alias level is fixed in grid units), so refining its mesh alone cannot
# reach a tighter target. The reference is SPME in Float64 with order 12, a fine
# mesh and a 12 Å cutoff; it is checked against an independent second
# reference (ESP, ε = 1e-10) before use. Then each stage (real space, spread,
# FFTs, spectral solve, gather) and the full force call are timed with CUDA
# events after warm-up.
#
# Usage:
#   julia --project=. scripts/bench_electrostatics.jl [n_per_side=40] [precision=f32] [rc=9.0] [ordered]
#
# n_per_side = 40 → N = 64 000 ions; 80 → 512 000.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using CUDA
using LinearAlgebra
using ParticleDynamics
using Printf
using Random

const ES = ParticleDynamics.Electrostatics
const SC = ParticleDynamics.SimulationCore
const KC = 14.399645          # eV·Å / e²

function rocksalt(n; a=2.82, disp=0.25, seed=1)
    rng = MersenneTwister(seed)
    L = n * a
    R = Matrix{Float64}(undef, 3, n^3)
    q = Vector{Float64}(undef, n^3)
    p = 0
    for i in 1:n, j in 1:n, k in 1:n
        p += 1
        R[:, p] .= (a * (i - 0.5), a * (j - 0.5), a * (k - 0.5)) .- L / 2 .+ disp .* randn(rng, 3)
        q[p] = isodd(i + j + k) ? 1.0 : -1.0
    end
    return R, q, (L, L, L)
end

function make_state(R, q, L, spec, ::Type{T}) where {T}
    N = size(R, 2)
    rc = spec.real_cutoff
    cap = Int32(ceil(1.4 * 4pi / 3 * (rc + 1.0)^3 * N / prod(L) + 32))
    st = build_simulation(N=N, box=L, nonbonded=:none, cutoff=rc, skin=1.0, cap=cap,
                          precision=T === Float64 ? :f64 : :f32, charges=q, spatial_reorder=false)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    attach_electrostatics!(st, spec)
    SC.evaluate_forces_into_f!(st, false)
    return st
end

forces(st) = vcat(Float64.(Array(st.fx))', Float64.(Array(st.fy))', Float64.(Array(st.fz))')
relerr(F, Fref) = sqrt(sum(abs2, F .- Fref) / sum(abs2, Fref))

function mesh_for(L, h, P)
    return ntuple(d -> max(ES.fft_friendly(ceil(Int, L[d] / h)), P), 3)
end

spme(rc, α, mesh, P) = ES.ElectrostaticsSpec(coulomb_prefactor=KC, real_cutoff=rc, split=ES.GaussianSplit(α),
                                             window=ES.BSplineWindow(P), mesh=mesh, influence=:spme)
p3m(rc, α, mesh, P, kind) = ES.ElectrostaticsSpec(coulomb_prefactor=KC, real_cutoff=rc, split=ES.GaussianSplit(α),
                                                  window=ES.BSplineWindow(P), mesh=mesh, influence=kind)
esp(rc, eps, mesh, P) = ES.ElectrostaticsSpec(coulomb_prefactor=KC, real_cutoff=rc,
    split=ES.ProlateSplit(rc, ES.prolate_shape_for_tolerance(eps)),
    window=ES.ProlateWindow(P, ES.prolate_shape_for_tolerance(eps / 2)), mesh=mesh, influence=:esp)

# Median of `reps` CUDA-event timings of `f()` (milliseconds).
function timed(f; reps=20)
    f(); f()
    ts = Float64[]
    for _ in 1:reps
        push!(ts, 1e3 * CUDA.@elapsed f())
    end
    sort!(ts)
    return ts[cld(reps, 2)]
end

function stage_times(st)
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
    t_total = timed(() -> ES.compute!(s, args..., false))
    t_real = timed(() -> ES.compute_real_space!(s, st.rx, st.ry, st.rz, st.charge, st.fx, st.fy, st.fz,
                                                st.Epot, st.virial_nonbonded, st.nbh, false))
    t_spread = timed(() -> begin
        fill!(ws.grid, zero(T))
        @cuda threads=pt blocks=pb ES._spread_kernel!(ws.grid_flat, st.rx, st.ry, st.rz, st.charge, order,
                                                      (L1, L2, L3), K1, K2, K3, s.stencil)
    end)
    t_fft = timed(() -> mul!(ws.spec, ws.plan_fwd, ws.grid))
    t_solve = timed(() -> @cuda threads=mt blocks=mb ES._solve_kernel!(ws.spec_flat, ws.G, ws.Vc, ws.acc,
                                                                       K1, K1 ÷ Int32(2) + Int32(1), nm, Val(false)))
    if ES.is_ik(s)
        t_field = timed(() -> for d in 1:3
            @cuda threads=mt blocks=mb ES._ik_field_kernel!(ws.scratch_flat, ws.spec_flat, K1, K2, K3, (L1, L2, L3), Int32(d))
        end)
        t_fft += 3 * timed(() -> mul!(ws.field[1], ws.plan_bwd, ws.scratch))
        ff = ws.field_flat
        t_gather = t_field + timed(() -> @cuda threads=pt blocks=pb ES._gather_ik_kernel!(st.fx, st.fy, st.fz, st.Epot,
            st.rx, st.ry, st.rz, st.charge, order, ff[1], ff[2], ff[3], ws.grid_flat, (L1, L2, L3), K1, K2, K3,
            s.stencil, s.bg, Val(false)))
    else
        t_fft += timed(() -> mul!(ws.grid, ws.plan_bwd, ws.spec))
        t_gather = timed(() -> @cuda threads=pt blocks=pb ES._gather_ad_kernel!(st.fx, st.fy, st.fz, st.Epot,
            st.rx, st.ry, st.rz, st.charge, order, ws.grid_flat, (L1, L2, L3), K1, K2, K3, s.stencil, s.bg, Val(false)))
    end
    return (total=t_total, real=t_real, spread=t_spread, fft=t_fft, solve=t_solve, gather=t_gather)
end

function main(args)
    n = length(args) >= 1 ? parse(Int, args[1]) : 40
    T = (length(args) >= 2 && args[2] == "f64") ? Float64 : Float32
    rc = length(args) >= 3 ? parse(Float64, args[3]) : 9.0
    R, q, L = rocksalt(n)
    N = size(R, 2)
    # Random storage order, as in a diffusing liquid without spatial reordering;
    # pass "ordered" to keep the lattice order (what spatial reordering provides).
    ordered = length(args) >= 4 && args[4] == "ordered"
    if !ordered
        perm = randperm(MersenneTwister(99), N)
        R = R[:, perm]; q = q[perm]
    end
    println(ordered ? "storage order: spatially ordered" : "storage order: random (no spatial reordering)")
    @printf("N = %d ions, box %.2f Å, r_c = %.1f Å, %s, %s\n", N, L[1], rc, T, CUDA.name(CUDA.device()))

    # Accuracy calibration on a replica at the same density (the relative mesh
    # error is intensive): converged Float64 references, cross-checked.
    ncal = min(n, 20)
    Rc, qc, Lc = rocksalt(ncal; seed=2)
    rref = 12.0
    FA = forces(make_state(Rc, qc, Lc, spme(rref, ES.alpha_from_tolerance(rref, 1e-13), mesh_for(Lc, 0.45, 12), 12), Float64))
    FB = forces(make_state(Rc, qc, Lc, esp(rref, 1e-10, mesh_for(Lc, pi * rref / ES.prolate_shape_for_tolerance(1e-10), 14), 14), Float64))
    GC.gc(); CUDA.reclaim()
    @printf("calibration replica: %d ions; reference agreement (SPME P=12 vs ESP ε=1e-10): %.2e\n\n",
            size(Rc, 2), relerr(FB, FA))

    espspec(Lx, eps) = esp(rc, eps, mesh_for(Lx, pi * rc / ES.prolate_shape_for_tolerance(eps), ESP_ORDER(eps)), ESP_ORDER(eps))
    rows = []
    for target in (1e-4, 1e-5)
        # Gaussian split with a real-space tail an order of magnitude below Δ
        α = ES.alpha_from_tolerance(rc, target / 10)
        methods = (("SPME", (Lx, h) -> spme(rc, α, mesh_for(Lx, h, 5), 5), 1.6 .* (0.92 .^ (0:40))),
                   ("P3M-AD", (Lx, h) -> p3m(rc, α, mesh_for(Lx, h, 5), 5, :p3m_ad), 1.6 .* (0.92 .^ (0:40))),
                   ("P3M-ik", (Lx, h) -> p3m(rc, α, mesh_for(Lx, h, 5), 5, :p3m_ik), 1.6 .* (0.92 .^ (0:40))),
                   ("ESP", espspec, nothing))
        for (name, mk, knobs) in methods
            knobs = knobs === nothing ? target .* (0.75 .^ (0:12)) : knobs
            chosen = nothing
            for knob in knobs
                spec_cal = mk(Lc, knob)
                maximum(ES.resolve_mesh(spec_cal, Lc)) * n / ncal > 400 && break   # full-system mesh cap
                st = make_state(Rc, qc, Lc, spec_cal, T)
                err = relerr(forces(st), FA)
                st = nothing; GC.gc(); CUDA.reclaim()
                if err <= target
                    chosen = (knob=knob, err=err)
                    break
                end
            end
            chosen === nothing && (println("  $(name): target $(target) not reached"); continue)
            st = make_state(R, q, L, mk(L, chosen.knob), T)
            tm = stage_times(st)
            mesh = ES.solver_mesh(st.electrostatics)
            P = window_order_of(st)
            push!(rows, (target, name, P, mesh, chosen.err, tm))
            @printf("Δ=%.0e %-7s P=%-2d mesh=%-15s err=%.2e | total %.3f ms (real %.3f, spread %.3f, FFT %.3f, solve %.3f, gather %.3f)\n",
                    target, name, P, string(mesh), chosen.err, tm.total, tm.real, tm.spread, tm.fft, tm.solve, tm.gather)
            st = nothing; GC.gc(); CUDA.reclaim()
        end
        println()
    end
    return rows
end

ESP_ORDER(eps) = ParticleDynamics.Workflow.esp_order_for_tolerance(eps)
window_order_of(st) = st.electrostatics.spec.window.order

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
