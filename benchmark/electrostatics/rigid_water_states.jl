# Shared by profile_engine_rigid.jl and water_mc_density.jl: rigid SPC/E water states from the
# GROMACS-equilibrated water boxes (LJ + ESP Δ = 1e-3, cluster backend at the GROMACS
# cluster-list buffer, exact SPC/E geometry, Maxwell velocities at 300 K projected by RATTLE).
include(joinpath(@__DIR__, "run_traversal_ab.jl"))
const RWC = ParticleDynamics.RigidWaterConstraints
const OH = 0.1; const HH = 0.16330

# exact SPC/E geometry: O fixed, the H–O–H plane and bisector kept
function snap!(R, box)
    s = HH / (2OH); c = sqrt(1 - s^2)
    for w in 1:(size(R, 2) ÷ 3)
        o = 3w - 2
        O = R[:, o]
        d1 = R[:, o + 1] .- O; d2 = R[:, o + 2] .- O
        d1 .-= box .* round.(d1 ./ box); d2 .-= box .* round.(d2 ./ box)
        b = normalize(d1 .+ d2); p = d1 .- d2; p .-= dot(p, b) .* b; p = normalize(p)
        R[:, o + 1] .= O .+ OH .* (c .* b .+ s .* p)
        R[:, o + 2] .= O .+ OH .* (c .* b .- s .* p)
    end
    return R
end

function rigid_state(f, names, spec, skin, v0, m)
    T = Float32; N = length(f); box = collect(f.box)
    R = snap!(copy(f.R), box)
    types = [_typeidx(names[i]) for i in 1:N]; σ, ε = lj_matrices()
    cap = Int32(ceil(1.3 * 4pi / 3 * (1.0 + skin)^3 * N / prod(f.box) + 32))
    st = SC.build_simulation(; N, box=f.box, nonbonded=:lj, cutoff=T(1.0), epsilon=T(maximum(ε)), sigma=T(maximum(σ)),
        skin=T(skin), cap, mass=T.(m), dt=T(0.002), precision=:f32, charges=f.q, spatial_reorder=false,
        neigh_interval=10_000)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(v0[1, :])); copyto!(st.vy, T.(v0[2, :])); copyto!(st.vz, T.(v0[3, :]))
    copyto!(st.typeid, Int32.(types))
    st.sigma_pair = CuArray(T.(σ)); st.epsilon_pair = CuArray(T.(ε))
    st.rcut_pair = CuArray([ε[a, b] > 0 ? T(1.0) : zero(T) for a in axes(ε, 1), b in axes(ε, 2)])
    attach_electrostatics!(st, spec; exceptions=f.exceptions)
    attach_rigid_water!(st, [(3w - 2, 3w - 1, 3w) for w in 1:(N ÷ 3)]; oh=OH, hh=HH)
    return st
end

function load_rigid_water(sysdir, name; seed=5)
    gro = joinpath(sysdir, name * ".gro"); top = joinpath(sysdir, name * ".top")
    gb = gromacs_buffers(joinpath(sysdir, name * ".log"))
    f = gromacs_fixture(gro; top, gmx=GMX, workdir=mktempdir()); nm = gro_names(gro)
    spec = esp_spec(f; rc=1.0, tolerance=1e-3)
    N = length(f); m = [startswith(nm[i], "OW") ? 15.9994 : 1.008 for i in 1:N]
    v0 = randn(Random.MersenneTwister(seed), 3, N) .* sqrt.(0.0083144626 * 300 ./ m')
    return rigid_state(f, nm, spec, gb.cluster, v0, m), (; f, nm, spec, gb, m, v0)
end

function constraint_errors(st)
    r = map(x -> Float64.(Array(x)), (st.rx, st.ry, st.rz)); v = map(x -> Float64.(Array(x)), (st.vx, st.vy, st.vz))
    box = Float64.(st.box3); dl = 0.0; dv = 0.0
    for w in 1:(length(r[1]) ÷ 3)
        ids = (3w - 2, 3w - 1, 3w)
        raw = ntuple(k -> RWC.SVector(ntuple(d -> r[d][ids[k]], 3)), 3)
        x = RWC.local_water(raw, raw[1], box)
        vv = ntuple(k -> RWC.SVector(ntuple(d -> v[d][ids[k]], 3)), 3)
        dl = max(dl, RWC.distance_error(x, (OH^2, OH^2, HH^2)))
        dv = max(dv, maximum(abs(dot(e, u)) for (e, u) in zip(RWC._edges(x), RWC._edges(vv))))
    end
    return dl, dv
end
