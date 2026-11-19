# ==============================================================================
# Canonical Benchmark & Validation Workloads for ParticleDynamics vs cuPotential
#   1. Molten NaCl (Tosi-Fumi / LJ + Coulomb, 10,648 or 64,000 ions)
#   2. Rigid SPC/E Water (10,000 or 32,000 waters = 30k / 96k atoms, WaterFastPath)
#   3. Inhomogeneous SPC/E Liquid-Vapor Slab (interfacial dispersion & surface tension)
#   4. Charged Bead-Spring Polyelectrolyte Melt (bonded exclusions + LJ + Coulomb)
# ==============================================================================
module BenchWorkloads

using CUDA
using LinearAlgebra
using Random
using StaticArrays
using ParticleDynamics
using ParticleDynamics: SimulationCore

export KB_KJ_MOL_K, COULOMB_NM_KJMOL,
       make_nacl_melt, make_spce_water, make_spce_slab, make_charged_polymer_melt

# Units: nm, kJ/mol, ps, elementary charge e, amu
const KB_KJ_MOL_K = 0.008314462618
const COULOMB_NM_KJMOL = 138.93545764

"""
    make_nacl_melt(::Type{T}; ncell=11, temp_K=1200.0, cutoff=1.0, skin=0.1,
                   vdw_modifier=:none, electrostatics_spec=nothing, seed=2026)

`8 * ncell^3` NaCl ion melt (`ncell=11 → 10,648` ions; `ncell=20 → 64,000` ions) in
nm / kJ·mol⁻¹ / ps / e units, rock-salt lattice thermally perturbed and relaxed.
"""
function make_nacl_melt(::Type{T}; ncell::Int=11, temp_K::Real=1200.0,
                        cutoff::Real=1.0, skin::Real=0.1, dt::Real=0.002,
                        vdw_modifier::Symbol=:none, electrostatics_spec=nothing,
                        seed::Int=2026) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    a = 0.60   # lattice constant in nm (liquid NaCl density ~ 1.5 g/cm³)
    L = ncell * a
    box = (L, L, L)
    basis = (
        (0.0, 0.0, 0.0,  1.0, Int32(1), 22.9898),
        (0.5, 0.5, 0.0,  1.0, Int32(1), 22.9898),
        (0.5, 0.0, 0.5,  1.0, Int32(1), 22.9898),
        (0.0, 0.5, 0.5,  1.0, Int32(1), 22.9898),
        (0.5, 0.0, 0.0, -1.0, Int32(2), 35.4530),
        (0.0, 0.5, 0.0, -1.0, Int32(2), 35.4530),
        (0.0, 0.0, 0.5, -1.0, Int32(2), 35.4530),
        (0.5, 0.5, 0.5, -1.0, Int32(2), 35.4530),
    )
    N = 8 * ncell^3
    R = Matrix{Float64}(undef, 3, N)
    q = Vector{Float64}(undef, N)
    ty = Vector{Int32}(undef, N)
    m = Vector{Float64}(undef, N)
    idx = 0
    for ix in 0:(ncell - 1), iy in 0:(ncell - 1), iz in 0:(ncell - 1), b in basis
        idx += 1
        x = (ix + b[1]) * a - 0.5 * L + 0.015 * randn(rng)
        y = (iy + b[2]) * a - 0.5 * L + 0.015 * randn(rng)
        z = (iz + b[3]) * a - 0.5 * L + 0.015 * randn(rng)
        R[1, idx] = x - L * round(x / L)
        R[2, idx] = y - L * round(y / L)
        R[3, idx] = z - L * round(z / L)
        q[idx] = b[4]; ty[idx] = b[5]; m[idx] = b[6]
    end
    kBT = KB_KJ_MOL_K * Float64(temp_K)
    V = randn(rng, 3, N) .* sqrt.(kBT ./ m')
    V .-= sum(V .* m'; dims=2) ./ sum(m)

    # Joung-Cheatham NaCl LJ parameters (nm, kJ/mol), Lorentz-Berthelot
    sig = [0.2439 0.3458; 0.3458 0.4477]
    eps = [0.3658 0.2340; 0.2340 0.1497]
    rcm = fill(Float64(cutoff), 2, 2)

    st = build_simulation(; N=N, box=T.(box), nonbonded=:lj, cutoff=T(cutoff), skin=T(skin),
                          cap=Int32(768), epsilon=T(maximum(eps)), sigma=T(maximum(sig)),
                          mass=T.(m), temperature=T(kBT), dt=T(dt),
                          precision=T === Float64 ? :f64 : :f32, charges=q,
                          spatial_reorder=false, nonbonded_backend=:cluster,
                          vdw_modifier=vdw_modifier)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    copyto!(st.typeid, ty)
    st.sigma_pair = CuArray(T.(sig))
    st.epsilon_pair = CuArray(T.(eps))
    st.rcut_pair = CuArray(T.(rcm))
    if electrostatics_spec !== nothing
        attach_electrostatics!(st, electrostatics_spec)
    end
    SimulationCore.invalidate_forces!(st)
    return st, (R=R, q=q, types=ty, masses=m, box=box, kBT=kBT, sigma_pair=sig, epsilon_pair=eps, exceptions=Tuple{Int,Int,Float64}[])
end

"""
    make_spce_water(::Type{T}; nx=22, ny=22, nz=21, temp_K=300.0, cutoff=1.0, skin=0.1,
                    vdw_modifier=:none, electrostatics_spec=nothing, seed=7)

Rigid 3-site SPC/E water box (`22×22×21 = 10,164` molecules = `30,492` atoms;
`32×32×32 = 32,768` molecules = `98,304` atoms) in nm / kJ·mol⁻¹ / ps / e units.
Automatically qualifies for `DeviceEngine.WaterFastPath`.
"""
function make_spce_water(::Type{T}; nx::Int=22, ny::Int=22, nz::Int=21,
                         temp_K::Real=300.0, cutoff::Real=1.0, skin::Real=0.1,
                         dt::Real=0.002, vdw_modifier::Symbol=:none,
                         electrostatics_spec=nothing, lz_scale::Real=1.0,
                         seed::Int=7) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    a = 0.3104   # ~0.998 g/cm³ liquid water density
    Lx = nx * a; Ly = ny * a; Lz = nz * a * Float64(lz_scale)
    box = (Lx, Ly, Lz)
    nw = nx * ny * nz
    N = 3 * nw
    oh = 0.10; hh = 0.16330
    theta = 2 * asin(hh / (2 * oh))

    R = Matrix{Float64}(undef, 3, N)
    triples = Vector{NTuple{3,Int}}(undef, nw)
    ex = Vector{Tuple{Int,Int,Float64}}(undef, 3 * nw)
    w = 0
    for ix in 0:(nx - 1), iy in 0:(ny - 1), iz in 0:(nz - 1)
        w += 1
        o = 3w - 2
        # For slab (lz_scale > 1), center the liquid film around z = 0
        ox = (ix + 0.5) * a - 0.5 * Lx + 0.008 * randn(rng)
        oy = (iy + 0.5) * a - 0.5 * Ly + 0.008 * randn(rng)
        oz = (iz + 0.5 - 0.5 * nz) * a + 0.008 * randn(rng)
        u = normalize(randn(rng, 3))
        p = normalize(cross(u, randn(rng, 3)))
        rO = SVector(ox, oy, oz)
        rH1 = rO + oh * SVector(u[1], u[2], u[3])
        rH2 = rO + oh * (cos(theta) * SVector(u[1], u[2], u[3]) + sin(theta) * SVector(p[1], p[2], p[3]))
        for (slot, r) in ((o, rO), (o + 1, rH1), (o + 2, rH2))
            R[1, slot] = r[1] - Lx * round(r[1] / Lx)
            R[2, slot] = r[2] - Ly * round(r[2] / Ly)
            R[3, slot] = r[3] - Lz * round(r[3] / Lz)
        end
        triples[w] = (o, o + 1, o + 2)
        ex[3w - 2] = (o, o + 1, 0.0)
        ex[3w - 1] = (o, o + 2, 0.0)
        ex[3w]     = (o + 1, o + 2, 0.0)
    end

    m = repeat([15.9994, 1.008, 1.008], nw)
    q = repeat([-0.8476, 0.4238, 0.4238], nw)
    ty = repeat(Int32[1, 2, 2], nw)
    kBT = KB_KJ_MOL_K * Float64(temp_K)
    V = randn(rng, 3, N) .* sqrt.(kBT ./ m')
    V .-= sum(V .* m'; dims=2) ./ sum(m)

    sig = [0.316557 0.0; 0.0 0.0]
    eps = [0.650629 0.0; 0.0 0.0]
    rcm = [Float64(cutoff) 0.0; 0.0 0.0]

    st = build_simulation(; N=N, box=T.(box), nonbonded=:lj, cutoff=T(cutoff), skin=T(skin),
                          cap=Int32(800), epsilon=T(0.650629), sigma=T(0.316557),
                          mass=T.(m), temperature=T(kBT), dt=T(dt),
                          precision=T === Float64 ? :f64 : :f32, charges=q,
                          spatial_reorder=false, neigh_interval=10_000,
                          nonbonded_backend=:cluster, vdw_modifier=vdw_modifier)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    copyto!(st.typeid, ty)
    st.sigma_pair = CuArray(T.(sig))
    st.epsilon_pair = CuArray(T.(eps))
    st.rcut_pair = CuArray(T.(rcm))
    if electrostatics_spec !== nothing
        attach_electrostatics!(st, electrostatics_spec; exceptions=ex)
    end
    attach_rigid_water!(st, triples; oh=oh, hh=hh)
    RW = ParticleDynamics.RigidWaterConstraints
    RW.constrain_velocities!(st.rigid_water.ws, st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, st.box3)
    SimulationCore.invalidate_forces!(st)
    return st, (R=R, q=q, types=ty, masses=m, box=box, kBT=kBT, sigma_pair=sig,
                epsilon_pair=eps, triples=triples, exceptions=ex, oh=oh, hh=hh)
end

"""
    make_spce_slab(::Type{T}; nx=16, ny=16, nz_film=20, lz_scale=3.0, kwargs...)

Inhomogeneous SPC/E liquid-vapor slab (`16×16×20 = 5,120` waters = `15,360` atoms
in an elongated `L_z = 3 × L_film` box with two vacuum interfaces).
"""
make_spce_slab(::Type{T}; nx::Int=16, ny::Int=16, nz_film::Int=20, lz_scale::Real=3.0, kwargs...) where {T} =
    make_spce_water(T; nx=nx, ny=ny, nz=nz_film, lz_scale=lz_scale, kwargs...)

"""
    make_charged_polymer_melt(::Type{T}; nchains=250, nbeads=32, cutoff=1.0, kwargs...)

Charged bead-spring polyelectrolyte + counterion melt (`32,000` total particles:
`250 × 32 = 8,000` harmonic-bonded chain beads, every 4th bead carrying `-1 e`, plus
`2,000` monovalent `+1 e` counterions and `22,000` neutral solvent beads) in nm / kJ·mol⁻¹ / ps / e.
"""
function make_charged_polymer_melt(::Type{T}; nchains::Int=250, nbeads::Int=32,
                                   nsolvent::Int=22_000, temp_K::Real=300.0,
                                   cutoff::Real=1.0, skin::Real=0.1, dt::Real=0.002,
                                   vdw_modifier::Symbol=:none, electrostatics_spec=nothing,
                                   seed::Int=42) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    npoly = nchains * nbeads
    ncounter = npoly ÷ 4
    N = npoly + ncounter + nsolvent
    rho = 24.0   # beads / nm³ (σ = 0.32 nm → ρ σ³ ≈ 0.79)
    L = cbrt(N / rho)
    box = (L, L, L)
    mimg(x) = x - L * round(x / L)

    R = Matrix{Float64}(undef, 3, N)
    q = zeros(Float64, N)
    bonds = Tuple{Int32,Int32}[]
    ex = Tuple{Int,Int,Float64}[]
    b_len = 0.33
    idx = 0
    for c in 1:nchains
        x = (rand(rng) - 0.5) * L; y = (rand(rng) - 0.5) * L; z = (rand(rng) - 0.5) * L
        for b in 1:nbeads
            idx += 1
            R[1, idx] = mimg(x); R[2, idx] = mimg(y); R[3, idx] = mimg(z)
            b % 4 == 0 && (q[idx] = -0.5)
            if b < nbeads
                push!(bonds, (Int32(idx), Int32(idx + 1)))
                push!(ex, (idx, idx + 1, 0.0))
            end
            u = normalize(randn(rng, 3))
            x += b_len * u[1]; y += b_len * u[2]; z += b_len * u[3]
        end
    end
    for k in 1:(ncounter + nsolvent)
        idx += 1
        R[1, idx] = (rand(rng) - 0.5) * L
        R[2, idx] = (rand(rng) - 0.5) * L
        R[3, idx] = (rand(rng) - 0.5) * L
        k <= ncounter && (q[idx] = +0.5)
    end
    @assert sum(q) == 0.0

    kBT = KB_KJ_MOL_K * Float64(temp_K)
    mass = 18.0
    V = randn(rng, 3, N) .* sqrt(kBT / mass)
    V .-= mean_vec(V)

    st = build_simulation(; N=N, box=T.(box), nonbonded=:lj, cutoff=T(cutoff), skin=T(skin),
                          cap=Int32(768), epsilon=T(0.65), sigma=T(0.32), mass=T(mass),
                          temperature=T(kBT), dt=T(dt),
                          bonds=bonds, bonding=ParticleDynamics.harmonic_bond(k=2500.0, r0=b_len),
                          exclude_bonded_pairs=true,
                          precision=T === Float64 ? :f64 : :f32, charges=q,
                          spatial_reorder=false, nonbonded_backend=:cluster,
                          vdw_modifier=vdw_modifier)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    if electrostatics_spec !== nothing
        attach_electrostatics!(st, electrostatics_spec; exceptions=ex)
    end
    # Brief soft/FIRE relaxation should be done by benchmark driver before NVE
    SimulationCore.invalidate_forces!(st)
    return st, (R=R, q=q, box=box, kBT=kBT, bonds=bonds, exceptions=ex)
end

mean_vec(V) = sum(V; dims=2) ./ size(V, 2)

end # module BenchWorkloads
