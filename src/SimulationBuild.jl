# =========================
#   Build simulation
# =========================
"""
    build_simulation(; N, box, cutoff=1, skin=0.4, cap=Int32(96),
                      neigh_interval=20, use_neighborlist=true,
                      epsilon=1, sigma=1, gamma=1, temperature=1,
                      noise_corr_time=nothing, dt=0.001,
                      mass=1, bonds=nothing, bonding=nothing,
                      nonbonded=:lj, softrep_params=nothing, backend=:cuda,
                      precision=:f32, unwrapped_positions::Bool=false)

Construct a [`SimulationState`](@ref) with GPU-resident SoA arrays and a
neighbor list configured for the requested potential. All inputs are keyword
arguments so that scripts can copy known-good parameter sets directly from the
`examples/` directory without ambiguity.

Key behaviors:
- The simulation dimensionality (2D vs 3D) is inferred from the length of
  `box`. Positions/velocities/forces allocate the corresponding CuArrays.
- For `nonbonded = :wca` the neighbor cutoff is forced to the physical WCA
  value `r_c = 2^(1/6) σ` even if a larger `cutoff` was passed, guaranteeing
  that kernels reuse the validated parameter sets from the packaged examples.
- `temperature` accepts either a scalar or length-`N` vector and is used to
  initialize velocities. Stochastic integrator parameters are constructed
  later via explicit specs such as [`velocityverlet`](@ref) and
  [`eulermaruyama`](@ref).
- Initial velocities are drawn from a Maxwell–Boltzmann distribution at
  `temperature` and then centered to remove center-of-mass drift.
- When `unwrapped_positions=true`, additional `rx_unwrap`/`ry_unwrap`/`rz_unwrap`
  buffers track continuous positions across periodic boundaries.

Example (mirrors `examples/2D_example.jl`, scaled down to N=4096 for testing):

```julia
st = build_simulation(N=4096, box=(250f0, 250f0),
                      cutoff=Float32(2^(1/6)), skin=Float32(2^(1/6))/2,
                      cap=Int32(250), neigh_interval=50,
                      epsilon=1f4, sigma=1f0,
                      gamma=615f0, temperature=10f0,
                      dt=1f-5, nonbonded=:wca, precision=:f32)
vv = velocityverlet(st; gamma=615f0, temperature=10f0, dt=1f-5)
step!(st, vv, 1f-5; compute_energy=false)
```

Returns a fully initialized `SimulationState`. Stochastic controls such as
`noise_corr_time` now belong to the explicit integrator spec rather than the
core simulation state.
"""
function build_simulation(;N::Int,
                           box,
                           cutoff::Real=1.0,
                           skin::Real=0.4,
                           cap::Int32=Int32(96),
                           neigh_interval::Int=20,
                           use_neighborlist::Bool=true,
                           epsilon::Real=1,
                           sigma::Real=1,
                           gamma::Union{AbstractVector{<:Real},Real,Nothing}=1,
                           temperature::Union{AbstractVector{<:Real},Real}=1,
                           noise_corr_time::Union{AbstractVector{<:Real},Real,Nothing}=nothing,
                           dt::Real=0.001,
                           mass::Real=1,
                           bonds::Union{Nothing,Vector{Tuple{Int32,Int32}}}=nothing,
                           bonding::Union{Nothing,Definitions.BondPotential}=nothing,
                           nonbonded::Symbol = :lj,
                           softrep_params::Union{Nothing,Definitions.SoftRepulsiveParams{<:Real}}=nothing,
                           backend::Union{Backends.AbstractBackend,Symbol}=:cuda,
                           precision::Symbol = :f32,
                           unwrapped_positions::Bool = false)

    backend_impl = Backends.normalize_backend(backend)
    Backends.ensure_available(backend_impl)

    D = length(box)

    if precision == :f32
        T = Float32
    elseif precision == :f64
        T = Float64
    else
        error("Unknown precision=$(precision). Use :f32 or :f64")
    end

    epsilonT = T(epsilon)
    sigmaT   = T(sigma)
    requested_cutoff = T(cutoff)
    nb_cutoff = (nonbonded === :wca) ? (sigmaT * T(WCA_RC_FACTOR)) : requested_cutoff
    rcut_factor = sigmaT == zero(T) ? T(1) : nb_cutoff / sigmaT

    rx = Backends.allocate_vector(backend_impl, T, N); ry = Backends.allocate_vector(backend_impl, T, N)
    vx = Backends.allocate_vector(backend_impl, T, N); vy = Backends.allocate_vector(backend_impl, T, N)
    fx = Backends.allocate_vector(backend_impl, T, N); fy = Backends.allocate_vector(backend_impl, T, N)
    rz = nothing; vz = nothing; fz = nothing
    rx_unwrap = unwrapped_positions ? Backends.allocate_vector(backend_impl, T, N) : nothing
    ry_unwrap = unwrapped_positions ? Backends.allocate_vector(backend_impl, T, N) : nothing
    rz_unwrap = nothing

    f0x = Backends.allocate_vector(backend_impl, T, N)
    f0y = Backends.allocate_vector(backend_impl, T, N)
    f0z = nothing

    if D == 3
        rz  = Backends.allocate_vector(backend_impl, T, N)
        vz  = Backends.allocate_vector(backend_impl, T, N)
        fz  = Backends.allocate_vector(backend_impl, T, N)
        f0z = Backends.allocate_vector(backend_impl, T, N)
        if unwrapped_positions
            rz_unwrap = Backends.allocate_vector(backend_impl, T, N)
        end
    end

    fill!(rx, zero(T)); fill!(ry, zero(T))
    rz === nothing || fill!(rz, zero(T))
    rx_unwrap === nothing || fill!(rx_unwrap, zero(T))
    ry_unwrap === nothing || fill!(ry_unwrap, zero(T))
    rz_unwrap === nothing || fill!(rz_unwrap, zero(T))

    fill!(fx, zero(T)); fill!(fy, zero(T)); fz === nothing || fill!(fz, zero(T))
    fill!(f0x, zero(T)); fill!(f0y, zero(T)); f0z === nothing || fill!(f0z, zero(T))

    temperature_vec = _device_particle_buffer(backend_impl, T, N, temperature, "temperature")

    if D == 2
        _init_vel2!(vx, vy, temperature_vec)
    else
        _init_vel3!(vx, vy, vz, temperature_vec)
    end

    if N > 0
        Vx = Backends.sum_elements(backend_impl, vx) / T(N)
        Vy = Backends.sum_elements(backend_impl, vy) / T(N)
        @. vx = vx - Vx
        @. vy = vy - Vy
        if D == 3
            Vz = Backends.sum_elements(backend_impl, vz) / T(N)
            @. vz = vz - Vz
        end
    end

    typeid = Backends.fill_vector(backend_impl, Int32(1), N)

    if use_neighborlist
        if D == 2
            nbh = NeighborLists.build_neighbors_dense!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=nb_cutoff, cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_dense!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=nb_cutoff, cap, skin=T(skin))
        end
    else
        if D == 2
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=nb_cutoff, cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=nb_cutoff, cap, skin=T(skin))
        end
    end

    lj = Definitions.LJParams{T}(epsilonT, sigmaT, nb_cutoff)
    noise_corr_time === nothing ||
        throw(ArgumentError("build_simulation no longer accepts `noise_corr_time`; pass it to an explicit integrator constructor such as velocityverlet(st; gamma=..., temperature=..., noise_corr_time=..., dt=...)."))

    Epot = Backends.allocate_vector(backend_impl, T, N); fill!(Epot, zero(T))
    dq   = Backends.allocate_vector(backend_impl, T, N); fill!(dq, zero(T))
    dU   = Backends.allocate_vector(backend_impl, T, N); fill!(dU, zero(T))
    Ekin = Backends.allocate_vector(backend_impl, T, N); fill!(Ekin, zero(T))
    virial = Backends.allocate_vector(backend_impl, T, N); fill!(virial, zero(T))
    nvirial = D == 2 ? 3 : 6
    virial_nonbonded = Backends.allocate_matrix(backend_impl, T, N, nvirial); fill!(virial_nonbonded, zero(T))
    virial_bonded = Backends.allocate_matrix(backend_impl, T, N, nvirial); fill!(virial_bonded, zero(T))
    virial_tensor = Backends.allocate_matrix(backend_impl, T, N, nvirial); fill!(virial_tensor, zero(T))
    Epot_accum = Backends.allocate_vector(backend_impl, T, N); fill!(Epot_accum, zero(T))
    Ekin_accum = Backends.allocate_vector(backend_impl, T, N); fill!(Ekin_accum, zero(T))
    virial_accum = Backends.allocate_vector(backend_impl, T, N); fill!(virial_accum, zero(T))
    virial_tensor_accum = Backends.allocate_matrix(backend_impl, T, N, nvirial); fill!(virial_tensor_accum, zero(T))

    local bondlist
    if bonds === nothing
        bondlist = nothing
    else
        bondlist = BondedForces.build_bondlist(N, bonds)
    end

    local nb_tag::UInt8
    local srp::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
    if nonbonded === :lj
        nb_tag = NB_KIND_LJ
        srp = nothing
    elseif nonbonded === :wca
        nb_tag = NB_KIND_WCA
        srp = nothing
    elseif nonbonded === :soft_repulsive || nonbonded === :softrep || nonbonded === :soft
        nb_tag = NB_KIND_SOFTREP
        srp = softrep_params === nothing ? Definitions.SoftRepulsiveParams{T}(epsilonT, sigmaT) : softrep_params
    else
        error("Unknown nonbonded=:$(nonbonded). Use :lj, :wca, or :soft_repulsive")
    end

    local bond_spec
    if bonding !== nothing
        if bonding isa Definitions.HarmonicBond
            p = (bonding::Definitions.HarmonicBond).params
            bond_spec = Definitions.HarmonicBond{T}(Definitions.HarmonicBondParams{T}(T(p.k), T(p.r0)))
        elseif bonding isa Definitions.FENEBond
            p = (bonding::Definitions.FENEBond).params
            bond_spec = Definitions.FENEBond{T}(Definitions.FENEParams{T}(T(p.k), T(p.R0)))
        else
            bond_spec = nothing
        end
    else
        bond_spec = nothing
    end

    st = SimulationState(rx, ry, rz, rx_unwrap, ry_unwrap, rz_unwrap, vx, vy, vz, fx, fy, fz,
                         f0x, f0y, f0z,
                         typeid,
                         nothing,
                         nothing,
                         nbh, neigh_interval, lj,
                         nothing, rcut_factor,
                         nothing, nothing, nothing,
                         bondlist, bond_spec,
                         T(mass), T(dt),
                         Epot, dq, dU, Ekin, virial, virial_nonbonded, virial_bonded, virial_tensor,
                         Epot_accum, Ekin_accum, virial_accum, virial_tensor_accum,
                         0, UInt8(0), nb_tag, srp,
                         FREEZE_NONE, -1, true, nothing, zero(T), nothing, nothing, nothing,
                         false, nothing, nothing, nothing)

    if D == 2
        st.box2 = (T(box[1]), T(box[2]))
        st.box3 = nothing
    else
        st.box2 = nothing
        st.box3 = (T(box[1]), T(box[2]), T(box[3]))
    end

    return st
end
