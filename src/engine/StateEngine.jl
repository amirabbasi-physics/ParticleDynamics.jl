# Included in ParticleDynamics.DeviceEngine. Public SimulationState integration
# and force evaluation replay captured CUDA graphs over the state's own arrays.

"""Per-case settings measured for Argon at
N = 1e4/1e5/1e6 and number density 0.05/0.3/0.8 (LJ, rc = 2.5σ). Classes in
between are provisional: the rule is validated only at the measured points."""
const ENGINE_SETTINGS = Dict(
    # (size class, density class) => (builder lanes, force lanes, skin, fused, cell_sub)
    (1,1)=>(8,8,1.00,true,1), (1,2)=>(8,8,0.70,true,1), (1,3)=>(8,8,0.35,true,1), (1,4)=>(16,8,0.35,true,2),
    (2,1)=>(1,2,0.55,true,1), (2,2)=>(1,4,0.35,false,1),(2,3)=>(8,2,0.35,true,1), (2,4)=>(8,2,0.35,true,1),
    (3,1)=>(1,1,0.55,true,1), (3,2)=>(1,2,0.45,true,1), (3,3)=>(8,2,0.45,false,1),(3,4)=>(8,2,0.45,false,1))

# Classes use expected neighbors per atom, ρ·4π/3·rc³, so the LJ (rc = 2.5)
# measurements carry over to other cutoffs: the measured densities 0.05/0.3/0.8
# give 3.3/19.6/52.4 neighbors; class borders at density 0.15/0.55 give 9.8/36.0.
# Class 4 (≥ 72 neighbors, the geometric mean of Argon 10k's 55 and molten NaCl 10.6k's 101):
# long rows, where 16 builder lanes and cells of half the list radius (fewer candidates per
# row) shorten the rebuild. Rows are bitwise identical across builder lanes;
# cell_sub changes the visit order (forces at summation noise). Measured at N ≈ 1e4 only:
# larger sizes keep the class-3 settings.
# Skin scales with rc/2.5 (provisional for non-LJ potentials until timed).
# Spread visit order of ELL charged runs: particle t → ((t−1)·S mod N) + 1 with S coprime to N
# near N/φ, for grids that stay in L2 (≤ SPREAD_STRIDE_BYTES); identity otherwise or with
# SPREAD_STRIDE[] = false. Sorted storage puts the particles of one warp and block next to each
# other, so their atomics contend for the same nodes; on a grid larger than L2 the strided order
# loses the locality instead (salt, ms/step: ESP 24³ at 10k 0.1169 → 0.1117, ESP 48³ at 97k
# 0.635 → 0.609, PME 144³ (12 MB) at 97k 0.737 → 0.909, PME 64³ at 10k unchanged).
const SPREAD_STRIDE=Ref(true)
const SPREAD_STRIDE_BYTES=2^21
function _spread_order(N,grid_bytes=0)
    SPREAD_STRIDE[] && N>2 && grid_bytes<=SPREAD_STRIDE_BYTES || return Int32.(1:N)
    S=round(Int,N/1.618033988749895)
    while gcd(S,N)!=1; S+=1; end
    return Int32[mod((t-1)*S,N)+1 for t in 1:N]
end

# Benchmark override of the policy skin (NaN: the table's). Tuning aid, not an API.
const ENGINE_SKIN=Ref(NaN)
# Device cell grid: cells of width rlist/cell_sub (when the box holds 2·cell_sub+1 per dimension).
# Benchmark override of the policy's cell_sub (0: the table's). Tuning aid, not an API.
const CELL_SUB=Ref(0)
# Neighbor-list diagnostics (candidate and accepted-pair counts, build causes) of state engines:
# off by default — they feed only `NL.device_neighbor_diagnostics` and cost an extra ballot per
# candidate chunk plus statistics kernels (salt 10k: 176 → 144 µs per build).
const ENGINE_DIAGNOSTICS=Ref(false)

function engine_settings(N::Integer,density::Real;cutoff::Real=2.5,dims::Integer=3)
    # 2D: area density, disc of radius rc (provisional: classes measured in 3D)
    neighbors=dims==3 ? density*4pi/3*cutoff^3 : density*pi*cutoff^2
    size_class=N<=30_000 ? 1 : N<=300_000 ? 2 : 3
    density_class=neighbors<9.8 ? 1 : neighbors<36.0 ? 2 : neighbors<72.0 ? 3 : 4
    b,f,s,u,c=ENGINE_SETTINGS[(size_class,density_class)]
    dims==2 && (b=1; u=false; c=1) # cooperative builder, fusion and finer cells are 3D-only
    return (;build_lanes=b,force_lanes=f,skin=s*cutoff/2.5,fused=u,cell_sub=c)
end

# Charged states on the dense list (15 Oct 2026): charges, a mesh solver without Coulomb
# exceptions (excluded pairs run on the cluster backend), 3D.
_ell_charged_ok(st)=st.electrostatics!==nothing && st.charge!==nothing && st.rz!==nothing &&
    length(st.electrostatics.exj)==0

"""Device pair model, its coefficient arrays and the neighbor-list cutoff
(largest pair cutoff) for the state's nonbonded description."""
function pair_model(st::SC.SimulationState{T}) where {T}
    m,data,rc=_lj_pair_model(st)
    s=st.electrostatics
    (s===nothing || st.nonbonded_backend===:cluster) && return m,data,rc
    coul=ClusterPairs.coulomb_polynomial(s.spec.split,Float64(s.C);T=T,cutoff=s.spec.real_cutoff,eshift=Float64(s.eshift))
    return ChargedPairs(m,coul,T(0.5*Float64(s.C)*Float64(s.selfc))),(data...,st.charge),max(rc,Float64(s.spec.real_cutoff))
end

function _lj_pair_model(st::SC.SimulationState{T}) where {T}
    lj=st.pair_lj
    # No pair term: the list only serves features; keep its radius within a quarter of the box
    # (ideal gases under a barostat shrink below the nominal cutoff).
    st.nb_kind==SC.NB_KIND_NONE && return NoPairs(),(),
        min(Float64(st.nbh isa NL.NeighborMatrix ? st.nbh.cutoff : lj.rcut),minimum(Float64.(_box(st)))/4)
    if st.nb_kind==SC.NB_KIND_SOFTREP
        sr=st.softrep
        return UniformSoftRepulsive(sr.ϵ,sr.σ),(),Float64(sr.σ)
    end
    wca=st.nb_kind==SC.NB_KIND_WCA
    shift=SC._lj_shifted(st)   # potential-shift modifier (LJ only)
    if st.sigma_pair!==nothing
        return (wca ? PairMatrixWCA() : PairMatrixLJ{shift}()),(st.typeid,st.sigma_pair,st.epsilon_pair,st.rcut_pair),
            Float64(maximum(st.rcut_pair))
    elseif st.sigma_particle!==nothing
        return (wca ? MixedSigmaWCA{T}(lj.ϵ,st.rcut_factor) :
                MixedSigmaLJ{T}(lj.ϵ,st.rcut_factor,shift ? lj_cutoff_energy(one(T),lj.ϵ,st.rcut_factor) : zero(T))),
            (st.sigma_particle,),Float64(st.rcut_factor)*Float64(maximum(st.sigma_particle))
    elseif wca
        m=UniformWCA(lj.ϵ,lj.σ)
        return m,(),max(Float64(lj.rcut),sqrt(Float64(m.rc2)))
    end
    return UniformLJ{T}(lj.ϵ,lj.σ,lj.rcut^2,shift ? lj_cutoff_energy(lj.σ,lj.ϵ,lj.rcut) : zero(T)),(),Float64(lj.rcut)
end

"""Configurations the device engine runs: 2D/3D, every integrator family,
on the dense ELL list with LJ or WCA (uniform, mixed σ or pair-matrix
coefficients) or uniform soft repulsion; unwrapped coordinates,
per-particle masses, freezing, collision counting, legacy
harmonic/FENE bonds, typed bonded families and bonded exclusions, sorted or not;
external force providers with a split graph per step (see
`external_supported`); the cluster-pair backend with or without electrostatics (see `cluster_supported`); small boxes with all-pairs rows; states with typed
stencil lists, whose per-type cutoffs the pair models apply on one list of the
largest radius."""
function engine_supported(st::SC.SimulationState,spec)
    spec isa SC.NVESpec || langevin_supported(st,spec) || csvr_supported(st,spec) || nhc_supported(st,spec) ||
        brownian_supported(st,spec) || return false
    st.external_potential===nothing || return external_supported(st)
    st.nonbonded_backend===:cluster && return cluster_supported(st,spec)
    kind_ok=st.nb_kind in (SC.NB_KIND_LJ,SC.NB_KIND_WCA,SC.NB_KIND_NONE) ||
        (st.nb_kind==SC.NB_KIND_SOFTREP && st.softrep!==nothing && st.sigma_pair===nothing && st.sigma_particle===nothing)
    pairs_ok=st.sigma_pair===nothing || (st.epsilon_pair!==nothing && st.rcut_pair!==nothing)
    _pressure_control_reason(st,spec)===nothing || return false   # device pressure control
    return (st.rz!==nothing ? st.box3!==nothing : st.box2!==nothing) && kind_ok && pairs_ok &&
        st.nonbonded_backend===:ell && st.nbh isa Union{NL.NeighborMatrix,NL.AllPairsNeighborMatrix,NL.StencilNeighborMatrix} &&
        bonded_supported(st) &&
        (st.electrostatics===nothing || _ell_charged_ok(st)) && st.external_potential===nothing &&
        _rigid_ok(st,spec) && st.cluster===nothing &&
        _list_feasible(st)
end

"""Why the device engine does not take `st` with `spec` (`nothing`: it does). Every
public step runs on the engine (no staged fallback), so these become errors."""
function engine_unsupported_reason(st::SC.SimulationState,spec)
    r=_integrator_reason(st,spec); r===nothing || return r
    if st.external_potential!==nothing
        external_supported(st) && return nothing
        return "external force providers need an unsorted state without bonds, electrostatics, rigid water or " *
            "the cluster backend; collision counting with a provider needs the cell-list backend"
    end
    r=_pressure_control_reason(st,spec); r===nothing || return r
    if st.nonbonded_backend===:cluster
        cluster_supported(st,spec) && return nothing
        try SC._cluster_supported(st) catch err; return sprint(showerror,err) end
        bonded_supported(st) || return _bonded_reason(st)
        _rigid_ok(st,spec) || return "rigid water runs with NVE or CSVR"
        return "this cluster-pair configuration (charges without a solver?) is not supported"
    end
    (st.rz!==nothing ? st.box3!==nothing : st.box2!==nothing) || return "the state has no periodic box"
    st.nb_kind==SC.NB_KIND_SOFTREP && (st.sigma_pair!==nothing || st.sigma_particle!==nothing) &&
        return "soft repulsion supports uniform parameters only (no pair matrices or per-particle σ)"
    st.sigma_pair===nothing || (st.epsilon_pair!==nothing && st.rcut_pair!==nothing) ||
        return "pair-matrix coefficients need sigma_pair, epsilon_pair and rcut_pair"
    (st.electrostatics===nothing || _ell_charged_ok(st)) ||
        return "long-range electrostatics with Coulomb exceptions run on the cluster-pair backend (set_nonbonded_backend!(st, :cluster))"
    st.nbh isa Union{NL.NeighborMatrix,NL.AllPairsNeighborMatrix,NL.StencilNeighborMatrix} ||
        return "unknown neighbor container $(typeof(st.nbh))"
    bonded_supported(st) || return _bonded_reason(st)
    _rigid_ok(st,spec) || return "rigid water runs with NVE or CSVR"
    _list_feasible(st) || return "the pair cutoff exceeds half the box (no periodic image is unique)"
    return engine_supported(st,spec) ? nothing : "unsupported configuration"
end
_bonded_reason(st)="bonded terms: legacy harmonic or FENE bonds and typed BondedTerms are supported; got $(typeof(st.bonding))"
function _integrator_reason(st,spec)
    spec isa SC.NVESpec && return nothing
    if spec isa LangevinSpec || spec isa BrownianFamily
        p=spec.params
        (p.corr_time===nothing || p.ou!==nothing) || return "coloured noise needs the OU workspace (set_ou_spectrum!)"
        length(p.gamma)==length(st.rx)==length(p.noise_scale) || return "the integrator was built for another particle count"
        spec isa LangevinSpec && p.mass!=st.mass && return "the integrator's mass differs from the state's"
        _friction_ok(spec,p.gamma) || return spec isa SC.VVSpec ? "friction must be nonnegative for every particle" :
            "friction must be positive for every particle"
        return nothing
    elseif spec isa SC.CSVRSpec
        p=spec.params; B=length(p.target_temperature)
        B<=CSVR_MAX_BATHS || return "at most $(CSVR_MAX_BATHS) CSVR baths"
        csvr_supported(st,spec) && return nothing
        return "CSVR needs positive targets and τ per bath and the state's mass"
    elseif spec isa SC.NHCSpec
        p=spec.params; B=length(p.target_temperature)
        B<=CSVR_MAX_BATHS || return "at most $(CSVR_MAX_BATHS) Nosé–Hoover baths"
        p.mass==st.mass || return "the integrator's mass differs from the state's"
        try SC.validate_integrator_inputs!(spec,st,st.dt) catch err; return sprint(showerror,err) end
        nhc_supported(st,spec) && return nothing
        return "unsupported Nosé–Hoover propagator $(p.propagator)"
    end
    return "integrator $(nameof(typeof(spec))) is not available"
end

"""Why the device pressure control cannot run on `st` with `spec` (`nothing`: it can)."""
function _pressure_control_reason(st,spec)
    st.pressure_control===nothing && return nothing
    if st.nonbonded_backend!==:cluster   # ELL (device pressure control on the dense list): 3D pair potentials without topology
        st.rz===nothing && return "device pressure control on the dense list is 3D"
        (st.bonds===nothing && st.bonded===nothing && (st.electrostatics===nothing || _ell_charged_ok(st)) &&
         st.rigid_water===nothing && st.freeze_mask===nothing) ||
            return "device pressure control with bonds, electrostatics, rigid water or freezing runs on the cluster-pair backend (set_nonbonded_backend!(st, :cluster))"
    end
    st.coll_enabled && return "device pressure control does not support collision counting"
    st.external_potential===nothing || return "device pressure control does not support external force providers"
    spec isa Union{SC.NVESpec,SC.CSVRSpec,SC.NHCSpec,SC.VVSpec,SC.BAOABSpec,SC.GSMSpec} ||
        return "device pressure control supports NVE, CSVR, Nosé–Hoover and the Langevin (BAOAB/GSM/VV) integrators"
    s=st.electrostatics
    (s===nothing || (s.bg==0 && s.ubg==0)) || return "device pressure control needs a neutral system (the mesh background term is not refreshed on the device)"
    return nothing
end

"""cluster-pair states (`set_nonbonded_backend!(st, :cluster)`): 3D, LJ (uniform or
pair table) or no vdW term, with or without an electrostatics solver; the staged
backend's own restrictions (no mixed σ, no bonded vdW exclusions, no collisions)
plus no rigid water and no external provider. Euler–Heun runs two cluster
evaluations per step."""
function cluster_supported(st::SC.SimulationState,spec)
    st.rz!==nothing && st.box3!==nothing || return false
    _pressure_control_reason(st,spec)===nothing || return false
    (try SC._cluster_supported(st); true catch; false end) || return false
    return bonded_supported(st) && _rigid_ok(st,spec) && st.external_potential===nothing &&
        (st.electrostatics===nothing || st.charge!==nothing)
end

# Rigid water supports NVE and CSVR integration.
_rigid_ok(st,spec)=st.rigid_water===nothing || spec isa Union{SC.NVESpec,SC.CSVRSpec}

const LangevinSpec=Union{SC.BAOABSpec,SC.GSMSpec,SC.BAOASpec,SC.VVSpec}

"""BAOAB/GSM with white noise (no OU/coloured noise), positive friction and
the state's uniform mass. Per-particle γ and temperature are supported."""
function langevin_supported(st::SC.SimulationState,spec)
    spec isa LangevinSpec || return false
    p=spec.params
    return (p.corr_time===nothing || p.ou!==nothing) && length(p.gamma)==length(st.rx)==length(p.noise_scale) &&
        p.mass==st.mass && _friction_ok(spec,p.gamma)
end
# Velocity Verlet (GJF) is well defined without friction (γ = 0: plain velocity Verlet);
# BAOAB, BAOA and GSM need γ > 0 (their staged validation says so).
_friction_ok(spec,gamma)=spec isa SC.VVSpec ? minimum(gamma)>=0 : minimum(gamma)>0

"""CSVR with up to $(CSVR_MAX_BATHS) baths, positive targets and τ, the state's uniform mass."""
function csvr_supported(st::SC.SimulationState,spec)
    spec isa SC.CSVRSpec || return false
    p=spec.params; B=length(p.target_temperature)
    return 1<=B<=CSVR_MAX_BATHS && length(p.tau)==B && all(>(0),p.target_temperature) && all(>(0),p.tau) &&
        p.mass==st.mass
end

"""NHC (any propagator, chain length and substeps) with up to $(CSVR_MAX_BATHS) baths and the
state's uniform mass."""
function nhc_supported(st::SC.SimulationState,spec)
    spec isa SC.NHCSpec || return false
    p=spec.params; B=length(p.target_temperature)
    p.propagator in (SC.NHC_PROPAGATOR_LEGACY,SC.NHC_PROPAGATOR_GROMACS,SC.NHC_PROPAGATOR_LAMMPS) || return false
    return 1<=B<=CSVR_MAX_BATHS && p.mass==st.mass && (try SC.validate_integrator_inputs!(spec,st,st.dt); true catch; false end)
end

const BrownianFamily=Union{SC.BrownianSpec,SC.EMSpec}

"""Euler–Maruyama and Euler–Heun with white noise and positive friction;
per-particle γ and temperature are supported."""
function brownian_supported(st::SC.SimulationState,spec)
    spec isa BrownianFamily || return false
    p=spec.params
    return (p.corr_time===nothing || p.ou!==nothing) && length(p.gamma)==length(st.rx)==length(p.noise_scale) &&
        minimum(p.gamma)>0
end

# The dense cell list needs >= 3 cells per dimension: cell size rc+skin <= L/3.
_max_skin(st)=0.999*(minimum(Float64.(_box(st)))/3-pair_model(st)[3])
# All-pairs rows: the list radius must stay within half the box (one image per pair).
_max_allpairs_skin(st)=0.999*(minimum(Float64.(_box(st)))/2-pair_model(st)[3])
_list_feasible(st)=_max_skin(st)>0 || _max_allpairs_skin(st)>=0
const ALLPAIRS_MAX_N=32_768
"""Graph launches between automatic host boundaries in `step!` (device status check,
capacity recovery). Above the 500 traced steps of a residency audit, which start at a
boundary."""
const BOUNDARY_LAUNCHES=1000
"""Neighbor rows for a box: cells with the desired skin while at least 3 cells per
side fit it; all pairs when the cell skin would be under a quarter of the desired one
(the list would be rebuilt almost every step) or no cell grid fits; else cells with
the largest fitting skin. All-pairs rows cost O(N²) per rebuild, so they are used up
to `ALLPAIRS_MAX_N` particles while a cell grid remains possible."""
function _list_mode(st,skin)
    cells=_max_skin(st)
    cells>=skin && return (false,skin)
    pairs=_max_allpairs_skin(st)
    if cells<=0 || (cells<0.25*skin && length(st.rx)<=ALLPAIRS_MAX_N && pairs>=0)
        return (true,max(min(skin,pairs),0.0))
    end
    return (false,cells)
end
_box(st)=st.rz===nothing ? st.box2 : st.box3
_coords(st)=st.rz===nothing ? (st.rx,st.ry) : (st.rx,st.ry,st.rz)
_velocities(st)=st.rz===nothing ? (st.vx,st.vy) : (st.vx,st.vy,st.vz)
_forces(st)=st.rz===nothing ? (st.fx,st.fy) : (st.fx,st.fy,st.fz)
_old_forces(st)=st.rz===nothing ? (st.f0x,st.f0y) : (st.f0x,st.f0y,st.f0z)

mutable struct StateEngine
    run::Any
    plain::Any
    sampled::Any
    fingerprint::Any
    spec::Any          # integrator spec of the capture (fingerprint checks outside step!)
    settings::NamedTuple
    launched::Int      # graph launches since capture
    device_step0::Int  # device step counter at capture
    host_call::Bool    # inside the host force call of a split step
    flags::Vector{Int8} # graph of every launch since the last boundary: 0 plain, 1 sampled, 2 pressure event
                        # (capacity recovery replays them)
    recoveries::Int    # capacity recoveries so far
    graph_box::Any     # box the graphs captured (box changes)
    stale::Bool        # the box differs from graph_box, or list storage was reallocated since capture
    stale_storage::Bool
    moved::Bool        # forces invalid only because `set_box!` moved the cell (and scaled coordinates) on the engine
    event::Any         # step graph with a device pressure event, or nothing
    virial::Any        # step graph before an event: virial trace only (kind 3), or nothing
    fast::Any          # plain step without list decisions between scheduled steps (HOST_SCHEDULE), or nothing
    sort::Any          # host-scheduled sort step of a fused ELL run (HOST_SORT), or nothing
    sort_due::Int      # device step of the next sort step
    tune::Any          # mesh-priority choice of the fast graph (MeshTune), or nothing
end

"""Choice of the fast step graph's mesh priority (`MESH_PRIORITY_TUNE`): after `MESH_TUNE_WARM`
fast launches, `MESH_TUNE_ROUNDS` pairs of `MESH_TUNE_BLOCK`-launch blocks alternate the
high-priority variant (`alt`) and the regular one, each launch timed by device events; the
variant with the lower median duration becomes `e.fast`. The physics is the same either way."""
mutable struct MeshTune
    alt::Any                    # high-priority fast graph (until decided)
    n::Int                      # fast launches seen
    samples::Vector{Any}        # (start event, stop event, high)
    decided::Bool
    high::Bool                  # the choice once decided
end
MeshTune(alt)=MeshTune(alt,0,Any[],false,false)
const MESH_TUNE_WARM=40
const MESH_TUNE_BLOCK=8
const MESH_TUNE_ROUNDS=4

_fast_high(e)=e.tune===nothing ? MESH_PRIORITY[] : e.tune.decided && e.tune.high

function _tuned_launch!(t::MeshTune,e)
    t.n+=1
    k=t.n-MESH_TUNE_WARM
    k<=0 && (CUDA.launch(e.fast); return nothing)
    high=isodd(cld(k,MESH_TUNE_BLOCK))
    a=CUDA.CuEvent(); b=CUDA.CuEvent()
    CUDA.record(a); CUDA.launch(high ? t.alt : e.fast); CUDA.record(b)
    push!(t.samples,(a,b,high))
    k<2*MESH_TUNE_BLOCK*MESH_TUNE_ROUNDS && return nothing
    CUDA.synchronize(b)
    med(h)=(x=sort([CUDA.elapsed(s[1],s[2]) for s in t.samples if s[3]==h]); x[cld(length(x),2)])
    t.high=med(true)<med(false)
    t.high && (e.fast=t.alt)
    t.alt=nothing; empty!(t.samples); t.decided=true
    return nothing
end

_particle_arrays(st)=(st.rx,st.ry,st.rz,st.vx,st.vy,st.vz,st.fx,st.fy,st.fz,st.f0x,st.f0y,st.f0z,
    st.Epot,st.dq,st.dU,st.Ekin,st.virial,st.Epot_accum,st.Ekin_accum,st.virial_accum,
    st.virial_nonbonded,st.virial_bonded,st.virial_tensor,st.virial_tensor_accum,st.typeid,st.tag)

# Anything a captured graph depends on: buffer addresses and physical/graph
# parameters, and the integrator (its kind and, for Langevin, its arrays).
_spec_signature(spec::SC.NVESpec)=(:nve,)
# Keyed on the parameter arrays, not the spec object: `step!(st, ::BAOABParams, dt)`
# wraps the same parameters in a new spec on every call.
_spec_signature(spec::LangevinSpec)=(typeof(spec).name.name,pointer(spec.params.gamma),
    pointer(spec.params.noise_scale),spec.params.mass,spec.params.dt)
function _spec_signature(spec::SC.CSVRSpec)
    ws=spec.workspace; p=spec.params
    return (:csvr,map(pointer,(ws.particle_bath_id,ws.target_temperature,ws.tau,ws.kinetic_total_per_bath,
        ws.cumulative_energy_exchange_per_bath,ws.last_velocity_scale_per_bath,ws.dof_per_bath)),
        Tuple(p.target_temperature),Tuple(p.tau),p.mass)
end
function _spec_signature(spec::SC.NHCSpec)
    ws=spec.workspace; p=spec.params
    return (:nhc,map(pointer,(ws.xi,ws.eta,ws.chain_force,ws.chain_masses,ws.target_temperature,ws.particle_bath_id,
        ws.dof_per_bath,ws.kinetic_total_per_bath,ws.kinetic_stage_start_per_bath,ws.cumulative_energy_exchange_per_bath,
        ws.thermostat_kinetic_per_bath,ws.thermostat_potential_per_bath,ws.last_velocity_scale_per_bath)),
        Tuple(p.target_temperature),Tuple(p.tau),p.substeps,p.chain_length,Tuple(p.chain_masses),p.propagator,p.mass)
end
_spec_signature(spec::BrownianFamily)=(typeof(spec).name.name,pointer(spec.params.gamma),
    pointer(spec.params.noise_scale),spec.params.dt)
_spec_signature(spec)=(:unsupported,objectid(spec))
_fingerprint(st,dt,spec)=(_spec_signature(spec),feature_signatures(st,spec),_fingerprint(st,dt)...)
_fingerprint(st,dt)=(map(a->a===nothing ? nothing : pointer(a),_particle_arrays(st)),
    map(a->a===nothing ? nothing : pointer(a),(st.sigma_particle,st.sigma_pair,st.epsilon_pair,st.rcut_pair)),
    length(st.rx),_box(st),st.pair_lj,st.nb_kind,st.softrep,st.rcut_factor,st.vdw_modifier,st.mass,dt,st.reorder_interval,
    _feature_signature(st))

# Everything `engine_supported` inspects besides coefficients: attaching any of
# these changes the fingerprint and forces a fresh support check.
_feature_signature(st)=(st.rz===nothing,st.nonbonded_backend,typeof(st.nbh),st.mass_particle===nothing,
    st.bonds===nothing,st.bonding===nothing,st.bonded===nothing,st.charge===nothing,st.electrostatics===nothing,
    st.external_potential===nothing,st.freeze_mask===nothing,st.coll_enabled,st.rigid_water===nothing,
    st.rx_unwrap===nothing,st.cluster===nothing,
    st.cluster===nothing ? nothing : objectid(st.cluster),st.electrostatics===nothing ? nothing : objectid(st.electrostatics),
    st.pressure_control===nothing ? nothing : objectid(st.pressure_control))

_thermostat(st,spec::SC.NVESpec)=nothing
function _thermostat(st,spec::SC.CSVRSpec)
    SC.ensure_integrator_workspace!(spec,st) # sizes/copies the per-bath arrays, as each staged step does
    ws=spec.workspace; B=length(spec.params.target_temperature)
    return CSVRData(ws.particle_bath_id,ws.target_temperature,ws.tau,ws.kinetic_total_per_bath,
        ws.cumulative_energy_exchange_per_bath,ws.last_velocity_scale_per_bath,ws.dof_per_bath,st.Ekin,
        CUDA.zeros(Float64,B),CUDA.zeros(Int32,B),rand(UInt64),CuArray(_constraint_dof_per_bath(st,ws.particle_bath_id,B)),
        CUDA.zeros(Float64,B),CUDA.zeros(Int32,B),Int32(spec.params.interval))
end

# 3 constraints per rigid water, counted in the bath of its atoms (a water split across
# baths is rejected because each constrained molecule must belong to one bath).
function _constraint_dof_per_bath(st,bath_id,B)
    c=zeros(Float64,B)
    st.rigid_water===nothing && return c
    ids=Array(bath_id)
    for (o,h1,h2) in Array(st.rigid_water.workspace.atoms)
        b=ids[o]
        ids[h1]==b && ids[h2]==b || throw(ArgumentError("A rigid water's atoms must share one thermostat bath"))
        1<=b<=B && (c[b]+=3)
    end
    return c
end
function _thermostat(st,spec::SC.NHCSpec)
    SC.ensure_integrator_workspace!(spec,st)
    SC._nhc_update_dof_per_bath!(spec,st)        # dof for the first pre-stage
    SC._ensure_nhc_kinetic_initialized!(spec,st) # K for the first pre-stage (host flag, once)
    B=length(spec.params.target_temperature)
    return NHCData(spec,st.Ekin,CUDA.zeros(Float64,B),CUDA.zeros(Int32,B))
end
_thermostat(st,spec::BrownianFamily)=BrownianData(spec isa SC.BrownianSpec,spec.params.gamma,
    spec.params.noise_scale,st.dq,st.dU,st.tag,rand(UInt64))
_scheme(::Union{SC.BAOABSpec,SC.GSMSpec})=LANGEVIN_BAOAB
_scheme(::SC.BAOASpec)=LANGEVIN_BAOA
_scheme(::SC.VVSpec)=LANGEVIN_GJF
_thermostat(st,spec::LangevinSpec)=LangevinData(_scheme(spec),spec isa SC.VVSpec ? _old_forces(st) : nothing,
    spec.params.gamma,spec.params.noise_scale,st.dq,st.dU,st.Ekin,
    st.tag,rand(UInt64)) # seed from the host RNG: `Random.seed!` makes runs reproducible

"""Capture an engine for `st`. With `evaluation=true` the engine only evaluates
forces outside `step!`: no graphs, no fusion and no spatial sorting (callers such as
the minimizer keep slot-indexed data), NVE features, and a fingerprint that never
matches a stepping spec, so the first `step!` captures a stepping engine."""
function _capture_engine(st::SC.SimulationState{T},dt::T,spec=SC.nve(st);evaluation::Bool=false) where {T}
    st.nonbonded_backend===:cluster && return _capture_cluster_engine(st,dt,spec;evaluation)
    N=length(st.rx); box=_box(st); D=length(box); density=N/prod(Float64.(box))
    model,model_data,list_cutoff=pair_model(st)
    s=engine_settings(N,density;cutoff=list_cutoff,dims=D)
    evaluation && (s=merge(s,(;fused=false)))
    isnan(ENGINE_SKIN[]) || (s=merge(s,(;skin=ENGINE_SKIN[])))
    CELL_SUB[]>0 && (s=merge(s,(;cell_sub=CELL_SUB[])))
    thermostat=_thermostat(st,spec)
    features=make_features(st,spec)  # Features.jl registry (unwrap, masses, freeze, OU, …)
    host=any(host_forces,features)   # forces from the host between graph segments
    # single-bath CSVR fuses too (its half-step K in the prefix, the scaled kick in the force kernel)
    ((thermostat===nothing || (thermostat isa CSVRData && nbaths(thermostat)==1 && D==3)) &&
        all(f->fusable(f) || (model isa ChargedPairs && f isa MassFeature),features) && !(model isa NoPairs)) ||
        (s=merge(s,(;fused=false))) # other thermostats, most features and pair-free states run unfused
    # the fused gather-kick differentiates the potential grid (16-lane AD gather): P3M-ik (and the
    # one-thread gathers) run unfused, through `mesh_gather!`
    (st.electrostatics===nothing || ES.gather_sets(st.electrostatics)) || (s=merge(s,(;fused=false)))
    # Spec-owned per-particle arrays are permuted with the particles by sorting.
    spec_real=thermostat isa Union{LangevinData,BrownianData} ? (thermostat.gamma,thermostat.noise) : ()
    spec_int=thermostat isa CSVRData ? (thermostat.bath_id,) :
        thermostat isa NHCData ? (spec.workspace.particle_bath_id,) : ()
    lj=st.pair_lj
    allpairs,skin_=_list_mode(st,s.skin)            # small boxes: all-pairs rows
    allpairs && st.pressure_control!==nothing && !evaluation &&
        throw(ArgumentError("Device pressure control needs a cell grid (the box is too small for the list radius)"))
    s=merge(s,(;skin=skin_)); allpairs && (s=merge(s,(;build_lanes=1)))
    skin=T(s.skin)
    reach=list_cutoff+s.skin
    # Uniform-density estimate; the user's `cap` (scaled to this radius) is a floor.
    cap=max(ceil(Int,1.5*(D==3 ? 4pi/3*reach^3 : pi*reach^2)*density+64),_user_cap(st,reach,D))
    combine=()->SC._combine_virial!(st)
    shared=(f=_forces(st),energy=st.Epot,virial=st.virial_nonbonded,tags=st.tag,
        extra_real=(_old_forces(st)...,st.dq,st.dU,st.Ekin,st.virial,st.Epot_accum,st.Ekin_accum,st.virial_accum,
            (st.sigma_particle===nothing ? () : (st.sigma_particle,))...,(st.charge===nothing ? () : (st.charge,))...,
            spec_real...,features_sorted_real(features)...),
        extra_int=(st.typeid,spec_int...,features_sorted_int(features)...),extra_mat=(st.virial_bonded,st.virial_tensor,st.virial_tensor_accum),
        slot_of=_topology_map(st)[2])   # sorted bonded states: id → slot, refreshed at every sort
    pc=st.pressure_control
    baro=(evaluation || pc===nothing) ? nothing : DeviceCRescale(st,pc.control,pc.interval,dt)
    mesh=(st.electrostatics===nothing || st.nonbonded_backend===:cluster) ? nothing :
        (solver=st.electrostatics,q=st.charge,Vglobal=st.virial_global,order=CuArray(Int32.(1:N)),
         sorder=CuArray(_spread_order(N,prod(st.electrostatics.ws.mesh)*sizeof(T))),   # spread visit order
         xq=CuArray{NTuple{4,VecElement{T}}}(undef,N))   # packed (x, y, z, q), written by the spread
    make(cap)=Run(_coords(st),_velocities(st),box;cutoff=T(list_cutoff),skin,cap,dt,mass=st.mass,barostat=baro,mesh,
        sigma=lj.σ,epsilon=lj.ϵ,step=st.step,spatial_reorder=!evaluation && st.tag!==nothing && !allpairs,allpairs,
        reorder_interval=st.reorder_interval,force_lanes=s.force_lanes,build_lanes=s.build_lanes,
        rebuild_policy=:displacement,fused=s.fused,shared,sample_hook=host ? nothing : combine,model,model_data,
        thermostat,features,cell_sub=s.cell_sub,diagnostics=ENGINE_DIAGNOSTICS[])
    # Size capacity from the measured maximum row, with headroom for later
    # density fluctuations; the estimate assumes uniform density. A row that
    # still overflows during a run latches an error reported at the next boundary.
    run=try
        make(cap)
    catch err
        err isa NL.NeighborCapacityError || rethrow()
        nothing
    end
    required=run===nothing ? cap_required(st,T(list_cutoff),skin,box;allpairs) : NL.device_neighbor_status(run.neighbors;check=false).required
    target=max(cap,ceil(Int,1.3*required)+16)
    if run===nothing || target>cap
        run=nothing; GC.gc(); CUDA.reclaim()
        run=make(target)
    end
    # Compile the hook's kernel outside capture (recomputes st.virial; harmless). The
    # staged provider path never combines the virial, so host-force runs have no hook;
    # their first forces (and energies) come from the provider at the current positions.
    host ? host_force!(_host_feature(run),st,true) : combine()
    CUDA.synchronize()
    plain,sampled=evaluation ? (nothing,nothing) : capture_graphs(run)
    event=(evaluation || baro===nothing) ? nothing : capture_event_graph(run)
    sortg=(evaluation || !host_sorted(run)) ? nothing : capture_sort_graph(run)
    status=NL.device_neighbor_status(run.neighbors)
    fp=evaluation ? _evaluation_fingerprint(st,dt) : _fingerprint(st,dt,spec)
    due=sortg===nothing ? 0 : status.step+run.ordering.interval
    return StateEngine(run,plain,sampled,fp,evaluation ? nothing : spec,s,0,status.step,false,Int8[],0,box,false,false,false,event,nothing,nothing,
        sortg,due,nothing)
end
"""Cluster-pair runs: the cluster backend supplies pair (and mesh) forces; the ELL
workspace is the control block. Its list is built only when a feature reads contact
rows (collision counting, the vdW cutoff plus the cluster skin)."""

function _capture_cluster_engine(st::SC.SimulationState{T},dt::T,spec;evaluation::Bool=false) where {T}
    N=length(st.rx); box=st.box3
    eng=SC._cluster_engine!(st)
    CP.setup!(eng,box)
    pc=st.pressure_control
    baro=(evaluation || pc===nothing) ? nothing : DeviceCRescale(st,pc.control,pc.interval,dt)
    backend=ClusterBackend(eng,st.electrostatics,st.charge,st.typeid,st.electrostatics===nothing ? nothing : st.virial_global;
        barostat=baro)
    thermostat=_thermostat(st,spec)
    features=make_features(st,spec)
    spec_real=thermostat isa Union{LangevinData,BrownianData} ? (thermostat.gamma,thermostat.noise) : ()
    spec_int=thermostat isa CSVRData ? (thermostat.bath_id,) :
        thermostat isa NHCData ? (spec.workspace.particle_bath_id,) : ()
    combine=()->SC._combine_virial!(st)
    shared=(f=_forces(st),energy=st.Epot,virial=st.virial_nonbonded,tags=st.tag,
        extra_real=(_old_forces(st)...,st.dq,st.dU,st.Ekin,st.virial,st.Epot_accum,st.Ekin_accum,st.virial_accum,
            spec_real...,features_sorted_real(features)...),
        extra_int=(st.typeid,spec_int...,features_sorted_int(features)...),extra_mat=(st.virial_bonded,st.virial_tensor,st.virial_tensor_accum))
    lj=st.pair_lj
    keep=any(needs_neighbors,features)   # collision counting: real ELL contact rows
    if keep
        ccut=Float64(st.rcut_pair===nothing ? st.pair_lj.rcut : maximum(Array(st.rcut_pair)))
        cskin=Float64(eng.skin); reach=ccut+cskin; density=N/prod(Float64.(box))
        lap=any(L->floor(L/reach)<3,Float64.(box))   # small boxes: all-pairs contact rows
        lap && 2reach>minimum(Float64.(box)) && throw(ArgumentError(
            "Collision counting needs the contact radius $(reach) within half the box $(box)"))
        lcut,lskin=T(ccut),T(cskin); lcap=min(N,ceil(Int,1.5*4pi/3*reach^3*density+64))
    else
        lcut,lskin,lcap,lap=T(minimum(Float64.(box))/4),zero(T),1,false   # control block only: any valid cell grid
    end
    run=Run(_coords(st),_velocities(st),box;cutoff=lcut,skin=lskin,cap=lcap,dt,mass=st.mass,
        sigma=lj===nothing ? one(T) : lj.σ,epsilon=lj===nothing ? one(T) : lj.ϵ,step=st.step,
        spatial_reorder=false,rebuild_policy=:displacement,fused=false,shared,sample_hook=combine,
        thermostat,features,cluster=backend,allpairs=lap)
    # one device box for the run: the cluster list reads the workspace's geometry
    eng.list.geom=run.neighbors.geom
    if keep   # first contact rows (grown here if the estimate is short), then the flags
        ws=run.neighbors
        _wrap_into_box!(st;contact_rows=true)   # rows need centered coordinates; drifts keep them so
        # every reference defined (the control block is marked valid; request_device_rebuild!
        # offsets only x): uninitialized y/z references could read as NaN in the decision
        foreach(copyto!,_refs(ws.list,Val(length(box))),_coords(st))
        for _ in 1:4
            NL.request_device_rebuild!(ws,_coords(st)); decide_policy!(ws,run.r,Val(:displacement))
            enqueue_rebuild!(run;reorder=false)
            status=NL.device_neighbor_status(ws;check=false)
            status.error==NL.DN_CAPACITY_ERROR || break
            NL.resize_capacity!(ws,ceil(Int,1.3*status.required)+16); NL.rearm_capacity_rebuild!(ws)
        end
        NL.device_neighbor_status(ws)   # any other latched error throws here, at capture
    end
    enqueue_force!(run;energy=false)   # first list (sized here) and forces
    combine()
    CUDA.synchronize()
    plain,sampled=evaluation ? (nothing,nothing) : capture_graphs(run)
    event=baro===nothing ? nothing : capture_event_graph(run)
    virial=(baro===nothing || !virial_trace_ok(run)) ? nothing : capture_virial_graph(run)
    status=NL.device_neighbor_status(run.neighbors)
    fp=evaluation ? _evaluation_fingerprint(st,dt) : _fingerprint(st,dt,spec)
    fast=(evaluation || !host_scheduled(run)) ? nothing : capture_fast_graph(run)
    tune=(fast===nothing || MESH_PRIORITY[] || !MESH_PRIORITY_TUNE[] || run.cluster.solver===nothing) ? nothing :
        MeshTune(capture_fast_graph(run;high=true))
    s=(;build_lanes=1,force_lanes=1,skin=Float64(eng.skin),fused=false,cell_sub=1)
    return StateEngine(run,plain,sampled,fp,evaluation ? nothing : spec,s,0,status.step,false,Int8[],0,box,false,false,false,event,virial,fast,
        nothing,0,tune)
end

_evaluation_fingerprint(st,dt)=(:evaluation,feature_signatures(st,SC.nve(st)),_fingerprint(st,dt)...)
"""The fingerprint the engine was captured with, recomputed for the current state."""
_engine_fingerprint(st,e)=e.spec===nothing ? _evaluation_fingerprint(st,e.run.dt) : _fingerprint(st,e.run.dt,e.spec)

"""The user's list capacity (`build_simulation(cap=…)`, sized for the staged radius
rc + skin), scaled to the engine's list radius `reach`. This is a floor, so
users who expect their system to densify can reserve room."""
_user_cap(st,reach,D)=st.nbh isa NL.NeighborMatrix ?
    ceil(Int,Int(st.nbh.cap)*(reach/(Float64(st.nbh.cutoff)+Float64(st.nbh.skin)))^D) : 0

# Required capacity after a failed estimate: build a throwaway list with ample
# capacity at the same radius (construction-time host boundary only).
function cap_required(st,cutoff,skin,box;allpairs::Bool=false)
    N=length(st.rx); T=eltype(st.rx)
    ws=NL.DeviceNeighborWorkspace(T,N,box;cutoff,skin=T(skin),cap=N,step=st.step,allpairs)
    NL.decide_device_neighbors!(ws,_coords(st)...)
    refs=_refs(ws.list,Val(length(box)))
    NL._device_rebuild!(ws,_coords(st),refs)
    return NL.device_neighbor_status(ws).required
end

"""Check device status at a host boundary: a neighbor-capacity overflow is repaired
(`_recover_capacity!`), other latched errors throw here, and the device step
counter must match the graph launches issued since capture."""
function SC._engine_boundary!(e::StateEngine,st)
    status=NL.device_neighbor_status(e.run.neighbors;check=false)
    status.error==NL.DN_CAPACITY_ERROR &&
        (e.run.cluster===nothing ? _recover_capacity!(e,st,status) : _recover_cluster!(e,st,status))
    features_at_boundary!(e.run)   # e.g. OU active_idx back to storage slots
    status=NL.device_neighbor_status(e.run.neighbors) # throws on latched errors
    status.step==e.device_step0+e.launched ||
        error("Device engine step counter $(status.step) differs from launches $(e.device_step0+e.launched)")
    # Ordinary cluster steps may leave bath totals lazy; public observations and
    # state hand-offs must see the completed-step K, including after recovery.
    csvr_lazy_report(e.run) && !st.energy_valid && enqueue_csvr_report!(e.run,e.run.thermostat)
    empty!(e.flags)
    _sync_pressure_control!(e,st)
    return nothing
end

"""Device pressure control at a host boundary: the host box mirrors and the barostat
counters follow the device (one small read); the fingerprint follows the box, so no
recapture is triggered."""
function _sync_pressure_control!(e::StateEngine,st)
    _has_barostat(e.run) || return nothing
    bar=_baro_ctx(e.run).barostat
    sync_device_box!(e,st)
    buf=Array(bar.buf); pc=st.pressure_control
    if pc!==nothing
        cr=pc.control
        cr.events=Int(buf[_BARO_EVENTS]); cr.volume_change=buf[_BARO_DV]
        cr.last_pressure=buf[_BARO_EVENTS]>0 ? buf[_BARO_P] : NaN
    end
    e.fingerprint=_engine_fingerprint(st,e)
    return nothing
end

"""Capacity recovery. A list that overflowed since the last
boundary stopped the device in the failed step's rebuild: after that step's drift
(and any due sort), with every later launch skipped. Grow the list from the
device-reported requirement, recapture the graphs on the same run (seeds,
thermostat and feature state and the sorting schedule are unchanged), redo the
failed rebuild and finish that step from its section, then replay the skipped
steps with their recorded energy flags. Noise is keyed on the device step, so the
trajectory is the one an ample list would have produced."""
function _recover_capacity!(e::StateEngine,st,status)
    run=e.run; ws=run.neighbors
    lost=e.launched-(status.step-e.device_step0)
    1<=lost<=length(e.flags) ||
        error("Device engine capacity recovery: $lost skipped steps, $(length(e.flags)) recorded")
    run.traversal.packed===nothing || error("Capacity recovery supports the ELL layout")
    flags=e.flags[end-lost+1:end]; section=NL.section_of(ws)
    old=Int(ws.list.cap); cap=max(ceil(Int,1.3*status.required)+16,ceil(Int,1.25*old))
    @info "Device engine: neighbor capacity $old → $cap (the system densified); replaying $lost step(s)"
    NL.resize_capacity!(ws,cap)
    _recapture!(e)   # the graphs captured the old storage
    NL.rearm_capacity_rebuild!(ws)
    sampled(f)=f in (1,2)   # 4: the sort step (no energies), replayed as captured
    g=sampled(flags[1]) ? e.sampled : e.plain
    if g isa SplitStep
        segs,_,_=split_plan(run); j,k=split_section(segs,section)
        _launch_split!(g,e,st;first=j,resume=resume_graph(run,segs[j][1],k,segs[j][2]))
    else
        sections,suffix=step_plan(run)
        CUDA.launch(resume_graph(run,sections,section,suffix(sampled(flags[1]))))
    end
    for f in flags[2:end]
        g=f==4 ? e.sort : sampled(f) ? e.sampled : e.plain
        g isa SplitStep ? _launch_split!(g,e,st) : CUDA.launch(g)
    end
    e.recoveries+=1
    return nothing
end

"""Cluster-pair capacity recovery: the outer list overflowed in the rebuild of the
failed step (after its drift). Grow it from the required entries, recapture, redo
that step from its decision (the drift already ran) and replay the skipped steps."""
function _recover_cluster!(e::StateEngine,st,status)
    run=e.run; ws=run.neighbors; L=run.cluster.eng.list
    lost=e.launched-(status.step-e.device_step0)
    1<=lost<=length(e.flags) ||
        error("Device engine capacity recovery: $lost skipped steps, $(length(e.flags)) recorded")
    flags=e.flags[end-lost+1:end]; section=NL.section_of(ws)
    c=Array(L.counts)
    if c[CP.CL_ERR]==CP.CL_ERR_CAPACITY   # the cluster outer list overflowed
        old=L.necap; need=max(Int(status.required),Int(c[CP.CL_REQ]))
        CP._alloc_entries!(L,ceil(Int,1.25*need)+1024)
        @info "Device engine: cluster-pair list capacity $old → $(L.necap) entries; replaying $lost step(s)"
        CP.request_build!(L)
        control_valid!(ws)
        _recapture!(e)
    else                                   # the ELL contact rows (collision counting)
        old=Int(ws.list.cap); cap=max(ceil(Int,1.3*status.required)+16,ceil(Int,1.25*old))
        @info "Device engine: contact-row capacity $old → $cap; replaying $lost step(s)"
        NL.resize_capacity!(ws,cap)
        _recapture!(e)
        NL.rearm_capacity_rebuild!(ws)
    end
    # the failed step resumes at its rebuild section (a pressure event before it already ran)
    CUDA.launch(CUDA.instantiate(cluster_step_graph(run,run.cluster;energy=flags[1]==1,virial=flags[1]==3,resume=section)))
    for f in flags[2:end]
        CUDA.launch(f==3 ? e.virial : f==2 ? e.event : f==1 ? e.sampled : e.plain)
    end
    e.recoveries+=1
    return nothing
end

"""Evaluate forces and optionally energies/virials at the current coordinates using
the engine that owns the state. The captured fingerprint and tether must match;
invalidated forces require a known coordinate-only move or an evaluation engine.
Otherwise the caller creates a new evaluation engine. The list is checked and
rebuilt when needed without sorting, so slot-indexed data remains valid.
Device errors (for example the FENE domain check) surface at the closing boundary."""
function SC._engine_evaluate!(e::StateEngine,st::SC.SimulationState,energy::Bool,spring::Bool;
                              moved::Bool=false,create::Bool=false)
    run=e.run
    # Evaluation-only engines (configuration evaluation): invalid forces only mean moved coordinates; the
    # fingerprint covers every structural change that also invalidates forces.
    ((moved || e.moved || st.force_valid || e.spec===nothing) && !e.host_call && spring==_tether_active(run) &&
        e.fingerprint==_engine_fingerprint(st,e)) || return false
    e.spec===nothing && !st.force_valid && _wrap_into_box!(st)   # evaluation-only: user edits
    try
        SC._engine_boundary!(e,st)    # earlier steps: device status, capacity recovery
        run.cluster===nothing && _list_needed(run) && _evaluation_list!(e)
        if _host_forces(run)
            e.host_call=true          # a provider must not evaluate through the state
            try
                host_force!(_host_feature(run),st,energy)
            finally
                e.host_call=false
            end
            features_after_force!(run;energy)
        else
            enqueue_force!(run;energy)
            energy && run.sample_hook!==nothing && run.sample_hook()
        end
        SC._engine_boundary!(e,st)    # device checks (e.g. the FENE domain) throw here, as staged
    catch
        _abandon_engine!(e,st)        # latched device state: the next evaluation starts afresh
        rethrow()
    end
    st.force_valid=true; st.energy_valid=energy; st.force_freeze_spring=spring; e.moved=false
    return true
end

"""Public evaluations of a state without an engine create an evaluation-only engine
(configuration evaluation; `create`, the default of `evaluate_current_configuration!`): no graphs,
no sorting, slot order unchanged. No staged fallback: coordinates a user
placed outside the box are wrapped to their image inside it first (`_wrap_into_box!`)."""
function SC._engine_evaluate!(::Nothing,st::SC.SimulationState{T},energy::Bool,spring::Bool;
                              moved::Bool=false,create::Bool=false) where {T}
    # Untethered forces of a tethered state (the tether is a restraint): an evaluation
    # engine captured with the freeze control off (its fingerprint differs, so the next
    # tethered evaluation or step captures again).
    if create && !spring && _freeze_active(st) && st.freeze_mode==SC.FREEZE_SPRING
        mode=st.freeze_mode; st.freeze_mode=SC.FREEZE_NONE
        try
            return SC._engine_evaluate!(nothing,st,energy,false;moved,create)
        finally
            st.freeze_mode=mode
        end
    end
    (create && _evaluation_eligible(st,spring)) || return false
    _wrap_into_box!(st)
    st.engine=_capture_engine(st,T(st.dt),SC.nve(st);evaluation=true)
    return SC._engine_evaluate!(st.engine,st,energy,spring;moved=true)
end

# Periodic image inside the engine's centered box [-L/2, L/2] for coordinates outside it
# (user edits between evaluations); coordinates already inside are untouched. Cluster-pair
# states take any image, so they are left as they are.
function _wrap_into_box_kernel!(x,L::T) where {T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(x) && return nothing
    @inbounds begin
        v=x[i]; h=T(0.5)*L
        (isfinite(v) && (v<-h || v>h)) && (x[i]=v-L*round(v/L))
    end
    return nothing
end
function _wrap_into_box!(st;contact_rows::Bool=false)
    # cluster pairs take any image; their ELL contact rows (collision counting) do not
    st.nonbonded_backend===:cluster && !contact_rows && return nothing
    for (x,L) in zip(_coords(st),_box(st))
        @cuda threads=256 blocks=cld(length(x),256) _wrap_into_box_kernel!(x,eltype(x)(L))
    end
    return nothing
end
_evaluation_eligible(st,spring)=isfinite(st.dt) && st.dt>0 && st.external_potential===nothing &&
    spring==(_freeze_active(st) && st.freeze_mode==SC.FREEZE_SPRING) && engine_supported(st,SC.nve(st))

"""Drop an engine whose device state latched an error (or that cannot follow a new
cell): the next step or evaluation captures afresh from the state, whose arrays the
engine shared; forces are invalid."""
function _abandon_engine!(e::StateEngine,st)
    st.engine===e || return nothing
    st.engine=nothing; _dispose!(e)
    st.force_valid=false; st.energy_valid=false
    return nothing
end
SC._engine_invalidated!(e::StateEngine)=(e.moved=false; nothing)
"""Settle the engine's pending steps and discard it (no staged list is rebuilt)."""
const _REENTRANT_PROVIDER="An external force provider must not evaluate forces through the state or step it " *
    "(called from `external_forces!` during a device-engine step)"
function SC._drop_engine!(e::StateEngine,st)
    st.engine===e || return nothing
    e.host_call && error(_REENTRANT_PROVIDER)
    try
        SC._engine_boundary!(e,st)
    finally
        st.engine=nothing; _dispose!(e)
    end
    return nothing
end
function SC._evaluation_refusal(st::SC.SimulationState,spring::Bool)
    (isfinite(st.dt) && st.dt>0) || return "the state's dt must be finite and positive"
    spring==(_freeze_active(st) && st.freeze_mode==SC.FREEZE_SPRING) ||
        return "the requested freeze tether ($(spring)) differs from the state's freeze control"
    return something(engine_unsupported_reason(st,SC.nve(st)),"unsupported configuration")
end
_tether_active(run)=any(f->f isa FreezeFeature && f.active && f.mode==FREEZE_SPRING_ENGINE,run.features)
SC._engine_neighbor_rows(e::StateEngine)=(e.run.neighbors.list.neighbors_flat,e.run.neighbors.list.counts)

"""Box change on the engine (`set_box!`, e.g. barostat events): the device
geometry follows and the next coverage check rebuilds the list. Graphs hold the box
in their kernel arguments, so they are re-instantiated lazily at the next `step!`; a
rejected trial restores the captured box and needs none, unless cell storage was
reallocated. Collision counting follows: the requested rebuild re-initializes the
contact flags from the re-snapshot positions. A box the engine's cell grid cannot
take drops the engine (the next step captures one with a smaller skin or all-pairs
rows)."""
function SC._engine_set_box!(e::StateEngine,st,box)
    try
        SC._engine_boundary!(e,st)    # earlier steps settle first (recovery replays them in the old box)
        if e.run.cluster!==nothing   # cluster list geometry; the control block keeps the volume
            realloc=CP.setup!(e.run.cluster.eng,box)
            bar=e.run.cluster.barostat   # device pressure control: grid reference = this setup
            bar===nothing || copyto!(view(bar.buf,_BARO_REF:_BARO_REF+2),Float64.(collect(box)))
            NL.set_device_box!(e.run.neighbors,box,_coords(st))
        else
            realloc=NL.set_device_box!(e.run.neighbors,box,_coords(st))
            e.run.barostat===nothing || copyto!(view(e.run.barostat.buf,_BARO_REF:_BARO_REF+2),Float64.(collect(box)))
        end
        e.stale_storage|=realloc
        e.stale=e.stale_storage || box!=e.graph_box
        e.fingerprint=_engine_fingerprint(st,e)
        e.moved=!st.force_valid
    catch err
        err isa ArgumentError || rethrow()
        _abandon_engine!(e,st)
    end
    return nothing
end
"""Recapture every step graph of `e` (regular, sampled and the pressure-event graph)."""
function _recapture!(e;warm::Bool=true)
    e.plain,e.sampled=capture_graphs(e.run;warm)
    e.event===nothing || (e.event=capture_event_graph(e.run))
    e.virial===nothing || (e.virial=capture_virial_graph(e.run))
    e.fast===nothing || (e.fast=capture_fast_graph(e.run;high=_fast_high(e)))
    if e.tune!==nothing && !e.tune.decided   # restart the choice on the new graphs
        e.tune.alt=capture_fast_graph(e.run;high=true); e.tune.n=0; empty!(e.tune.samples)
    end
    e.sort===nothing || (e.sort=capture_sort_graph(e.run))
    return nothing
end

function _refresh_graphs!(e::StateEngine,st)
    SC._engine_boundary!(e,st)
    _recapture!(e)
    e.graph_box=_box(st); e.stale=false; e.stale_storage=false
    return nothing
end

"""List maintenance for an evaluation outside `step!`: the displacement check at the
current coordinates and, when it requests one, a rebuild without sorting. An
overflow here is outside any step (step recovery cannot replay it), so it is
repaired here: grow the list, recapture the graphs, rebuild again."""
function _evaluation_list!(e::StateEngine)
    run=e.run; ws=run.neighbors
    decide_policy!(ws,run.r,Val(run.rebuild_policy))
    NL.device_neighbor_status(ws;check=false).requested || return nothing
    enqueue_rebuild!(run;reorder=false)
    status=NL.device_neighbor_status(ws;check=false)
    status.error==NL.DN_CAPACITY_ERROR || return nothing
    old=Int(ws.list.cap); cap=max(ceil(Int,1.3*status.required)+16,ceil(Int,1.25*old))
    @info "Device engine: neighbor capacity $old → $cap (the system densified; evaluation outside step!)"
    NL.resize_capacity!(ws,cap)
    e.plain===nothing || _recapture!(e)   # evaluation-only engines have none
    NL.rearm_capacity_rebuild!(ws)
    enqueue_rebuild!(run;reorder=false)
    return nothing
end

"""Test hook: shrink the list to the current maximum row and rebuild it now, so the
next growth of a row overflows (capacity recovery tests). The first rebuild measures
the rows at the current positions (the last build's maximum may be smaller, and a
rebuild that overflows outside a step cannot be recovered); the second fills the
tightened list. Graphs are recaptured."""
function tighten_capacity!(st)
    e=st.engine; run=e.run; ws=run.neighbors
    NL.rearm_capacity_rebuild!(ws); enqueue_rebuild!(run)
    NL.resize_capacity!(ws,max(1,NL.device_neighbor_status(ws).required))
    NL.rearm_capacity_rebuild!(ws); enqueue_rebuild!(run)
    _recapture!(e)
    return st
end
"""Test hook (cluster-pair runs): shrink the outer list to half the current need, so
the next rebuild (requested here) overflows and is recovered at the next boundary.
Graphs are recaptured."""
function tighten_cluster_capacity!(st)
    e=st.engine; L=e.run.cluster.eng.list
    need=Int(Array(L.counts)[CP.CL_NE])
    CP._alloc_entries!(L,max(1,need÷2)); CP.request_build!(L)
    _recapture!(e;warm=false)   # no eager build: the graph overflows
    return st
end

"""Test hook: the same two extra rebuilds as `tighten_capacity!`, without shrinking."""
function force_rebuild!(st)
    e=st.engine; run=e.run
    for _ in 1:2
        NL.rearm_capacity_rebuild!(run.neighbors); enqueue_rebuild!(run)
    end
    return st
end

"""Free the GPU workspaces a replaced engine owns (never the state's arrays), in stream
order on the task stream. Otherwise their finalizers run later on other threads and can
land inside a graph capture of the replacing engine, invalidating it (seen 13 Oct once R3
phase 2 made every first evaluation create an evaluation-only engine)."""
function _dispose!(e::StateEngine)
    run=e.run
    # the state's cluster list outlives the run: it keeps its own copy of the device box
    if run.cluster!==nothing && run.cluster.eng.list.geom===run.neighbors.geom
        run.cluster.eng.list.geom=copy(run.neighbors.geom)
    end
    for x in (run.neighbors,run.neighbors.list,run.traversal,run.observables,run.fusion)
        x===nothing || _free_fields!(x)
    end
    run.cluster===nothing || run.cluster.order===nothing || CUDA.unsafe_free!(run.cluster.order)
    e.plain=nothing; e.sampled=nothing; e.event=nothing; e.virial=nothing; e.fast=nothing; e.sort=nothing; e.tune=nothing
    return nothing
end
function _free_fields!(x)
    for f in fieldnames(typeof(x))
        v=getfield(x,f)
        if v isa CuArray
            CUDA.unsafe_free!(v)
        elseif v isa AbstractVector && eltype(v)<:CuArray
            foreach(CUDA.unsafe_free!,v)
        end
    end
    return nothing
end

"""Hand the state back to the staged path."""

SC.step!(st::SC.SimulationState{T},spec::SC.NVESpec{T},dt::Real;compute_energy::Bool=true) where {T<:AbstractFloat}=
    _engine_step!(st,spec,dt,compute_energy)
SC.step!(st::SC.SimulationState{T},spec::Union{SC.BAOABSpec{T},SC.GSMSpec{T},SC.BAOASpec{T},SC.VVSpec{T},SC.CSVRSpec{T},SC.NHCSpec{T},SC.BrownianSpec{T},SC.EMSpec{T}},dt::Real;compute_energy::Bool=true) where {T<:AbstractFloat}=
    _engine_step!(st,spec,dt,compute_energy)

function _engine_step!(st::SC.SimulationState{T},spec,dt::Real,compute_energy::Bool) where {T<:AbstractFloat}
    dtT=T(dt)
    all_before_step!(st)   # host-only feature hooks (e.g. an expired freeze turns off, as staged)
    isfinite(dtT) && dtT>zero(T) || throw(ArgumentError("dt must be finite and positive"))
    e=st.engine
    e isa StateEngine && e.host_call && error(_REENTRANT_PROVIDER)
    # A cell change (`set_box!`) only moved the cell and coordinates the engine follows:
    # forces at the new cell come from the engine instead of a recapture.
    (e isa StateEngine && e.moved && !st.force_valid && e.fingerprint==_fingerprint(st,dtT,spec)) &&
        SC._engine_evaluate!(e,st,false,_tether_active(e.run))
    e=st.engine
    # Fast path: host-only fingerprint (addresses, parameters, feature signature).
    # The support check (which may reduce coefficient arrays on the GPU) runs only
    # when the fingerprint changes, i.e. when a capture is due anyway.
    # `invalidate_forces!` (manual position/velocity edits) also recaptures: the
    # graph's first kick would otherwise consume forces of the old positions.
    if !(e isa StateEngine) || !st.force_valid || e.fingerprint!=_fingerprint(st,dtT,spec)
        # Staged input checks (e.g. a Langevin spec's noise is built for its own dt)
        # throw before anything changes; dt is in the fingerprint, so they run on each change.
        SC.validate_integrator_inputs!(spec,st,dtT)
        if !engine_supported(st,spec)   # no staged fallback
            r=engine_unsupported_reason(st,spec)
            throw(ArgumentError("The device engine does not run this state: " * something(r,"unsupported configuration")))
        end
        e===nothing || (SC._engine_boundary!(e,st); _dispose!(e))
        st.engine=nothing
        # Staged NVE/CSVR clear the stochastic heat/work channels after another integrator.
        if spec isa Union{SC.NVESpec,SC.CSVRSpec,SC.NHCSpec} && st.last_integrator!=SC.integrator_id(spec)
            fill!(st.dq,zero(T)); fill!(st.dU,zero(T))
        end
        st.engine=e=_capture_engine(st,dtT,spec)
    end
    e.stale && _refresh_graphs!(e,st)  # the box changed since capture (barostat event)
    # A periodic boundary bounds how long a latched overflow can wait for its recovery
    # (later launches are no-ops until then) and the recorded flags of a long run.
    length(e.flags)>=BOUNDARY_LAUNCHES && SC._engine_boundary!(e,st)
    kind=_launch_kind(st,e,compute_energy)
    g=kind==3 ? e.virial : kind==2 ? e.event : kind==1 ? e.sampled : e.plain
    # host-scheduled fixed intervals: plain steps between scheduled ones need no list decision
    (kind==0 && e.fast!==nothing && !scheduled_step(e.run,e.device_step0+e.launched)) && (g=e.fast)
    # host-scheduled sorting: a plain step at or past the due step is the sort step (flag 4)
    if kind==0 && e.sort!==nothing && e.device_step0+e.launched>=e.sort_due
        g=e.sort; kind=Int8(4); e.sort_due=e.device_step0+e.launched+e.run.ordering.interval
    end
    if g===e.fast && e.tune!==nothing && !e.tune.decided
        _tuned_launch!(e.tune,e)   # the first fast steps choose the mesh priority
    else
        g isa SplitStep ? _launch_split!(g,e,st) : CUDA.launch(g)
    end
    e.launched+=1; e.moved=false
    push!(e.flags,kind)
    # Device steps end with forces (and, when sampled, energies) at the new positions.
    st.force_valid=true
    st.energy_valid=kind==1
    # an event step's graph computes no energies (the barostat read its virial a step
    # earlier); energies requested on it come from one evaluation at the new positions
    kind==2 && compute_energy && SC._engine_evaluate!(e,st,true,_tether_active(e.run))
    all_after_step!(st)
    st.step+=1
    st.last_integrator=SC.integrator_id(spec)
    return nothing
end

"""Graph of the next launch: 2 for a device pressure event (every `interval` steps of the
host step counter), 1 for a sampled step (requested, or the step before an event, whose
end-of-step virial the event reads), 0 otherwise. Host bookkeeping only."""
function _launch_kind(st,e,compute_energy::Bool)
    pc=st.pressure_control
    (pc===nothing || e.event===nothing) && return Int8(compute_energy)
    (st.step+1)%pc.interval==0 && return Int8(2)
    compute_energy && return Int8(1)
    (st.step+2)%pc.interval==0 && return e.virial===nothing ? Int8(1) : Int8(3)
    return Int8(0)
end

"""Launch the segments of a split step with the host force hook between them.
If the hook throws, the step stops after its drift, as on the staged path: forces
are marked invalid, so the next step recaptures from the current positions (the
device step counter has not advanced)."""
function _launch_split!(g::SplitStep,e::StateEngine,st;first::Int=1,resume=nothing)
    f=_host_feature(e.run); n=length(g.segments)
    for k in first:n
        CUDA.launch(k==first && resume!==nothing ? resume : g.segments[k])
        k==n && break
        e.host_call=true
        try
            host_force!(f,st,g.energy && k==n-1)
        catch
            st.force_valid=false; st.energy_valid=false
            rethrow()
        finally
            e.host_call=false
        end
    end
    return nothing
end

SC.step_graph!(st::SC.SimulationState{T},spec::Union{SC.NVESpec{T},SC.BAOABSpec{T},SC.GSMSpec{T},SC.BAOASpec{T},SC.VVSpec{T},SC.CSVRSpec{T},SC.NHCSpec{T},SC.BrownianSpec{T},SC.EMSpec{T}},dt::Real;compute_energy::Bool=true) where {T<:AbstractFloat}=
    SC.step!(st,spec,dt;compute_energy)

"""Explicit host boundary for a state advanced by the device engine (no-op otherwise)."""
synchronize_state!(st::SC.SimulationState)=(SC._engine_boundary!(st.engine,st); st)

# Direct integrator-observable reads need the same freshness as full step samples.
function SC._ensure_csvr_report_sample!(e::StateEngine,st,spec)
    csvr_lazy_report(e.run) && SC._engine_boundary!(e,st)
    return nothing
end
