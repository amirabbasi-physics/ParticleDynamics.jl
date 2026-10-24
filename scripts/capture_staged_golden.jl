#!/usr/bin/env julia
# Capture script that recorded the golden data used by the device-engine tests, from
# the earlier staged engine. Rerunning it now records current device-engine results,
# which are not an independent reference. Deterministic cases store forces at a
# fixed configuration, the state after one step and after 50 steps; stochastic
# integrators store observable time series for statistical comparison.
# Usage: julia --project=. scripts/capture_staged_golden.jl NEW_OUTPUT_DIR [deterministic|statistical|all|equilibrated|equilibrated-3d]
# Note: the v1 records were captured from the earlier staged engine, before NVE on
# 3D LJ states moved to the device engine.
using CUDA, ParticleDynamics, TOML, SHA, Random, Dates, Statistics
const SC=ParticleDynamics.SimulationCore
const NL=ParticleDynamics.NeighborLists

function lattice(D,N,a,::Type{T}) where {T}
    m=ceil(Int,N^(1/D))
    pos=[T[a*(mod((i-1)÷m^(d-1),m)-(m-1)/2)+0.05a*sin(i*(0.7+0.3d)) for i in 1:N] for d in 1:D]
    return pos,m*a
end
velocities(D,N,scale,::Type{T}) where {T}=[T[scale*sin(i*(0.31+0.17d))*cos(i*0.11d) for i in 1:N] for d in 1:D]

function build(D,::Type{T},pot,N;a,temperature=0,gamma=1,dt=0.002,unwrapped=false,interval=5) where {T}
    pos,L=lattice(D,N,a,T); L=max(L,T(12))
    box=ntuple(_->T(L),D)
    st=SC.build_simulation(;N,box,cutoff=T(2.5),skin=T(0.3),cap=Int32(160),neigh_interval=interval,
        use_neighborlist=true,epsilon=T(1),sigma=T(1),gamma=T(gamma),temperature=T(temperature),dt=T(dt),
        mass=T(1),nonbonded=pot,precision=T===Float64 ? :f64 : :f32,unwrapped_positions=unwrapped)
    return st
end

# Write positions/velocities explicitly (build_simulation draws its own).
function place!(st,D,pos,vel)
    for (x,p) in zip((st.rx,st.ry,st.rz)[1:D],pos); copyto!(x,p); end
    for (v,p) in zip((st.vx,st.vy,st.vz)[1:D],vel); copyto!(v,p); end
    if D==2
        NL.update_neighbors_inplace!(st.nbh,st.rx,st.ry;box=st.box2,step=st.step)
    else
        NL.update_neighbors_inplace!(st.nbh,st.rx,st.ry,st.rz;box=st.box3,step=st.step)
    end
    st.rx_unwrap===nothing || SC.sync_unwrapped!(st)
    return st
end

# Host copies in build-time particle order (st.tag maps slot -> build id).
function ordered(st,x)
    a=Float64.(Array(x))
    st.tag===nothing && return a
    out=similar(a); tags=Array(st.tag)
    if ndims(a)==1
        out[tags].=a
    else
        out[tags,:].=a
    end
    return out
end
function snapshot(st,D)
    s=Dict{String,Any}("step"=>st.step)
    for (name,arr) in zip(("rx","ry","rz")[1:D],(st.rx,st.ry,st.rz)[1:D]); s[name]=ordered(st,arr); end
    for (name,arr) in zip(("vx","vy","vz")[1:D],(st.vx,st.vy,st.vz)[1:D]); s[name]=ordered(st,arr); end
    for (name,arr) in zip(("fx","fy","fz")[1:D],(st.fx,st.fy,st.fz)[1:D]); s[name]=ordered(st,arr); end
    s["Epot"]=ordered(st,st.Epot)
    hasproperty(st,:virial) && st.virial isa CuArray && (s["virial"]=vec(ordered(st,st.virial)))
    return s
end
numeric(obs)=Dict(string(k)=>Float64(v) for (k,v) in pairs(obs) if v isa Real && !(v isa Bool) && isfinite(v))

function integrator(st,name,T)
    name===:nve && return SC.nve(st)
    name===:nhc && return SC.nosehooverchain(st;temperature=T(1),tau=T(0.5))
    name===:csvr && return SC.csvr(st;temperature=T(1),tau=T(0.5))
    name===:vv && return SC.velocityverlet(st;gamma=T(1),temperature=T(1))
    name===:baoab && return SC.baoab(st;gamma=T(1),temperature=T(1))
    name===:baoa && return SC.baoa(st;gamma=T(1),temperature=T(1))
    name===:gsm && return SC.gsm(st;gamma=T(1),temperature=T(1))
    name===:eulerheun && return SC.eulerheun(st;gamma=T(1),temperature=T(1))
    name===:eulermaruyama && return SC.eulermaruyama(st;gamma=T(1),temperature=T(1))
    error("unknown integrator $name")
end

function deterministic_case(D,T,pot,integ;N=257)
    Random.seed!(0xE0); CUDA.seed!(UInt64(0xE0))
    a=pot===:soft_repulsive ? 0.9 : 1.12
    pos,_=lattice(D,N,a,T); vel=velocities(D,N,0.5,T)
    st=build(D,T,pot,N;a)
    place!(st,D,pos,vel)
    rec=Dict{String,Any}("D"=>D,"precision"=>string(T),"potential"=>string(pot),"integrator"=>string(integ),
        "N"=>N,"box"=>collect(Float64.(D==2 ? st.box2 : st.box3)),"dt"=>Float64(st.dt),"lattice_spacing"=>a)
    SC.evaluate_forces_into_f!(st,true); CUDA.synchronize()
    rec["forces"]=snapshot(st,D)
    spec=integrator(st,integ,T)
    SC.step!(st,spec,st.dt;compute_energy=true); CUDA.synchronize()
    rec["step1"]=snapshot(st,D)
    for _ in 2:50; SC.step!(st,spec,st.dt;compute_energy=true); end
    CUDA.synchronize()
    rec["step50"]=snapshot(st,D)
    rec["step50_observables"]=numeric(SC.collect_step_observables(st,spec))
    return rec
end

function statistical_case(D,integ;N=1024,warmup=2000,samples=200,every=20)
    T=Float32
    Random.seed!(0xE1); CUDA.seed!(UInt64(0xE1))
    brownian=integ in (:eulerheun,:eulermaruyama)
    dt=brownian ? 1e-4 : 0.002
    pos,_=lattice(D,N,1.3,T); vel=velocities(D,N,brownian ? 0.0 : 1.0,T)
    st=build(D,T,:lj,N;a=1.3,temperature=1,gamma=1,dt,unwrapped=true)
    place!(st,D,pos,vel)
    spec=integrator(st,integ,T)
    for _ in 1:warmup; SC.step!(st,spec,st.dt;compute_energy=false); end
    ref=[ordered(st,x) for x in (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D]]
    series=Dict{String,Vector{Float64}}()
    msd=Float64[]
    for k in 1:samples
        for i in 1:every; SC.step!(st,spec,st.dt;compute_energy=i==every); end
        for (name,v) in numeric(SC.collect_step_observables(st,spec)); push!(get!(series,name,Float64[]),v); end
        cur=[ordered(st,x) for x in (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D]]
        push!(msd,sum(sum(abs2,c.-r) for (c,r) in zip(cur,ref))/N)
    end
    CUDA.synchronize()
    return Dict{String,Any}("D"=>D,"precision"=>"Float32","potential"=>"lj","integrator"=>string(integ),
        "N"=>N,"dt"=>dt,"gamma"=>1.0,"temperature"=>1.0,"warmup"=>warmup,"samples"=>samples,"every"=>every,
        "series"=>series,"msd_series"=>msd,
        "summary"=>Dict(k=>Dict("mean"=>mean(v),"std"=>std(v)) for (k,v) in series))
end

# Equilibrated statistics (the v1 statistical records are transients). Independent replicas differ in seed and initial velocities, so the
# replica spread gives the statistical uncertainty of each mean; first/second
# half means of each replica are stored to check stationarity.
# Lattice spacing: 2D a=1.3 gives ρ=0.592 (supercritical at T=1). In 3D, a=1.3
# gives ρ=0.350, inside liquid–vapour coexistence of LJ cut at 2.5σ (slow
# coarsening, non-stationary Epot; v2-eq 3D records rejected). The 3D liquid
# uses ρ=0.8 with N=1728 (full 12³ lattice, L=12.93 > the 12σ minimum box of
# `build`, which would otherwise pad N=1024 back to ρ=0.59, still two-phase).
equilibrated_size(D)=D==2 ? 1024 : 1728
equilibrated_spacing(D,N)=D==2 ? 1.3 : (N/0.8)^(1/3)/ceil(Int,N^(1/3)-1e-9)
function equilibrated_case(D,integ,seed;N=equilibrated_size(D),samples=500)
    T=Float32
    Random.seed!(seed); CUDA.seed!(UInt64(seed))
    brownian=integ in (:eulerheun,:eulermaruyama)
    dt=brownian ? 1e-4 : 0.002
    warmup=brownian ? 200_000 : 20_000   # 20 and 40 time units
    every=brownian ? 400 : 40
    rng=Xoshiro(seed)
    a=equilibrated_spacing(D,N)
    pos,_=lattice(D,N,a,T)
    vel=[brownian ? zeros(T,N) : T.(randn(rng,N)) for _ in 1:D]
    brownian || foreach(v->(v.-=sum(v)/N),vel)
    st=build(D,T,:lj,N;a,temperature=1,gamma=1,dt,unwrapped=true)
    place!(st,D,pos,vel)
    spec=integrator(st,integ,T)
    for _ in 1:warmup; SC.step!(st,spec,st.dt;compute_energy=false); end
    ref=[ordered(st,x) for x in (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D]]
    series=Dict{String,Vector{Float64}}()
    msd=Float64[]
    for k in 1:samples
        for i in 1:every; SC.step!(st,spec,st.dt;compute_energy=i==every); end
        for (name,v) in numeric(SC.collect_step_observables(st,spec)); push!(get!(series,name,Float64[]),v); end
        cur=[ordered(st,x) for x in (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D]]
        push!(msd,sum(sum(abs2,c.-r) for (c,r) in zip(cur,ref))/N)
    end
    CUDA.synchronize()
    h=samples÷2
    return Dict{String,Any}("D"=>D,"precision"=>"Float32","potential"=>"lj","integrator"=>string(integ),"seed"=>seed,
        "N"=>N,"dt"=>dt,"gamma"=>1.0,"temperature"=>1.0,"warmup"=>warmup,"samples"=>samples,"every"=>every,
        "density"=>N/prod(Float64.(D==2 ? st.box2 : st.box3)),"series"=>series,"msd_series"=>msd,
        "summary"=>Dict(k=>Dict("mean"=>mean(v),"std"=>std(v),"first_half_mean"=>mean(v[1:h]),
            "second_half_mean"=>mean(v[h+1:end])) for (k,v) in series))
end

function save(dir,name,rec)
    file=joinpath(dir,name*".toml")
    ispath(file) && error("Refusing to overwrite $file")
    open(io->TOML.print(io,rec;sorted=true),file,"w")
    open(io->println(io,bytes2hex(sha256(read(file)))," ",name*".toml"),joinpath(dir,"SHA256SUMS"),"a")
    println("saved ",name); flush(stdout)
end

function main(args)
    out=abspath(args[1]); mode=length(args)>1 ? args[2] : "all"
    mkpath(out)
    CUDA.functional() || error("CUDA is required")
    CUDA.allowscalar(false)
    meta=joinpath(out,"capture.toml")
    isfile(meta) || open(io->TOML.print(io,Dict("created_utc"=>string(now(UTC)),"gpu"=>CUDA.name(CUDA.device()),
        "julia"=>string(VERSION),"cuda"=>string(pkgversion(CUDA)),"depot"=>DEPOT_PATH,
        "purpose"=>"device-engine golden references (current engine, not independent)")),meta,"w")
    if mode in ("deterministic","all")
        for D in (2,3), T in (Float32,Float64), pot in (:lj,:wca,:soft_repulsive), integ in (:nve,:nhc)
            save(out,"det_$(D)d_$(T)_$(pot)_$(integ)",deterministic_case(D,T,pot,integ))
        end
    end
    if mode in ("statistical","all")
        for D in (2,3), integ in (:nhc,:csvr,:vv,:baoab,:baoa,:gsm,:eulerheun,:eulermaruyama)
            save(out,"stat_$(D)d_$(integ)",statistical_case(D,integ))
        end
    end
    if mode in ("equilibrated","equilibrated-3d") # not part of "all": captured separately
        for D in (mode=="equilibrated" ? (2,3) : (3,)), integ in (:nhc,:csvr,:vv,:baoab,:baoa,:gsm,:eulerheun,:eulermaruyama), seed in 0xE301:0xE304
            save(out,"eq_$(D)d_$(integ)_s$(string(seed;base=16))",equilibrated_case(D,integ,Int(seed)))
        end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    main(ARGS)
end
