# Shared by the thermostat device-engine tests: staged equilibrated golden records and
# device replicas of their protocol (scripts/capture_staged_golden.jl, mode
# `equilibrated`), without unwrapped coordinates (not used by the dynamics).
# Requires capture_staged_golden.jl, Test, SHA, TOML, Statistics, Random, SC, DE.
const EQ2D=golden_directory(:equilibrium_2d)
# 3D v2-eq records are two-phase (rejected); 3D uses the homogeneous liquid set.
const EQ3D=golden_directory(:equilibrium_3d)


function eq_golden(D,integ)
    dir=D==2 ? EQ2D : EQ3D
    files=sort(filter(f->startswith(f,"eq_$(D)d_$(integ)_"),readdir(dir)))
    sums=Dict(split(l)[2]=>split(l)[1] for l in eachline(joinpath(dir,"SHA256SUMS")))
    return map(files) do f
        @test bytes2hex(sha256(read(joinpath(dir,f))))==sums[f]
        TOML.parsefile(joinpath(dir,f))
    end
end

# The equilibrated protocol of capture_staged_golden.jl without unwrapped
# coordinates (not used by the dynamics; not needed for the reference comparison).
function device_replica(D,integ,seed;N=equilibrated_size(D))
    T=Float32; rng=Xoshiro(seed); Random.seed!(seed)
    brownian=integ in (:eulerheun,:eulermaruyama)          # protocol of `equilibrated_case`
    dt=brownian ? 1e-4 : 0.002; warmup=brownian ? 200_000 : 20_000; every=brownian ? 400 : 40
    a=equilibrated_spacing(D,N); pos,_=lattice(D,N,a,T)
    vel=[brownian ? zeros(T,N) : T.(randn(rng,N)) for _ in 1:D]; brownian || foreach(v->(v.-=sum(v)/N),vel)
    st=build(D,T,:lj,N;a,temperature=1,gamma=1,dt)
    place!(st,D,pos,vel)
    spec=integrator(st,integ,T)
    for _ in 1:warmup; SC.step!(st,spec,st.dt;compute_energy=false); end
    s=Dict(k=>Float64[] for k in ("Ekin_total","Epot_total","virial_total"))
    for _ in 1:500
        for i in 1:every; SC.step!(st,spec,st.dt;compute_energy=i==every); end
        o=SC.collect_step_observables(st,spec)
        for k in keys(s); push!(s[k],Float64(getfield(o,Symbol(k)))); end
    end
    st.engine isa DE.StateEngine || error("replica did not run on the device engine")
    return s
end

# Block means (10 blocks of 50 samples = 2,000 steps, much longer than the
# correlation times) pooled over replicas: 40 nearly independent values per side.
# The standard error of the replica means alone has only 3 degrees of freedom.
block_means(x;nb=10)=[mean(x[(k-1)*(length(x)÷nb)+1:k*(length(x)÷nb)]) for k in 1:nb]


"""Device vs staged replica statistics: |Δmean| < 4.5 combined block standard errors."""
function compare_to_golden(D,integ;seeds=0xE3A1:0xE3A4)
    gold=eq_golden(D,integ); @test length(gold)==4 && all(r["N"]==equilibrated_size(D) for r in gold)
    dev=[device_replica(D,integ,s) for s in seeds]
    for k in (integ in (:eulerheun,:eulermaruyama) ? ("Epot_total","virial_total") : ("Ekin_total","Epot_total","virial_total"))
        g=reduce(vcat,[block_means(Float64.(r["series"][k])) for r in gold])
        d=reduce(vcat,[block_means(x[k]) for x in dev])
        se=sqrt(var(g)/length(g)+var(d)/length(d))
        @test abs(mean(d)-mean(g))<4.5se
        println(D,"D ",integ," ",k,": device ",mean(d)," ± ",std(d)/sqrt(length(d)),", staged ",mean(g),
            " ± ",std(g)/sqrt(length(g)),"  (",round((mean(d)-mean(g))/se,digits=2)," SE)")
    end
end
