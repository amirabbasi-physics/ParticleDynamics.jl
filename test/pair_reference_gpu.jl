module PairReferenceGPU
# Independent Float64 all-pairs CUDA reference for the pair models. It does not
# call production pair functions, periodic helpers or neighbor lists: plain
# round-based minimum image in Float64, direct formulas per potential.
# kind: 1 LJ, 2 WCA (uniform rc² = 1.2599211σ²; shift ε), 3 soft repulsive (harmonic, rc=σ).
# style: 1 uniform (σ, ε, cutoff); 2 mixed σ (σij=(σi+σj)/2, rc=factor·σij);
#        3 pair matrix (σ, ε, rc by typeid). Scalar, type-stable device code.
using CUDA

@inline function _coefficients(style::Int,kind::Int,i,j,σu::Float64,εu::Float64,rcu::Float64,
                               σp,factor::Float64,typeid,σm,εm,rcm)
    if style==1
        rc2=kind==2 ? 1.2599211*σu*σu : kind==3 ? σu*σu : rcu*rcu
        return σu,εu,rc2
    elseif style==2
        s=0.5*(Float64(σp[i])+Float64(σp[j])); rc=factor*s
        return s,εu,rc*rc
    end
    a=typeid[i]; b=typeid[j]; rc=Float64(rcm[a,b])
    return Float64(σm[a,b]),Float64(εm[a,b]),rc*rc
end

@inline _mi(x::Float64,L::Float64)=x-L*round(x/L)

function _reference_kernel!(out,rx,ry,rz,Lx::Float64,Ly::Float64,Lz::Float64,dim::Int,ids,kind::Int,style::Int,
                            σu::Float64,εu::Float64,rcu::Float64,σp,factor::Float64,typeid,σm,εm,rcm)
    t=threadIdx().x; sample=blockIdx().x; i=ids[sample]; N=length(rx)
    fx=0.0; fy=0.0; fz=0.0; e=0.0
    j=t
    while j<=N
        if j!=i
            dx=_mi(Float64(rx[i])-Float64(rx[j]),Lx)
            dy=_mi(Float64(ry[i])-Float64(ry[j]),Ly)
            dz=dim==3 ? _mi(Float64(rz[i])-Float64(rz[j]),Lz) : 0.0
            r2=dx*dx+dy*dy+dz*dz
            σ,ε,rc2=_coefficients(style,kind,i,j,σu,εu,rcu,σp,factor,typeid,σm,εm,rcm)
            if 0.0<r2<rc2
                fr=0.0; u=0.0
                if kind==3
                    rr=sqrt(r2); x=1.0-rr/σ; fr=(ε/σ)*x/rr; u=0.5*ε*x*x
                else
                    s6=(σ*σ/r2)^3; fr=24.0*ε*s6*(2.0*s6-1.0)/r2; u=4.0*ε*s6*(s6-1.0)+(kind==2 ? ε : 0.0)
                end
                fx+=fr*dx; fy+=fr*dy; fz+=fr*dz; e+=0.5*u
            end
        end
        j+=blockDim().x
    end
    scratch=CuStaticSharedArray(Float64,(256,4))
    scratch[t,1]=fx; scratch[t,2]=fy; scratch[t,3]=fz; scratch[t,4]=e
    sync_threads()
    stride=128
    while stride>0
        if t<=stride
            for q in 1:4; scratch[t,q]+=scratch[t+stride,q]; end
        end
        sync_threads(); stride>>=1
    end
    if t==1
        for q in 1:4; out[q,sample]=scratch[1,q]; end
    end
    return nothing
end

"""Forces (D×n) and per-particle energies at the sampled atoms `ids` (2D or 3D)."""
function reference_forces(r::Tuple,box,ids::CuVector{Int32};kind::Int,style::Int=1,sigma=1.0,epsilon=1.0,
                          cutoff=2.5,sigma_particle=nothing,factor=1.0,typeid=nothing,
                          sigma_pair=nothing,epsilon_pair=nothing,rcut_pair=nothing)
    D=length(r); T=eltype(r[1])
    out=CuArray{Float64}(undef,4,length(ids))
    rz=D==3 ? r[3] : r[1]; Lz=D==3 ? Float64(box[3]) : 1.0
    one1=CUDA.zeros(T,1); onei=CUDA.ones(Int32,1); one2=CUDA.zeros(T,1,1)
    @cuda threads=256 blocks=length(ids) _reference_kernel!(out,r[1],r[2],rz,Float64(box[1]),Float64(box[2]),Lz,D,ids,
        kind,style,Float64(sigma),Float64(epsilon),Float64(cutoff),something(sigma_particle,one1),Float64(factor),
        something(typeid,onei),something(sigma_pair,one2),something(epsilon_pair,one2),something(rcut_pair,one2))
    return (F=out[1:D,:],energy=out[4,:])
end
end
