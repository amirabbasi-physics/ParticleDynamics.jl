module GPUForceReference
using CUDA
export sampled_lj_oracle_gpu, gpu_force_errors

# Allocating, independent CUDA reduction oracle, used only outside timing.
# Explicit adjacent half-step velocities deliberately avoid the production
# v⊗v + a⊗a identity, and generic reductions do not reuse its shared tree.
function observable_reference_gpu(energy,virial,v,f,mass,dt,box)
    velocity=Float64.(hcat(v...)); force=Float64.(hcat(f...))
    minus=velocity .- (Float64(dt)/(2Float64(mass))).*force
    plus=velocity .+ (Float64(dt)/(2Float64(mass))).*force
    pairs=((1,1),(2,2),(3,3),(1,2),(1,3),(2,3))
    kinetic=hcat([vec(sum((Float64(mass)/4).*(minus[:,a].*minus[:,b] .+
        plus[:,a].*plus[:,b]);dims=1)) for (a,b) in pairs]...)
    k=vec(sum(kinetic[:,1:3];dims=2))
    u=vec(sum(Float64.(energy);dims=1))
    w=vec(sum(Float64.(virial);dims=1))
    full=vec(sum((Float64(mass)/2).*sum(abs2,velocity;dims=2);dims=1))
    temp=2 .*k./(3size(velocity,1))
    pressure=(2 .*vec(kinetic).+w)./prod(Float64.(box))
    p=vec(sum(pressure[1:3];dims=1))./3
    return vcat(u,w,vec(kinetic),full,k,temp,u.+k,p,pressure)
end

function observable_errors(run;pair_reference=nothing)
    e=pair_reference===nothing ? run.energy : pair_reference.energy
    w=pair_reference===nothing ? run.virial : permutedims(pair_reference.virial)
    ref=observable_reference_gpu(e,w,run.v,run.f,run.mass,run.dt,run.neighbors.box)
    delta=abs.(run.observables.values.-ref)
    scaled=delta./max.(abs.(ref),1.0)
    return Dict("max_scaled_error"=>maximum(scaled),"max_absolute_error"=>maximum(delta),
        "reference"=>(pair_reference===nothing ? "independent CUDA global sums and explicit adjacent-half-step velocities; stored per-particle U/W" :
         "independent CUDA all-pair U/W, global sums and adjacent-half-step velocities"))
end

# Independent direct pair summation in Float64 on the GPU. One block per
# sampled atom; no production neighbor list, pair functor or MIC helper.
function _direct_lj_reference!(out,rx,ry,rz,ids,Lx::Float64,Ly::Float64,Lz::Float64,
                               sigma::Float64,epsilon::Float64,cutoff::Float64)
    t=threadIdx().x; sample=blockIdx().x
    i=ids[sample]
    xi=Float64(rx[i]); yi=Float64(ry[i]); zi=Float64(rz[i])
    fx=0.0; fy=0.0; fz=0.0; energy=0.0; scale=0.0; count=0.0
    wxx=0.0; wyy=0.0; wzz=0.0; wxy=0.0; wxz=0.0; wyz=0.0
    j=t
    @inbounds while j<=length(rx)
        if j!=i
            dx=xi-Float64(rx[j]); dx-=Lx*round(dx/Lx)
            dy=yi-Float64(ry[j]); dy-=Ly*round(dy/Ly)
            dz=zi-Float64(rz[j]); dz-=Lz*round(dz/Lz)
            r2=dx*dx+dy*dy+dz*dz
            if 0.0<r2<cutoff*cutoff
                s6=(sigma*sigma/r2)^3
                a=24.0*epsilon*s6*(2.0*s6-1.0)/r2
                fx+=a*dx; fy+=a*dy; fz+=a*dz
                energy+=2.0*epsilon*s6*(s6-1.0)
                wxx+=0.5*a*dx*dx; wyy+=0.5*a*dy*dy; wzz+=0.5*a*dz*dz
                wxy+=0.5*a*dx*dy; wxz+=0.5*a*dx*dz; wyz+=0.5*a*dy*dz
                scale+=abs(a)*sqrt(r2); count+=1.0
            end
        end
        j+=blockDim().x
    end
    scratch=CuStaticSharedArray(Float64,(256,12))
    values=(fx,fy,fz,energy,wxx,wyy,wzz,wxy,wxz,wyz,scale,count)
    @inbounds for q in 1:12; scratch[t,q]=values[q]; end
    sync_threads()
    stride=Int32(128)
    while stride>0
        if t<=stride
            @inbounds for q in 1:12; scratch[t,q]+=scratch[t+stride,q]; end
        end
        sync_threads()
        stride>>=1
    end
    if t==1
        @inbounds for q in 1:12; out[q,sample]=scratch[1,q]; end
    end
    return
end

function sampled_lj_oracle_gpu(rx,ry,rz,box,ids::CuVector{Int32},sigma,epsilon,cutoff)
    length(rx)==length(ry)==length(rz) || error("Coordinate lengths differ")
    isempty(ids) && error("Empty GPU reference sample")
    out=CuArray{Float64}(undef,12,length(ids))
    @cuda threads=256 blocks=length(ids) _direct_lj_reference!(out,rx,ry,rz,ids,
        Float64.(box)...,Float64(sigma),Float64(epsilon),Float64(cutoff))
    return (; F=out[1:3,:],energy=out[4,:],virial=out[5:10,:],pair_scale=out[11,:],counts=out[12,:])
end

function gpu_force_errors(fx,fy,fz,ids::CuVector{Int32},reference)
    observed=permutedims(hcat(fx[ids],fy[ids],fz[ids]))
    delta=Float64.(observed).-reference
    err2=sum(abs2,delta); ref2=sum(abs2,reference)
    peak=maximum(abs,delta)
    return Dict("rms_absolute"=>sqrt(err2/length(ids)),
        "rms_relative"=>sqrt(err2/max(ref2,eps(Float64))),
        "max_component_absolute"=>peak,"reference_rms"=>sqrt(ref2/length(ids)))
end
end
