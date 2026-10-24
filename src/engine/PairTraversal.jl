# Included in ParticleDynamics.DeviceEngine.
# Dimension- and potential-generic cooperative traversal. For D=3 and UniformLJ
# it performs the same operations in the same order as `_lj_cooperative_kernel!`
# (periodic_delta, nested muladd r², the staged LJ terms, half-counted virials),
# so it can replace that kernel without changing results.

@inline _r2(dr::NTuple{3,T}) where {T}=muladd(dr[1],dr[1],muladd(dr[2],dr[2],dr[3]*dr[3]))
@inline _r2(dr::NTuple{2,T}) where {T}=muladd(dr[1],dr[1],dr[2]*dr[2])
@inline _half_virial(dr::NTuple{3,T},f::NTuple{3,T}) where {T}=NB._half_virial3(dr...,f...)
@inline _half_virial(dr::NTuple{2,T},f::NTuple{2,T}) where {T}=NB._half_virial2(dr...,f...)
@inline _virial_zero(::Val{3},::Type{T}) where {T}=ntuple(_->zero(T),Val(6))
@inline _virial_zero(::Val{2},::Type{T}) where {T}=ntuple(_->zero(T),Val(3))
# Tuple helpers as top-level functions: closures inside the kernel that capture
# reassigned accumulators would be boxed (Core.Box), which does not compile.
@inline _tadd(a::NTuple{K,T},b::NTuple{K,T}) where {K,T}=ntuple(k->a[k]+b[k],Val(K))
# Group reduction of a tuple: one shuffle round over all elements per shift.
@inline _tshfl(a::NTuple{K,T},shift,::Val{G}) where {K,T,G}=
    ntuple(k->a[k]+CUDA.shfl_down_sync(UInt32(0xffffffff),a[k],shift,G),Val(K))
@inline function _tsum(a::NTuple{K,T},::Val{G}) where {K,T,G}
    shift=G÷2
    while shift>0
        a=_tshfl(a,shift,Val(G))
        shift÷=2
    end
    return a
end
@inline _tstore!(arrays::NTuple{K},i,vals::NTuple{K}) where {K}=(ntuple(k->(@inbounds arrays[k][i]=vals[k]; nothing),Val(K)); nothing)
@inline _rowstore!(M,i,vals::NTuple{K}) where {K}=(ntuple(k->(@inbounds M[i,k]=vals[k]; nothing),Val(K)); nothing)
@inline _tkick!(v::NTuple{K},i,c,a::NTuple{K}) where {K}=(ntuple(k->(@inbounds v[k][i]=muladd(c,a[k],v[k][i]); nothing),Val(K)); nothing)
# CSVR kick v = s·v + c·a, returning the new velocity (top level: see the closure note above)
@inline _tskick!(v::NTuple{K},i,c,s,a::NTuple{K}) where {K}=ntuple(k->(@inbounds v[k][i]=muladd(c,a[k],s*v[k][i])),Val(K))

@inline _xyz(p::NTuple{4,VecElement{T}},::Val{D}) where {T,D}=ntuple(d->p[d].value,Val(D))
# Add pair (i, j) (from `_pair_at`) to the lane's force, energy and virial.
@inline function _pair_acc(a::NTuple{D,T},e::T,w,p,fx,i,j,::Val{E}) where {D,T,E}
    dr,r2,inside,fr,u=p
    if inside && !excluded(fx,i,j)   # feature bundle: bonded exclusions
        fij=ntuple(d->fr*dr[d],Val(D))
        if E
            e+=T(0.5)*u
            w=_tadd(w,_half_virial(dr,fij))
        end
        a=_tadd(a,fij)
    end
    return a,e,w
end
# Displacement, r² and pair terms of neighbor j: from the coordinate arrays, or (xqp) from the
# packed (x, y, z, w), one 16-byte load per neighbor instead of D + 1 separate ones.
@inline function _pair_at(::Nothing,r::NTuple{D},ri,j,box,model,data,ctx) where {D}
    T=eltype(ri)
    dr=ntuple(d->NB.periodic_delta(ri[d],(@inbounds r[d][j]),T(0.5)*box[d],box[d]),Val(D))
    r2=_r2(dr)
    inside,fr,u=pair_eval(model,data,ctx,j,r2)
    return dr,r2,inside,fr,u
end
@inline function _pair_at(xqp,r::NTuple{D},ri,j,box,model,data,ctx) where {D}
    T=eltype(ri)
    pj=@inbounds xqp.xq[j]
    rj=_xyz(pj,Val(D))
    dr=ntuple(d->NB.periodic_delta(ri[d],rj[d],T(0.5)*box[d],box[d]),Val(D))
    r2=_r2(dr)
    inside,fr,u=pair_eval_w(model,data,ctx,j,r2,pj[4].value)
    return dr,r2,inside,fr,u
end

function _pair_cooperative_kernel!(r::NTuple{D},f::NTuple{D},energy,virial,indices,counts,
                                   slot_groups,box,model,data,fx,
                                   ::Val{G},::Val{L},::Val{E},kick) where {D,G,L,E}
    T=eltype(r[1]); box=NL._boxlen(box,Val(D))   # host tuple or device geometry
    tid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    i=tid÷G+1; lane=tid%G; N=length(r[1])
    a=ntuple(_->zero(T),Val(D)); e=zero(T); w=_virial_zero(Val(D),T)
    if i<=N
        @inbounds begin
            xqp=_find(fx,Val(:xq))   # packed (x, y, z, w) of this step, when the graph provides them
            ri=xqp===nothing ? ntuple(d->(@inbounds r[d][i]),Val(D)) : _xyz(xqp.xq[i],Val(D))
            ctx=pair_context(model,data,i)
            slot=lane; cnt=counts[i]
            # two neighbors per iteration: both loads in flight before either is used (the 10k
            # loop is latency-bound: one wave, ~19 dependent neighbor loads per lane); the
            # accumulation order per lane is unchanged (slot, slot + G, slot + 2G, …)
            while G<=4 && slot+G<cnt   # (8 lanes: no gain, salt 31.9 → 33.7 µs; argon NVE 0.0315 → 0.0324 ms/step)
                j1=indices[_traversal_neighbor_index(i,slot,N,slot_groups,Val(G),Val(L))]
                j2=indices[_traversal_neighbor_index(i,slot+G,N,slot_groups,Val(G),Val(L))]
                p1=_pair_at(xqp,r,ri,j1,box,model,data,ctx)
                p2=_pair_at(xqp,r,ri,j2,box,model,data,ctx)
                a,e,w=_pair_acc(a,e,w,p1,fx,i,j1,Val(E))
                a,e,w=_pair_acc(a,e,w,p2,fx,i,j2,Val(E))
                slot+=2G
            end
            while slot<cnt   # the remainder (one neighbor), or every neighbor when G > 4
                j=indices[_traversal_neighbor_index(i,slot,N,slot_groups,Val(G),Val(L))]
                a,e,w=_pair_acc(a,e,w,_pair_at(xqp,r,ri,j,box,model,data,ctx),fx,i,j,Val(E))
                slot+=G
            end
        end
    end
    # Every lane reaches every shuffle (empty rows, groups beyond N).
    a=_tsum(a,Val(G))
    if E
        e=_cooperative_sum(e,Val(G))
        w=_tsum(w,Val(G))
    end
    if lane==0 && i<=N
        @inbounds begin
            _tstore!(f,i,a)
            if E
                energy[i]=e+pair_self_energy(model,data,i,T)
                _rowstore!(virial,i,w)
            end
            if kick!==nothing # fused final kick: identical to SC's kick kernels
                v,c,_,cs=kick
                if cs===nothing
                    _tkick!(v,i,c,a)
                else   # CSVR: v(t+h) = s·v(t+h/2) + h/2m·f, Ekin of the completed step
                    u=_tskick!(v,i,c,T(cs[1][1]),a)
                    ekin=T(0.5)*cs[7]*_r2(u)
                    cs[2][i]=ekin
                end
            end
        end
    end
    kick===nothing || _fused_kick_report!(kick,lane==0 && i<=N,i,N,Val(G),Val(E))
    return nothing
end

# Tail of a fused final kick (G lanes per particle; `active`: the lane that kicked particle i):
# on energy steps (E) the CSVR K of the completed step summed per block into its slot
# (`enqueue_fused_k_report!` adds the slots: the reported K is what observables read, so
# plain steps skip it); the step counter.
@inline function _fused_kick_report!(kick,active::Bool,i,N,::Val{G},::Val{E}) where {G,E}
    if E && kick[4]!==nothing   # CSVR K of the completed step: this block's partial
        cs=kick[4]
        ek=active ? Float64(@inbounds cs[2][i]) : 0.0
        sh=CUDA.CuStaticSharedArray(Float64,32)
        ks=_fused_block_sum(ek,sh)
        threadIdx().x==1 && (@inbounds cs[3][blockIdx().x]=ks)
    end
    tid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    if tid==0 # fused step counter; the wrapper guarantees a valid list
        @inbounds kick[3][NL.DN_STEP]+=1
    end
    return nothing
end

# Explicit `kick` argument (never a default): see the Argon log, 9 Oct 2026.
function _pair_cooperative_noE_kernel!(args...)
    @inline _pair_cooperative_kernel!(args...,Val(false),nothing)
    return nothing
end
function _pair_cooperative_observable_kernel!(args...)
    @inline _pair_cooperative_kernel!(args...,Val(true),nothing)
    return nothing
end
function _pair_cooperative_noE_kernel_kick!(kick,args...)
    @inline _pair_cooperative_kernel!(args...,Val(false),kick)
    return nothing
end
function _pair_cooperative_observable_kernel_kick!(kick,args...)
    @inline _pair_cooperative_kernel!(args...,Val(true),kick)
    return nothing
end
