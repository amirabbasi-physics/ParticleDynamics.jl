# Ornstein–Uhlenbeck (coloured) noise added to the
# white thermal noise of every stochastic integrator, as staged does
# (`_vv_apply_ou*`, `_apply_ou*`): for each active particle (OU column j) and mode
# k, η_k ← a[k,j]·η_k + c[k,j]·N(0,1) per component, and β += Σ_k η_k.
#
# Staged indexes OU columns by storage slot (`active_idx[j]` = slot). Sorting
# moves particles, so the engine keeps its own slot → column map (`col`, 0 =
# inactive), permuted with the particles, and writes `active_idx` back at every
# host boundary (`sync_ou!`). OU states are advanced once per step by the first
# kernel that uses the noise; a second kernel of the same step (Euler–Heun
# final, GJF velocity) redraws only the white part and reads the updated states.
# Philox streams: white 0; OU mode k uses stream k (Float64: 2k and 2k+1).

struct OUFeature{M<:CuArray,S<:Tuple} <: EngineFeature
    col::CuArray{Int32,1}     # engine: OU column of each slot
    active::CuArray{Int32,1}  # spec: active_idx (slot of each column), synced back
    a::M
    c::M
    state::S                  # spec workspace ou_x, ou_y[, ou_z] (modes × columns)
end

_ou_spec(spec)=hasproperty(spec,:params) && hasproperty(spec.params,:ou) ? spec.params.ou : nothing
function make_feature(::Type{OUFeature},st,spec)
    ou=_ou_spec(spec)
    ou===nothing && return nothing
    SC.ensure_integrator_workspace!(spec,st)                 # OU state matrices (stationary start)
    SC._refresh_ou_coefficients!(ou,spec.params.dt)
    ws=spec.workspace; N=length(st.rx)
    col=CUDA.zeros(Int32,N)
    K=length(ou.active_idx)
    K>0 && (@cuda threads=256 blocks=cld(K,256) _ou_col_kernel!(col,ou.active_idx))
    state=st.rz===nothing ? (ws.ou_x,ws.ou_y) : (ws.ou_x,ws.ou_y,ws.ou_z)
    return OUFeature(col,ou.active_idx,ou.coeff_a,ou.coeff_c,state)
end
# The OU spectrum object and its arrays (replaced by `Filters.set_ou_spectrum!`).
function feature_signature(::Type{OUFeature},st,spec)
    ou=_ou_spec(spec)
    ou===nothing && return nothing
    ws=spec.workspace
    return (objectid(ou),map(pointer,(ou.active_idx,ou.coeff_a,ou.coeff_c)),ou.dt,
        map(x->x===nothing ? nothing : pointer(x),(ws.ou_x,ws.ou_y,ws.ou_z)))
end
device_part(f::OUFeature)=(col=f.col,a=f.a,c=f.c,state=f.state)
sorted_int(f::OUFeature)=(f.col,)
at_boundary!(f::OUFeature)=sync_ou!(f)

function _ou_col_kernel!(col,active)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    j<=length(active) && (@inbounds col[active[j]]=Int32(j))
    return nothing
end
function _ou_sync_kernel!(active,col)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=length(col)
        @inbounds j=col[i]
        j>0 && (@inbounds active[j]=Int32(i))
    end
    return nothing
end

"""Write the current slot of every OU column back into the spec's `active_idx`."""
function sync_ou!(ou::OUFeature)
    N=length(ou.col)
    @cuda threads=256 blocks=cld(N,256) _ou_sync_kernel!(ou.active,ou.col)
    return nothing
end

"""β plus the OU modes of particle `i` (advanced when `update`, else read)."""
@inline noise_add(fx::Tuple,β,::Type{T},::Val{D},seed,step,id,i,update::Bool) where {T,D}=
    _ou_add(β,_find(fx,Val(:col)),T,Val(D),seed,step,id,i,update)

@inline _ou_add(β,::Nothing,::Type{T},::Val{D},seed,step,id,i,update::Bool) where {T,D}=β
@inline function _ou_add(β,ou::NamedTuple,::Type{T},::Val{D},seed,step,id,i,update::Bool) where {T,D}
    j=@inbounds ou.col[i]
    j==0 && return β
    acc=β
    for k in 1:size(ou.a,1)
        if update
            z=philox_normals(T,Val(D),seed,step,id,UInt32(k))
            ak=@inbounds ou.a[k,j]; ck=@inbounds ou.c[k,j]
            n=ntuple(d->ak*(@inbounds ou.state[d][k,j])+ck*z[d],Val(D))
            for d in 1:D; @inbounds ou.state[d][k,j]=n[d]; end
        else
            n=ntuple(d->(@inbounds ou.state[d][k,j]),Val(D))
        end
        acc=map(+,acc,n)
    end
    return acc
end
