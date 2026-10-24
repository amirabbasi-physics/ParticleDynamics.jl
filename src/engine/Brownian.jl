# overdamped Brownian dynamics inside the graph.
# Staged (BrownianIntegrators, SimulationCore): ξ = noise·N(0,1) with
# noise = √(2γT·dt_spec), μ = 1/γ, positions wrapped with `_wrap_centered`.
#   Euler–Maruyama: x += μ f(x) dt + μ ξ;  dq, dU += f(x)·Δx_mic
#   Euler–Heun (midpoint, `BrownianSpec`): x_m = x + ½(μ f(x) dt + μ ξ) stored in
#     the velocity slots; f(x_m); x = x + μ f(x_m) dt + μ ξ (same ξ);
#     dq, dU += f(x_m)·Δx_mic; then f at the new x.
# Engine: the same arithmetic in T. ξ is Philox keyed on the particle tag and the
# device step, so the final Heun kernel redraws the midpoint's ξ instead of
# storing it. For Heun, positions move to x_m (x is saved in the velocity slots,
# which sorting permutes), the neighbor list is checked and possibly rebuilt,
# forces are evaluated at x_m, then the final kernel restores x, advances it and
# leaves x_m in the velocity slots, as staged does. Two neighbor checks and two
# force kernels per step (`chained_conditional_graph`).

struct BrownianData{T,V<:CuArray{T,1},I}
    midpoint::Bool
    gamma::V        # spec-owned, permuted in place by sorting
    noise::V        # spec-owned, permuted in place
    dq::V
    dU::V
    tags::I
    seed::UInt64
end

@inline _bd_wrap(x::T,L::T) where {T}=BrownianIntegrators._wrap_centered(x,L)

@inline function _bd_noise(::Type{T},::Val{D},noise,tags,state,seed,i,fx,update::Bool) where {T,D}
    s=@inbounds noise[i]; step=UInt64(@inbounds state[NL.DN_STEP]); id=_noise_id(tags,i)
    z=philox_normals(T,Val(D),seed,step,id,UInt32(0))
    return noise_add(fx,map(x->s*x,z),T,Val(D),seed,step,id,i,update)
end

# Euler–Maruyama step, and the Heun final step (f at x_m, x0 from `saved`).
function _bd_step_kernel!(r::NTuple{D,<:CuDeviceVector{T}},saved,f::NTuple{D,<:CuDeviceVector{T}},
                          gamma,noise,dq,dU,tags,state,seed::UInt64,dt::T,box::NTuple{D,T},fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        μ=one(T)/gamma[i]
        ξ=_bd_noise(T,Val(D),noise,tags,state,seed,i,fx,saved===nothing) # Heun final: OU already advanced
        fi=_get(f,i)
        x0=saved===nothing ? _get(r,i) : _get(saved,i)
        Δ=map((b,c)->μ*b*dt+μ*c,fi,ξ)
        xw=map((a,p,L)->_bd_wrap(a+p,L),x0,Δ,box)
        displace!(fx,i,Δ)                             # staged: unwrap += Δ (midpoint moves excluded)
        w=_dot(fi,map((a,b,L)->_bd_wrap(a-b,L),xw,x0,box))
        saved===nothing || _put!(saved,i,_get(r,i))   # leave x_m in the velocity slots
        _put!(r,i,xw)
        dq[i]+=w; dU[i]+=w
    end
    return nothing
end

# Heun midpoint: save x in the velocity slots, move to x_m.
function _bd_midpoint_kernel!(r::NTuple{D,<:CuDeviceVector{T}},saved,f::NTuple{D,<:CuDeviceVector{T}},
                              gamma,noise,tags,state,seed::UInt64,dt::T,box::NTuple{D,T},fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        μ=one(T)/gamma[i]
        ξ=_bd_noise(T,Val(D),noise,tags,state,seed,i,fx,true)
        x=_get(r,i)
        _put!(saved,i,x)
        _put!(r,i,map((a,b,c,L)->_bd_wrap(a+T(0.5)*(μ*b*dt+μ*c),L),x,_get(f,i),ξ,box))
    end
    return nothing
end

_bd_step_args(run,b::BrownianData)=(run.r,b.midpoint ? run.v : nothing,run.f,b.gamma,b.noise,b.dq,b.dU,b.tags,
    run.neighbors.state,b.seed,run.dt,run.neighbors.box,run.fx)
_bd_mid_args(run,b::BrownianData)=(run.r,run.v,run.f,b.gamma,b.noise,b.tags,run.neighbors.state,b.seed,run.dt,
    run.neighbors.box,run.fx)

function _bd_launch!(run,kernel,args;launch::Bool=true)
    s=run.neighbors.state; N=length(run.r[1])
    k=@cuda launch=false NL._device_if_valid!(s,kernel,args...)
    launch && k(s,kernel,args...;threads=256,blocks=cld(N,256))
    return nothing
end
enqueue_bd_step!(run,b::BrownianData;launch::Bool=true)=_bd_launch!(run,_bd_step_kernel!,_bd_step_args(run,b);launch)
enqueue_bd_midpoint!(run,b::BrownianData;launch::Bool=true)=_bd_launch!(run,_bd_midpoint_kernel!,_bd_mid_args(run,b);launch)

"""Conditional graph with several (prefix, rebuild) sections in sequence, then
`suffix` (`nothing`: none). Each section's prefix ends by setting its own
conditional handle from the device rebuild request (as
`NL.device_conditional_graph`, which has one)."""
function chained_conditional_graph(sections,suffix,state)
    CC=NL.ConditionalCUDA
    graph=CUDA.CuGraph(); deps=CC.CUgraphNode[]
    for (prefix,rebuild) in sections
        handle=Ref{CC.CUgraphConditionalHandle}()
        CC.cuGraphConditionalHandleCreate(handle,graph,graph.ctx,UInt32(0),CC.CU_GRAPH_COND_ASSIGN_DEFAULT)
        CUDA.@cuda launch=false NL._device_set_rebuild_condition_kernel!(handle[],state)
        CUDA.synchronize()
        before=NL.capture_graph() do
            prefix()
            CUDA.@cuda threads=1 blocks=1 NL._device_set_rebuild_condition_kernel!(handle[],state)
        end
        first=NL._add_child_graph!(graph,before,deps)
        conditional,body=NL._add_conditional_graph!(graph,handle[],first)
        NL._add_child_graph!(body,NL.capture_graph(rebuild),CC.CUgraphNode[])
        deps=CC.CUgraphNode[conditional]
    end
    suffix===nothing || NL._add_child_graph!(graph,NL.capture_graph(suffix),deps)
    return graph
end
