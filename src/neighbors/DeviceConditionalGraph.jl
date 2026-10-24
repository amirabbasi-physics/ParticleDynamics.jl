# Small bridge to the driver bindings already supplied by CUDA.jl. This code
# does not create graphs or query the driver when the module is loaded.
const ConditionalCUDA = CUDA.CUDACore

function _device_set_rebuild_condition_kernel!(handle::UInt64,state)
    value = UInt32(_device_rebuild_requested(state))
    ccall("extern cudaGraphSetConditional",llvmcall,Cvoid,(UInt64,UInt32),handle,value)
    return
end

"""
    capture_graph(f) -> CuGraph

`CUDA.capture(f)` on a fresh stream. CUDA.jl frees an array on the stream that
last used it, from a finalizer that may run on any thread. If that stream is the
one being captured, the free lands in the capture: it fails ("does not refer to
an active allocation" or "not permitted when stream is capturing"), and a child
graph that holds a memory node cannot be embedded (`cuGraphAddChildGraphNode` →
CUDA_ERROR_NOT_SUPPORTED, "Child graph ownership must be moved…"). Seen 12 Oct with
16 Julia threads. Neither disabling finalizers on the capturing thread nor pausing
GC (`CUDA.capture` already does) prevents it: finalizers queued earlier still run
on other threads. A stream created here owns no memory, so no finalizer can target
it during the capture. Relaxed mode: launching on this stream moves the ownership
of arrays last used on the task stream, which synchronizes that (non-capturing)
stream, an API call that global mode forbids during any capture. The graph is
launched later on the task stream, as before.

The stream is non-blocking. A finalizer that frees an array owned by the legacy
default stream (`cuMemFreeAsync` on CU_STREAM_LEGACY) while a *blocking* stream is
capturing fails with CUDA_ERROR_STREAM_CAPTURE_IMPLICIT and invalidates the capture
(seen 14 Oct, about once per full test suite). A non-blocking stream has no
implicit synchronization with the legacy stream. As a backstop, an invalidated
capture (which executes no GPU work) is repeated on a fresh stream, up to
`attempts` times.
"""
# `high=true`: capture on the highest-priority stream; its kernel nodes keep that priority in
# graphs instantiated with CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY (`GRAPH_FLAGS`).
function capture_graph(f;attempts::Int=4,high::Bool=false)
    for _ in 1:attempts
        s=high ? CUDA.CuStream(;flags=ConditionalCUDA.STREAM_NON_BLOCKING,priority=last(CUDA.priority_range())) :
                 CUDA.CuStream(;flags=ConditionalCUDA.STREAM_NON_BLOCKING)
        g=CUDA.stream!(s) do
            CUDA.capture(f;flags=ConditionalCUDA.STREAM_CAPTURE_MODE_RELAXED,throw_error=false)
        end
        g===nothing || return g
        CUDA.synchronize()
    end
    error("CUDA graph capture was invalidated $attempts times in a row.")
end

function _add_child_graph!(parent,child,dependencies)
    node=Ref{ConditionalCUDA.CUgraphNode}()
    ConditionalCUDA.cuGraphAddChildGraphNode(node,parent,dependencies,length(dependencies),child)
    return node[]
end

function _add_conditional_graph!(parent::CUDA.CuGraph,handle,dependency)
    CC=ConditionalCUDA
    params=Ref(CC.CUgraphNodeParams(ntuple(_->UInt8(0),256)))
    node=Ref{CC.CUgraphNode}()
    deps=CC.CUgraphNode[dependency]
    GC.@preserve params parent begin
        p=Base.unsafe_convert(Ptr{CC.CUgraphNodeParams},params)
        p.type=CC.CU_GRAPH_NODE_TYPE_CONDITIONAL
        p.conditional=CC.CUDA_CONDITIONAL_NODE_PARAMS(handle,CC.CU_GRAPH_COND_TYPE_IF,
            UInt32(1),C_NULL,Base.unsafe_convert(CC.CUcontext,parent.ctx))
        CC.cuGraphAddNode_v2(node,parent,deps,C_NULL,length(deps),params)
        # The DRIVER supplies this array. It is not caller-allocated storage,
        # and neither it nor the body graph must be independently destroyed.
        body=unsafe_load(params[].conditional.phGraph_out)
        return node[],body
    end
end

"""
    device_conditional_graph(prefix, rebuild, suffix, state)

Build an experimental graph whose GPU predicate selects the rebuild body.
`prefix` must write DN_REQUEST/DN_ERROR on device; suffix consumers must guard
list validity themselves because a rebuild can fail. Callables must already
be capture-safe and warmed. This function captures but executes no GPU work.

The root owns the conditional handle/body. Child graphs are ordinary captured
graphs copied by the driver, without conditional nodes of their own. Retain
all referenced GPU arrays for the graph lifetime. Driver/compiler support is
checked by actual graph creation and instantiation, not an assumed version.
"""
function device_conditional_graph(prefix::F,rebuild::G,suffix::H,state) where {F,G,H}
    CC=ConditionalCUDA
    graph=CUDA.CuGraph()
    handle=Ref{CC.CUgraphConditionalHandle}()
    CC.cuGraphConditionalHandleCreate(handle,graph,graph.ctx,UInt32(0),CC.CU_GRAPH_COND_ASSIGN_DEFAULT)
    # Compile and device-link the setter BEFORE capture, without executing it
    # outside its owning graph. CUDA.jl links the extern against libcudadevrt.
    CUDA.@cuda launch=false _device_set_rebuild_condition_kernel!(handle[],state)
    CUDA.synchronize()
    before=capture_graph() do
        prefix()
        CUDA.@cuda threads=1 blocks=1 _device_set_rebuild_condition_kernel!(handle[],state)
    end
    first=_add_child_graph!(graph,before,CC.CUgraphNode[])
    conditional,body=_add_conditional_graph!(graph,handle[],first)
    inside=capture_graph(rebuild)
    _add_child_graph!(body,inside,CC.CUgraphNode[])
    after=capture_graph(suffix)
    _add_child_graph!(graph,after,CC.CUgraphNode[conditional])
    return graph
end

function _device_set_flag_condition_kernel!(handle::UInt64,flags,idx::Int32)
    value=UInt32((@inbounds flags[idx])!=zero(eltype(flags)))
    ccall("extern cudaGraphSetConditional",llvmcall,Cvoid,(UInt64,UInt32),handle,value)
    return
end

"""
    flag_conditional_graph(sections, suffix; parallel=nothing, join=nothing) -> CuGraph

Chain of `(prefix, flags, idx, body)` sections: each `prefix` runs, then `body`
runs if and only if `flags[idx] != 0` on the device at that point; `suffix` (or
`nothing`) follows the last section. Used by the cluster-pair backend,
whose step has two device decisions (rebuild, prune). `parallel` (a callable)
is captured as a separate root node that runs concurrently with the whole
chain, and `join` runs after both (the mesh transform overlaps the list
maintenance and pair pass; the gather joins). With `parallel_after = k` the
parallel node waits for the prefix of section `k` (engine steps: the drift). Every part is captured on its own
fresh stream, so no array changes stream ownership inside a capture.
`parallel_high`: the parallel node on the highest-priority stream (honored by graphs instantiated with
node priorities).
Callables must be capture-safe and warmed; this captures but executes no GPU work.
"""
function flag_conditional_graph(sections,suffix;parallel=nothing,join=nothing,parallel_after::Integer=0,side=nothing,
                                parallel_high::Bool=false)
    CC=ConditionalCUDA
    graph=CUDA.CuGraph(); deps=CC.CUgraphNode[]; firsts=CC.CUgraphNode[]
    for (prefix,flags,idx,body) in sections
        if body===nothing   # a section without a decision: its prefix only
            first=_add_child_graph!(graph,capture_graph(prefix),deps)
            push!(firsts,first); deps=CC.CUgraphNode[first]
            continue
        end
        handle=Ref{CC.CUgraphConditionalHandle}()
        CC.cuGraphConditionalHandleCreate(handle,graph,graph.ctx,UInt32(0),CC.CU_GRAPH_COND_ASSIGN_DEFAULT)
        CUDA.@cuda launch=false _device_set_flag_condition_kernel!(handle[],flags,Int32(idx))
        CUDA.synchronize()
        before=capture_graph() do
            prefix()
            CUDA.@cuda threads=1 blocks=1 _device_set_flag_condition_kernel!(handle[],flags,Int32(idx))
        end
        first=_add_child_graph!(graph,before,deps)
        push!(firsts,first)
        conditional,b=_add_conditional_graph!(graph,handle[],first)
        _add_child_graph!(b,capture_graph(body),CC.CUgraphNode[])
        deps=CC.CUgraphNode[conditional]
    end
    suffix===nothing || (deps=CC.CUgraphNode[_add_child_graph!(graph,capture_graph(suffix),deps)])
    if parallel!==nothing
        pdeps=parallel_after>0 ? CC.CUgraphNode[firsts[parallel_after]] : CC.CUgraphNode[]
        push!(deps,_add_child_graph!(graph,capture_graph(parallel;high=parallel_high),pdeps))
    end
    if side!==nothing   # a further branch after the same prefix (or the root), joined like `parallel`
        sdeps=parallel_after>0 ? CC.CUgraphNode[firsts[parallel_after]] : CC.CUgraphNode[]
        push!(deps,_add_child_graph!(graph,capture_graph(side),sdeps))
    end
    join===nothing || _add_child_graph!(graph,capture_graph(join),deps)
    return graph
end
