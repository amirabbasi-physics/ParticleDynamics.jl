module DeviceConditionalGraphGPUChecks
using Test, CUDA, ParticleDynamics
const NL=ParticleDynamics.NeighborLists

function _prefix!(state,input,counters)
    @inbounds begin
        state[NL.DN_REQUEST]=input[1]
        state[NL.DN_ERROR]=input[2]
        counters[1]+=1
    end
    return
end
function _body!(counters)
    @inbounds counters[2]+=1
    return
end
function _suffix!(counters)
    @inbounds counters[3]+=1
    return
end

@testset "CUDA conditional branch driven by device state" begin
    state=CUDA.zeros(Int64,9); input=CUDA.zeros(Int64,2); counters=CUDA.zeros(Int64,3)
    prefix=() -> (@cuda threads=1 blocks=1 _prefix!(state,input,counters); nothing)
    body=() -> (@cuda threads=1 blocks=1 _body!(counters); nothing)
    suffix=() -> (@cuda threads=1 blocks=1 _suffix!(counters); nothing)
    # Warm ordinary kernels, then clear counters before graph construction.
    prefix(); body(); suffix(); fill!(counters,Int64(0)); CUDA.synchronize()
    graph=NL.device_conditional_graph(prefix,body,suffix,state)
    exec=CUDA.instantiate(graph)
    @test all(counters .== 0) # graph creation/capture must execute nothing
    checks=0; taken=0
    for (request,err) in ((0,0),(1,0),(0,0),(1,0),(1,1),(0,1),(1,0))
        copyto!(input,Int64[request,err])
        CUDA.launch(exec)
        checks+=1; taken+=(request!=0 && err==0)
        @test Array(counters)==[checks,taken,checks]
    end
end
end
