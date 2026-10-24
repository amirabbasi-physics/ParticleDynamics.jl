# Included in ParticleDynamics.DeviceEngine. Experimental list-maintenance policies.
# :adaptive is the original decision (skin/2 displacement OR adaptive age).
# :displacement rebuilds only for an invalid list or a strict skin/2
# violation; the age trigger and interval adaptation are removed. Coverage:
# a list built at rc+skin holds every pair within rc while no particle has
# moved more than skin/2 since the build, checked every step at the actual
# force coordinates.

function validate_policy(policy)
    policy in (:adaptive,:displacement) || throw(ArgumentError("Rebuild policy must be :adaptive or :displacement"))
    return nothing
end

function _displacement_decide_kernel!(state,scratch::CuDeviceVector{T},threshold::T) where {T}
    @inbounds state[NL.DN_ERROR]!=0 && return nothing
    value=zero(T)
    i=threadIdx().x
    while i<=length(scratch)
        @inbounds value=max(value,scratch[i])
        i+=blockDim().x
    end
    largest=NL._displacement_block_max(value)
    if threadIdx().x==1
        @inbounds begin
            scratch[1]=largest
            state[NL.DN_CHECKS]+=1
            state[NL.DN_REQUEST]=0
            diagnostic=state[NL.DN_DIAGNOSTICS]!=0
            diagnostic && (state[NL.DN_REASON]=0)
            if !isfinite(largest)
                state[NL.DN_ERROR]=NL.DN_COORDINATE_ERROR
                state[NL.DN_VALID]=0
            elseif state[NL.DN_VALID]==0
                state[NL.DN_REQUEST]=1
                diagnostic && (state[NL.DN_REASON]=NL.DN_REASON_INVALID)
            elseif largest>threshold
                state[NL.DN_REQUEST]=1
                state[NL.DN_VALID]=0
                diagnostic && (state[NL.DN_REASON]=NL.DN_REASON_DISPLACEMENT)
            end
        end
    end
    return nothing
end

"""Enqueue the coverage check and rebuild decision for the run's policy."""
function decide_policy!(ws::NL.DeviceNeighborWorkspace{T,D},coords,::Val{P}) where {T,D,P}
    if P===:adaptive
        return NL.decide_device_neighbors!(ws,coords...)
    end
    nb=ws.list
    NL._validate_neighbor_arrays(nb,coords...)
    refs=_refs(nb,Val(D))
    @cuda threads=256 blocks=cld(Int(nb.N),256) NL._device_displacement_kernel!(ws.displacement,ws.state,coords,refs,ws.geom)
    @cuda threads=256 blocks=1 _displacement_decide_kernel!(ws.state,ws.displacement,T(0.25)*nb.skin*nb.skin)
    return nothing
end
