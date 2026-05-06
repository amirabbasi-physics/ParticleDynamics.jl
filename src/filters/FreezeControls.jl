"""
    freeze_particles!(st; filter=All(), mode=:hold, steps=nothing, k=0, include_energy=true)

Freeze the selected particles for a fixed number of steps.
- `mode=:hold` clamps positions while allowing velocities to update (Langevin).
- `mode=:spring` adds a harmonic tether with stiffness `k` to the current positions.
- `steps`: number of steps to keep the freeze active; `nothing` keeps it on until
  [`unfreeze_particles!`](@ref) is called.
- `include_energy`: add tether energy to `Epot` when `mode=:spring`.
"""
function freeze_particles!(st::SimulationState{T};
                           filter::Filter=All(),
                           mode::Symbol=:hold,
                           steps::Union{Nothing,Integer}=nothing,
                           k::Real=0,
                           include_energy::Bool=true) where {T<:AbstractFloat}
    sel = selection(st, filter)
    freeze_particles!(st, sel; mode, steps, k, include_energy)
    return sel
end

function freeze_particles!(st::SimulationState{T}, sel::Selection;
                           mode::Symbol=:hold,
                           steps::Union{Nothing,Integer}=nothing,
                           k::Real=0,
                           include_energy::Bool=true) where {T<:AbstractFloat}
    freeze_particles!(st, sel.device; mode, steps, k, include_energy)
    return sel
end

function freeze_particles!(st::SimulationState{T}, idx::CuArray{Int32,1};
                           mode::Symbol=:hold,
                           steps::Union{Nothing,Integer}=nothing,
                           k::Real=0,
                           include_energy::Bool=true) where {T<:AbstractFloat}
    if steps !== nothing
        steps < 0 && throw(ArgumentError("steps must be >= 0"))
    end
    if mode === :hold
        st.freeze_mode = SimulationCore.FREEZE_HOLD
        st.freeze_k = zero(T)
    elseif mode === :spring
        k <= 0 && throw(ArgumentError("spring k must be > 0"))
        st.freeze_mode = SimulationCore.FREEZE_SPRING
        st.freeze_k = T(k)
    else
        throw(ArgumentError("mode must be :hold or :spring"))
    end
    st.freeze_include_energy = include_energy
    st.freeze_until = steps === nothing ? -1 : st.step + Int(steps)

    if length(idx) == 0
        st.freeze_mode = SimulationCore.FREEZE_NONE
        st.freeze_until = -1
        return idx
    end

    if st.freeze_mask === nothing || length(st.freeze_mask) != length(st.rx)
        st.freeze_mask = CUDA.fill(UInt8(0), length(st.rx))
    else
        fill!(st.freeze_mask, UInt8(0))
    end
    assign_scalar!(st.freeze_mask, idx, UInt8(1))

    if st.freeze_rx === nothing || length(st.freeze_rx) != length(st.rx)
        st.freeze_rx = similar(st.rx)
        st.freeze_ry = similar(st.ry)
        st.freeze_rz = st.rz === nothing ? nothing : similar(st.rz)
    elseif st.rz !== nothing && st.freeze_rz === nothing
        st.freeze_rz = similar(st.rz)
    end

    copyto!(st.freeze_rx, st.rx)
    copyto!(st.freeze_ry, st.ry)
    if st.rz !== nothing
        copyto!(st.freeze_rz, st.rz)
    end
    return idx
end

"""
    unfreeze_particles!(st)

Disable any active freeze/tethering.
"""
function unfreeze_particles!(st::SimulationState)
    st.freeze_mode = SimulationCore.FREEZE_NONE
    st.freeze_until = -1
    return st
end
