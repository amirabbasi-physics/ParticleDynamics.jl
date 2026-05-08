# =========================
#   Energy accumulation (GPU)
# =========================
function _accumulate_energies!(Ekin_accum, Epot_accum, Ekin, Epot)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(Ekin); if i > N; return; end
    @inbounds begin
        Ekin_accum[i] += Ekin[i]
        Epot_accum[i] += Epot[i]
    end
    return
end

"""
    accumulate_energies!(st)

Add the instantaneous `Ekin`/`Epot` buffers into their per-interval accumulators.
Called once per logging interval in `examples/TwoT_2D_LD_VV.jl` before computing
entropy production.
"""
function accumulate_energies!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.Ekin)
    threads = min(256, N)
    blocks  = cld(N, threads)
    k = CUDA.@cuda launch=false _accumulate_energies!(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot)
    k(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot; threads, blocks)
    return nothing
end
