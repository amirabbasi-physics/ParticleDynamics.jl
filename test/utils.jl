module TestUtils

using Random
using CUDA
using NonEqSimGPU
using NonEqSimGPU: Simulation

export gpu_required, seed_all!,
       build_tiny2d, build_tiny3d,
       set_positions_2d!, set_positions_3d!,
       set_velocities_2d!, set_velocities_3d!,
       msd_2d, msd_3d, kinetic_moments,
       gpu_allfinite, state_allfinite

const DEFAULT_SEED = 0x1234ABCD

function gpu_required()
    ok = CUDA.functional()
    if !ok
        @info "Skipping GPU-only tests: CUDA.functional() == false"
    end
    return ok
end

function seed_all!(seed::Integer=DEFAULT_SEED)
    Random.seed!(seed)
    CUDA.seed!(UInt64(seed))
    return seed
end

function _grid_positions_2d(N::Integer, ::Type{T}; spacing::T=T(1.75)) where {T<:AbstractFloat}
    nx = max(1, ceil(Int, sqrt(N)))
    rx = Vector{T}(undef, N)
    ry = Vector{T}(undef, N)
    for k in 1:N
        i = (k - 1) % nx
        j = (k - 1) ÷ nx
        rx[k] = (T(i) - T(nx - 1) / T(2)) * spacing
        ry[k] = (T(j) - T(cld(N, nx) - 1) / T(2)) * spacing
    end
    return rx, ry
end

function _grid_positions_3d(N::Integer, ::Type{T}; spacing::T=T(1.75)) where {T<:AbstractFloat}
    nx = max(1, ceil(Int, cbrt(N)))
    nxy = nx * nx
    rx = Vector{T}(undef, N)
    ry = Vector{T}(undef, N)
    rz = Vector{T}(undef, N)
    for k in 1:N
        i = (k - 1) % nx
        j = ((k - 1) ÷ nx) % nx
        l = (k - 1) ÷ nxy
        rx[k] = (T(i) - T(nx - 1) / T(2)) * spacing
        ry[k] = (T(j) - T(nx - 1) / T(2)) * spacing
        rz[k] = (T(l) - T(ceil(Int, N / nxy) - 1) / T(2)) * spacing
    end
    return rx, ry, rz
end

function set_positions_2d!(st, rx::AbstractVector, ry::AbstractVector)
    T = eltype(st.rx)
    @assert length(st.rx) == length(rx) == length(ry)
    copyto!(st.rx, T.(rx))
    copyto!(st.ry, T.(ry))
    return st
end

function set_positions_3d!(st, rx::AbstractVector, ry::AbstractVector, rz::AbstractVector)
    T = eltype(st.rx)
    @assert st.rz !== nothing
    @assert length(st.rx) == length(rx) == length(ry) == length(rz)
    copyto!(st.rx, T.(rx))
    copyto!(st.ry, T.(ry))
    copyto!(st.rz, T.(rz))
    return st
end

function set_velocities_2d!(st, vx::AbstractVector, vy::AbstractVector)
    T = eltype(st.vx)
    @assert length(st.vx) == length(vx) == length(vy)
    copyto!(st.vx, T.(vx))
    copyto!(st.vy, T.(vy))
    return st
end

function set_velocities_3d!(st, vx::AbstractVector, vy::AbstractVector, vz::AbstractVector)
    T = eltype(st.vx)
    @assert st.vz !== nothing
    @assert length(st.vx) == length(vx) == length(vy) == length(vz)
    copyto!(st.vx, T.(vx))
    copyto!(st.vy, T.(vy))
    copyto!(st.vz, T.(vz))
    return st
end

function build_tiny2d(;
    N::Int=16,
    T::Type{<:AbstractFloat}=Float32,
    box=(T(20), T(20)),
    cutoff::Real=T(2.5),
    skin::Real=T(0.3),
    cap::Int32=Int32(32),
    neigh_interval::Int=5,
    use_neighborlist::Bool=true,
    epsilon::Real=T(1),
    sigma::Real=T(1),
    gamma::Union{Nothing,Real}=T(1),
    temperature::Real=T(1),
    noise_corr_time::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    dt::Real=T(1e-3),
    nonbonded::Symbol=:lj,
    precision::Symbol=(T == Float64 ? :f64 : :f32),
    unwrapped_positions::Bool=false,
)
    st = Simulation.build_simulation(
        N=N, box=box, cutoff=cutoff, skin=skin, cap=cap, neigh_interval=neigh_interval,
        use_neighborlist=use_neighborlist, epsilon=epsilon, sigma=sigma, gamma=gamma,
        temperature=temperature, dt=dt,
        nonbonded=nonbonded, precision=precision, unwrapped_positions=unwrapped_positions,
    )
    rx, ry = _grid_positions_2d(N, T)
    set_positions_2d!(st, rx, ry)
    set_velocities_2d!(st, fill(zero(T), N), fill(zero(T), N))
    NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
    if unwrapped_positions
        Simulation.sync_unwrapped!(st)
    end
    return st
end

function build_tiny3d(;
    N::Int=16,
    T::Type{<:AbstractFloat}=Float32,
    box=(T(20), T(20), T(20)),
    cutoff::Real=T(2.5),
    skin::Real=T(0.3),
    cap::Int32=Int32(32),
    neigh_interval::Int=5,
    use_neighborlist::Bool=true,
    epsilon::Real=T(1),
    sigma::Real=T(1),
    gamma::Union{Nothing,Real}=T(1),
    temperature::Real=T(1),
    noise_corr_time::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    dt::Real=T(1e-3),
    nonbonded::Symbol=:lj,
    precision::Symbol=(T == Float64 ? :f64 : :f32),
    unwrapped_positions::Bool=false,
)
    st = Simulation.build_simulation(
        N=N, box=box, cutoff=cutoff, skin=skin, cap=cap, neigh_interval=neigh_interval,
        use_neighborlist=use_neighborlist, epsilon=epsilon, sigma=sigma, gamma=gamma,
        temperature=temperature, dt=dt,
        nonbonded=nonbonded, precision=precision, unwrapped_positions=unwrapped_positions,
    )
    rx, ry, rz = _grid_positions_3d(N, T)
    set_positions_3d!(st, rx, ry, rz)
    set_velocities_3d!(st, fill(zero(T), N), fill(zero(T), N), fill(zero(T), N))
    NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    if unwrapped_positions
        Simulation.sync_unwrapped!(st)
    end
    return st
end

function msd_2d(rx0::CuArray{T,1}, ry0::CuArray{T,1},
                rx1::CuArray{T,1}, ry1::CuArray{T,1}) where {T<:AbstractFloat}
    @assert length(rx0) == length(ry0) == length(rx1) == length(ry1)
    return Float64(CUDA.sum((rx1 .- rx0).^2 .+ (ry1 .- ry0).^2) / T(length(rx0)))
end

function msd_3d(rx0::CuArray{T,1}, ry0::CuArray{T,1}, rz0::CuArray{T,1},
                rx1::CuArray{T,1}, ry1::CuArray{T,1}, rz1::CuArray{T,1}) where {T<:AbstractFloat}
    @assert length(rx0) == length(ry0) == length(rz0) == length(rx1) == length(ry1) == length(rz1)
    return Float64(CUDA.sum((rx1 .- rx0).^2 .+ (ry1 .- ry0).^2 .+ (rz1 .- rz0).^2) / T(length(rx0)))
end

function kinetic_moments(vx::CuArray{T,1}, vy::CuArray{T,1};
                         mass::T=one(T)) where {T<:AbstractFloat}
    v2mean = CUDA.sum(vx.^2 .+ vy.^2) / T(length(vx))
    return (mean_v2=Float64(v2mean), mean_kinetic=Float64(T(0.5) * mass * v2mean))
end

function kinetic_moments(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1};
                         mass::T=one(T)) where {T<:AbstractFloat}
    v2mean = CUDA.sum(vx.^2 .+ vy.^2 .+ vz.^2) / T(length(vx))
    return (mean_v2=Float64(v2mean), mean_kinetic=Float64(T(0.5) * mass * v2mean))
end

function gpu_allfinite(x::CuArray{T,1}) where {T<:AbstractFloat}
    bad = CUDA.sum(ifelse.(isfinite.(x), Int32(0), Int32(1)))
    return bad == 0
end

gpu_allfinite(::Nothing) = true

function state_allfinite(st)
    ok = gpu_allfinite(st.rx) && gpu_allfinite(st.ry) &&
         gpu_allfinite(st.vx) && gpu_allfinite(st.vy) &&
         gpu_allfinite(st.fx) && gpu_allfinite(st.fy) &&
         gpu_allfinite(st.Epot) && gpu_allfinite(st.Ekin) &&
         gpu_allfinite(st.dq) && gpu_allfinite(st.dU)
    if st.rz !== nothing
        ok = ok && gpu_allfinite(st.rz) && gpu_allfinite(st.vz) && gpu_allfinite(st.fz)
    end
    return ok
end

end # module
