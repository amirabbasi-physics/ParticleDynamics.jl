using ParticleDynamics
using CUDA

include(joinpath(@__DIR__, "_example_utils.jl"))

"""
Tiny 2D example that writes the total per-particle configurational virial tensor
into a GSD trajectory.

This uses the `write_virial=true` flag of `write_gsd_frame!`, which adds custom
`particles/virial` and `particles/property/virial` chunks. The component order
matches `virial_components(st)`, so in 2D each row stores `(xx, yy, xy)`.

Virial buffers are refreshed on steps with `compute_energy=true`, so the example
logs only after those force evaluations.
"""

function initialize_square_lattice!(st, box::NTuple{2,T}) where {T<:AbstractFloat}
    N = length(st.rx)
    n_side = ceil(Int, sqrt(N))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side

    rx_host = Vector{T}(undef, N)
    ry_host = Vector{T}(undef, N)

    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = linear ÷ n_side
        rx_host[i] = (ix + T(0.5)) * spacing_x - box[1] / 2
        ry_host[i] = (iy + T(0.5)) * spacing_y - box[2] / 2
    end

    copyto!(st.rx, rx_host)
    copyto!(st.ry, ry_host)
    return st
end

N = maybe_override_int(256, "SIM_NPARTICLES")
box = (40.0f0, 40.0f0)
r_cut = Float32(2^(1 / 6))
sigma = 1.0f0
epsilon = 10.0f0
dt = 2.0f-4
gamma = 50.0f0
temperature = 1.0f0
N_steps = maybe_override_int(1000, "SIM_MAX_STEPS")
N_log = maybe_override_interval(100, N_steps)

st = build_simulation(
    N = N,
    box = box,
    dt = dt,
    cutoff = r_cut,
    skin = 0.4f0,
    cap = Int32(64),
    neigh_interval = 25,
    epsilon = epsilon,
    sigma = sigma,
    gamma = gamma,
    temperature = temperature,
    nonbonded = :wca,
    precision = :f32,
)

initialize_square_lattice!(st, box)
vv = velocityverlet(st; gamma=gamma, temperature=temperature, dt=dt)

gsd_path = joinpath(@__DIR__, "traj2d_virial.gsd")
types = ["A"]

gsd_open(gsd_path) do gsdh
    # Populate force and virial buffers before the first dump.
    step!(st, vv, dt; compute_energy = true)
    write_gsd_frame!(gsdh, st;
                     diameter = sigma,
                     types_names = types,
                     step = st.step,
                     write_virial = true)

    @time for s in 2:N_steps
        if s % N_log == 0
            step!(st, vv, dt; compute_energy = true)
            write_gsd_frame!(gsdh, st;
                             diameter = sigma,
                             types_names = types,
                             step = st.step,
                             write_virial = true)
        else
            step!(st, vv, dt; compute_energy = false)
        end
    end
end

frame = ParticleDynamics.read_gsd_frame!(gsd_path)
virial = frame.particle_properties[:virial]
println("Wrote trajectory with virial tensors to $(gsd_path)")
println("Virial component order: $(virial_components(st))")
println("Read back virial matrix with size $(size(virial)) from the last frame")
