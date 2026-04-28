# Getting Started

This page gives a minimal, test-aligned startup path for running `NonEqSimGPU` on a CUDA-capable GPU.

## 1) Install and verify CUDA

```julia
using Pkg
Pkg.add(url="https://github.com/<your-org>/NonEqSimGPU.jl")  # or local dev path
```

```julia
using CUDA
CUDA.functional() || error("CUDA is not functional on this machine.")
```

`NonEqSimGPU` is GPU-only: simulation state arrays are `CuArray`s and integrators/forces are implemented for GPU execution.

## 2) First simulation (2D, tiny and fast)

```julia
using NonEqSimGPU
using CUDA

N = 64
dt = 2.0f-4
sigma = 1.0f0
rcut = Float32(2^(1/6)) * sigma  # WCA cutoff

st = build_simulation(
    N=N,
    box=(40.0f0, 40.0f0),
    cutoff=rcut,
    skin=0.4f0,
    cap=Int32(32),
    neigh_interval=10,
    epsilon=10.0f0,
    sigma=sigma,
    gamma=50.0f0,
    temperature=1.0f0,
    nonbonded=:wca,
    precision=:f32,
)

# Preferred stepping path: keep an explicit integrator spec and reuse it.
vv = velocityverlet(st; gamma=50.0f0, temperature=1.0f0, dt=dt)
for _ in 1:200
    step!(st, vv, dt; compute_energy=false)
end

@show st.step
```

Expected result: `st.step == 200` and finite state arrays.

## 3) Explicit integrator selection

```julia
# Langevin BAOAB
bao = baoab(st; gamma=50.0f0, temperature=1.0f0, dt=dt)
step!(st, bao, dt; compute_energy=true)

# Brownian Euler-Maruyama
em = eulermaruyama(st; gamma=50.0f0, temperature=1.0f0, dt=dt)
step!(st, em, dt; compute_energy=false)
```

Explicit specs are the stepping API. Stochastic parameters such as `gamma`,
temperature, and OU correlation time belong to the integrator constructor, not
to `SimulationState`.

### Important runtime policy

For stochastic paths that divide by friction, `gamma` must be strictly positive.  
Tests enforce informative errors for invalid `gamma <= 0` usage on BAOAB/Brownian/EM paths.

## 4) Simple output writing

```julia
write_xyz!("traj.xyz", st, st.step)
write_observables_csv!("obs.csv", st.step; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)

gsdh = gsd_open("traj.gsd")
write_gsd_frame!(gsdh, st; diameter=1.0f0, types_names=["A"], step=st.step)
gsd_close(gsdh)
```

## 5) Reproducibility basics

```julia
using Random, CUDA
Random.seed!(0xBADC0DE)
CUDA.seed!(UInt64(0xBADC0DE))
```

This gives reproducible runs at the statistical/moment level.  
Bitwise-identical trajectories across different GPUs/toolchains are not guaranteed.

## 6) Where to copy realistic parameter sets

Use repository examples as templates:

- `examples/SingleT_2D_LD_VV.jl`
- `examples/TwoT_2D_LD_BAOAB.jl`
- `examples/TwoT_2D_BD_EH.jl`
- `examples/2D_active_OU_brownian.jl`
- `examples/3D_BD.jl`

These scripts contain production-like parameter scales; for quick iteration/tests, downscale `N` and total steps first.
