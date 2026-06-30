# Examples

This directory contains the public runnable examples for the package.

## How to run examples

Run examples from the repository root:

```bash
julia --project examples/3D_BD.jl
```

Use environment variables such as `SIM_NPARTICLES`, `SIM_MAX_STEPS`,
`SIM_LOG_INTERVAL`, `SIM_WARMUP_STEPS`, and `SIM_MAX_SECONDS` to shrink long
runs when you only want a smoke test.

## Public examples

- `2D_active_OU_brownian.jl`
- `2D_active_OU_brownian_free_msd.jl`
- `2D_active_multi_OU_brownian_free_msd.jl`
- `2D_active_thermal_OU_brownian.jl`
- `2D_active_thermal_OU_brownian_free_msd.jl`
- `2D_active_thermal_multi_OU_brownian_free_msd.jl`
- `2D_example_virial.jl`
- `2D_polymer_bonded.jl`
- `3D_BD.jl`
- `3D_LJ_NVE.jl`
- `3D_polymer_melt.jl`
- `SingleT_2D_MD_CSVR.jl`
- `TwoT_LJ2D_LD.jl`

## Helper files

The following files support the examples and are not meant to be run directly:

- `_example_utils.jl`
- `_free_aoup_msd_utils.jl`
- `_restart_workflow.jl`
