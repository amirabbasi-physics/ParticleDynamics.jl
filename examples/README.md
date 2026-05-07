# Examples

This directory contains runnable workflow examples plus a small number of helper
files used by those examples.

## How to run examples

Run examples from the repository root:

```bash
julia --project examples/2D_example.jl
```

Use environment variables such as `SIM_NPARTICLES`, `SIM_MAX_STEPS`,
`SIM_LOG_INTERVAL`, `SIM_WARMUP_STEPS`, and `SIM_MAX_SECONDS` to shrink long
runs when you only want a smoke test.

## Example categories

### Basic workflow

- `2D_example.jl`
- `3D_BD.jl`
- `3D_quicktest.jl`

### Force and virial diagnostics

- `2D_example_forces.jl`
- `2D_example_virial.jl`
- `2D_allpairs_quicktest.jl`

### Soft-repulsive and Brownian systems

- `2D_soft_repulsive.jl`
- `2D_soft_repulsive_BD.jl`
- `2D_soft_repulsive_double.jl`

### Active and mixed OU noise

- `2D_active_OU_brownian.jl`
- `2D_active_OU_particles.jl`
- `2D_active_OU_free_msd.jl`
- `2D_mixed_OU_Brownian_BD.jl`
- `2D_mixed_multi_OU_Brownian_BD.jl`

### Mixed-size and pair-table systems

- `2D_wca_mixed_pairs.jl`
- `2D_stencil_two_sizes.jl`
- `2D_stencil_two_sizes_BD.jl`
- `3D_stencil_two_sizes.jl`
- `3D_stencil_two_sizes_BD.jl`
- `TwoT_2D_LD_LJ_pair_eps.jl`

### Bonded polymers

- `2D_polymer_bonded.jl`
- `2D_polymer_bonded_BP.jl`
- `3D_polymer_bonded.jl`

### Two-temperature and filter-driven workflows

- `3D_filters.jl`
- `TwoT_2D_BD_EH.jl`
- `TwoT_2D_LD_BAOA.jl`
- `TwoT_2D_LD_BAOAB.jl`
- `TwoT_2D_LD_GSM.jl`
- `TwoT_2D_LD_VV.jl`
- `TwoT_2D_LD_frac.jl`
- `TwoT_2D_LD_Circle.jl`
- `TwoT_2D_LD_slab.jl`
- `TwoT_LJ2D_LD.jl`

### Freeze and collision workflows

- `TwoT_2D_LD_freeze_hold.jl`
- `TwoT_2D_LD_freeze_spring.jl`

### Molecular dynamics with thermostats

- `SingleT_2D_MD_CSVR.jl`
- `TwoT_LJ2D_MD_CSVR.jl`
- `TwoT_LJ2D_MD_CSVR_slab.jl`
- `TwoT_LJ2D_MD_NHC.jl`
- `TwoT_LJ3D_MD_CSVR.jl`
- `TwoT_LJ3D_MD_NHC.jl`
- `TwoT_softrepulsive2D_MD_NHC.jl`
- `TwoT_SR2D_MD_CSVR_slab.jl`
- `TwoT_2D_MD_NHC_LJ_slab.jl`
- `TwoT_2D_MD_NHC_softrepulsive_slab.jl`

### Restart

- `restart_from_gsd.jl`

## Helper files

The following files support other examples and are not meant to be run
directly:

- `_example_utils.jl`
- `_restart_workflow.jl`
