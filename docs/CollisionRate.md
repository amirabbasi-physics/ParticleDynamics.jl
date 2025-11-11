# Collision Event Rate in NonEqSimGPU

This document describes the GPU implementation that measures collision event rates during MD/LD runs. It explains what is counted, how it is computed, how to configure type-based bins and pairwise cutoffs, and how to consume the results in user job files.

## Overview

- A “collision” is defined as a pair of particles entering the contact region with center–center distance `r_ij < r_cut,ij`.
- The quantity measured is an event rate (entries per unit time), not an occupancy time. Persistent contacts are not repeatedly counted every step.
- Counting is fully on GPU and integrated into the step loop for all supported integrators (VV/GJF, BAOAB-like). It incurs minimal overhead.
- Each geometric pair `(i,j)` is counted at most once per entry by enforcing `i < j` when evaluating edges; no post-hoc division by 2 is needed.

## Definitions

- Contact threshold: `r_cut,ij` is either
  - a global value taken from `st.pair_lj.rcut`, or
  - a per-type pair value provided by the user (see “Per-Pair Cutoffs”).
- Neighbor list: collisions are evaluated over the CSR neighbor graph built by `NeighborLists.NeighborMatrix`. A small skin and reasonable rebuild period are recommended.

## Algorithm (GPU)

- For each particle `i`, iterate its neighbor list entries `(i → j)` from the CSR structure: `neighbors_index`, `neighbors_flat`, `counts`.
- Compute MIC-wrapped separation and squared distance `r2 = |r_i - r_j|^2`.
- Current contact state: `cur = (r2 > 0) && (r2 < r_cut,ij^2)`.
- Previous contact state is stored per edge in a device array `coll_prev` (one byte per edge).
- If `(i < j) && (prev == 0) && cur`, an entry event is detected for the pair (one event per contact entry). Increment the appropriate type bin using a device atomic.
- Update `coll_prev` for this edge to `cur`.
- On neighbor rebuild, the `coll_prev` array is resized (if needed) and re-initialized to the current contact state to avoid spurious “entry” counts caused by reindexing.

Implementation details

- Files: see `src/Collisions.jl` for kernels and API, `src/Simulation.jl` for integration points (called after each positions update; re-init called on NL rebuild).
- Counting is done with CUDA atomics on a short device vector `coll_counts` (one slot per bin).
- Periodic boundary conditions use the same minimum-image logic as force kernels.

## Binning by Particle Type

- When collisions are enabled, a small lookup table `coll_bins[ti,tj]` (device matrix of Int32) maps an unordered type pair `(ti,tj)` to a bin index.
- The provided API currently supports an automatic scheme `bins=:all_pairs`, which builds bins for every unordered pair `(ti,tj)` with `1 ≤ ti ≤ tj ≤ ntypes`.
  - For example, `ntypes=2` yields bins in the order: `(1,1)`, `(1,2)`, `(2,2)`.
  - The LUT stores `-1` for pairs that should be ignored (none in `:all_pairs`).

Enabling and reading results

```
using NonEqSimGPU

# after build_simulation(...)
enable_collision_counting!(st; ntypes=2, bins=:all_pairs)

# inside your logging interval on host
counts = collisions_read_counts!(st)             # Vector{Int64}
rate_11 = counts[1] / (dt * log_interval)
rate_12 = counts[2] / (dt * log_interval)
rate_22 = counts[3] / (dt * log_interval)

# reset device counters for next interval
collisions_reset_counts!(st)
```

Single-type systems

- Use `enable_collision_counting!(st; ntypes=1)`. The only non-zero bin is the first one (`(1,1)`); remaining printed columns can be left as zeros for consistent log formatting.

## Per-Pair Cutoffs (Optional)

- To use different contact radii for different type pairs, upload a dense matrix of distances (not squared) with

```
# rcut_pair[i,j] in distance units; kernels square internally
set_collision_pair_cutoffs!(st, rcut_pair)
```

- If `st.rcut_pair` is set in the simulation for nonbonded forces (e.g., LJ pairs), collisions will automatically use it. Otherwise they fall back to `st.pair_lj.rcut`.

## Integration Points in the Step Loop

- The event update runs immediately after positions are advanced and any neighbor rebuild is performed:
  - VV/GJF: `src/Simulation.jl` right after `vv_positions_soa!` (2D/3D).
  - BAOAB-like: after the `A` substeps (position updates) at the same stage where forces are re-evaluated.
  - Brownian dynamics (EH/EM): right after the final positions update `bd_finish_step_2d!/3d!`.
- On NL rebuild, `coll_prev` is reinitialized:
  - See `src/Simulation.jl` hooks guarded with `_collisions_reinit_on_rebuild!(st)`.

## Normalization (Rates)

- The device counters accumulate entry events since the last reset. If you log every `M = log_interval` steps with step size `Δt = dt`, convert counts to a rate by

```
rate = counts_bin / (M * Δt)
```

- Optionally, divide by `N` to get a per-particle rate, depending on your reporting needs.

## Performance & Limits

- Overhead is typically low: one boolean and one atomic increment per (rare) event on a CSR edge.
- Memory overhead: `coll_prev` is a `CuArray{UInt8,1}` with the same length as the flattened neighbor list (`neighbors_flat`).
- Assumes a `NeighborMatrix` (cell list). The current implementation does not support the `AllPairsNeighborMatrix` sentinel.
- Extremely short contacts may be missed if they occur entirely between neighbor list builds and never appear in the CSR; mitigate by using a small `skin` and reasonable `neigh_interval`.

## API Summary

- `enable_collision_counting!(st; ntypes, bins=:all_pairs)`
  - Allocates device buffers, builds a type-pair LUT, and initializes contact state.
- `set_collision_pair_cutoffs!(st, rcut_pair::AbstractMatrix)`
  - Optional per-pair contact distances; kernels square internally.
- `collisions_read_counts!(st) -> Vector{Int64}`
  - Copies the current device counters to host.
- `collisions_reset_counts!(st)`
  - Zeros the device counters (does not change per-edge contact state).
- `disable_collision_counting!(st)`
  - Releases buffers and disables hooks.

## Examples

Two-temperature example (already wired)

- File: `examples/TwoT_2D_LD_VV.jl`
  - Enables counting with `ntypes=2` and logs rates for `(1,1)`, `(1,2)`, `(2,2)`.

Single-temperature examples (updated)

- Files: `examples/SingleT_2D_LD_VV.jl`, `examples/SingleT_2D_LD_VV_Circle.jl`
  - Enable counting with `ntypes=1` and log the single bin `(1,1)` as “cold/cold coll rate”; other columns are printed as zeros to keep the format consistent.

## Code References

- Kernels and API: `src/Collisions.jl`
- Step-loop hooks and rebuild integration: `src/Simulation.jl`
- Example usage: `examples/TwoT_2D_LD_VV.jl`, `examples/SingleT_2D_LD_VV.jl`
