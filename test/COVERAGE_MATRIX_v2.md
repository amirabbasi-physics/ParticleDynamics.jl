# Coverage Matrix v2 (Phase 4A)

Date: 2026-02-13

| Test file | Testset | Component(s) | Method | Expected outcome |
|---|---|---|---|---|
| `test/test_phase4a_forces.jl` | `LJ magnitude, sign, and momentum balance` | `src/NonBondedForces.jl` (LJ all-pairs) | 2-particle analytic force check at repulsive and attractive radii | force magnitude/sign matches formula; `f1 = -f2`; net force near zero |
| `test/test_phase4a_forces.jl` | `WCA repulsive branch and cutoff behavior` | `src/NonBondedForces.jl` (WCA all-pairs) | 2-particle analytic force below cutoff and zero-force above cutoff | force matches LJ branch for `r < r_c`; force vanishes for `r > r_c` |
| `test/test_phase4a_forces.jl` | `Soft-repulsive analytic force and sign` | `src/NonBondedForces.jl` (soft repulsive all-pairs) | 2-particle analytic force below `sigma` and zero-force above `sigma` | force matches harmonic-rep expression; force vanishes outside support |
| `test/test_phase4a_pair_pbc.jl` | `Pair-matrix parameter selection (type-dependent WCA)` | `src/NonBondedForces.jl` (pair-matrix kernels), `src/Simulation.jl` type IDs | hand-constructed 2-type pair matrix and analytic reference | selected `(sigma, epsilon, rcut)` entry drives computed force |
| `test/test_phase4a_pair_pbc.jl` | `Periodic translation invariance (forces + energy)` | `src/Definitions.jl` PBC conventions, `src/NonBondedForces.jl`, `src/NeighborLists.jl` | translate all particles by box lattice vectors and recompute | forces and total potential energy unchanged |
| `test/test_phase4a_neighbors.jl` | `AllPairs vs Dense parity (LJ, small N)` | `src/NonBondedForces.jl`, `src/NeighborLists.jl` dense backend | cross-backend parity on matched coordinates | force components and total energy agree within tight tolerance |
| `test/test_phase4a_neighbors.jl` | `Dense vs Stencil parity (uniform cutoff)` | `src/NonBondedForces.jl`, `src/NeighborLists.jl` stencil backend | rebuild stencil list and compare with dense backend | force components and total energy agree within tolerance |
| `test/test_phase4a_neighbors.jl` | `Rebuild trigger threshold (controlled displacement)` | `src/NeighborLists.jl` rebuild criterion | deterministic displacement just below/above `skin/2` threshold | rebuild false below threshold, true above threshold |
