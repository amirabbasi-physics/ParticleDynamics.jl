# Interactions

Describe interactions with force objects and pass them to `Integrator(...; forces=...)`.
All parameters use the same consistent units as your particle configuration.

## Pair interactions

| Force | Model | Example |
|:--|:--|:--|
| `LennardJones` | Attraction and short-range repulsion | `LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5)` |
| `WCA` | Purely repulsive Lennard-Jones interaction | `WCA(epsilon=1.0, sigma=1.0)` |
| `SoftRepulsive` | Soft repulsive particles | See the [workflow reference](workflow.md#Nonbonded-forces) for parameters |
| `PME`, `PPPM`, `ESP` | Long-range Coulomb interactions | See [electrostatics](electrostatics.md) |

```julia
force = LennardJones(
    epsilon=1.0, sigma=1.0, cutoff=2.5,
    pairs=:neighborlist,
    neighborlist=CellList(buffer=0.5, capacity=256, rebuild_interval=10),
)
```

For small examples, `pairs=:all` avoids a neighbor-list setup. Its cost grows
quadratically with particle count; use neighbor lists for larger systems.

## Choose a neighbor list

`CellList` is a useful starting point for short-range fluids. Charged systems
use the cluster-pair backend automatically. To choose it explicitly, pass
`neighborlist=ClusterPairList(buffer=0.1)` on a force. Cluster pairs support
3D LJ and WCA systems; use the default cell-list route for 2D and soft repulsion.

The buffer extends the list beyond the interaction cutoff. Choose list
capacity for the densest configuration you expect, especially under pressure
control. Overflow stops the simulation instead of silently dropping neighbors.

```@docs
ParticleDynamics.ClusterPairList
```

## Lennard-Jones energy at the cutoff

Choose the energy convention explicitly when comparing models or engines:

- `vdw_modifier=:none` (default) truncates the potential at the cutoff.
- `vdw_modifier=:potential_shift` subtracts the value at the cutoff from each
  interacting pair's energy.

```julia
LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5,
             vdw_modifier=:potential_shift)
```

Forces inside the cutoff are the same for both conventions. Energies differ,
which matters for energy comparisons and Monte Carlo volume moves. Use the
same convention on both sides of a comparison.

## Bonded models and mixtures

Use `Topology` to supply connectivity and combine pair forces with
`HarmonicBondForce`, `FENEBondForce`, angle, dihedral or improper descriptors.
The [bonded guide](bonded_minimization_npt.md) covers parameters and labeling.
For type-dependent pair parameters, use `PairTable`; its constructor is in
the [workflow reference](workflow.md#Nonbonded-forces).

## FENE domain

FENE requires finite `k ≥ 0`, positive finite `R0` with a representable positive
finite square, and finite minimum-image bond lengths strictly below `R0`.
Evaluation throws `DomainError` identifying an incident particle if a bond
reaches or exceeds the limit. The potential and force are not clamped or
continued beyond the physical domain. Near the limit they diverge; choose the
timestep and initial configuration accordingly.

## Physical helpers

These helpers convert between common particle properties and characteristic
times. Keep their inputs consistent with the units of your model.

```@docs
ParticleDynamics.StokesFrictionCoefficient
ParticleDynamics.SphereMass
ParticleDynamics.InertialTime
ParticleDynamics.DiffusiveTime
```

Try the [fluid and polymer examples](../examples.md) for complete scripts.
