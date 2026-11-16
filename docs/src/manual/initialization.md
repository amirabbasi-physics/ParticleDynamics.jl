# Initial Configurations

Lattice-based generators for starting configurations, evaluated on the host.
They return the box, positions centered on the origin, and the lattice-site
indices of each region, so a script can assign types or temperatures per
region. Each docstring lists the returned fields.

## Box size from packing fraction

```@docs
ParticleDynamics.box_from_phi_2d
ParticleDynamics.box_from_phi_3d
```

## 2D generators

```@docs
ParticleDynamics.hex_sites_in_box_2d
ParticleDynamics.hex_random_2d
ParticleDynamics.hex_circle_2d
ParticleDynamics.hex_circle_in_box_2d
ParticleDynamics.hex_circle_plus_random_2d
ParticleDynamics.hex_slab_coexistence_2d
```

## 3D generators

```@docs
ParticleDynamics.fcc_sites_in_box_3d
ParticleDynamics.fcc_random_3d
ParticleDynamics.fcc_slab_coexistence_3d
```

