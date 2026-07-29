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
- `3D_KG_melt_showcase.jl`
- `3D_polymer_melt.jl`
- `SingleT_2D_MD_CSVR.jl`
- `TwoT_LJ2D_LD.jl`

## Kremer--Grest showcase movie

`3D_KG_melt_showcase.jl` writes a 3,200-bead, 100-chain production trajectory,
chain observables, and final bond lengths to `kg_out/`. The companion
`kg_fresnel_movie.py` turns those outputs into a 16:9 Fresnel path-traced movie
with live chain-size comparisons against Table I of the included Kremer--Grest
(1990) paper. The paper's N=25 values are rescaled by N−1 to the N=32 system.

Fresnel 0.13.8 is distributed on conda-forge (the similarly named PyPI package
is unrelated). Create the renderer environment once:

```bash
micromamba create -f examples/kg_fresnel_environment.yml
```

Render a single final-frame dashboard first:

```bash
micromamba run -n particledynamics-kg-fresnel \
  python examples/kg_fresnel_movie.py --preview --samples 16 --light-samples 4
```

Then render the 12-second, 1080p movie:

```bash
micromamba run -n particledynamics-kg-fresnel \
  python examples/kg_fresnel_movie.py --samples 16 --light-samples 4
```

For a smoother social-media master, use 32 path-tracing samples and a slightly
lower CRF:

```bash
micromamba run -n particledynamics-kg-fresnel \
  python examples/kg_fresnel_movie.py --samples 32 --light-samples 4 --crf 16
```

LinkedIn re-encodes uploads aggressively. Make a delivery copy with a 6 Mbps
H.264 video stream, Rec.709 metadata, and a silent AAC track before uploading:

```bash
ffmpeg -y -i examples/kg_out/kg_showcase.mp4 \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -map 0:v:0 -map 1:a:0 -c:v libx264 -preset slow -tune animation \
  -profile:v high -level:v 4.1 -pix_fmt yuv420p -r 30 -g 60 \
  -b:v 6M -maxrate 6M -bufsize 12M \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a 128k -shortest -movflags +faststart \
  examples/kg_out/kg_showcase_linkedin.mp4
```

The default `--samples 8 --light-samples 2` is faster. Use
`--tracer preview` for a quick direct-lighting draft, or `--stride 2` to path
trace every other saved configuration while retaining the same movie length.

## Helper files

The following files support the examples and are not meant to be run directly:

- `_example_utils.jl`
- `_free_aoup_msd_utils.jl`
- `_restart_workflow.jl`
- `kg_fresnel_environment.yml`
