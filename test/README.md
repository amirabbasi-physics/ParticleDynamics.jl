# Running tests

The simulation tests require a working CUDA device. Run the suite with:

```sh
julia --project --startup-file=no test/runtests.jl
```

A small set of algebra and reference checks also runs without CUDA. These
validate the mathematics used by the GPU kernels; the GPU testset reports a
skip if no device is available.

The runner prints a time estimate before loading CUDA and starting the checks.
Allow **30–45 minutes for the complete suite on an RTX 3090 with Julia 1.12**,
based on recent runs of about 28 and 39 minutes. First-time compilation, different
hardware and concurrent workloads can increase this. A run that skips external
references may be shorter; its duration has not been separately calibrated.

## External golden references

Historical trajectory and equilibrium records are kept outside this repository.
Before starting, a terminal run offers to download missing references. The
answer defaults to **no**: press Enter or answer `n` to skip. The archive is
24.8 MB (68.1 MB unpacked); download time is additional to the suite estimate.
The archive SHA-256 and extracted tree hash are pinned in the test helper, and
each dataset's records are checked against its checksum manifest. Downloaded
data stays in Julia's artifact cache, outside the repository, and is reused.

**Publication pending:** the default download URL is not configured yet. Until
the reference archive is published, set `PD_GOLDEN_URL` to its URL (a `file://`
URL to the same archive also works), or supply existing data via
`PD_GOLDEN_ROOT`. A mirror must serve the exact pinned archive. With no URL,
the runner explains that downloading is unavailable; it never requests consent
for a nonexistent download.

Unattended runs, including CI, never wait for keyboard input. Use
`PD_GOLDEN_DOWNLOAD=yes` to explicitly permit a download, or
`PD_GOLDEN_DOWNLOAD=no` to skip it. The default, `ask`, prompts only when stdin
is a terminal and `CI` is not `true` or `1`. A failed or corrupt download is an
error, not a silently downgraded test run. Already available references are
used regardless of this download setting.

Set `PD_GOLDEN_ROOT` to an existing copy of the reference collection, or
override individual dataset directories:

| Dataset | Subdirectory of `PD_GOLDEN_ROOT` | Individual override |
|---|---|---|
| Deterministic trajectories | `v1-20261009T122031` | `PD_GOLDEN_DIR` |
| Engine regression states | `v3-engine-tests` | `PD_ENGINE_GOLDEN_DIR` |
| 2D equilibrium ensembles | `v2-eq-20261009T180331` | `PD_GOLDEN_EQ2D` |
| 3D equilibrium ensembles | `v2-eq3d-liquid-20261009T190013` | `PD_GOLDEN_EQ3D` |

Explicit directory settings take precedence and are never overwritten by the
downloader. Existing `test/golden/` datasets remain supported; otherwise the
runner looks in Julia's artifact cache. Keep local copies or symlinks untracked;
prefer an external directory. Each dataset must contain its `SHA256SUMS`
manifest and the records named there.

When a dataset directory is absent, its comparison testsets emit a warning and
record a skip (`Broken` in Julia's test summary). Other testsets in the same
file continue. Assertions bundled into a golden-dependent testset are skipped
with that comparison; therefore a successful run with missing fixtures is
**partial validation**, not a replacement for the complete regression suite.
An existing but incomplete dataset, malformed checksum manifest, or checksum
mismatch is an error. Checksums are verified once per dataset per process; the
reference readers retain their individual record checks.

For release validation, require both CUDA and all reference datasets:

```sh
PD_GOLDEN_ROOT=/path/to/golden-staged PD_REQUIRE_GOLDEN=1 \
  julia --project --startup-file=no -e \
  'using CUDA; CUDA.functional(true) || error("CUDA required"); include("test/runtests.jl")'
```

`PD_REQUIRE_GOLDEN=1` makes missing data an error. No fixture files need to be
added to Git. The old capture helpers require the historical staged engine;
they cannot recreate its reference data with the current engine.

Once the download URL is configured, unattended full validation can instead use
`PD_GOLDEN_DOWNLOAD=yes PD_REQUIRE_GOLDEN=1`. Declining a download in strict
mode stops before the tests; ordinary mode runs the available checks and marks
the missing comparisons as skipped.

Ordinary test configurations are constructed during the tests. The external
records serve a different purpose: they preserve results from an earlier,
independent implementation. Replacing them with outputs of the implementation
under test would allow a regression to change both sides of the comparison.
`PD_CAPTURE_STAGED=1` is a historical capture option, not a supported way to
prepare these references with the current package.
