# Single-band publication campaign concurrency check

Date: 2026-08-29

## Purpose

Choose how many independent galaxy fits to run at once on the Mac Studio. This
is an operational throughput check, not a replacement for the controlled Stage
4 timing benchmark.

## Conditions

- Host: Mac Studio with Apple M1 Ultra, 16 performance cores, 4 efficiency
  cores, and 128 GiB memory.
- Workload: eight existing S4G `wide_z010` images spanning the measured image
  range from 77 x 77 to 509 x 509 pixels.
- Per galaxy: Isoster `ref_default`, photutils `baseline_median`, and AutoProf
  `baseline`, run serially in that order.
- Total per run: 8 galaxies and 24 fits.
- Numerical-library thread limits: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` all 1.
- The screen was kept awake with `caffeinate`. Python bytecode, Matplotlib, and
  Numba caches were placed outside Dropbox.
- Background load was not publication-clean. The sampled one-minute load was
  3.46 before two workers, 3.53 before four workers, 5.04 before eight workers,
  and 4.31 before the warm serial repeat. Spotlight and macOS media analysis
  were active before the series. Results therefore describe practical campaign
  throughput under the recorded conditions.

## Results

The first serial run took 156.87 s but included a one-time Matplotlib font-cache
build, so the warm 141.60 s repeat is the comparison baseline.

| Concurrent galaxies | Wall time (s) | Fits per second | Speedup vs warm serial | Failed fits | Swap events | Thermal/performance warning |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 141.60 | 0.1695 | 1.000 | 0 | 0 | none |
| 2 | 76.24 | 0.3148 | 1.857 | 0 | 0 | none |
| 4 | 49.08 | 0.4890 | 2.885 | 0 | 0 | none |
| 8 | 29.20 | 0.8219 | 4.849 | 0 | 0 | none |

Eight concurrent galaxies are selected for the first-round mock campaign. It
had the highest measured throughput and did not add execution failures, swaps,
or thermal warnings. Tools and arms remain serial inside each galaxy.

## Repeatability observation

Across the 1/2/4/8 outputs, all eight Isoster profile tables and all eight
photutils profile tables were byte-identical. AutoProf profile tables differed
in all eight galaxies; some runs also stopped with a neighboring number of
isophotes. This is not specific to parallel execution: the warm serial repeat
also differed from the first serial run. AutoProf 1.3.4 reseeds NumPy from the
process ID and clock time inside `Pipeline.Process_Image`, and its isophote fit
uses randomized perturbations. The campaign should retain this as tool-native
stochastic behavior rather than claim bitwise AutoProf repeatability.

## Retained evidence

The five campaign directories, including the initial cold serial run and the
warm repeat, are under:

`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/concurrency/`

Each directory is about 22 MiB and contains its resolved campaign and
environment snapshots, 24 run records, profiles, and residual models.
