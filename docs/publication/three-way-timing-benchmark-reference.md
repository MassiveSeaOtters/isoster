# Three-way timing benchmark reference

Date of campaign: 2026-08-26 to 2026-08-27 (Asia/Shanghai)  
Date of this record: 2026-08-27  
Branch: `benchmarks/three-way-comparison`  
Stage 3 parameter fingerprint: `aa979185be5dbab4`  
Status: Stage 4 complete; raw data retained outside Git

## Purpose and authority

This is the detailed reference for the controlled three-way timing comparison
of Isoster, `photutils.isophote`, and AutoProf. It records the benchmark's
development, frozen design, machine conditions, execution outcome, numerical
results, and limitations. The scientific publication should give a shorter
description and point back to this record when more detail is needed.

This benchmark is **not a survey of galaxy morphology or observing
conditions**. It deliberately uses a small synthetic grid and many repeated
measurements so that timing variation can be reported. It does not replace the
existing two-way archive in `benchmarks/performance/reference_speedup.json`.

The source hierarchy is:

| Role | Source |
|---|---|
| Frozen scientific inputs and accuracy limits | `benchmarks/timing/frozen_stage1_contract.json` |
| Frozen timing plan | `benchmarks/timing/frozen_stage3_parameters.json` |
| Campaign implementation | `benchmarks/timing/run_stage4_campaign.py` |
| Full design history and amendments | `docs/specs/2026-08-22-three-way-benchmark-comparison-design.md` |
| Consolidated raw result | external `stage4_timing_summary.json`, identified below by SHA-256 |
| Human-readable result and caveat record | this document |

If prose here disagrees with a frozen JSON contract or the identified raw
summary, the machine-readable source wins.

## Executive result

Stage 4 completed on its first attempt:

- 132 timing arms were run;
- 9,900 records were retained;
- 147,225 timed calls were made after applying the frozen per-arm batch sizes;
- all 9,900 executions succeeded;
- all 9,900 profiles met the frozen radial-coverage rule;
- all 9,900 records were free of a recorded thermal warning or recognized
  competing process; and
- all 9,900 records were timing-eligible.

The primary measurement is **fit plus harness**, defined below. For this
measurement, Isoster was faster than photutils, and photutils was faster than
AutoProf, in every one of the 44 matched configuration triples. The ranges of
the three session medians did not overlap in any of the 132 pairwise tool
comparisons. These ranges are descriptive run-to-run ranges, not confidence
intervals.

The following ratios are calculated independently for every matched
configuration and then summarized across configurations. They are not ratios
of pooled, unmatched samples.

| Scope | Ratio | Minimum | Median | Maximum | Configurations |
|---|---:|---:|---:|---:|---:|
| Fixed aperture | photutils / Isoster | 11.04 | 29.90 | 128.50 | 20 |
| Fixed aperture | AutoProf / Isoster | 103.65 | 341.18 | 2709.34 | 20 |
| Fixed aperture | AutoProf / photutils | 9.31 | 12.18 | 21.14 | 20 |
| End to end | photutils / Isoster | 19.03 | 37.33 | 97.68 | 24 |
| End to end | AutoProf / Isoster | 20.00 | 63.01 | 105.41 | 24 |
| End to end | AutoProf / photutils | 1.03 | 1.31 | 2.47 | 24 |

The very large fixed-aperture AutoProf ratios include its real worker harness.
When only the AutoProf pipeline step is used, while the Isoster and photutils
timers remain unchanged, the comparison is narrower:

| Scope | Fit-only ratio | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Fixed aperture | AutoProf / Isoster | 38.06 | 93.93 | 276.06 |
| Fixed aperture | AutoProf / photutils | 2.12 | 3.34 | 3.75 |
| End to end | AutoProf / Isoster | 16.02 | 60.20 | 88.50 |
| End to end | AutoProf / photutils | 0.82 | 1.19 | 2.44 |

Thus, the defensible user-facing result is the ordering from the primary
fit-plus-harness measurement. It would be incorrect to turn that result into a
claim that every internal AutoProf fitting step is slower than photutils: some
end-to-end AutoProf fit-only medians are lower than the corresponding
photutils medians.

## Raw archive and retention policy

The consolidated result is deliberately outside the repository:

```text
/Users/shuang/.cache/isoster/stage4_campaign_20260826/stage4_timing_summary.json
```

| Item | Value |
|---|---|
| Size | 213,357,401 bytes (about 203.5 MiB) |
| SHA-256 | `f31bb2c7773ef2a3f0d70b18cfd5c27f19f3758e43459a1f18f8a072525de579` |
| Record count | 9,900 |
| Timing-summary arm count | 132 |
| Accuracy-evaluation exceptions | 0 |

The completed campaign initially occupied about 25 GB because each of three
8.3 GB session workspaces preserved per-call FITS, `.prof`, `.aux`, log, and
duplicate profile files. On 2026-08-27, after the benchmark result and this
reference were accepted, those three intermediate workspaces were permanently
deleted. They contained 19,581 files and occupied 24.721 GiB. The consolidated
summary, three session JSON files, baseline, monitoring trace, and six AutoProf
logs were retained. The external campaign directory is now about 414 MB.

Before deletion, the measured portions of the three session record streams
were compared with the consolidated summary after excluding the five verdicts
calculated only during final aggregation: `timing_eligible`, the three accuracy
statuses, and `headline_eligible`. Both sides contained 9,900 records and 9,900
profiles and produced the same canonical SHA-256 digest:
`23a8ebaddc0609c50fb030c94d9da801d1690e94247782752059310e3788708d`.

The retained JSON and monitoring files are:

| File | SHA-256 |
|---|---|
| `baseline.json` | `d23e4257373cf9ccc8de65e1bc1f193e6016c64616d1e2924c1339b018f7bf1a` |
| `attempt_01/attempt.json` | `06a49c82d0081ecc9532a0343aaeda00eae0a7d9d44a4f6cacf294f781de4f71` |
| `attempt_01/session_00.json` | `f5f8ca07221c1c081f07b25c4b0fc976595c5df27d80b1ab3c0dd61cc7dc35fd` |
| `attempt_01/session_01.json` | `1a0afc5c5b8cda92de3d1f6e4a7e84dbe611139d4fb02a771625a2ae92405aa3` |
| `attempt_01/session_02.json` | `076d75a844cd5b598e5059cf4a5f2e2de42c381749fc6aa2109f08b412ffe726` |

The six logs were copied to `attempt_01/preserved_logs/` and verified against
their sources before deletion:

| Preserved log | SHA-256 |
|---|---|
| `session_00_autoprof.log` | `3879821c9d33ea7c41db1ae005c8974d95e6962b9f570c1183d0c3c6eb68c467` |
| `session_00_worker.stderr.log` | `f2ee753306087e800842ddee6ee6c1a8b189c927d7e91a6ff16d9601b8ae8812` |
| `session_01_autoprof.log` | `4a66067378734598d57dd15eb58a6d923d1f23ff6d92c937f982ab34e1a37419` |
| `session_01_worker.stderr.log` | `9b09429978633580ca71641ea9a2ff9ee89a5231f671db92127b08d605888c14` |
| `session_02_autoprof.log` | `5cd582bdc0c21670396150fd07afe83caafae5d88e97a303e9ee8157bb417af8` |
| `session_02_worker.stderr.log` | `e38a8b76788681aba9bd46af29e4baf9c41e68b28cb469c7b13c1fbcc168209e` |

None of the remaining external archive belongs in Git. This tracked document
contains only reduced numerical results, checksums, and the information needed
to interpret them.

## How the benchmark reached Stage 4

### Part A: coefficient convention

Before timing higher harmonics, Part A established how the three tools encode
their third- and fourth-order coefficients. It used exact one-dimensional
response tests and planted-deviation images at matched apertures. This was
necessary because a timing comparison of nominally identical harmonic tasks is
not meaningful if the outputs use different signs, angle bases, or
normalizations. Part A is archived and gated separately; its detailed record
is in the design specification and the `benchmarks/harmonic_scale/` archives.

### Stage 1: frozen scientific contract

Stage 1 froze the fixtures, planted deviations, two noise arms, radial grids,
accuracy definitions, thread limits, and benchmark host. This prevented the
scientific task or acceptance rules from being adjusted after timings were
seen.

The contract was amended for the Mac Studio rather than silently reusing the
original MacBook Pro M1 Max assumptions. The final host contract is the Apple
Silicon model `Mac13,2` with 20 logical CPU cores.

### Stage 2: calibration and recovery

Stage 2 measured per-arm duration and run-to-run spread to choose batching and
session counts. Several early runs exposed weaknesses in the initial machine
contamination policy:

1. one isolated load sample could stop a long otherwise clean session;
2. AutoProf's numerical libraries initially used 20 threads despite the
   intended single-thread policy; and
3. the one-minute load average remained elevated after legitimate benchmark
   work on this 20-core machine, so treating it as an in-session abort signal
   repeatedly rejected the benchmark's own load.

The final policy retained a clean idle preflight, one thread for numerical
libraries, immediate rejection for thermal warnings or recognized competing
processes, and **record-only** in-session load. This change is scientifically
important: the campaign does not pretend the load average stayed below an idle
ceiling while the benchmark itself was running.

The retained Stage 2 calibration initially contained 1,818 failed AutoProf
records. The persistent AutoProf worker had accumulated nested sampling-probe
wrappers and eventually exited, which erased all later AutoProf coverage. The
shared worker was fixed so probe installation replaces previous wrappers and a
pipeline failure no longer terminates the worker. A separate recovery selected
only the failed AutoProf request identifiers, recreated their frozen inputs,
and reran 606 records in each of the three original sessions. It did not
overwrite the source calibration or alter the accuracy thresholds.

The recovered Stage 2 summary contained 9,900 successful, complete, clean
records. Its frozen checksum is
`889f533d4b1176e4b06de31906b40a09df53580cf3533234281ef94e67d2887a`.

### Stage 3: timing freeze and accuracy decision

Stage 2 recommended three sessions for 124 of 132 arms. Eight Isoster arms
would have required between 5 and 33 sessions to meet the calibration's 5%
relative target. Because this timing comparison is supporting evidence and the
matched speed differences are much larger, Stage 3 froze three sessions for
all arms and required the resulting session-median ranges to be reported.

Stage 3 also froze:

- 25 repetitions per arm per session;
- an independent campaign seed block, `60260822`;
- tool-order seed `20260826`;
- one per-arm batch size measured to produce at least 0.1 seconds of work in
  Stage 2; and
- primary reporting from `fit_plus_harness_s`.

Only 15 of the 132 Stage 2 arms passed every frozen accuracy condition, and no
end-to-end arm passed. The numerical limits were not relaxed. Instead, Stage 3
made accuracy descriptive and defined timing eligibility only from successful
execution, complete radial coverage, and clean thermal/process indicators.
That separation carried unchanged into Stage 4.

### Stage 4: independent campaign

Stage 4 used the independent campaign noise seeds and a new output directory.
It did not reuse Stage 2 timings. The runner performed a full idle preflight,
calibrated the two Python interpreters, opened fresh session processes, used a
persistent AutoProf worker within each session, monitored external conditions,
and accepted only a complete clean attempt. The first attempt completed all
three sessions successfully.

An equivalent invocation is shown below. The runner will now refuse this exact
command because the output directory already exists; an intentional rerun must
use a new output directory so the accepted archive cannot be overwritten.

```bash
/Users/shuang/.venvs/isoster/bin/python \
  benchmarks/timing/run_stage4_campaign.py \
  --output /Users/shuang/.cache/isoster/stage4_campaign_20260826 \
  --autoprof-python /Users/shuang/.venvs/isoster_autoprof_py310/bin/python
```

The runner itself injects the frozen single-thread environment into each
session process.

## Scientific inputs and factorial design

### Common image model

All fixtures are analytic Sérsic images rendered at pixel centers without
subpixel integration. Their common inputs are:

| Input | Frozen value |
|---|---|
| Effective-radius intensity, `I_e` | 100 |
| Ellipticity | 0.3 |
| Position angle | 0 radians |
| Background | 0 |
| PSF | none |
| Mask | none |
| Variance | none for noiseless; constant for Gaussian arm |
| Harmonic basis | physical polar angle from the major axis |
| Planted cosine amplitudes | order 3: 0.02; order 4: 0.03 |
| Planted sine amplitudes | order 3: 0.015; order 4: 0.01 |

The six fixtures are:

| Fixture | Sérsic `n` | `R_e` (px) | Image shape | Fixed radii (px) | Scope |
|---|---:|---:|---:|---|---|
| `sersic_n2_compact` | 2 | 25 | 241 x 241 | 12, 18, 25, 35, 45 | both |
| `sersic_n4_extended` | 4 | 40 | 321 x 321 | 18, 28, 40, 55, 70 | both |
| `size_ladder_481` | 2 | 50 | 481 x 481 | 25, 37.5, 50, 70, 90 | both |
| `size_ladder_961` | 2 | 100 | 961 x 961 | 50, 75, 100, 140, 180 | both |
| `size_ladder_1921` | 2 | 200 | 1921 x 1921 | 100, 150, 200, 280, 360 | both |
| `wide_canvas_961` | 2 | 25 | 961 x 961 | not used | end to end only |

The size ladder scales both the galaxy and canvas. `wide_canvas_961` instead
holds the compact `n=2` galaxy fixed and expands only the canvas, so it tests
whole-image overhead rather than a larger galaxy. It is therefore excluded
from fixed-aperture extraction.

### Noise arms

The two noise arms are:

- `noiseless`: the analytic image, repeated for timing but with no random
  realization;
- `gaussian_reference`: independent Gaussian noise from NumPy PCG64, with
  `sigma = I_e / 100 = 1`, plus a constant variance map where the tool route
  accepts one.

The 25 Gaussian realizations use seeds `60260822` through `60260846`. The same
realization seeds are reused across tools and sessions, so tool comparisons are
matched rather than confounded by different random images.

These mocks are not HSC-like or representative of real survey noise. They have
no correlated noise, PSF, mask, sky-estimation uncertainty, or detector
artifacts.

### Arm count

The 132 arms are the Cartesian product:

| Scope | Fixtures | Tools | Noise arms | Harmonics | Arms |
|---|---:|---:|---:|---:|---:|
| Fixed aperture | 5 | 3 | 2 | off/on | 60 |
| End to end | 6 | 3 | 2 | off/on | 72 |
| Total | | | | | 132 |

Each arm has 25 records in each of three sessions, giving 75 records per arm
and 9,900 records overall. The tool order is deterministically shuffled for
each session and realization so that one tool does not always run first or
last.

## What was timed

### Fixed-aperture extraction

This scope imposes the same five centers, ellipticities, position angles, and
semi-major axes on all three tools. It measures extraction and optional
third/fourth-order harmonic evaluation. It does **not** measure geometry
fitting.

- Isoster uses template-based forced photometry.
- Photutils builds fixed `EllipseSample` objects at the requested radii.
- AutoProf uses its standard forced pipeline and is required to report
  line-interpolated sampling rather than isophotal-band sampling.
- Output writing is excluded as a scientific task in this scope.

### End-to-end fitting

This scope starts each tool from the same input center, ellipticity, position
angle, and initial semi-major axis, then allows the tool to use its native
radial grid, convergence rule, sigma clipping, interpolation, background
handling, and stopping rule. Output writing is included.

The result is a comparison of real user-facing pipelines, not an assertion
that all three codes performed an identical number of rings or iterations. The
median ring counts are therefore printed beside every end-to-end timing below.

### Timing fields

`time.perf_counter()` is used for wall-clock timing.

- `fit_only_s` is the measured fitting or extraction step. For Isoster and
  photutils it is identical to the outer timer.
- `fit_plus_harness_s` is the primary user-facing measurement. For AutoProf it
  includes request serialization, IPC, required file preparation and reading,
  and the measured pipeline step. Persistent-worker startup is excluded and
  recorded separately.
- `harness_s` is the difference between those two fields.
- A batch repeats an arm the frozen number of times, then divides by that
  count. Batch counts range from one for slow arms to 153 for the fastest
  fixed-aperture Isoster arm.
- Each table timing is the median over all 75 per-call records. Brackets give
  the minimum and maximum of the three per-session medians.

The AutoProf harness is substantial in fixed-aperture mode: its median share
across arms is 72.23%, with a range from 62.93% to 89.87%. In end-to-end mode,
its median share is 6.27%, with a range from 1.21% to 20.21%. This is why both
the primary and fit-only summaries are retained.

## Machine and execution conditions

### Hardware and software

| Component | Value |
|---|---|
| Machine | Mac Studio, model `Mac13,2` |
| Chip | Apple M1 Ultra |
| CPU | 20 cores: 16 performance and 4 efficiency |
| Memory | 128 GB |
| Operating system | macOS 15.7.3, build 24G419 |
| Project Python | CPython 3.12.11 |
| Isoster | 1.0.0 |
| NumPy / SciPy | 2.3.5 / 1.17.0 |
| Astropy / photutils | 7.2.0 / 2.3.0 |
| Numba / llvmlite | 0.63.1 / 0.46.0 |
| AutoProf Python | CPython 3.10.21 |
| AutoProf | 1.3.4 |
| AutoProf NumPy / SciPy | 1.26.4 / 1.15.3 |
| AutoProf Astropy / photutils | 5.3.4 / 1.5.0 |

The AutoProf Python and package versions are embedded in every Stage 4 session
record. The full project package list above was verified from the retained
virtual environment on 2026-08-27. The Stage 4 summary itself does not embed
the project Git commit or the complete Isoster environment. The repository was
clean at `0b3e2476e8e036792487ae54f4223ef973fb0d8d` when this record was prepared,
with no newer commit after the campaign, but that association is post-run
provenance rather than a field embedded in the archive.

### Thread controls

Every session recorded the following environment:

```text
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

This prevents a tool from receiving an accidental multi-thread advantage and
was especially important for the AutoProf environment.

### Preflight and monitoring

The idle preflight took 30 one-minute-load samples at 10-second intervals. The
frozen acceptance ceiling applied to the **median**, not every sample.

| Preflight statistic | Value |
|---|---:|
| Minimum load | 1.86 |
| Median load | 2.895 |
| Maximum load | 5.27 |
| Frozen median ceiling | 4.0 |
| Thermal warnings | none |
| Recognized competing processes | none |
| Host-contract mismatch | none |

The accepted attempt ran from 2026-08-26 10:26:18 UTC to 17:24:34 UTC, or
18:26:18 to 01:24:34 local time. The monitored interval was 6.97 hours.

| In-session statistic | Value |
|---|---:|
| Monitoring samples | 2,490 |
| Minimum load | 2.09 |
| Median load | 3.25 |
| Maximum load | 8.87 |
| Samples with thermal warning | 0 |
| Samples with recognized competing process | 0 |
| Accepted attempt | 1 |

The maximum load of 8.87 is retained rather than hidden. Under the final frozen
policy, in-session load is descriptive because it includes work from the
benchmark itself. A thermal warning or recognized `codex`, `claude`, `pytest`,
or busy `uv` process would have contaminated the whole attempt.

### Interpreter calibration

The same CPU-bound calibration workload was executed seven times in both
interpreters before the sessions:

| Interpreter | Median calibration time |
|---|---:|
| Project Python 3.12 | 0.224149 s |
| AutoProf Python 3.10 | 0.176323 s |
| AutoProf / project ratio | 0.786635 |

The calibration is reported but never subtracted or used to rescale a tool's
time. On this workload, the older AutoProf interpreter was actually faster, so
the measured AutoProf disadvantage cannot be summarized as a simple Python
3.10 penalty.

Persistent AutoProf worker startup took 1.033, 1.136, and 1.144 seconds in the
three sessions. Startup is not included in per-arm timing.

## Detailed timing results

The tables below report primary `fit_plus_harness_s` timings. Each cell is
`median [minimum session median, maximum session median]`. `H` indicates
whether orders 3 and 4 were measured. `P/I`, `A/I`, and `A/P` are photutils /
Isoster, AutoProf / Isoster, and AutoProf / photutils. The ring-count triplet is
Isoster / photutils / AutoProf.

### Fixed-aperture extraction

Times are milliseconds. Every tool returned the five requested rings.

| Fixture | Noise | H | Isoster | photutils | AutoProf | P/I | A/I | A/P |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| `sersic_n2_compact` | Gaussian | off | 0.493 [0.490, 0.497] | 9.681 [9.674, 9.687] | 98.476 [98.309, 98.805] | 19.62 | 199.59 | 10.17 |
| `sersic_n2_compact` | Gaussian | on | 0.953 [0.947, 0.958] | 10.527 [10.512, 10.547] | 99.264 [99.106, 99.785] | 11.04 | 104.12 | 9.43 |
| `sersic_n2_compact` | noiseless | off | 0.493 [0.487, 0.496] | 9.654 [9.639, 9.684] | 97.722 [96.942, 98.262] | 19.59 | 198.32 | 10.12 |
| `sersic_n2_compact` | noiseless | on | 0.950 [0.945, 0.958] | 10.505 [10.477, 10.531] | 98.470 [97.918, 99.122] | 11.06 | 103.65 | 9.37 |
| `sersic_n4_extended` | Gaussian | off | 0.531 [0.525, 0.531] | 14.434 [14.416, 14.443] | 142.326 [142.321, 142.591] | 27.18 | 268.02 | 9.86 |
| `sersic_n4_extended` | Gaussian | on | 1.023 [1.016, 1.052] | 15.378 [15.366, 16.022] | 143.796 [143.307, 144.688] | 15.03 | 140.55 | 9.35 |
| `sersic_n4_extended` | noiseless | off | 0.531 [0.526, 0.533] | 14.434 [14.366, 14.482] | 141.170 [140.841, 141.458] | 27.18 | 265.87 | 9.78 |
| `sersic_n4_extended` | noiseless | on | 1.022 [1.015, 1.024] | 15.358 [15.340, 15.376] | 142.994 [142.542, 143.311] | 15.03 | 139.96 | 9.31 |
| `size_ladder_481` | Gaussian | off | 0.561 [0.555, 0.563] | 18.725 [18.676, 18.863] | 234.277 [233.410, 235.046] | 33.36 | 417.37 | 12.51 |
| `size_ladder_481` | Gaussian | on | 1.104 [1.101, 1.140] | 19.791 [19.743, 20.141] | 236.016 [235.839, 236.453] | 17.92 | 213.69 | 11.93 |
| `size_ladder_481` | noiseless | off | 0.561 [0.557, 0.563] | 18.718 [18.700, 18.762] | 232.623 [232.201, 233.121] | 33.34 | 414.33 | 12.43 |
| `size_ladder_481` | noiseless | on | 1.108 [1.099, 1.143] | 19.873 [19.732, 20.461] | 233.545 [233.212, 234.609] | 17.94 | 210.80 | 11.75 |
| `size_ladder_961` | Gaussian | off | 0.685 [0.680, 0.687] | 43.131 [42.584, 43.985] | 714.229 [712.196, 717.185] | 62.97 | 1042.82 | 16.56 |
| `size_ladder_961` | Gaussian | on | 1.381 [1.374, 1.382] | 45.441 [43.980, 45.855] | 715.379 [714.750, 716.122] | 32.92 | 518.20 | 15.74 |
| `size_ladder_961` | noiseless | off | 0.685 [0.679, 0.687] | 43.418 [42.763, 43.615] | 705.681 [705.403, 706.887] | 63.40 | 1030.52 | 16.25 |
| `size_ladder_961` | noiseless | on | 1.381 [1.377, 1.386] | 45.042 [44.921, 45.196] | 704.856 [704.300, 705.272] | 32.62 | 510.47 | 15.65 |
| `size_ladder_1921` | Gaussian | off | 0.939 [0.934, 0.961] | 120.320 [119.562, 121.179] | 2543.172 [2541.503, 2545.610] | 128.18 | 2709.34 | 21.14 |
| `size_ladder_1921` | Gaussian | on | 1.929 [1.922, 1.939] | 121.848 [121.547, 122.415] | 2538.179 [2535.114, 2543.646] | 63.16 | 1315.62 | 20.83 |
| `size_ladder_1921` | noiseless | off | 0.933 [0.929, 0.960] | 119.940 [119.508, 120.684] | 2506.375 [2505.623, 2510.082] | 128.50 | 2685.22 | 20.90 |
| `size_ladder_1921` | noiseless | on | 1.939 [1.930, 1.956] | 121.740 [121.265, 122.083] | 2501.889 [2501.182, 2503.596] | 62.80 | 1290.61 | 20.55 |

### End-to-end fitting

Times are seconds. Ring counts differ because each pipeline used its native
radial grid and stopping rule.

| Fixture | Noise | H | Isoster | photutils | AutoProf | P/I | A/I | A/P | Rings I/P/A |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `sersic_n2_compact` | Gaussian | off | 0.044 [0.044, 0.044] | 2.325 [2.319, 2.331] | 2.871 [2.810, 2.970] | 52.77 | 65.17 | 1.23 | 58/60/55 |
| `sersic_n2_compact` | Gaussian | on | 0.060 [0.059, 0.061] | 2.327 [2.324, 2.346] | 2.966 [2.959, 3.004] | 38.73 | 49.36 | 1.27 | 58/60/55 |
| `sersic_n2_compact` | noiseless | off | 0.044 [0.044, 0.045] | 1.345 [1.340, 1.345] | 3.003 [2.971, 3.062] | 30.33 | 67.72 | 2.23 | 58/60/55 |
| `sersic_n2_compact` | noiseless | on | 0.059 [0.059, 0.060] | 1.352 [1.349, 1.357] | 3.025 [2.998, 3.057] | 22.80 | 51.03 | 2.24 | 58/60/55 |
| `sersic_n4_extended` | Gaussian | off | 0.048 [0.048, 0.048] | 2.229 [2.218, 2.231] | 4.279 [4.236, 4.307] | 46.26 | 88.80 | 1.92 | 61/63/58 |
| `sersic_n4_extended` | Gaussian | on | 0.071 [0.071, 0.072] | 2.236 [2.234, 2.241] | 4.327 [4.274, 4.416] | 31.36 | 60.68 | 1.93 | 61/63/58 |
| `sersic_n4_extended` | noiseless | off | 0.048 [0.048, 0.050] | 1.748 [1.747, 1.749] | 4.321 [4.313, 4.348] | 36.06 | 89.15 | 2.47 | 61/63/58 |
| `sersic_n4_extended` | noiseless | on | 0.070 [0.070, 0.071] | 1.758 [1.751, 1.761] | 4.312 [4.288, 4.453] | 25.09 | 61.52 | 2.45 | 61/63/58 |
| `size_ladder_481` | Gaussian | off | 0.055 [0.055, 0.056] | 3.601 [3.601, 3.603] | 3.916 [3.740, 3.985] | 64.94 | 70.62 | 1.09 | 65/67/63 |
| `size_ladder_481` | Gaussian | on | 0.094 [0.094, 0.094] | 3.617 [3.615, 3.633] | 3.801 [3.793, 3.826] | 38.60 | 40.56 | 1.05 | 65/67/63 |
| `size_ladder_481` | noiseless | off | 0.055 [0.055, 0.056] | 2.511 [2.504, 2.523] | 3.960 [3.925, 4.053] | 45.25 | 71.35 | 1.58 | 65/67/63 |
| `size_ladder_481` | noiseless | on | 0.093 [0.092, 0.094] | 2.522 [2.517, 2.527] | 4.004 [3.967, 4.044] | 27.15 | 43.10 | 1.59 | 65/67/63 |
| `size_ladder_961` | Gaussian | off | 0.073 [0.073, 0.074] | 5.151 [5.134, 5.166] | 5.951 [5.873, 6.032] | 70.36 | 81.28 | 1.16 | 72/74/70 |
| `size_ladder_961` | Gaussian | on | 0.196 [0.193, 0.198] | 5.183 [5.134, 5.202] | 5.752 [5.611, 5.825] | 26.46 | 29.36 | 1.11 | 72/74/70 |
| `size_ladder_961` | noiseless | off | 0.073 [0.073, 0.073] | 5.005 [4.986, 5.025] | 6.097 [6.010, 6.135] | 68.50 | 83.45 | 1.22 | 72/74/70 |
| `size_ladder_961` | noiseless | on | 0.198 [0.195, 0.199] | 5.027 [5.003, 5.033] | 6.062 [5.919, 6.096] | 25.44 | 30.68 | 1.21 | 72/74/70 |
| `size_ladder_1921` | Gaussian | off | 0.111 [0.108, 0.113] | 10.704 [10.659, 10.763] | 11.050 [10.946, 11.169] | 96.23 | 99.34 | 1.03 | 80/82/77 |
| `size_ladder_1921` | Gaussian | on | 0.563 [0.558, 0.566] | 10.725 [10.668, 10.790] | 11.271 [11.085, 11.393] | 19.03 | 20.00 | 1.05 | 80/82/77 |
| `size_ladder_1921` | noiseless | off | 0.110 [0.106, 0.110] | 10.709 [10.661, 10.769] | 11.557 [11.122, 11.710] | 97.68 | 105.41 | 1.08 | 80/82/77 |
| `size_ladder_1921` | noiseless | on | 0.556 [0.548, 0.568] | 10.709 [10.694, 10.760] | 11.533 [11.002, 11.669] | 19.26 | 20.74 | 1.08 | 80/82/77 |
| `wide_canvas_961` | Gaussian | off | 0.063 [0.062, 0.063] | 3.001 [2.973, 3.011] | 4.038 [3.985, 4.156] | 47.94 | 64.50 | 1.35 | 72/63/62 |
| `wide_canvas_961` | Gaussian | on | 0.096 [0.094, 0.096] | 3.013 [3.010, 3.015] | 4.026 [3.975, 4.064] | 31.55 | 42.16 | 1.34 | 72/63/62 |
| `wide_canvas_961` | noiseless | off | 0.072 [0.072, 0.073] | 5.227 [5.213, 5.258] | 7.026 [6.943, 7.175] | 72.16 | 96.99 | 1.34 | 72/75/70 |
| `wide_canvas_961` | noiseless | on | 0.185 [0.184, 0.189] | 5.246 [5.231, 5.274] | 7.086 [6.913, 7.094] | 28.33 | 38.27 | 1.35 | 72/75/70 |

## Secondary timing observations

### Harmonic-evaluation cost

For each fixture and noise arm, the harmonics-on median was divided by its
matched harmonics-off median:

| Scope | Tool | Minimum | Median | Maximum |
|---|---|---:|---:|---:|
| Fixed aperture | Isoster | 1.924 | 1.970 | 2.077 |
| Fixed aperture | photutils | 1.013 | 1.059 | 1.088 |
| Fixed aperture | AutoProf | 0.998 | 1.006 | 1.013 |
| End to end | Isoster | 1.337 | 1.682 | 5.071 |
| End to end | photutils | 1.000 | 1.004 | 1.006 |
| End to end | AutoProf | 0.966 | 1.003 | 1.033 |

Isoster's harmonic work is directly visible and grows with ring size in the
end-to-end fits. For photutils and AutoProf, harmonic evaluation is small
relative to the native pipeline work under these settings. This does not mean
that those tools perform no harmonic calculation; it means the incremental
cost is not resolved above their larger pipeline cost here.

### Session-to-session variation

For each arm, the width of the three-session median range was divided by the
overall median. The table summarizes that relative width across arms:

| Scope | Tool | Median range width | Maximum range width |
|---|---|---:|---:|
| Fixed aperture | Isoster | 1.33% | 3.92% |
| Fixed aperture | photutils | 0.76% | 4.26% |
| Fixed aperture | AutoProf | 0.42% | 1.35% |
| End to end | Isoster | 1.87% | 4.48% |
| End to end | photutils | 0.61% | 1.31% |
| End to end | AutoProf | 2.83% | 6.26% |

These values support the resolved primary ordering, but three session medians
are too few to assign a formal sampling distribution. The bracketed ranges
must not be described as 68% or 95% confidence intervals.

### Noise dependence

Fixed-aperture timing is almost insensitive to the Gaussian realization: the
Gaussian/noiseless median ratio stays within about 1.5% for every tool. The
end-to-end ratios vary more because noise changes native convergence and
stopping behavior. Across matched fixtures and harmonic settings, the
Gaussian/noiseless ratio ranges are 0.516--1.017 for Isoster, 0.574--1.729 for
photutils, and 0.568--1.003 for AutoProf. This is a real property of the
end-to-end task, not measurement noise to be averaged away.

## Accuracy outcome and its separation from timing

Timing eligibility is not a scientific-accuracy verdict. An arm is
`headline_eligible` only if it is timing-eligible and also passes every
applicable frozen accuracy family: intensity, harmonics when enabled, and
geometry for end-to-end fits.

Stage 4 retained the following record-level outcomes:

| Outcome | Pass/available | Fail/unavailable/not applicable |
|---|---:|---:|
| Headline eligible | 975 | 8,925 not eligible |
| Intensity accuracy | 1,275 pass | 8,625 fail |
| Geometry accuracy | 2,550 pass | 2,850 fail; 4,500 not applicable |
| Harmonic accuracy | 2,325 pass | 2,625 fail; 4,950 not applicable |
| Intensity availability | 9,150 available | 750 unavailable |

Because every arm contains 75 records, the 975 eligible records correspond to
13 of 132 complete arms:

| Tool | Scope | Fully eligible arms |
|---|---|---:|
| Isoster | fixed aperture | 8 of 20 |
| photutils | fixed aperture | 0 of 20 |
| AutoProf | fixed aperture | 5 of 20 |
| All tools | end to end | 0 of 72 |

The 13 arms are:

- Isoster, `size_ladder_961`, both noise arms, harmonics off and on;
- Isoster, `size_ladder_1921`, both noise arms, harmonics off and on;
- AutoProf, `sersic_n2_compact`, noiseless, harmonics on;
- AutoProf, `size_ladder_481`, noiseless, harmonics on;
- AutoProf, `size_ladder_961`, both noise arms, harmonics on; and
- AutoProf, `size_ladder_1921`, noiseless, harmonics on.

The per-tool and per-scope metric counts explain why no end-to-end arm is fully
eligible:

| Scope and tool | Intensity pass | Geometry pass | Harmonic pass when applicable |
|---|---:|---:|---:|
| Fixed Isoster | 600 / 1,500 | not applicable | 750 / 750 |
| Fixed photutils | 0 / 1,500 | not applicable | 300 / 750 |
| Fixed AutoProf | 675 / 1,500 | not applicable | 450 / 750 |
| End-to-end Isoster | 0 / 1,800 | 1,650 / 1,800 | 225 / 900 |
| End-to-end photutils | 0 / 1,800 | 900 / 1,800 | 150 / 900 |
| End-to-end AutoProf | 0 / 1,800 | 0 / 1,800 | 450 / 900 |

AutoProf's 750 fixed-aperture, harmonics-off records have unavailable
intensity accuracy because AutoProf 1.3.4 emits the Fourier mean `b0` only when
coefficient extraction is enabled. Enabling it would change the timed
harmonics-off task. AutoProf's median `I` column is a different estimator and
was not substituted.

The accuracy result prevents a broad claim of cross-tool scientific
equivalence. In particular, the end-to-end speed ratios compare complete
native profiles but **none** of those arms met every common accuracy condition.
The timings remain useful as engineering measurements; they cannot be used
alone to say that the three tools produced interchangeable scientific
profiles.

## Caveats and boundaries of interpretation

1. **Small synthetic grid.** Six fixtures cannot represent the distribution of
   real galaxies, masks, backgrounds, PSFs, correlated noise, neighboring
   sources, or detector defects.
2. **Fixed aperture is not geometry fitting.** Its large speed ratios apply to
   matched extraction and harmonic evaluation at five imposed rings.
3. **End-to-end tasks are native rather than identical.** The tools use
   different radial grids, sampling, clipping, convergence, background, and
   stopping rules. Ring counts are reported because runtime partly reflects
   those choices.
4. **Accuracy is not established end to end.** No end-to-end arm passed all
   common scientific criteria. The timing study is supporting evidence, not a
   proof of equivalent recovered profiles.
5. **Harness asymmetry is explicit.** AutoProf runs through a persistent
   Python 3.10 worker and filesystem-oriented pipeline. Its fit-plus-harness
   time is the user-facing primary result, while fit-only is retained so the
   harness is not mistaken for its inner algorithm.
6. **One machine and software stack.** Absolute times and even ratios may
   change with hardware, operating system, compiler libraries, package
   versions, and storage. These numbers describe the recorded Mac Studio.
7. **Interpreter versions differ.** AutoProf required Python 3.10 while the
   project used Python 3.12. The measured calibration ratio is reported and no
   correction is applied.
8. **Descriptive session ranges.** Three sessions provide a useful stability
   check but not formal statistical confidence intervals.
9. **Load was recorded, not kept below the idle ceiling during work.** The
   accepted run reached a one-minute load of 8.87. It had no recorded thermal
   warning or recognized competing process under the frozen policy.
10. **Post-run provenance gap.** AutoProf's environment is embedded, but the
    Stage 4 archive does not embed the project Git SHA or complete project
    package list. This record associates the clean branch head and retained
    environment immediately afterward; future campaigns should embed both.
11. **Local archive durability.** The checksum makes accidental changes
    detectable, but a cache directory is not a backup. Long-term preservation
    requires copying the external archive to a non-Git archival location.
12. **Per-call native files were removed after validation.** The consolidated
    summary retains every normalized profile and timing record, and the six
    AutoProf logs remain available. Inspecting a native per-call FITS, `.prof`,
    or `.aux` file would now require recreating that deterministic input and
    rerunning the corresponding tool; the original timed file no longer
    exists.

## Safe use in the scientific publication

A short publication description can state that, on the frozen Mac Studio
synthetic benchmark and using the primary fit-plus-harness measurement,
Isoster was faster than photutils and AutoProf in all 44 matched conditions.
Across the end-to-end configurations, the median speed ratios were 37.3 for
photutils/Isoster and 63.0 for AutoProf/Isoster, with full configuration ranges
of 19.0--97.7 and 20.0--105.4. Across fixed-aperture extraction, the
corresponding median ratios were 29.9 and 341.2, with the explicit warning that
the AutoProf value includes substantial required harness work.

The publication should state in the same passage that the experiment is a
small controlled synthetic study, that end-to-end tools used their native
grids and stopping rules, and that accuracy was recorded separately and did
not establish general end-to-end equivalence. Claims broader than those
sentences require a separate real-galaxy or scientifically accuracy-matched
campaign.
