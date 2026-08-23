# Design: three-way benchmark comparison (isoster / photutils / AutoProf)

Date: 2026-08-22 (revised through five review rounds)
Branch: `benchmarks/three-way-comparison`
Location: tracked in `docs/specs/`, excluded from the published site. Moved
here from the gitignored `docs/agent/` on 2026-08-22 — a design this branch
depends on should not live only on one machine.
Status (2026-08-23): **Part A complete, measured, archived and gated**, and
revised once after review — criterion 2 was withdrawn, and two prose figures
were corrected against the archives. **Part B specified but not implemented**;
its accuracy contract is being redesigned after review (see B2–B4) and must be
frozen before the pilot produces any measurement.

Closes the two items recorded as open in
`docs/publication/method-code-consistency-audit.md`:

1. AutoProf's harmonic scale is unverified in both directions.
2. The photutils speedup sweep is a single timing per configuration in a
   fixed tool order, not a controlled timing study.

## Revision note

The first draft of this spec contained three errors of substance, all
found in review and all since checked against AutoProf itself. They are
recorded here rather than quietly deleted, because two of them would have
produced a wrong fixture and the third would have put a false statement
into the publication.

1. **"AutoProf's normalization is flux-calibration dependent" was wrong.**
   Both the FFT numerator and its DC denominator scale with a
   multiplicative flux factor, so the ratio does not. Measured: `a4` and
   `b4` are identical to twelve decimal places across image scalings of
   1, 10 and 10⁶. The real sensitivity is to an **additive background
   error**, which changes the denominator only: a `+100` offset on a ring
   of mean 50 moved `a4` from 0.01500 to 0.00500, exactly the factor of 3
   the mean ratio predicts. Bender normalization has the opposite
   character — insensitive to a constant offset, because a constant does
   not change the radial gradient. The grid now tests background offset
   instead of flux scaling.

2. **The raw reconstruction is exact, not approximate.** The first draft
   proposed recovering AutoProf's DC term from its `SB` column and
   flagged a mean-versus-median error. Unnecessary: AutoProf already
   writes `b0 = Re(fft(I)[0])/N`, the exact mean of the vector that
   entered the FFT, as a column next to `a3`/`b3`/`a4`/`b4`. Confirmed in
   real `.prof` output, and confirmed to scale exactly ×10 under a ×10
   image. The mean-minus-median gap remains an interesting diagnostic but
   is **not** an error term in the reconstruction.

3. **"Exactly" was wrong in the truth derivation.** The planted-shape
   result `a_n = ε_n^sin`, `b_n = ε_n^cos` holds only to **first order**.
   Sérsic curvature and products between simultaneously planted modes
   generate additional harmonics. The validation is restructured around a
   response matrix and a numerically integrated truth.

A fourth correction came out of trying to measure item 1, and is a
finding in its own right — see "Why apertures must be matched" below.

**Second review round** produced four further corrections, all checked
against AutoProf's source before being written in:

5. **`|b₀|`, not `b₀`.** AutoProf divides by `|fft(I)[0]|`. A noisy outer
   ring can have a non-positive mean, so dropping the absolute value
   would flip the reconstructed sign exactly where the data are worst.
6. **Both conventions are invariant to multiplicative flux scaling.**
   Calling Bender normalization "the reverse" was still wrong: in each
   convention numerator and denominator scale together. They differ only
   in their response to an *additive* background error.
7. **Band suppression must be deterministic.** Raising S/N or setting
   `ap_isoband_start = 0` does not guarantee ring sampling, because the
   governing clause is `medflux > …`, which fails for a non-positive
   median. The `or isobandwidth < 0.5` clause does guarantee it.
8. **`ap_set_*` does not fix an aperture.** It initializes an ordinary
   fit. The forced pipeline is a different, explicitly named route, and
   Part A depends on it entirely.

**Third review round** corrected the forced route itself, plus four
operational details:

9. **`isophotefit forced` is not part of the standard forced pipeline
   and must not be added.** AutoProf's forced mode is
   `background → psf → center forced → isophoteinit forced →
   isophoteextract forced → writeprof` (`Pipeline.py:322-332`). The
   forced extractor already reads `R`, `ellip` and `pa` from the forcing
   profile, so an inserted fitting step is redundant and is one more
   place for geometry to drift. The first draft invented it.
10. **The companion `.aux` file is unnecessary.** `Center_Forced` checks
    `ap_set_center` first and returns immediately (`Center.py:90-108`);
    the `.aux` parse is only the fallback. Passing `ap_set_center` is
    simpler and drops a file-format dependency.
11. **The sampling-mode assertion needs instrumentation.** Stock `.prof`
    output has no line-versus-band column, so the check demanded in the
    Background section cannot be implemented from existing output; the
    worker has to record it.
12. **Pilot seeds must be disjoint from validation seeds**, or the
    tolerances are still partly selected on the data they will judge.
    Also: an invalid conversion stores NaN with
    `harmonic_conversion_valid=False`, not an "empty" column — the output
    is a fixed-schema FITS table where absence must be a value.

**Fourth review round** found that three of the four fixed-aperture routes
did not work as written, one of them because of a real defect in isoster:

13. **`ap_set_center` does not remove the `.aux` dependency after all.** It
    short-circuits `Center_Forced`, but `Isophote_Init_Forced` opens the
    companion `.aux` unconditionally. Correction 10 was half right: the
    `.aux` is still required by the standard pipeline.
14. **`ap_process_mode` is inert via `Process_Image`.** It only selects the
    forced pipeline through `Process_ConfigFile`, and the existing adapter
    calls `Process_Image` directly.
15. **isoster's forced photometry did not measure harmonics at all** — it
    emitted `a_n`/`b_n` pre-filled with `0.0` and never ran the solve. Had
    this spec been implemented as written, isoster would have contributed
    fabricated zeros to the cross-tool comparison and the conclusion would
    have been drawn against them. **Fixed on this branch** before any
    benchmark work, with regression coverage.
16. **The masked arm could not have masked anything.** No mask-loading step
    is composed into the forced pipeline. The mask axis is deferred.

Also: `photutils`'s fixed-aperture route needs `sample.update(...)` before
`Isophote(...)`, or it raises.

**Fifth review round** found two more paths that could still publish
plausible but incorrect coefficients, both now fixed on this branch:

17. **The gradient fallback was being treated as a measurement.**
    `compute_gradient` substitutes `-1.0` (or a scaled previous gradient)
    when it cannot form a comparison ring, and the first version of the
    forced-harmonic fix accepted any finite non-zero value. Reproduced
    with a mask: a clean ring at `sma = 44` whose comparison ring is fully
    masked published `b4 = 3.5e-05` marked `valid=True`, normalized by the
    sentinel. `compute_deviations` likewise returns zeros when singular or
    underdetermined. Both now expose an opt-in explicit status, and forced
    photometry writes a harmonic only when both statuses say *measured* —
    testing status, never value, because `-1.0` is a legal gradient and
    `0.0` a legal amplitude.
18. **`config=None` silently overrode explicit arguments.** The gradient
    config is now built from the effective `integrator` and
    `use_eccentric_anomaly` arguments, so the comparison ring cannot follow
    different settings from the ring being measured. The mismatch was worth
    2% in `b4` in the regression case.

Also: `simultaneous_harmonics=True` now warns in forced mode instead of
being silently downgraded to the per-order solve. **Part A uses and records
`simultaneous_harmonics=False`.**

**Sixth round — from the A2 measurement rather than from review.** Running
A2 changed A3, which is the order these should happen in:

19. **`ap_iso_interpolate_start` is promoted from a recorded setting to a
    grid axis.** It turned out to be the largest effect on the grid by an
    order of magnitude — 13–25% against sub-percent for everything else —
    and it decides whether the headline reads "the tools agree to 0.1%" or
    "AutoProf reads 13–25% high". Both readings are true at their own
    setting, so the setting cannot be fixed; it has to be crossed. A3
    carries the reasoning and the added **radius × interpolation start**
    interaction.
20. **The interpolation threshold is set, not measured — the reverse of what
    this note first claimed.** `ap_iso_interpolate_start` multiplies
    `results["psf fwhm"]`, and the forced pipeline's `psf` step is
    `PSF_Assumed`, which hardcodes 4.0 px. So the switch radius does *not*
    move between fixtures at a fixed setting; it is 20 px at the default,
    everywhere. The realized threshold is archived per case and the per-ring
    mode is instrumented anyway, for the same reason correction 11
    instrumented line-versus-band sampling: the mode is the measurand, and
    the PSF step is swappable.

## Background: what the scales actually are

Read from source, then checked numerically.

**isoster** (`isoster/fitting.py:589-594`) fits
`I(φ) = c₀ + S_n·sin(nφ) + C_n·cos(nφ)` and stores

    a_n = S_n / (sma · |dI/da|)        b_n = C_n / (sma · |dI/da|)

**photutils matches isoster exactly.**
`photutils/isophote/isophote.py::_compute_deviations` computes
`a = up_coeffs[1]/sma/abs(gradient)` and
`b = up_coeffs[2]/sma/abs(gradient)` against the model
`x[0] + x[1]·sin(nφ) + x[2]·cos(nφ)` — same normalization, same
`a`-is-sine convention, same polar-angle-from-major-axis basis. Checked,
not assumed.

**AutoProf** (`autoprof/pipeline_steps/Isophote_Extract.py:170-181`)
stores, with `coefs = fft(I)`:

    a_n = Im(coefs[n]) / |coefs[0]|    b_n = Re(coefs[n]) / |coefs[0]|
    a_0 = Im(coefs[0]) / N  (= 0)      b_0 = Re(coefs[0]) / N  (= mean)

Since `coefs[n] = (N/2)(C_n − i·S_n)` and `coefs[0] = N·b₀`, and AutoProf
divides by `|coefs[0]|`:

    a_n = −S_n / (2·|b₀|)              b_n = +C_n / (2·|b₀|)

The absolute value is not cosmetic. A noisy outer ring can have a
non-positive mean, and `|b₀|` is what the code actually uses; writing
`b₀` would flip the reconstructed sign exactly where the data are worst.

So AutoProf is the sole outlier of the three, differing in four ways:

1. **Normalization by mean intensity** rather than `sma·|dI/da|`.
   **Both** conventions are invariant to multiplicative flux scaling —
   in each case numerator and denominator scale together — so that is not
   a discriminator. They differ in their response to an **additive
   background error**: a constant offset changes AutoProf's mean-intensity
   denominator, and leaves the Bender radial gradient untouched.
2. **Factor of 2**, from the FFT convention.
3. **Sign flip on `a` only.** AutoProf's `a` is the negated sine
   coefficient; `b` is not negated.
4. **Angle basis, conditional on a runtime flag.** With no mask and
   `ap_isoclip=False`, the FFT runs on raw samples, uniform in eccentric
   anomaly. With a mask or `ap_isoclip=True` — the campaign default —
   `Isophote_Extract.py:166` re-interpolates onto uniform *polar* angle
   carrying the `+PA` rotation from `SharedFunctions.py:640`. isoster
   (default) and photutils are unconditionally polar-from-major-axis, so
   AutoProf is the only tool whose basis changes at runtime.

A fifth behaviour matters for the measurement even though it is not a
convention difference. At `Isophote_Extract.py:95-103` the sampling mode
is chosen by

    if medflux > background_noise · ap_isoband_start  or  isobandwidth < 0.5:
        ring ("line") sampling
    else:
        isophotal-band sampling over R ± isobandwidth

which changes the estimand from a ring to a radial band in the faint
outskirts.

**Suppress it deterministically, not statistically.** Raising S/N or
setting `ap_isoband_start = 0` is not a guarantee: the first clause is
`medflux > …`, which fails for a noisy ring whose median flux is
non-positive. The second clause is an **or**, so it wins unconditionally.
For calibration set

    ap_isoband_fixed = True        # isobandwidth becomes a constant …
    ap_isoband_width < 0.5         # … which is below the 0.5 px threshold

and then **assert from archived per-ring metadata that every calibration
ring actually used line sampling**, rather than trusting the settings.
Isophotal-band mode is tested separately, and only enters a published
claim if that separate test supports it.

The campaign fitter currently passes AutoProf's `a3/b3/a4/b4` straight
into the isoster-schema columns of the same name
(`benchmarks/exhausted/fitters/autoprof_fitter.py:671-674`), and does not
read `a0`/`b0` at all. Both must change.

## Part A: settle the harmonic scale

Six parts, A1 through A6, ordered cheapest and most conclusive first.
Each is a gate on the next.

### A1. Exact one-dimensional convention test — no images, no fitting

Feed **known ring samples** directly to each tool's harmonic
implementation and check the returned coefficients against closed-form
truth. Construct `I(φ) = I₀ + S·sin(nφ) + C·cos(nφ)` on a uniform φ grid
and call `isoster.fitting.compute_deviations`,
`photutils.isophote.harmonics.fit_upper_harmonic`, and the AutoProf FFT
expression, with no galaxy, no noise and no geometry fitting anywhere.

This isolates convention from everything else, and it runs in
milliseconds with no AutoProf venv required for the isoster and photutils
arms. It pins:

- the factor of 2 and the sign on `a`;
- the `a`-is-sine assignment for all three;
- the PA rotation formula of A4;
- background-offset sensitivity, by adding a constant to `I` and checking
  that AutoProf's coefficients move and the Bender ones do not.

**Use a response matrix, not four simultaneous modes.** Inject one mode
at a time — `(n, sin)` and `(n, cos)` for n = 3 and 4 — and record the
full 4×4 matrix of responses. A response matrix exposes sign swaps, phase
rotation and cross-order leakage individually; four simultaneously
planted amplitudes let those effects hide in each other. Run the combined
case afterwards as a superposition check.

### A2. Planted-deviation image fixture, at fixed geometry

Neither existing generator can plant a harmonic: `create_sersic_model`
(`tests/fixtures/sersic_factory.py`) and `create_sersic_image*`
(`benchmarks/utils/sersic_model.py`) both produce plain ellipses, and
`_fit_boxy_galaxy` in `tests/unit/test_harmonic_normalization.py` is
named for an intent it does not implement — its harmonics come from
noise.

Add one function to `benchmarks/utils/sersic_model.py`, reusing its
`sersic_1d` and `compute_bn`, building `I(x,y) = I_sersic(r_eff)` with

    r_eff = r_ellipse / D(φ),   D(φ) = 1 + Σ_n [ε_n^cos·cos(nφ) + ε_n^sin·sin(nφ)]

φ being the polar angle from the major axis — the basis isoster
(`use_eccentric_anomaly=False`) and photutils both use natively.

**Truth is computed, not linearized.** The first-order expansion
`I(a/D) ≈ I(a) − a·I'(a)·(D−1)` gives `a_n ≈ ε_n^sin`, `b_n ≈ ε_n^cos`,
but only to first order; Sérsic curvature and mode products add
higher-order terms. So the reference values come from **dense numerical
Fourier integration of the analytic renderer** on the exact sampling
ellipse — evaluate `I_sersic(a/D(φ))` on a fine φ grid and integrate. The
linear expansion is retained only as a printed diagnostic, so the size of
the non-linearity is reported rather than assumed negligible.

**Why apertures must be matched.** Attempting the flux-scaling experiment
end-to-end produced a result worth recording: on the *same* image scaled
by 10, AutoProf's fitted geometry drifted by up to 0.047 in ellipticity
and 8.8° in position angle, and **none** of 58 rings had identical
geometry. The apparent coefficient differences were dominated by the
tools landing on different ellipses, not by the harmonic convention. Raw
amplitudes depend on flux units, radius and the exact fitted ellipse, so
they are not portable across differing apertures.

Part A therefore requires, for the convention measurement:

- the same input image in the same intensity units;
- fixed and identical center, ellipticity, PA, and semi-major axes across
  all three tools;
- explicit ring-to-ring matching, verified rather than assumed;
- isophotal-band mode suppressed (see Background);
- **free-geometry end-to-end recovery reported separately**, as a
  different question with a different answer.

**The exact fixed-aperture route per tool.** `ap_set_center`,
`ap_isoinit_ellip_set` and `ap_isoinit_pa_set` only *initialize* an
ordinary AutoProf fit; they do not impose an aperture. This step is
load-bearing — without it Part A cannot separate convention from
geometry — so each route is named rather than left to the implementer:

| Tool | Fixed-aperture route |
|---|---|
| AutoProf | AutoProf's standard forced pipeline: `background` → `psf` → `center forced` → `isophoteinit forced` → `isophoteextract forced` → `writeprof` (`Pipeline.py:322-332`), driven by `ap_forcing_profile` **and its companion `.aux`** — see the two traps below. |
| isoster | `fit_image(image, ..., template=…)` — template-based forced photometry. Valid **only as of the harmonic fix on this branch**; see below. |
| photutils | per target `sma`: `EllipseGeometry(x0, y0, sma, eps, pa)` → `EllipseSample(image, sma=sma, geometry=g)` → `sample.update(fixed_parameters=np.ones(4, dtype=bool))` → `Isophote(sample, niter=0, valid=True, stop_code=0)`. The `update()` call is **required** — without it the sample has extracted no intensities and computed no gradient, and constructing `Isophote` raises `TypeError: 'NoneType' object is not subscriptable`. (`fit_image(..., fix_center=True, fix_pa=True, fix_eps=True)` fixes the shape but still chooses its own radial grid, so it does not guarantee matched radii.) |

**The forcing CSV's units and conventions**, fixed before the file is
generated rather than discovered by debugging it:

| Field | Unit / convention |
|---|---|
| `R` | **arcsec**, i.e. `sma_pixels * ap_pixscale` — the forced extractor divides by the pixel scale on read |
| `ellip` | dimensionless, `1 − b/a` |
| `pa` | **astronomical degrees**, the existing `isoster_pa_to_autoprof_init()` conversion, taken modulo 180° |

**isoster's route needed a code fix to be usable at all.**
`extract_forced_photometry` emitted `a_n`/`b_n` pre-filled with `0.0` and never
ran the harmonic solve, so template-based forced photometry would have
contributed fabricated zeros to the cross-tool comparison. Fixed on this
branch: harmonics are measured, unmeasurable rings report NaN, and `grad` is
exposed under `debug=True`. Coverage in
`tests/unit/test_forced_photometry_harmonics.py`.

**Trap 1: `ap_set_center` does not remove the `.aux` dependency.** It
short-circuits `Center_Forced` (`Center.py:90-108`), but the *next* step,
`Isophote_Init_Forced`, opens `ap_forcing_profile[:-4] + "aux"`
unconditionally with no `ap_set_*` check (`Isophote_Initialize.py:75`). The
standard forced pipeline therefore **requires** the companion `.aux`. Preferred
route: **generate a minimal `.aux` alongside the forcing profile**, and record —
in the spec and in the code — that its *global* ellipticity and PA are
initialization values that do **not** control the per-ring extraction geometry;
the CSV rows do. (The alternative is to drop `isophoteinit forced` and
demonstrate identical profiles, but that pipeline may then no longer be
described as standard.)

**Trap 2: `ap_process_mode` is inert when calling `Process_Image` directly.**
It selects the forced pipeline only through `Process_ConfigFile`
(`Pipeline.py:322`). The current adapter calls `Process_Image(options)` directly
(`benchmarks/utils/autoprof_adapter.py:212`), where the option does nothing.
The worker must either call `Process_ConfigFile`, or call `UpdatePipeline` with
the exact forced step list before `Process_Image` — and archive the realized
step sequence, so what actually ran is on the record.

**Two things not to do here, both of which the first draft got wrong.**

*Do not add `isophotefit forced`.* It is not part of AutoProf's standard
forced pipeline, and it is redundant: `Isophote_Extract_Forced` already
reads `R`, `ellip` and `pa` straight from the forcing profile. Inserting
a fitting step between the profile and the extractor only creates another
chance for geometry to drift. If a concrete reason to add it ever
appears, document the reason next to the step.

**Then check, do not assume.** After every run, assert ring-by-ring that
the returned `(R, ellip, pa, center)` equal the requested values to
tolerance. A silently re-fitted ring invalidates the comparison it feeds.

**The sampling-mode assertion needs instrumentation to exist.** AutoProf's
`.prof` output carries no column recording whether a ring used line or
isophotal-band sampling — the columns are `R`, the flux pair, `totflux*`,
`ellip`, `pa`, their errors, `pixels`, `maskedpixels`, `a0`, `b0` and the
requested `a_n`/`b_n`, and nothing else. The assertion in the Background
section therefore **cannot be implemented from stock output**. The
subprocess worker must instrument the mode per ring and archive it
alongside the profile; only then can `harmonic_sampling_mode` be
populated and the all-line-sampling check actually run.

### A2 result: the scales agree; the apparent gap is pixel-grid aliasing

Measured once A1 and A2 were in place, on the planted fixture at fixed
geometry, against the integrated analytic truth. Ratios of measured to true
`C_4`, varying only `ap_iso_interpolate_start` — the option deciding which
rings AutoProf samples with Lanczos interpolation and which it samples by
rounding to the nearest pixel (`SharedFunctions.py:673`):

| `ap_iso_interpolate_start` | sma=15 | sma=25 | sma=35 | sma=45 |
|---|---|---|---|---|
| 100 (Lanczos everywhere) | 0.993 | 0.999 | 1.000 | 1.001 |
| 5 (the default) | 0.993 | 1.247 | 1.132 | 1.084 |
| 0 (nearest-neighbour everywhere) | 1.357 | 1.247 | 1.132 | 1.084 |

**With interpolation used everywhere, AutoProf agrees with analytic truth to
0.1%.** So the four-part conversion derived in the Background section is
*correct*, and the 13–25% excess first seen with default settings is not a
scale difference at all — it is nearest-neighbour sampling of the ring.

The mechanism is specific and was confirmed by prediction. A square pixel grid
is four-fold symmetric, so rounding sample positions to the nearest pixel
should alias into `m=4` far more than into `m=3`. Planting equal amplitudes at
both orders:

| Sampling | order | sma=15 | sma=25 | sma=35 | sma=45 |
|---|---|---|---|---|---|
| Lanczos | m=3 | 0.995 | 1.000 | 0.997 | 0.999 |
| Lanczos | m=4 | 0.993 | 0.999 | 1.000 | 1.001 |
| nearest | m=3 | 0.981 | 1.063 | 0.927 | 0.996 |
| nearest | m=4 | 1.357 | 1.245 | 1.136 | 1.084 |

`m=3` shows scatter with no systematic bias; `m=4` is systematically inflated.

**The radius dependence is not monotonic, and an earlier draft said it was.**
On this fixture the excess happens to fall — 24.5% / 13.4% / 8.4% at
`sma = 25 / 35 / 45` — which invited the explanation that a half-pixel
displacement matters less relative to a larger ring. The second fixture
falsifies that: at `sma = 28 / 40 / 55 / 70` it reads 16.3% / **1.2%** / 7.2%
/ 6.1%, with the 40 px ring almost clean between two badly aliased
neighbours. The size of the aliasing depends on how a particular ring's
sample positions happen to land on the integer grid, which is a
commensurability between radius, ellipticity and sample count rather than a
smooth function of radius. Expect the effect to be *large and erratic* above
the switch, not *large and decreasing*.

**Consequences for the rest of Part A.**

- The publication claim is **not** "AutoProf reads 13–25% high". Nor, as an
  earlier draft of this document said, "the three tools agree to ~0.1–0.3%":
  a review caught that as contradicted by the archive. That figure describes
  isoster and interpolated AutoProf at the largest radii only; photutils never
  reaches it.

  Across the tested matched apertures all three tools recover the raw amplitudes
  to within 2.4%, and the agreement is strongly radius-dependent rather than
  uniform. On the compact n=2 fixture the three tools agree with the analytic
  truth to better than 2.3%; on the extended n=4 fixture the three tools agree
  with the analytic truth to better than 2.4%. The tools separate at large
  radius: on the compact n=2 fixture at the largest tested radius, sma = 45 px,
  isoster reaches 0.13% and AutoProf 0.13%, while photutils retains 1.85%; on the
  extended n=4 fixture at the largest tested radius, sma = 70 px, isoster reaches
  0.08% and AutoProf 0.19%, while photutils retains 1.85%.

  The isoster-versus-AutoProf agreement at large radius is the strong result;
  the photutils residual is a separate finding (see A3 result). Both clauses
  above are bound to the archive by `check_harmonic_scale.py`.
- `ap_iso_interpolate_start` becomes a **grid axis**, not a default to
  inherit and not a fixed recorded setting. A3 below states why: it is the
  largest effect on the grid by an order of magnitude, so pinning it to
  either value answers a different question than the one being asked.
- It is also a finding about `m=4` measurements generally: any tool sampling
  isophotes by nearest-pixel lookup can contaminate the four-fold mode --- and
  did inflate it on both fixtures tested here, though two fixtures do not
  establish that it always will. It is
  precisely the boxy/discy diagnostic. Worth stating in its own right.
- The earlier reading that the excess "shrinks with radius" is a partial view
  of this: what matters first is which side of the switch a ring falls on, and
  only then its radius. (An earlier draft attributed the non-monotonicity to a
  PSF that differs at eps=0.6. That was wrong — `PSF_Assumed` returns 4.0 px
  regardless of the image, measured identical across all fifteen grid cases.
  The switch radius is the same everywhere at a fixed setting.)

Separately confirmed, and it settles difference 4: with `ap_isoclip=False` the
eccentric-anomaly path degrades badly with ellipticity — ratios 0.99 at eps=0,
0.88–1.10 at eps=0.3, and **0.26–0.35 at eps=0.6**. That is the order mixing
the spec predicted, now measured, and it confirms that path must not be
converted by a same-order rotation.

### A3. Measurement grid

`benchmarks/harmonic_scale/run_harmonic_scale.py`. The grid tests every
claim above rather than only the PA one:

| Axis | Values | Why |
|---|---|---|
| `ap_iso_interpolate_start` | 100 (Lanczos everywhere), 5 (AutoProf's default) | **The largest single effect measured, by an order of magnitude.** See below |
| Ellipticity | ~0 (control), 0.3, 0.6 | At eps≈0 polar angle and eccentric anomaly coincide, so the basis question vanishes and everything else is isolated; the non-circular cases make it bite |
| PA | 0°, 30° | At PA=0 AutoProf's polar basis coincides with major-axis polar, so differences 1–3 show alone; 30° adds difference 4 |
| `ap_isoclip` | off, on | Selects AutoProf's angle basis |
| Radius | ≥2 values | Raw amplitudes are radius-dependent; one radius cannot show that; and the interpolation switch is itself a radius threshold |
| Background offset | 0, +δ | Tests the corrected claim 1 directly |
| Noise | several realizations at each S/N | One realization measures a realization, not a distribution |
| Mask | **deferred from the initial campaign** | See below |

**Why sampling is an axis and not a setting.** The A2 result above measured a
13–25% AutoProf excess at the default and 0.1% with Lanczos interpolation
everywhere. Nothing else on this grid moves the answer by remotely that much:
the background offset, the noise level and the position angle are all
sub-percent to few-percent effects on the reconstructed amplitude. Fixing
`ap_iso_interpolate_start` at either value would therefore have produced a
calibration that is *correct but unrepresentative* — pinned at 100 it would
certify a scale agreement no default-settings user ever sees, and pinned at 5
it would report a scale disagreement that is not a scale disagreement at all.
Crossing it with the rest of the grid is what separates the two readings, and
it is why the axis is listed first.

**It is a radius threshold, and the PSF it is measured in is assumed rather
than measured.** `ap_iso_interpolate_start` is not a boolean and not a radius
in pixels: `Isophote_Extract.py:110-115` multiplies it by
`results["psf fwhm"]` to get `rad_interp`, and `SharedFunctions.py:653` then
samples with Lanczos where `Rlim < rad_interp` and by rounding to the nearest
pixel otherwise.

**Corrected after measurement.** An earlier version of this section said the
switch radius was data-dependent because AutoProf measures the PSF from the
image. It does not, on this pipeline. The `psf` step resolves to
`PSF_Assumed` (`Pipeline.py:53`), which **hardcodes 4.0 px** unless
`ap_set_psf` or `ap_guess_psf` is supplied; measuring requires selecting
`psf starfind` or another variant explicitly. Confirmed across all fifteen
grid cases, which report `psf fwhm = 4.0` at every ellipticity, noise level
and background offset. The threshold is therefore fully predictable here —
20 px at the default setting, 32 px at 8, 400 px at 100 — and the archive
records it per case as a fact rather than as a discovery.

Two consequences the runner must still respect:

- **The per-ring mode is instrumented, never inferred.** The runner records,
  for every ring, whether it was interpolated or rounded, by observing whether
  AutoProf's interpolator actually ran. That the threshold is predictable does
  not make the prediction a measurement: the per-ring mode is the quantity the
  whole study turns on, recomputing it here would check our arithmetic rather
  than AutoProf's behaviour, and the PSF step is swappable, so a prediction
  that holds today holds only for this configuration. This is the same
  treatment `ap_isoband_fixed` already gets — the band-versus-line probe
  exists for the identical reason.
- **The value 100 is a hypothesis about this fixture, not a guarantee.** It is
  chosen to put every ring below the switch; whether it did is reported.

Because the PSF is a constant here, a *second fixture will not move the switch
radius on its own*. Varying it independently of `ap_iso_interpolate_start`
requires `ap_set_psf`, which is why the second-fixture campaign adds that as
an axis: two different options that both act only through `rad_interp` should
give identical answers at matched sampling mode, which is a stronger test of
the mechanism than either alone.

The `m=4` finding generalizes past AutoProf and is reported in its own right:
a square pixel grid is four-fold symmetric, so *any* tool that samples an
isophote by nearest-pixel lookup will inflate the four-fold mode specifically —
which is the boxy/discy diagnostic. `m=3` on the same rings showed scatter
without bias.

**The mask axis is deferred, because the forced pipeline cannot honour it.**
AutoProf's standard forced sequence contains no mask-loading step: `Bad_Pixel_Mask`,
`Star_Mask` and `Mask_Segmentation_Map` exist in the step *registry*
(`Pipeline.py:75-78`) but none is composed into the forced pipeline, so nothing
populates `results["mask"]` even though `_Generate_Profile` would consume it. A
masked-sector case run on the standard pipeline would silently mask nothing.
Either add a precisely documented mask step and archive the realized sequence,
or leave the mask axis out of the first campaign — the latter for now. The
`ap_isoclip=True` arm still exercises the polar-resampling conversion, which is
what the mask axis was mainly there to reach.

**Do not run the full Cartesian product.** Six active axes multiplied
together (the seventh, mask, is deferred above) is both large and
uninformative: when a case fails, a saturated design cannot say which
factor caused it. The structure is instead

1. one **clean reference configuration** — the case everything else is a
   perturbation of;
2. **one factor at a time** off that reference;
3. only the interactions with a scientific reason to exist:
   **radius × interpolation start**, **ellipticity × basis**,
   **PA × polar resampling**, **noise × clipping**. (Mask × resampling is
   dropped with the mask axis — see above.)

   Radius × interpolation start is first because it is the only pair on the
   list whose two factors are not independent: `ap_iso_interpolate_start`
   *is* a radius threshold, so a ring's sampling mode is decided by both
   factors jointly and neither one alone predicts it. Crossing them is what
   puts rings on each side of the switch within a single case, which is the
   configuration the A2 table stumbled into by accident and this grid should
   reach on purpose.
4. one final **combined stress case**, as a check rather than a
   measurement.

**Pre-register before archiving.** Exact S/N values, noise seeds, radii,
planted amplitudes and acceptance tolerances are fixed in advance and
recorded. Tolerances are not guessed: run a small pilot first, measure
the numerical scatter, then freeze tolerances justified by it. Choosing a
tolerance after seeing the result is how a measurement becomes a
formality.

**Pilot seeds and fixtures must be disjoint from the archived validation
seeds and fixtures.** Tuning a tolerance on the same realizations it will
later judge is selection on the evaluated data, just at one remove — the
tolerance ends up fitted to the noise it is supposed to survive. Draw the
pilot from its own seed block and record both blocks.

Reported per case, for both tracks of A4, with provenance following
`benchmarks/draft_timings/`: environment block, fixture fingerprint,
clean-tree check, refusal to archive from a dirty tree.

### A3 result: sampling mode is the cause, and it is the only large effect

Measured on the designed grid, pilot on seed block 900000 and validation on
20260822, with tolerances frozen from the pilot and committed before the
validation run. Archived in `benchmarks/harmonic_scale/reference_harmonic_scale.json`.

**The headline.** In the clean configuration — noiseless, clipping on, and on
only those rings AutoProf actually interpolated — on the compact n=2 fixture
the three tools agree with the analytic truth to better than 2.3%, and the
agreement is strongly radius-dependent:

| worst \|ratio − 1\| | sma=12 | sma=18 | sma=25 | sma=35 | sma=45 |
|---|---|---|---|---|---|
| isoster | 2.18% | 1.02% | 0.57% | 0.25% | 0.13% |
| photutils | 1.90% | 1.70% | 1.25% | 0.88% | 1.85% |
| AutoProf | 2.33% | 0.81% | 0.99% | 0.45% | 0.13% |

The pooled worst case is set by the smallest ring, where the fixture itself is
marginally resolved: at `sma = 12` and `eps = 0.3` a quarter of an `m=4` cycle
spans only a few pixels. Quoting the pooled number alone would present
pixelation of the fixture as the tools' agreement floor. isoster and AutoProf
converge to 0.13% at `sma = 45`; photutils does not, for a reason given below.

**Sampling mode is the cause of the AutoProf excess, and this now says so
rather than inferring it.** AutoProf reads 24.5% high at sma = 25 px when that
ring is sampled by rounding to the nearest pixel; it reads 13.4% high at
sma = 35 px when that ring is sampled by rounding to the nearest pixel; and it
reads 8.4% high at sma = 45 px when that ring is sampled by rounding to the
nearest pixel. The
radius × interpolation-start interaction is what makes this a causal statement
instead of a correlation:

| ring | `start=100` | `start=8` | `start=5` |
|---|---|---|---|
| sma=25 | **0.35%** (interpolated) | **0.35%** (interpolated) | **24.5%** (rounded) |
| sma=35 | **0.40%** (interpolated) | **13.4%** (rounded) | **13.4%** (rounded) |
| sma=45 | **0.13%** (interpolated) | **8.4%** (rounded) | **8.4%** (rounded) |

Same ring, same mode, different setting gives the *same* answer; same ring,
different mode gives an answer that differs by a factor of seventy. Across the
whole grid, `mode_matched_spread` — the largest disagreement between two cases
whose sampling mode at a ring agreed — is **0** to the six decimals the archive
stores. The setting does not matter except through the mode it selects, and
the radius does not matter except through which side of the switch it falls.

**Two other results, both corrections to what this spec previously said.**

- **The background axis does not test what the revision note claimed, for
  AutoProf.** Revision item 1 measured the FFT expression directly and found a
  ring-mean-ratio sensitivity. Through the *pipeline* there is none: the forced
  sequence begins with a `background` step, so an additive sky offset is
  subtracted before extraction and never reaches the normalization. Measured: a
  `+50` offset moved `b0` from 99.35842 to 99.35843 and left native `b4`
  identical to six decimals. Raw amplitudes are invariant for all three tools
  to floating-point round-off — 4e-15 for isoster, 1e-14 for AutoProf, and
  2e-7 for photutils, the last being the `leastsq` floor A1 already
  characterized rather than a background sensitivity. Both statements are true
  at their own level; A1 covers the formula, this axis covers the pipeline.

- **photutils' residual is consistent with cross-order leakage from uneven
  angular sampling, to which the other two are not exposed.** They sample at
  evenly spaced angles, so *this particular* non-orthogonality cannot arise in
  them; that is not a claim that they are free of cross-order effects in
  general, which nothing here tested. *(Downgraded 2026-08-23 after review: the causal
  claim is not established. The archived leakage metric compares differences
  between worst errors, possibly at different radii, and is not a cross-order
  response coefficient. Settling it needs a joint fit of all four n=3,4
  components on photutils' own sampled angles, or the 4x4 response matrix built
  from those angles — an in-process run needing no AutoProf.)* Its
  1.85% floor at `sma = 45` is not a scale error. Its ring samples are not
  evenly spaced in polar angle — measured spacing varies by a factor of 1.4
  around one ring — so the harmonic basis is not orthogonal on them
  (`⟨sin 3φ, cos 4φ⟩ = 6e-4`, against `1e-18` on an even grid), which *would*
  let a fit that models one order at a time absorb part of the others. The
  single-mode control is consistent with that: with only `m=3 cos` planted,
  photutils recovers it to 0.24%.

  What is observed: the residual appears only on multi-mode rings, and the
  non-orthogonality is arithmetic and certain. What is hypothesis: that the
  non-orthogonality is *the cause* of the residual's size. The control shows
  the excess needs other modes present; it does not measure how much of the
  excess those modes explain, and the archived leakage metric compares
  differences between worst errors rather than a cross-order response
  coefficient. A joint four-component fit on photutils' own sampled angles, or
  the 4×4 response matrix built from them, would settle it. Until then the
  excess is an estimator-dependent effect rather than a scale error — which is
  all the archive needs, since the control case is what keeps it separable
  from the scale question.

**Difference 4 confirmed and quantified.** With clipping off, AutoProf is in
the eccentric-anomaly basis and the same-order rotation is not a valid
conversion. The cost of applying it anyway is 12.3% at `eps = 0.3` and 68.2%
at `eps = 0.6` — and, unlike the sampling excess, it is nearly independent of
radius, which is what a basis error rather than a sampling error looks like.

**What the noisy arms do and do not establish.** At S/N = 30 the ratio scatters
with a standard deviation near 0.4, so five realizations could not pin a
median; the count is 25 and those arms characterize a distribution rather than
calibrating a scale, which the noiseless arms do. Note also that 30 of the 36
frozen claims come from noiseless cases and so reproduce *exactly* between the
two seed blocks: they test determinism, and only the six scatter claims test
whether the frozen tolerances were well chosen. The checker reports that split
rather than printing a bare pass count.

### A4. The two tracks

**Track 1 — raw amplitudes (primary).** Reconstruct the raw sine and
cosine amplitudes each tool measured, in image intensity units:

| Tool | Reconstruction | Exact? |
|---|---|---|
| isoster | `S_n = a_n·sma·\|grad\|`, `C_n = b_n·sma·\|grad\|` | Yes — `sma` always; `grad` under `debug=True`, in forced mode as well as free |
| photutils | identical formula and convention | Yes |
| AutoProf | `S_n = −2·\|b₀\|·a_n^AP`, `C_n = +2·\|b₀\|·b_n^AP` | **Yes** — `b₀` is the exact DC term of the same vector |

All three are exact. This is the primary diagnostic of whether the tools
measured the same residual signal, and it avoids the uncertain radial
gradient entirely.

It is **not** a general-purpose portable quantity: raw amplitudes are
comparable only within identical images, identical intensity units and
matched apertures. That constraint is a property of the comparison, and
is why A2 fixes the geometry.

**Track 2 — Bender-normalized (secondary).** Retained as the morphology
comparison, since it is the dimensionless quantity a reader wants:

    a_n^isoster = −2·|b₀| · a_n^AP / (sma·|dI/da|)
    b_n^isoster = +2·|b₀| · b_n^AP / (sma·|dI/da|)

AutoProf reports no radial gradient, so `dI/da` must be finite-differenced
from its own radial profile — and **which profile** matters. Use the
**`b₀` profile**, not the default median `SB` profile: `b₀` is the mean of
the exact vector that entered the FFT, so it is the estimator consistent
with the harmonic numerator. Finite-differencing median `SB` would build
the denominator from one ring estimator and the numerator from another.

A3 reports all three gradients side by side — from `b₀`, from median `SB`,
and the fixture's analytic value — which quantifies the normalization
uncertainty directly instead of asserting it is small. Where the
reconstruction is not defensible, rows are marked invalid with a reason
rather than filled with a noisy number under a column name implying
calibration.

**Angle-basis conversion, pinned by tests rather than described.** For
AutoProf's polar-resampled path, first reconstruct sky-frame raw
coefficients as above, then rotate to the major-axis frame with
`α = n·PA`:

    S_major = S_sky·cos α − C_sky·sin α
    C_major = S_sky·sin α + C_sky·cos α

**The eccentric-anomaly path is not convertible this way.** Changing
between the polar and eccentric-anomaly bases mixes harmonic *orders*, so
no same-order two-component rotation can express it. Therefore:

- the supported adapter conversion **requires** AutoProf's polar-resampled
  path (mask or `ap_isoclip=True`);
- eccentric-anomaly output stays native and explicitly labelled, or is
  converted by resampling the original ring signal — never by
  transforming `a_n` and `b_n` alone.

### A3 result, second campaign: what survives a change of galaxy

The first campaign had six axes and one galaxy, so every number in it was
conditional on that galaxy. A second campaign — Sérsic `n = 4`, `R_e = 40`,
321 px, on its own seed blocks with its own tolerances frozen from its own
pilot — says which conclusions were about the tools and which were about the
fixture. Archived in `reference_harmonic_scale_n4.json`.

**Agreement improves for two tools and not for the third.** On the extended
n=4 fixture the three tools agree with the analytic truth to better than 2.4%,
but that pooled number hides two opposite behaviours:

| worst \|ratio − 1\| | sma=18 | sma=28 | sma=40 | sma=55 | sma=70 |
|---|---|---|---|---|---|
| isoster | 1.04% | 0.26% | 0.27% | 0.07% | 0.08% |
| photutils | 1.67% | 2.01% | 2.12% | 2.41% | 1.85% |
| AutoProf | 0.81% | 0.86% | 0.27% | 0.20% | 0.19% |

isoster and AutoProf improve on this fixture — its rings are larger in pixels,
so there is less pixelation — and fall to 0.08% and 0.19%. photutils does not
improve at all: it is flat-to-rising across the tested radii and ends worse
than it started. Its leakage metric grew from 1.19% to 2.03% on the steeper
profile. Note this is a finite fixed-aperture experiment, so "flat-to-rising
across the tested radii" is what the data support; it is not an asymptotic
convergence test and no statement about the limit is being made.

**That establishes the photutils floor as reproducible, not its cause.** On one
fixture its 1.85% at large radius could have been an artefact of that galaxy.
Across two, with the other two tools reaching below 0.2% on both, it is
estimator-dependent rather than a scale error. The proposed mechanism —
photutils' ring samples are not evenly spaced in polar angle, so the harmonic
basis is not orthogonal on them and a fit modelling one order at a time could
absorb part of the others — remains a hypothesis; the observation that a
steeper profile shows a larger effect is consistent with it and does not
establish it.

**The mechanism claim is now much harder to explain away.** Because
`PSF_Assumed` fixes the PSF at 4.0 px, `ap_set_psf` is a second, independent
route to the same threshold, and three cases exploit that:

- **`threshold_matched_control`.** `ap_iso_interpolate_start = 10` with
  `ap_set_psf = 2` reaches the same 20 px threshold as the default arm's
  `5 × 4`. Both factors differ; the product does not. Every ring returns a
  ratio identical to the default arm's to **0.0** — not "within tolerance",
  identically zero on all five rings and all four components.
- **`psf_x_interpolate`.** Holding `ap_iso_interpolate_start = 5` and doubling
  only the PSF moves the switch from 20 px to 40 px, and `sma = 28` flips from
  16.3% to 0.56%. The interpolation setting never changed.
- **`psf_set_8`.** Doubling the PSF while the threshold is already far outside
  the fixture changes nothing, to 0.0 — so `ap_set_psf` acts through the
  threshold and nowhere else.

Only the product matters. `mode_matched_spread` is 0 on both campaigns.

**Reproduced across galaxies:** the eccentric-anomaly basis error, 12.3% on
both fixtures at `eps = 0.3` (12.32% and 12.26%) and 68.2% / 63.4% at
`eps = 0.6`; exact background-offset invariance for all three tools; and
`mode_matched_spread = 0`.

**Falsified by the second campaign:** that the nearest-pixel excess falls with
radius. On this fixture AutoProf reads 16.3% high at sma = 28 px when that ring
is sampled by rounding to the nearest pixel; it reads 1.2% high at sma = 40 px
when that ring is sampled by rounding to the nearest pixel; it reads 7.2% high
at sma = 55 px when that ring is sampled by rounding to the nearest pixel; and
it reads 6.1% high at sma = 70 px when that ring is sampled by rounding to the
nearest pixel. A nearly clean ring sits between two badly aliased ones, which
no smooth function of radius can produce.

**Still a limitation.** Two galaxies is not a survey. Neither fixture is
PSF-convolved, both are Sérsic profiles on a square grid, and both use the same
four planted amplitudes. This narrows the generalization gap; it does not close
it.

### A4 decision, first pass (superseded): Track 1 delivered, Track 2 not yet reconstructed

**Track 1 is done.** Raw amplitudes are reconstructed exactly for all three
tools and measured against integrated analytic truth across the whole grid.
That is the primary diagnostic, and it needed no gradient.

**Track 2 is deliberately not filled, and the schema says so per row.** The
grid as run did not measure AutoProf's gradient at all — the adapter records
`NaN` for it, because AutoProf reports none. Licensing the Bender conversion
needs its own measurement, pre-registered on its own terms: finite-difference
the `b0` profile, compare that gradient against isoster's measured gradient
and against the fixture's analytic value on the same rings, and quantify the
normalization uncertainty rather than assert it is small. Until then, an
AutoProf arm's `a3`/`b3`/`a4`/`b4` are `NaN` with
`harmonic_conversion_valid = False` and `harmonic_conversion_reason =
"no_radial_gradient_reported"`.

Two things must not be quietly conflated when that measurement happens.
First, use the **`b0` profile**, not the median `SB` profile: `b0` is the mean
of the exact vector that entered the FFT, so it is the estimator consistent
with the harmonic numerator, and differencing `SB` would build the denominator
from one ring estimator and the numerator from another. Second, that
measurement is a *separate* pre-registration — it must not be bolted onto the
frozen A3 grid, because changing the grid changes the fixture fingerprint and
invalidates the archive the current tolerances judge.

### A4 Track 2 pre-registration: match the convention, not the truth

Written before the campaign it describes was run, and committed before its
tolerances were frozen. Prompted by a scouting measurement whose result
changed the design — the same order A2 → A3 followed.

**The finding that forced the design.** The Bender coefficient is
`a_n = S_n / (sma·|dI/da|)`. Track 1 already recovers `S_n` exactly, so every
bit of Track 2's error lives in the denominator. The obvious move is to
finite-difference AutoProf's `b0` profile into the most *accurate* gradient
available. That would be wrong.

isoster and photutils do not divide by the radial derivative. They divide by a
**forward secant** over `sma → sma·(1 + astep)`, with `astep = 0.1` by default
(`isoster/fitting.py`: `gradient_sma = sma * (1.0 + step)`, then
`gradient = (mean_g - mean_c) / delta_r`). On the Part A fixture that secant
sits **11–14% below the point derivative**, systematically, at every radius.
Measured, on rings 12–50 px:

| gradient | vs analytic point derivative |
|---|---|
| central difference of AutoProf `b0` | 0.5% median, 2.0% worst |
| isoster's own measured gradient | 12.0% median, 14.5% worst |
| photutils' own measured gradient | 11.9% median, 14.5% worst |

The two tools are not 12% wrong. They are computing a different quantity, and
they agree with *each other*. Had Track 2 used the accurate point derivative,
AutoProf's `a_n` would have differed from isoster's by ~12% **by
construction**, and the campaign would have reported a definition mismatch as a
disagreement between tools.

**So Track 2's denominator is a matched secant.** Take AutoProf's `b0` at
`sma` and at `sma·(1 + astep)` and difference them over the same interval
isoster uses. Measured against isoster's own gradient on the same rings, that
reproduces it to **0.05% median, 0.13% worst** — against 11–14% for the point
derivative.

This is not circular. What is *shared* is the interval, which both tools must
use or the comparison is meaningless. What is *measured* is the value, computed
from AutoProf's own `b0` and nothing else — no isoster quantity enters the
reconstruction. That the two land within 0.13% is a result, not an
identity.

**It is also a finding in its own right, and belongs in the publication.**
Bender-normalized harmonic amplitudes are convention-dependent at the ~12%
level through the gradient definition alone. Anyone comparing `a4` between two
tools, or against a published value, needs the gradient step to match; a
half-decade of `astep` differences would move `a4` by more than most of the
effects people use it to argue about.

**What the campaign measures, per ring, and reports side by side:**

| gradient | why it is there |
|---|---|
| `b0` matched secant | the reconstruction Track 2 proposes |
| median-flux matched secant | the *wrong* estimator, quantified rather than dismissed. AutoProf's `I` column (`SB` in magnitude mode) is a **median**; `b0` is the mean of the exact vector that entered the FFT, so only `b0` is consistent with the harmonic numerator |
| isoster's measured gradient | the comparison target |
| analytic point derivative | the fixture's truth, which is what shows the secant offset is a convention and not an error |

**Cost, stated rather than hidden.** Every measurement radius needs its
comparison radius measured too, so the AutoProf ring count doubles and the
paired rings must be requested explicitly. The realized pairing is archived.

**Acceptance, fixed in advance — ORIGINAL PRE-REGISTRATION, SUPERSEDED.**
Retained for history because a pre-registration that is quietly edited is
worthless. Criterion 2 below was **withdrawn on 2026-08-23**; the current model
is in "A4 Track 2 result" further down, and this section is not an instruction.
As written in advance, Track 2 was licensed — meaning the schema may write
`a_n`/`b_n` for an AutoProf arm instead of NaN — only if **both** held on the
clean configuration:

1. the `b0` matched secant reproduces isoster's gradient **decisively better
   than the point derivative does**, and
2. the resulting Bender `a_n`/`b_n` agree with isoster's no worse than Track 1's
   raw-amplitude agreement plus that gradient error — that is, normalizing
   introduces no *new* systematic. Both sides measured
   AutoProf-against-isoster.

**Both criteria are threshold-free, and the third correction is why.** As
first written, criterion 1 asked whether the reconstruction agreed "within the
frozen tolerance" — but those tolerances are *derived from the pilot's own
values*, so the test would have passed by construction. A criterion that
cannot fail is precisely what this procedure exists to prevent, and it had
crept into the procedure itself.

Both are now comparisons rather than thresholds. Criterion 1 weighs two
candidate reconstructions against the same target; criterion 2 weighs a
normalized quantity against the unnormalized one it is built from. Neither
needs a number chosen in advance. The frozen tolerances keep their proper and
separate job: testing that a validation run on a disjoint seed block
reproduces the pilot, not deciding whether the result is good.

If either fails, Track 2 stays unlicensed and A5's columns keep their NaN and
their stated reason. A criterion that can only be met is not a criterion.

**Criterion 2 sharpened after the pilot, before any tolerance was frozen.**
As first written it compared a raw-versus-*truth* number against a
Bender-versus-*isoster* number. Those are not comparable. Under noise both
tools see the same realization, so their errors are correlated and the
tool-to-tool gap is much smaller than either tool's gap to truth — the pilot
read raw 34.5% against Bender 9.9% at S/N = 30, which would have suggested
that *normalizing improves accuracy*. The baseline is now
AutoProf-versus-isoster for the raw amplitudes too: same pair of tools, same
realization, only the normalization differing. That is the only form in which
"does normalizing add a systematic" is a well-posed question.

**Licensing is expected to be regime-dependent, and the schema must say so.**
The pilot already shows the reconstruction is not uniformly good: excellent
where the rings are cleanly sampled and moderately elliptical, and degrading
where they are not. A single global yes/no would therefore be the wrong
output. Where the conditions do not hold, the Bender columns keep their NaN
and a reason naming the condition, exactly as they do today for the missing
gradient.

**Scope.** Only where the conversion is already valid: the polar-resampled
path, `ap_isoclip = True`. Never on the eccentric-anomaly path, whose order
mixing Part A measured at 12% and 63%.

**Separate campaign.** Its own grid, fingerprint, archive, frozen tolerances
and seed block. The two A3 archives are frozen and gated; nothing here may
move them.

### A4 Track 2 result: the method is validated; applicability is per row

Both fixtures were run: pilot on one seed block, tolerances frozen from it,
validation on a disjoint block, archived as
`reference_gradient_reconstruction_sersic_n2_compact.json` and
`..._sersic_n4_extended.json`. Both are gated by
`check_gradient_reconstruction.py`, the fourth gate in the docs CI job, and
every number below is bound to those archives by its prose check — none of
them is transcribed by hand.

**Criterion 1 passes decisively, on both galaxies.** On the compact n=2
fixture the matched secant reproduces isoster's gradient on the clean
configuration to 0.131% at the worst ring and 0.041% at the typical one; on
the extended n=4 fixture the matched secant reproduces isoster's gradient on
the clean configuration to 0.081% at the worst ring and 0.049% at the typical
one. Against the same target, the point derivative is two orders of magnitude
further away: on the compact n=2 fixture the matched secant beats the point
derivative by 101x, and on the extended n=4 fixture the matched secant beats
the point derivative by 162x. The criterion asked only that the margin be
decisive, with ten times fixed in advance as the bar; it is not close.

**The convention offset is real, reproducible, and publishable on its own.**
On the compact n=2 fixture the forward secant and the point derivative
disagree by 13.2% at the worst ring, and on the extended n=4 fixture the
forward secant and the point derivative disagree by 13.1% at the worst ring.
Both figures are **maxima over rings**, and a review correctly caught an
earlier version of this paragraph generalizing them into "a property of the
definition, not of the profile". That is false. The *existence* of a
secant-versus-derivative gap is definitional, but its size depends on profile
curvature, radius, Sersic index and step, and the archive shows it varying
strongly with radius:

| fixture | | | | | |
|---|---|---|---|---|---|
| compact n=2 (sma 12/18/25/35/45) | 8.01% | 9.34% | 10.55% | 11.97% | 13.19% |
| extended n=4 (sma 18/28/40/55/70) | 10.34% | 11.18% | 11.92% | 12.56% | 13.10% |

It rises monotonically with radius and nearly doubles across the n=2 fixture.
The two galaxies agree at ~13% only because both *maxima* land there; comparing
two maxima and calling the result a constant was the error.

The defensible statement is narrow: **with `astep = 0.1`, the largest tested
differences were 13.2% and 13.1% on these two fixtures.** The consequence for
anyone comparing Bender amplitudes across tools survives unchanged and does not
depend on the magnitude: `a_4` from two codes is comparable only when their
radial step matches. Had Track 2 used
the accurate point derivative, this campaign would have reported a definition
mismatch as a 13% disagreement between tools.

**Criterion 2 was withdrawn on 2026-08-23, after review.** As pre-registered it
asked that the Bender agreement be no worse than the raw agreement plus the
gradient error. But the Bender amplitude *is* the raw amplitude divided by the
gradient, so the criterion compared a quantity against the two quantities it is
constructed from: an arithmetic identity check, not independent evidence. It
could not have licensed anything whatever numbers it returned.

Two further defects came with it. It compared maxima drawn from different rings
and different components, and it added percentage errors linearly where the
exact relation carries a denominator:

    |B - 1| <= (|R - 1| + |G - 1|) / (1 - |G - 1|)

That approximation is the whole explanation for the small paired "failures"
this design previously cited as a reason to keep the unpaired form — excesses
of 3.4e-4 and 6.6e-5. Under the exact bound, every noiseless case satisfies it
to rounding (worst excess 0.000000 and 0.000001). The earlier reading, that the
paired form "fails everywhere and so is no escape", was wrong.

**Validity is therefore structural, row-level, and observed.** A conversion is
applicable for a given **ring pair** where the realized harmonic basis is the
polar one, *both* the ring and its comparison partner at `sma·(1 + astep)` were
sampled by interpolation rather than nearest-pixel rounding, and both were
measured so a secant exists.

Two corrections here, both from review. These are read from **realized
provenance**, not from the requested configuration — an earlier version
inferred them from `spec["isoclip"]` and `spec["interpolate_start"]`, so an
archive whose realized behaviour disagreed with its request still reported
valid, inverting this campaign's own rule to instrument the sampling mode and
never predict it. And they are evaluated **per pair**, because a secant needs
both rings and a case-wide boolean cannot express a case that mixes modes.

The three things the withdrawn "licensed" boolean had merged are now separate
fields, because they answer different questions:

| field | scope | question |
|---|---|---|
| `conversion_method_validated` | campaign | did the method get empirical support? (criterion 1 — itself an accuracy comparison) |
| `harmonic_conversion_valid` | ring pair | does this row structurally support conversion? |
| per-regime accuracy | regime | how close was it here? gates nothing |

Criterion 1 remains an accuracy comparison, so the earlier claim that accuracy
"gates nothing" was too broad: accuracy supports the *method*, campaign-wide.
It does not gate any row, and no regime is excluded for being inaccurate.

How close it came is reported per regime:

| fixture | regime | gradient (worst) | gradient (typical) | raw | Bender | valid rows |
|---|---|---|---|---|---|---|
| compact n=2 | `reference` | 0.13% | 0.04% | 1.95% | 1.96% | 5/5 |
| compact n=2 | `eps_high` | 5.54% | 0.35% | 7.98% | 14.32% | 5/5 |
| compact n=2 | `noise_snr30` | 1.33% | 0.59% | 9.19% | 11.06% | 5/5 |
| extended n=4 | `reference` | 0.08% | 0.05% | 0.51% | 0.50% | 5/5 |
| extended n=4 | `eps_high` | 1.24% | 0.16% | 7.89% | 6.57% | 5/5 |
| extended n=4 | `noise_snr30` | 1.10% | 0.33% | 8.74% | 9.02% | 5/5 |

The last column counts **ring pairs**, not cases. Validity is row-level and is
read from realized provenance -- the recorded harmonic basis and each ring's
recorded sampling mode -- never from what the run requested. A case-wide
boolean would be wrong for `interpolate_default`, which contains interpolated
and nearest-pixel rings at once.

Every row is bound to the archives by `check_gradient_reconstruction.py`.

**This revises a pre-registered criterion after seeing data, which needs saying
plainly.** It is defensible only because the objection is structural rather
than outcome-driven: criterion 2 could not have been evidence whatever it
returned, and the revision neither loosens nor tightens a bar in order to reach
a conclusion. The withdrawn criterion and this reason are both preserved in the
archives, under `withdrawn_criterion_2`. Criterion 1 — which does weigh two
independent candidate reconstructions against one target — is unchanged and
still carries the verdict.

**Consequence for the earlier regime verdict.** The previous version of this
section reported that Track 2 was licensed on the reference configuration and
"nowhere else", because `eps_high` and `noise_snr30` failed criterion 2 on one
fixture and passed on the other. That verdict rested entirely on the withdrawn
criterion. The regimes are structurally valid; what varies between them is
*accuracy*, which the table above states directly. A user should read the
Bender column and decide whether 14% is tolerable for their purpose, rather
than be handed a yes/no derived from an identity.

**Applicability rules — per row, never per case.** A5 may write Bender
`a_n`/`b_n` for a given AutoProf **ring pair** instead of NaN when, and only
when, all of the following hold. Each is read from realized provenance
recorded in the archive, not from the run's requested configuration:

- the realized harmonic basis is `polar_from_image_x_axis` — never the
  eccentric-anomaly basis, whose order mixing Part A measured at 12% and 63%;
- the ring's realized sampling mode is interpolated, not nearest-pixel;
- the comparison partner at `sma·(1 + astep)` has a realized sampling mode that
  is also interpolated;
- both were measured, so a secant exists.

Where any fails, that row keeps its NaN and a `harmonic_conversion_reason`
naming the condition. Rows in the same case may differ: `interpolate_default`
has 2 of 5 rows applicable on both fixtures.

**Ellipticity and noise are deliberately absent from this list.** An earlier
version added "moderate ellipticity and clean data", which contradicted the
table above — `eps_high` and `noise_snr30` are structurally applicable on both
fixtures. Those regimes are less *accurate*, which the table states directly.
Accuracy is not an applicability condition, and folding it back in as one was
the withdrawn licensing model returning under another name.

The comparison-partner condition is the binding one in practice, and it is why
none of the exhausted-benchmark campaign's rows qualify: that campaign's
AutoProf arm is a free fit, so in general no ring exists at `sma·1.1` to
difference against. Track 2 there would need either forced paired rings or an
interpolated `b0`, and interpolating `b0` has not been tested. Until one of
those is built, A5 keeps its NaN on the campaign. See
`docs/09-exhausted-benchmark.md`.

**Two supporting numbers, archived rather than quoted loosely.** The wrong
ring estimator — a median rather than a mean — costs about one percent in the
reconstructed gradient, which is small enough that it is not the explanation
for anything but large enough to be worth naming. And AutoProf's default
nearest-pixel sampling perturbs the secant more than it perturbs either `b0`
it is built from, which is the same aliasing A3 measured, arriving in a
difference where it is amplified.

**What this result does not establish.** Two galaxies, neither PSF-convolved,
both noiseless in the reference configuration. Method validation is a statement
about a conversion under stated conditions, not a survey of how AutoProf
behaves on real data.

### A5. Schema: preserve native values, never overwrite

The pass-through columns must not simply be replaced. `profile.fits` from
an AutoProf arm carries all of:

| Column | Meaning |
|---|---|
| `autoprof_a3_native`, `autoprof_b3_native`, … | exactly what AutoProf wrote, untouched |
| `autoprof_b0` | the DC term, needed for the raw reconstruction |
| `s3_raw_major`, `c3_raw_major`, … | reconstructed raw amplitudes rotated to the major-axis frame |
| `a3`, `b3`, … | Bender-converted values — **written only when valid** |
| `harmonic_basis` | which angle basis produced the native values |
| `harmonic_sampling_mode` | line versus isophotal-band, per ring |
| `harmonic_conversion_valid` | boolean |
| `harmonic_conversion_reason` | why, when not valid |
| `harmonic_measurement_status` | the producing tool's own failure reason for a NaN row |

`harmonic_measurement_status` carries isoster's internal `GRADIENT_*` /
`DEVIATIONS_*` status verbatim (`empty_comparison_ring`, `singular`,
`underdetermined`, …) so a NaN in the archive says *why* it is NaN. This is
diagnostic provenance rather than a prerequisite: a NaN is already correct
without it, but "the outer rings are NaN" and "the outer rings are NaN because
the comparison ring fell off the frame" are different amounts of information
when someone reads the archive a year later.

**Native and converted values must never share the column name `a3`.**
That is the whole failure this section exists to prevent. The bare names
`a3`/`b3` keep their established meaning — Bender-normalized, major-axis,
comparable with isoster and photutils.

When the conversion does not hold, those columns carry **NaN** (or a
masked value) together with `harmonic_conversion_valid = False` and a
populated `harmonic_conversion_reason`. "Left empty" was loose wording:
the output is a fixed-schema FITS table, where every row has every
column, so the absence has to be represented by a value rather than by
omission. NaN also fails loudly in arithmetic, which a silent 0.0 would
not.

This change requires a **schema version increment** and a migration note,
because existing archived campaign products contain an `a3` column whose
contents are native AutoProf values under the old pass-through. Without
the version bump, old and new files are indistinguishable while meaning
different things.

### A6. Tests and CI

Meaningful coverage must not sit behind an AutoProf-available skip, since
CI has no AutoProf venv.

- **Always-running unit tests**: FFT scaling and signs, the `a`-is-sine
  assignment, the PA rotation formula, background-offset sensitivity, the
  schema conversion, and the validity-flag logic. These are pure
  arithmetic on synthetic rings (A1) and need no AutoProf.
- **Optional integration test**: the image-level planted fixture through
  a real AutoProf run, skipping cleanly when the venv is absent — the
  same treatment the campaign fitter gives a missing venv.

  The always-running A1 arm *reimplements* AutoProf's short FFT
  expression rather than executing AutoProf. That is acceptable for a
  unit test, but it means the unit tests cannot notice if an AutoProf
  upgrade changes the convention underneath us. So the integration test
  must additionally confirm that the **installed** AutoProf still emits
  `a_n`, `b_n` and `b0` with the same meaning, and the archive must record
  the AutoProf version and a digest of
  `pipeline_steps/Isophote_Extract.py`. A convention verified against a
  version we did not record is a convention we cannot defend later.
- **Archive plus checker**: `reference_harmonic_scale.json` and
  `check_harmonic_scale.py`, wired into the docs CI job beside the two
  existing gates, storing both tracks and every diagnostic.

## Part B: controlled three-way timing

### B0. The contract — authoritative, frozen 2026-08-23

**Everything below this section is historical.** B1–B5 record how the design
was reached and were then corrected twice; reading them as instructions
produces contradictions, because successive corrections were appended rather
than folded in. This section supersedes them wherever they disagree. Nothing in
Part B may be implemented against B1–B5 directly.

**Scope, and what each number is called.**

1. **Fixed-aperture extraction and harmonic evaluation** (the controlled
   scope). Geometry is imposed, so nothing about geometry *fitting* is timed.
   It is not a comparison of fitting algorithms and must never be described as
   one. Initialization and background supplied, output writing excluded,
   AutoProf on a persistent worker.
2. **End-to-end** (the natural scope). Same scientific input and required
   output for all three; each tool uses its own radial grid and stopping rule.
   Matched radii are *not* imposed. Achieved coverage and partial profiles are
   reported per tool.

These are archived separately. A single ratio spanning both is a category
error.

**The accuracy contract.**

3. **One common threshold per metric and per fixture, applied to all three
   tools.** Never a per-tool bar: that preserves whatever accuracy a tool
   already has, so a faster but less accurate implementation passes its own
   lower bar and wins.
4. Thresholds are justified from **scientific requirements or independently
   established numerical accuracy** — never from any tool's own pilot result.
   The phase-2 timing calibration sets repetition counts and quantifies timing
   scatter, and nothing else; it never touches a threshold.
5. **Completion, radial coverage and scientific accuracy are three separate
   outcomes**, recorded separately and never collapsed into one verdict.
6. **Every timing is archived**, including failed and ineligible ones.
   Eligibility affects which timings enter a headline summary; it never affects
   retention, and never affects failure accounting.
7. **Harmonic accuracy criteria apply only to harmonics-on arms.**
8. **End-to-end accuracy is evaluated on each tool's own returned apertures,
   over a defined common radial overlap.** Part A's fixed-ring truth does not
   transfer to a free fit.
9. **Accuracy is computed outside the timed region.** A tool must not be
   charged for the harness measuring it.

**Environment and abort rules.**

10. The **interpreter gap is measured on its own calibration line** — one
    identical CPU-bound workload timed in both interpreters — and reported
    beside every AutoProf timing. Never subtracted: the penalty is real for a
    user, and never hidden: it is not part of AutoProf's algorithm.
11. **Harness cost** — serialization, FITS round-trip, IPC — is timed apart
    from the fit. Both fit-only and fit-plus-harness figures are published.
12. **Contamination aborts key on external indicators** (machine load, thermal
    state, competing processes) **or abort an entire campaign.** Individual
    sessions are never rejected for having dispersed timings: discarding
    measurements because of their values is outcome-based selection.
    Dispersion is reported, not used as a filter.

**Structure.** Few fixtures, many repetitions — roughly six to ten
configurations across size, Sersic index and signal-to-noise, timed over
several sessions in separate interpreters with interleaved arm order. This buys
an uncertainty interval on every number. **It is not a survey**, and the
archive must say so in those words. Harmonics on and off is a grid axis.

### B0.1 Stage 1 contract — the scientific terms, frozen

*(Renamed: this document uses **Part A / Part B** for the two halves of the
study, so the sub-steps of Part B are **Stages 1–4**. An earlier draft called
them "Phase 1–4", which collided confusingly with Part A/B.)*

Everything Stage 1 must fix, fixed. Nothing here may be revised by looking at a
timing result; revision requires a committed protocol amendment saying what was
learned and why.

**1. Metrics, named in the schema's own vocabulary.** An earlier draft called
the raw components `a3,b3,a4,b4`, which the Schema-1 reference reserves for
**Bender-normalized** coefficients. The raw ones are `s_n`/`c_n`:

| metric | components | definition |
|---|---|---|
| `raw_amplitude_error_pct` | `s3_raw_major`, `c3_raw_major`, `s4_raw_major`, `c4_raw_major` | \|measured/truth − 1\| × 100 |
| `ring_intensity_error_pct` | ring mean | \|measured/analytic − 1\| × 100 |
| `center_error_px` | centre only | \|measured − true centre\|, pixels |
| `eps_error` | ellipticity | \|measured − true\| |
| `pa_error_deg` | position angle | \|measured − true\| mod 180° |

The last three are free-fit arms only. Centre alone is insufficient — a fit can
have the right centre with wrong ellipticity or PA and so sample quite
different apertures — which is why `eps_error` and `pa_error_deg` were added
and the centre metric renamed from `geometry_error_px` to `center_error_px`.

**Reduction order, fixed:** over **component** (max of the four), then
**radius** (per-ring, not pooled — the bars are per-ring), then **seed** (see
§2), then **session** (median, with the full spread archived).

**2. Thresholds — derived, per component, by
`benchmarks/timing/accuracy_thresholds.py`.** The whole contract is committed
as `frozen_stage1_contract.json` and gated field by field by
`check_accuracy_thresholds.py`; this prose describes it and is not the source
of truth. Two regimes:

*Noiseless fixtures gate systematic accuracy.* With no noise there is no
sampling variance, so a departure from analytic truth is the tool's own
numerics. The bar for each **component** on each ring is that component's own
ideal 1σ at S/N = 100, obtained from the exact raw amplitude by dense Fourier
integration (`integrated_harmonic_truth`) — not from a planted fraction times
the ring intensity, which an earlier draft used. Raw harmonic truth depends on
the radial gradient, Sérsic curvature, mode products and each component's own
planted amplitude, so one bar for all four was simultaneously too strict for
the large modes and too loose for the small ones, and since eligibility takes
the worst component it penalised the smallest systematically. For
`sersic_n2_compact`, s3/c3/s4/c4:

| sma | s3 | c3 | s4 | c4 |
|---|---|---|---|---|
| 12 | 2.771% | 2.078% | 4.156% | 1.385% |
| 45 | 7.995% | 5.996% | 11.991% | 3.997% |

**The justification is declared, not borrowed from any tool.** The bar is
`IDEAL_SIGMA_FRACTION = 1.0` times the uncertainty of an *ideal* estimator at
the reference depth — a systematic at or below the noise a user faces cannot be
detected in their data, whoever wrote the code. An earlier draft additionally
argued that a tighter bar should be rejected "because no current implementation
meets it". That is exactly the reasoning B0.4 forbids and it has been removed.
Whether existing tools clear these bars is a **result**, reported separately in
the A3 archive, and never an input to the bar.

*Noisy fixtures are judged on ensemble bias, at the arm level.* The mean over
R = 25 realizations has standard error σ/√R. A per-test 3σ screen across the
120 component–ring tests would fail an unbiased tool about 10% of the time by
chance alone, so the decisive statistic is **one χ² per arm**: the sum of
squared standardized mean residuals over all tests, which is χ²(120) under the
no-bias hypothesis, compared against its 99th percentile (158.95). Multiplicity
is handled by construction rather than by a screen that cries wolf.

Geometry bars: `center_error_px ≤ 0.5`, `eps_error ≤ 0.01`,
`pa_error_deg ≤ 1.0`.

**Rejected alternative, recorded.** Gating a single realization's worst error at
a statistical 1σ gives an unbiased ring at σ = 9.8% an 8.1% chance of passing
and a *perfect* tool a vanishing chance of passing every test at once. That
gate measures noise, not accuracy.

**3. Fixture grid and seeds.** Six galaxy configurations, each run with
harmonics on and off — a full factorial. All six are **defined executably** in
`benchmarks/timing/stage1_fixtures.py` with frozen fixed-aperture radii; an
earlier draft named four of them in prose only, so no accuracy bar could be
computed for them and Stage 2 could not have checked most of its arms.

| fixture | n | R_e | shape | radii | scope |
|---|---|---|---|---|---|
| `sersic_n2_compact` | 2 | 25 | 241² | 12 / 18 / 25 / 35 / 45 | both |
| `sersic_n4_extended` | 4 | 40 | 321² | 18 / 28 / 40 / 55 / 70 | both |
| `size_ladder_481` | 2 | 50 | 481² | 25 / 37.5 / 50 / 70 / 90 | both |
| `size_ladder_961` | 2 | 100 | 961² | 50 / 75 / 100 / 140 / 180 | both |
| `size_ladder_1921` | 2 | 200 | 1921² | 100 / 150 / 200 / 280 / 360 | both |
| `wide_canvas_961` | 2 | 25 | 961² | 12 / 18 / 25 / 35 / 45 | end-to-end only |

The ladder scales `R_e` and the radii with the canvas, so ring count and
samples-per-ring grow together and the extraction workload genuinely changes.
`wide_canvas_961` holds the galaxy fixed and grows only the image, which
changes whole-image overhead but barely touches fixed-ring extraction — so it
is confined to the end-to-end scope, where overhead is legitimately part of the
task.

Seed blocks: calibration **100000**, campaign **60260822**, disjoint from all
four Part A blocks.

**4. Radial range, coverage and overlap.**

*Fixed-aperture scope:* identical requested radii for all tools, so coverage is
exact by construction and a missing ring is a failure, not a coverage
difference.

*End-to-end scope:* each tool chooses its own radii, so "coverage" needs a
target that exists independently of what any tool returned. The frozen,
tool-independent **scientific target interval** is `[0.3 R_e, 3.0 R_e]` per
fixture — inside the PSF-free core at one end and out to where the planted
harmonics remain above the noise at the other. Coverage is measured against
*that*, never against the tools' own extents.

The **common overlap** — `[max(r_min,i), min(r_max,i)]` across the three tools
— is used **only** for cross-tool accuracy comparison, never for coverage
scoring. Three tools that all truncate early would otherwise show an excellent
mutual overlap while covering little of the science.

*Truth on a free-fit aperture* is defined operationally: for each returned
aperture `(x0, y0, sma, eps, pa)`, the analytic harmonic truth is integrated
**on that ellipse**, by the same routine Part A uses on the planted reference
ellipse (`integrated_harmonic_truth`), evaluated with the tool's own geometry
rather than the planted geometry. Where a tool's geometry differs from the
planted one, that difference shows up in `eps_error`/`pa_error_deg`, not
silently in the amplitude comparison.

**5. Outcome fields — five, not one status.** An earlier draft required
exactly one status per timing, which re-collapses precisely the separation B0
demands: a run can be both incomplete and inaccurate. Five independent fields
are archived, and eligibility is *derived* rather than assigned:

| field | values |
|---|---|
| `execution_status` | `ok` / `failed` (raised, crashed, or returned no profile) |
| `coverage_status` | `complete` / `partial` (< 60% of the target interval) |
| `accuracy_status` | `pass` / `fail` (against §2) / `not_evaluated` (harmonics-off arms are not judged on harmonic metrics) |
| `contamination_status` | `clean` / `contaminated` |
| `headline_eligible` | derived: `execution_status == ok and coverage_status == complete and accuracy_status != fail and contamination_status == clean` |

Every timing is archived under all five regardless of value. Only
`headline_eligible` affects summaries; nothing affects retention.

**6. Contamination — measured, and the first bound was wrong.** The rule keys
on external state, never on the timings.

*What was measured, and why it is not a baseline.* 30 samples at 10 s gave
1-minute load average **min 2.30, median 4.00, p90 6.96, max 8.56**. That
window ran concurrently with this session's own test suites, so it is an
**upper bound on a busy machine, not an idle baseline**, and it is recorded
here as such rather than dressed up as one. It did its job regardless: an
earlier draft proposed a bound of `≤ 1.0`, which this host does not reach even
at its quietest observed moment of 2.30. Measuring before freezing caught that.

*The absolute ceiling is therefore derived from the hardware, not from that
sample* — a contaminated measurement cannot set the bar it was meant to
validate. The host has **10 cores**, so a 1-minute load of 2.0 is 20% of
capacity already committed before the benchmark starts.

*Frozen rule:*

- **Pre-campaign baseline**: 30 samples at 10 s **with no other work at all**,
  agent sessions included. Require `median ≤ 2.0` (20% of 10 cores). If the
  machine cannot meet this when genuinely idle, that is a finding about the
  machine, to be reported — not a reason to raise the bar.
- **During sessions**: sample every 10 s, not merely before and after — a
  before/after pair misses contention in between. Abort if load exceeds
  `baseline_median + 2.0`. The benchmark is single-threaded on a 10-core host,
  so it contributes ≈ 1; the remaining 1.0 is jitter allowance.
- **Thermal**: `pmset -g therm` sampled per session; any recorded thermal or
  performance warning aborts. On this Apple Silicon host
  `kern.thermalpressurelevel` and `machdep.xcpm.cpu_thermal_level` do not
  exist, so this signal is binary rather than graded.
- **On abort**: the whole campaign is re-run; individual sessions are never
  dropped. **Every aborted campaign is retained and archived**, with its
  indicator trace. **Maximum 3 retries**; if a fourth would be needed, stop and
  report the machine as unfit rather than continuing to draw until a quiet run
  appears.

**7. Partial profiles.** Neither discarded nor padded. Achieved ring count and
coverage are archived, accuracy is computed on the rings returned, and the run
is classified by §5. Cost-per-isophote is reported beside total time, since a
tool taking fewer, coarser rings is not thereby faster at the same task.

**Four stages, in order, each frozen by commit before the next begins.** An
earlier version of this section said pilots set repetition counts *and* that
repetition counts must be fixed before the pilot — a contradiction, because it
left "the pilot" naming two different things.

1. **Stage 1 — freeze the science.** Metrics, the numeric value of each common threshold
   with its scientific justification, fixtures, coverage rules, contamination
   indicators and their bounds, and the success/failure taxonomy including
   partial profiles. **No timing is run in this phase**, so no threshold can be
   chosen after seeing what a tool achieved.
2. **Stage 2 — timing calibration** — explicitly named as such, and *not* the benchmark.
   Its only *decisional* outputs are batching, repetition counts and session
   structure, chosen from observed timing scatter.

   Its accuracy numbers are **retained in the calibration archive and are
   explicitly non-decisional**: they may not inform any Stage 1 threshold. An
   earlier version said they were discarded. That was wrong — freezing the
   thresholds before Stage 2 already prevents outcome-driven selection, so
   deleting the evidence buys nothing and costs the ability to notice that
   calibration timed a broken arm. They are therefore checked *against* the
   already-frozen thresholds, and a failure stops the work for investigation.
   Changing a threshold afterwards requires an explicit, committed protocol
   revision that says what was learned and why — never a quiet edit, and never
   the deletion of the measurement that prompted it.
3. **Stage 3 — commit those timing parameters.**
4. **Stage 4 — run the campaign**, independent of Stages 2 and 3's data.

Stage 1 is the blocker. Implementation scaffolding and small functional smoke
tests may be written now; the scientific timing work may not start until the
thresholds exist and are committed.

**A new archive, not a replacement.** See B5, which stands unchanged.


### B1. Where the protocol goes

`archive_speedup.py` *summarizes*; the timing protocol belongs in the
**runner**, not the archiver. The controlled protocol —
sessions in separate interpreters, shuffled interleaved arm order within
each repetition, batching for in-process work — is added at the point
where fits are executed.

### B2. The timed task must be defined, not implied

Left unspecified, "the same fit" is not a defined quantity. Fix and
record, identically for all three tools:

- whether higher harmonics are computed at all (they cost real time);
- the common radial range and radial step;
- geometry freedom: free-fit versus fixed-geometry forced photometry;
- sigma-clipping settings and interpolation scheme;
- whether writing outputs to disk is inside or outside the timed region.

### B3. Comparable work, honestly bounded

AutoProf's `Process_Image` runs background estimation, PSF measurement
and centering that the in-memory isoster and photutils calls do not.

**Report both measurements, not one or the other.** The earlier draft
offered them as alternatives; that was a decision left undecided.

The two have **different scopes, and the difference is deliberate**:

- **Core fit — controlled, matched.** Initialization and background
  supplied, output writing excluded, run against a **persistent AutoProf
  worker** so process spawn and interpreter startup do not contaminate it
  (startup reported as its own line). This comparison *may* require
  matched radii and matched coverage, because the geometry is imposed
  rather than discovered. It is the number that compares fitting
  algorithms.

- **End-to-end — natural, uncontrolled.** The same scientific input and
  the same required output for all three, but **each tool uses its own
  radial grid and its own stopping rule**. Do not force matched radii
  here: that would measure a configuration nobody runs. Instead report
  achieved radial coverage and partial profiles per tool, so the reader
  sees that the tools sampled differently rather than being told they
  sampled the same. It is the number a user experiences.

Reporting a single speed ratio across these two scopes would be a
category error; they answer different questions and are archived
separately.

**Do not claim identical algorithms.** Interpolation and clipping cannot
be made identical across the three tools. Define the intended *task*,
document each tool's actual setting alongside its timing, and let the
difference stand in the open.

Record alongside every timing: accuracy against truth, successful radial
coverage, number of rings fitted, and where obtainable the sampling and
iteration counts. Cost per isophote is not equivalent work — a tool that
takes fewer, coarser rings is not thereby faster at the same task.

### B4. Frozen before the full run

The following are fixed and recorded in advance, not settled while
looking at results:

- fixture grid and noise seeds;
- session and repetition structure;
- success and failure accounting — what counts as a completed fit;
- the radial-coverage accuracy requirement a fit must meet to be timed at
  all;
- treatment of partial profiles;
- summary statistics and the uncertainty interval reported with each;
- **worker-state reset between AutoProf cases**, so one case cannot
  inherit another's cached state.

### B2-B4 settled with the reviewer, 2026-08-22

B2-B4 stated principles but left the decisions that actually determine the
answer open. Five were taken with the reviewer **before any implementation**,
and are recorded here rather than settled while looking at results.

**Measured asymmetries the protocol must live with.** These were checked, not
assumed:

| | isoster | photutils | AutoProf |
|---|---|---|---|
| interpreter | Python 3.12.7 | 3.12.7 | **3.10.21** |
| invocation | in-process | in-process | **subprocess, FITS round-trip** |
| photutils | — | 2.3.0 | pins `<=1.5.0` |

AutoProf pins `numpy<2` and `photutils<=1.5`, so it cannot share the project
environment. The interpreter gap is therefore not a defect in the harness to
be fixed; it is a property of the tool as anyone can obtain it today.

**1. The interpreter gap is measured and reported on its own line.** A
calibration arm runs one identical CPU-bound workload in both interpreters and
publishes the ratio beside every AutoProf timing. CPython gained a large
general speedup after 3.10, so a raw three-way ratio would charge AutoProf an
interpreter penalty on top of its algorithm. The penalty is real for a user,
which is why it is not removed — but a reader must be able to separate
"AutoProf is slower" from "AutoProf's stack is older", and a single number
cannot express both.

**2. The accuracy gate is Part A's fixtures and Part A's truth — with a
common bar, revised after review.** As first written this said the bar would be
derived "per tool and per fixture" from a pilot. That is wrong, and a review
caught it: a per-tool bar preserves whatever accuracy each tool already has, so
a faster but less accurate implementation passes its own lower bar and wins the
comparison. The bar must be **one common threshold per metric and per fixture,
applied to all three tools**, chosen from scientific requirements or from
independently established numerical accuracy — never from any tool's own pilot
result. Pilots then set repetition counts and quantify timing scatter, and
nothing else.

Four consequences follow, all of which must be frozen before the pilot runs:
completion, radial coverage and scientific accuracy are accounted **separately**
rather than rolled into one verdict; failed and ineligible timings are
**archived**, and excluded only from headline ratios, never from failure
accounting; the harmonic accuracy gate applies **only to harmonics-on arms**;
and for the end-to-end free fits, truth is evaluated on each tool's own returned
apertures over a defined **common radial overlap**, because Part A's fixed-ring
truth does not transfer to a free fit.

The original wording follows, superseded:

> **2. The accuracy gate is Part A's fixtures and Part A's truth.** B4 required
"an accuracy requirement a fit must meet to be timed at all" without saying
what it was, and that one line does more work than the rest of B4 together: a
tool that converges loosely is faster, so an ungated timing rewards giving up
early. Before its timing counts, each arm must recover the planted harmonics
and the intensity profile on the Part A planted fixtures to a bar **frozen in
advance**, derived from a pilot exactly as Part A's tolerances were. This
reuses instrumentation that already exists and is already gated, and ties
"the same task" to a measured quantity rather than to matching config files.

**3. Harness cost is a separate line and is never silently subtracted.** B3's
persistent worker removes process spawn, but serialization, the FITS
write/read and IPC remain, and those are our harness rather than AutoProf's
algorithm. The worker is instrumented so they are timed apart from the fit.
Both figures are published: fit-only, and fit-plus-harness. A single number
with the overhead quietly removed would describe a program nobody can run.

**4. Few fixtures, many repetitions.** A small frozen grid — roughly six to
ten configurations spanning image size, Sersic index and signal-to-noise —
each timed across several sessions in separate interpreters with interleaved
arm order. This buys a genuine uncertainty interval and a dispersion check on
every number, which the existing two-way archive (243 configurations, a single
timing each, no error bar) does not have. **It is not a survey**, and the
archive must say so in those words: it trades coverage for the ability to
state how well each number is determined.

**5. Harmonics are a grid axis, not a fixed setting.** Each tool is timed with
`a3/b3/a4/b4` on and off. What measuring higher harmonics costs is a finding
in its own right, and Part A has just established that the three tools'
harmonics are comparable quantities, so the comparison is meaningful. The
small grid absorbs the doubling cheaply.

**Naming, corrected after review:** the fixed-aperture core measurement is an
**extraction and harmonic-evaluation comparison**, not a comparison of
geometry-fitting algorithms. Geometry is imposed there, so nothing about
geometry fitting is being timed, and calling it that would overclaim what the
headline number covers.

**Consequent, and following from B3 rather than newly decided:** the core-fit
scope is fixed-geometry with matched radii and output writing excluded; the
end-to-end scope is a free fit on each tool's own radial grid and stopping
rule, with output writing included. Sigma-clipping and interpolation cannot be
made identical across the three; each tool's actual setting is recorded beside
its own timing and the difference stands in the open.

**Still to be fixed by measurement before the full run**, in the same manner
as Part A's tolerances — frozen from a pilot, committed, then validated:

- ~~the accuracy bar itself, per tool and per fixture~~ — **superseded by B0.3**:
  one common threshold per metric and fixture, applied to all three tools;
- session and repetition counts, chosen so the reported interval is narrower
  than the effect being claimed;
- the **contamination abort rule**. An earlier draft of this section proposed
  rejecting individual sessions whose timings were dispersed. A review caught
  that as outcome-based selection: discarding measurements because of their
  values is how a benchmark quietly flatters itself. The rule must instead key
  on **external contamination indicators** — machine load, thermal state,
  competing processes — or abort an **entire campaign**, never individual
  sessions chosen after seeing their numbers. Dispersion is then *reported*,
  not used as a filter.
- success and failure accounting, and the treatment of partial profiles.

### B5. A new archive, not a replacement

Create a **new three-way archive** rather than overwriting
`reference_speedup.json`. The established two-way result and the
published median-45× claim it supports stay valid until the new protocol
is accepted on its own merits. Replacement, if any, is a later and
separate decision, and any change to a published number is printed by the
checker rather than hand-edited.

## Sequencing

Part A first, because it is cheap and it is a correctness gate on any
harmonic comparison.

But **Part A does not logically gate Part B**: harmonic-scale validation
only constrains a timing study if harmonic measurement is inside the
timed task (see B2). If harmonics are excluded from the timed task, the
two parts are independent and may proceed in either order.

## What is deliberately not in scope

- The real-data benchmark plan named in the handover. This branch makes
  the three-way machinery trustworthy; choosing real galaxies and
  defining real-image success criteria is separate work.
- Any change to isoster's own harmonic convention. The invariant pinned
  by `tests/unit/test_harmonic_normalization.py` stands.
- The `sma = 0` central-pixel row, whose `0.0` harmonics are a **legacy
  sentinel** rather than a measurement: there is no ring at zero radius.
  Part A excludes that row from every comparison and every archived
  statistic. It is documented as a sentinel in `docs/04-architecture.md`
  and is deliberately left as it is.
- Cutting the `v1.0.0` tag or correcting `CITATION.cff`. Both open, both
  unrelated.
