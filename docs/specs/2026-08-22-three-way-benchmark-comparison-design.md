# Design: three-way benchmark comparison (isoster / photutils / AutoProf)

Date: 2026-08-22 (revised through five review rounds)
Branch: `benchmarks/three-way-comparison`
Location: tracked in `docs/specs/`, excluded from the published site. Moved
here from the gitignored `docs/agent/` on 2026-08-22 — a design this branch
depends on should not live only on one machine.
Status: **Part A ready for implementation**; **Part B deliberately
unfrozen** — its choices are written down (B2, B3, B4) but are not
finalized, and are best frozen after the Part A calibration work is
complete and has informed them.

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

`m=3` shows scatter with no systematic bias; `m=4` is systematically inflated,
decreasingly so with radius as the half-pixel displacement matters less
relative to the ring.

**Consequences for the rest of Part A.**

- The publication claim is now "the three tools measure the same harmonic
  signal to ~0.1–0.3% once each is sampled comparably", with the sampling
  caveat stated — not "AutoProf reads 13–25% high".
- `ap_iso_interpolate_start` becomes a **required, recorded** setting of the
  calibration, not a default to inherit. The grid must archive it.
- It is also a finding about `m=4` measurements generally: any tool sampling
  isophotes by nearest-pixel lookup will bias the four-fold mode, which is
  precisely the boxy/discy diagnostic. Worth stating in its own right.
- The earlier reading that the excess "shrinks with radius" was a partial view
  of this: at eps=0.6 the measured PSF differs, moving the switch radius, so
  the excess is not monotonic in ellipticity. Radius dependence is real but
  secondary to which side of the switch a ring falls.

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
| Ellipticity | ~0 (control), 0.3, 0.6 | At eps≈0 polar angle and eccentric anomaly coincide, so the basis question vanishes and everything else is isolated; the non-circular cases make it bite |
| PA | 0°, 30° | At PA=0 AutoProf's polar basis coincides with major-axis polar, so differences 1–3 show alone; 30° adds difference 4 |
| `ap_isoclip` | off, on | Selects AutoProf's angle basis |
| Radius | ≥2 values | Raw amplitudes are radius-dependent; one radius cannot show that |
| Background offset | 0, +δ | Tests the corrected claim 1 directly |
| Noise | several realizations at each S/N | One realization measures a realization, not a distribution |
| Mask | **deferred from the initial campaign** | See below |

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
   **ellipticity × basis**, **PA × polar resampling**,
   **noise × clipping**. (Mask × resampling is dropped with the mask
   axis — see above.)
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
