# Multi-Band Isoster (Experimental)

> **Status: experimental.** Stage-1 (joint fit) shipped 2026-04-30; the
> Stage-3 backport campaign + Phase 39 finalization shipped 2026-05-04
> on `feat/multiband-feasibility`. API and output schema are subject
> to change before the feature is merged to `main`. The pre-merge code
> review (see `docs/agent/journal/`) cleared three correctness blockers
> (B1/B2/B3) and six high-priority issues (H1–H6). See
> `docs/agent/plan-2026-04-29-multiband-feasibility.md` for the locked
> design record (24 decisions captured from a structured interview
> before any code was written) plus Phase-38 / Phase-39 update.

## Status

The default configuration — shared validity, joint intercepts, the
geometry-parameterised solve and `independent` higher harmonics, in OLS or WLS —
is **supported**, not experimental, as of 2026-08-15. Two features keep their own
warnings and are called out where they appear below:

- `multiband_higher_harmonics ∈ {simultaneous_in_loop, simultaneous_original}`,
  because the single-band equivalent has benchmark regressions;
- `loose_validity=True`, which is repaired and tested end to end but not yet
  the default.

Real-data validation is concentrated on a small number of targets, and the B=5
path has synthetic coverage only — the asteris five-band cutouts are not
available on the development machine.

## Shared higher harmonics mean a shared *shape*

`multiband_higher_harmonics='shared'` fits one dimensionless, Bender-normalised
shape coefficient across bands, and writes each band's own raw amplitude back
into Schema 1.

This matters because a common shape deviation produces raw amplitude
`A_n_raw = -A_n_norm · sma · grad_b` in band `b` — proportional to *that band's*
radial gradient. Writing one identical raw value into every column therefore
cannot represent one shared shape once the gradients differ, and because
plotting divides by each band's own gradient, the reported normalised amplitudes
came out band-dependent *and* dependent on the bands' arbitrary flux units. For
a planted shape of 0.02 in both bands:

| units | reported normalised a4, band g | band r |
|---|---|---|
| original | 0.02000 | 0.02000 |
| band r × 10 | 0.11000 | 0.01100 |
| band r × 0.1 | 0.01100 | 0.11000 |

After the reparameterisation both bands report 0.02000 at every rescaling. The
raw per-band columns now legitimately differ — that is what a shared shape looks
like in raw units — and the invariant to check is that they *normalise* to one
value.

**The `simultaneous_in_loop` and `simultaneous_original` modes still stamp one
identical raw amplitude into every band**, so they represent a shared raw
intensity residual rather than a shared shape. That is a further reason those
modes keep their experimental warning, and it is why they no longer agree with
`shared` column-by-column. The default `independent` mode is unaffected: it fits
each band separately from its own gradient and was always self-consistent.

## Backport status: shipped vs not-shipped vs experimental

The table below tracks every Stage backport from single-band into the
multi-band path, plus the Phase 39 finalization. Use it to decide
whether the feature you want is available in this code path.

| Stage | Feature | Status |
|---|---|---|
| 1 | Joint multi-band fit (shared geometry, per-band intensities + harmonics) | ✅ shipped |
| A | `integrator='median'` for `intens_<b>` (decoupled mode) | ✅ shipped |
| B | `outer_reg_*` damping (Tikhonov, default `outer_reg_weights = {1, 0, 0}` — center only) | ✅ shipped |
| C | `lsb_auto_lock` family (matrix-mode lock, anchor capture, debounce) | ✅ shipped |
| D | `compute_cog` per-band curve-of-growth + `sky_offsets` post-hoc kwarg | ✅ shipped |
| F | `central_reg_*` (LSB-central geometry penalty) | ✅ shipped |
| H | Forced-photometry mode (`template_isophotes`) | ✅ shipped |
| H.1 | Post-hoc per-band harmonics in forced mode | ✅ shipped |
| I | ASDF I/O symmetry (`isophote_results_mb_to_asdf` / `_from_asdf`) | ✅ shipped |
| J | Parallel CLI (`isoster-mb` console script, separate from `isoster`) | ✅ shipped |
| 6 | Higher-order harmonic modes: `independent` / `shared` / `simultaneous_in_loop` / `simultaneous_original` | ✅ shipped (`simultaneous_*` carries an experimental warning) |
| E | `outer_reg_mode='solver'` (reference-pull solver mode) | ⏸ deprioritized — same failure mode as `{1, 1, 1}` damping default. Reactivate on user request. |
| G | `'adaptive'` integrator | ⏸ deprioritized — superseded by Stage-C `lsb_auto_lock`. |
| — | ISOFIT (`simultaneous_harmonics` flag from single-band) | ❌ not ported. The Section-6 `multiband_higher_harmonics` enum is the multi-band replacement. |

## Known limitations and gotchas

Read this list before running multi-band on real data. Most items are
real-data findings codified during the Stage-3 campaign and the
pre-merge review pass.

1. **`outer_reg_weights = {1, 1, 1}` over-pins on barred / spiral / BCG-ICL
   systems.** The Stage-B follow-up sweep on PGC006669 showed Tikhonov
   `α` saturates above 99.99 % for any positive eps / pa weight in the
   user-intuition range — the eps / pa choice is **binary, not graded**.
   Default flipped to `{1, 0, 0}` (center-only). Opt back into
   `{1, 1, 1}` only for confirmed-static outer envelopes (cD galaxies,
   isolated massive ellipticals).

2. **`lsb_auto_lock` is for cD / massive ellipticals only.** On barred /
   spiral targets the lock fires inside the disc (sma ~ 100 px, well
   before the LSB regime) and pins outer geometry to the bar shape. The
   MAD-based scatter metric does NOT catch this; visual QA on the
   comparison panel does. The field docstring carries the galaxy-type
   caveat.

3. **Forced-photometry mode silently ignores several iteration-only
   features.** `lsb_auto_lock`, `use_outer_center_regularization`,
   `use_central_regularization`, `harmonic_combination='ref'`,
   and `multiband_higher_harmonics ∈ {shared,
   simultaneous_in_loop, simultaneous_original}` are no-ops under
   forced extraction. The driver emits a `UserWarning` listing which
   features were dropped (review fix B2).

4. **Bender-normalized harmonics are required for interpretation.** Raw
   `a_n_<b>` / `b_n_<b>` values scale with local flux and are NOT
   directly comparable across bands or rings. Always plot
   `A_n_norm = -A_n / (a · dI/da_b)` per band. The repo plotting helpers
   enforce this convention (see `CLAUDE.md`).

5. **PSF-matched inputs assumed.** No PSF handling in the driver. If
   the per-band PSFs differ, isophotes whose SMA is comparable to the
   worst PSF FWHM will carry PSF-mismatch artifacts; the user must
   accept that or PSF-match upstream.

6. **Variance maps are all-or-nothing.** Either every band has one
   (full WLS) or no band has one (full OLS). Mixed-mode rejection at
   driver entry.

7. **`compute_cog`'s raw curve-of-growth dips past the LSB transition**
   on real data because the joint solver's `I0_b` carries residual sky
   (~ −1×10⁻³ e⁻/s on denoised cutouts, even after upstream
   sky-subtraction). Use the `sky_offsets` post-hoc kwarg on
   `compute_cog_mb` to apply an outermost-tail-derived offset before
   plotting `cog`/`r1/2`. Demos under
   `examples/example_asteris_denoised/` apply this explicitly between
   FITS write and QA plot generation; raw FITS files always preserve
   uncorrected `intens_<b>`.

8. **Single-band `integrator='median'` requires
   `fit_per_band_intens_jointly=False`.** The matrix-mode joint LS
   solve cannot host a median (non-linear). The validator hard-errors
   the combination at config construction with a remediation hint.

9. **Sample-validity is shared across bands by default.** A sample is
   dropped from the joint solve if any band's mask flags it, any band
   has NaN there, or any band's variance is non-positive. Set
   `IsosterConfigMB.loose_validity=True` to relax this — see the
   "Loose validity (D9 backport)" subsection below for per-band-drop
   semantics, the `n_valid_<b>` column, and the optional
   `loose_validity_band_normalization` knob.

10. **Forced-mode `intens_err_<b>` (OLS) uses the population std.**
    `extract_forced_photometry_mb` returns `np.std(intens) / sqrt(n)`
    for the mean integrator and `sqrt(π/2) · np.std(intens) / sqrt(n)`
    for the median integrator (Gaussian-asymptotic median SEM),
    matching single-band `extract_forced_photometry`. Review fix H4
    originally used `ddof=1` here, leaving the two paths disagreeing by
    `sqrt(N/(N-1))` — about 0.8% at 64 samples. For `n < 2` the error
    is `NaN` rather than a misleading 0.0; single-band reports 0.0
    there, which reads as measured certainty rather than an absent
    measurement. WLS branch is unchanged — `1 / sqrt(Σ 1/var_i)` is the
    exact MLE error, and like single-band it reports the
    inverse-variance-weighted mean regardless of `integrator`.

11. **Central-pixel `intens_err_<b>` reads the variance map under
    WLS** (review fix H3). In OLS mode (no variance map provided) the
    central-pixel record still reports `intens_err_<b> = 0.0` because
    a single sample admits no internal error estimate. `rms_<b>`
    stays `0.0` in both modes.

12. **`fit_image_multiband` does NOT mutate the user's
    `IsosterConfigMB`** (review fix H1). Internal resolution like the
    WLS / OLS variance-mode tag goes onto a `model_copy` so the
    caller's instance is safe to reuse across runs.

13. **`loose_validity=True` is supported but not yet the default.** Every
    band keeps its own surviving samples instead of the cross-band
    intersection. This matters more than it sounds: shared validity
    drops a sample from *every* band whenever *any* band rejects it, and
    on the HSC demo with its real per-band masks that discards 18-27% of
    usable samples routinely — and at one radius, where band g is fully
    masked, it discards **everything**, failing an isophote that r and i
    could each measure with 370 samples.

    Four defects were repaired in the 2026-08-15 maturity pass:

    - Forced photometry now honours the flag end to end (per-band
      sampling, clipping, statistics and jagged harmonics). An isophote
      fails only when *every* band is empty.
    - `intens_err_<b>` keeps its joint covariance when
      `fit_per_band_intens_jointly=True`; the direct ring SEM was
      understating by ~34% under incomplete angular coverage.
    - B=1 combined with `loose_validity=True` is now rejected at
      configuration time rather than failing every isophote mid-fit.
    - A test asserting that *all* locked rows share geometry was
      over-tight and has been corrected — see the note below.

    `compute_joint_gradient` now samples in the same validity mode as the
    solve, so the numerator and denominator of every geometry correction
    describe one sample set. It previously sampled shared-validity
    regardless, which let a single fully-masked band collapse the joint
    gradient to the "no usable ring" sentinel while other bands had full
    rings.

    **Not a defect:** the `lsb_auto_lock` trigger row carries its own
    geometry rather than the anchor's. It is fitted before the locked
    config exists, and `lsb_locked=True` there means "the lock committed
    here", not "this row is frozen"; the anchor is identified separately
    via `lsb_auto_lock_anchor_sma`. Single-band behaves identically. Only
    the rows with `lsb_auto_lock_anchor=False` are frozen, and those do
    share the anchor's geometry exactly in both validity modes.

## What it does

`isoster.multiband.fit_image_multiband` fits elliptical isophotes
simultaneously on multiple aligned, same-pixel-grid images of the same
target (e.g. HSC g/r/i/z/y coadds). It produces a **single shared
geometry per SMA** with **per-band intensities and per-band harmonic
deviations**. This replaces the traditional forced-photometry workflow
("fit one band, apply the geometry to others") with a joint fit where
every band contributes to the geometry.

The joint design matrix per ellipse, B bands, N kept samples:

```
[ 1_g 0    0    sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_g]    [intens_g]
[ 0   1_r  0    sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_r]  = [intens_r]
[ 0   0    1_i  sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_i]    [intens_i]
                                              [A1   ]
                                              [B1   ]
                                              [A2   ]
                                              [B2   ]
```

Free parameters: `(5 + B)` per ellipse (per-band background `I0_b` plus
shared geometric harmonic coefficients). Solved once per iteration in
WLS or OLS mode. Per-band weights `w_b` enter the joint normal
equations as a diagonal weight matrix: each band's row block
contributes `Aᵀ W A` with `W = diag(w_b)` in OLS, or
`W = diag(w_b / variance_<b>(pixel))` in WLS.

### Per-band intercept mode

The ``IsosterConfigMB.fit_per_band_intens_jointly`` boolean controls
how each band's per-isophote intercept ``I0_b`` (reported as
``intens_<b>``) is computed. The default ``True`` is the **coupled /
joint** mode described above: ``B`` per-band intercept columns sit in
the ``(B*N, B + 4)`` joint design matrix, so the per-band intercept
and the shared harmonic deformations are co-fit. On full rings the
sin/cos columns are mathematically orthogonal to the constant column
over ``[0, 2π]``, so the joint coefficient is numerically equivalent
to the per-band inverse-variance-weighted ring mean. On partial rings
(masking, loose validity), orthogonality breaks and the joint solve
absorbs the harmonic-shape coupling between bands when reporting each
band's intercept — this is where the joint mode adds information
across bands relative to a per-band ring statistic.

Setting ``fit_per_band_intens_jointly=False`` switches to a
**decoupled / ring-mean** mode: the leading ``B`` per-band intercept
columns are dropped from the design matrix; the solve becomes a
4-column shared ``(A1, B1, A2, B2)`` system. Per-band ``intens_<b>``
is then reported as the band's ring-mean intensity (IVW under WLS,
simple mean under OLS), independent of the harmonic fit, and
``intens_err_<b>`` is the band's own SEM. Use this when you want the
per-band intercept to be a clean ring statistic decoupled from the
harmonic block — for example, when you have already done a high-
quality sky subtraction upstream. Mutually exclusive with
``harmonic_combination='ref'`` (ref-mode bypasses the joint solve).

> **Integrator scope (Stage-3, plan section 7 S1–S2):** the
> ``integrator`` config field affects ``intens_<b>`` reporting at
> converged isophotes only in the **decoupled** intercept mode.
> ``integrator='mean'`` is unconditionally legal — both intercept
> modes give a (weighted) ring mean, numerically identical on full
> rings since ``sin(nφ)``/``cos(nφ)`` are orthogonal to the constant
> column. ``integrator='median'`` requires
> ``fit_per_band_intens_jointly=False``: the matrix-mode joint LS
> solve cannot host a median (non-linear), so the validator hard-
> errors on ``integrator='median'`` ∧
> ``fit_per_band_intens_jointly=True`` with a remediation hint.
> Under ``integrator='median'`` the decoupled path replaces the
> per-band ring mean with a per-band ``np.median`` of the surviving
> ring samples (sample sigma-clipping has already been applied
> upstream by the sclip/nclip pipeline). Use this when partial-ring
> contaminants would pull a ring mean (e.g. faint companion sectors,
> diffraction spikes that the mask did not catch). The integrator
> field still also drives (a) per-band gradient computation and
> (b) the forced-photometry fallback for the central pixel.

Renamed in Section 6 cleanup from the deprecated
``fix_per_band_background_to_zero`` (with inverted polarity:
``fix_per_band_background_to_zero=True`` ↔
``fit_per_band_intens_jointly=False``). The old field name is
rejected at config construction with a clear error pointing at the
new name; FITS files written by the new code carry only the new
name. The multi-band path is pre-production so we take a clean
break with no auto-translation.

## Performance

On the asteris denoised dataset (768×768 cutouts, 74 isophotes, all five
HSC bands), the joint multi-band fit runs in **~2× single-band wall
time end-to-end** (0.49 s for B=5 vs 0.25 s for single-band on i-band,
including FITS I/O). This is well within the Stage-1 quality bar of
≤2.5× and reflects two key optimizations: (1) the
``(B·N × (B + 4))`` joint design matrix builder is numba-accelerated
with a NumPy fallback (``isoster/multiband/numba_kernels_mb.py``);
(2) the driver pre-resolves image / mask / variance arrays once per
fit and threads them through every per-iteration sampler call instead
of re-allocating per call. See decision D19 in the plan doc.

## Public API

```python
from isoster.multiband import fit_image_multiband, IsosterConfigMB

config = IsosterConfigMB(
    bands=["g", "r", "i", "z", "y"],
    reference_band="i",
    band_weights={"g": 1.0, "r": 1.0, "i": 1.0, "z": 1.0, "y": 1.0},
    harmonic_combination="joint",  # or "ref" for reference-band fallback
    sma0=10.0, maxsma=384.0, astep=0.1,
    debug=True, compute_deviations=True,
)
result = fit_image_multiband(
    images=[g_image, r_image, i_image, z_image, y_image],
    masks=object_mask,                    # single ndarray broadcast or list per band
    variance_maps=[g_var, r_var, ...],    # all-or-nothing
    config=config,
)
```

### Command-line interface (Stage-J, ``isoster-mb``)

The multi-band path ships a **separate** console script ``isoster-mb``
(entry point ``isoster.multiband.cli_mb:main``). This is intentionally a
parallel CLI to the stable single-band ``isoster`` script: while the
multi-band path remains experimental the two CLIs do **not** share
implementation, so multi-band-specific fixes cannot regress the
single-band entry point. Argument layout mirrors single-band for user
familiarity (``--config``, ``--output``, ``--x0``, ``--y0``, ``--sma0``,
``--fix-center``, ``--fix-eps``, ``--fix-pa``, ``--template``) but
takes one positional FITS path per band plus a ``--bands`` flag.

The CLI prints an ``EXPERIMENTAL`` banner per invocation while in beta;
suppress with ``--quiet``.

```bash
# Joint multi-band fit (FITS output uses Schema-1 multi-HDU writer).
isoster-mb image_g.fits image_r.fits image_i.fits \
    --bands g r i --reference-band i \
    --config config.yaml \
    --output isophotes_mb.fits

# Forced-photometry workflow: fit a deep band first, reuse its geometry
# across every band in one call (more efficient than running single-band
# B times — see Stage-H above).
isoster image_i.fits --output template_i.fits --config single_band.yaml
isoster-mb image_g.fits image_r.fits image_i.fits image_z.fits image_y.fits \
    --bands g r i z y --reference-band i \
    --template template_i.fits \
    --output isophotes_mb_forced.fits

# Per-band masks / variance maps + WLS:
isoster-mb image_g.fits image_r.fits \
    --bands g r --reference-band g \
    --masks mask_g.fits mask_r.fits \
    --variance-maps var_g.fits var_r.fits \
    --output isophotes_mb.fits
```

The ``--config`` YAML accepts any field of
:class:`isoster.multiband.IsosterConfigMB`; CLI flags layer on top.
``bands`` and ``reference_band`` may be set in YAML in lieu of
``--bands`` / ``--reference-band``. Output extension drives the writer:
``.fits`` (Schema-1 multi-HDU), ``.asdf`` (Stage-I native tree), or
anything else (astropy ``Table.write`` of the per-isophote table).

## Input contract (placeholder)

- `images`: `list[ndarray]` of length B, all of shape `(H, W)`.
- `masks`: `None`, single `(H, W)` boolean ndarray (broadcast to all
  bands), or `list[ndarray | None]` of length B. `None` per band means
  "no bad pixels in that band."
- `variance_maps`: all-or-nothing. Either `None` (full OLS), a single
  `(H, W)` ndarray (broadcast), or a `list[ndarray]` of length B.
  Entries that are non-finite (NaN or infinite) or non-positive (zero or
  negative) carry no usable information about the noise, so they are
  marked `NaN` with a `RuntimeWarning` and the affected samples are
  dropped during sampling — the same policy single-band `fit_image`
  applies (see `docs/04-architecture.md`, section "Invalid-variance
  policy"). Under the default shared validity a dropped sample is
  dropped from every band, exactly as a masked pixel already is; set
  `loose_validity=True` for per-band treatment.

    This retires the earlier sentinel scheme, in which non-finite entries
    became `1e30` and non-positive entries were clamped to `1e-30`.
    Neither did what its comment claimed: `1e30` is finite and strictly
    positive, so the sampler's validity rule *kept* it, and `1e-30` gave
    one pixel near-infinite weight, collapsing every
    inverse-variance-weighted error by fourteen orders of magnitude.
- `bands`: list of strings, regex `^[A-Za-z][A-Za-z0-9_]*$`, no
  duplicates. Strings appear verbatim as column suffixes (`intens_g`,
  `intens_r`, ...).
- `reference_band`: string in `bands`. Used for diagnostics only; does
  not affect joint geometry.
- `band_weights`: dict (every band as key) or list (length B) of
  positive finite floats. Default uniform 1.0.

## Output schema (Schema 1)

`fit_image_multiband` returns a dict with `'isophotes'` (list of dicts,
one per SMA) and the multi-band-specific top-level keys:

```
result['bands']                       : list[str]
result['multiband']                   : True
result['harmonic_combination']        : 'joint' | 'ref'
result['reference_band']              : str
result['band_weights']                : dict[str, float]
result['variance_mode']               : 'wls' | 'ols'
result['fit_per_band_intens_jointly'] : bool
result['loose_validity']              : bool
result['multiband_higher_harmonics']  : 'independent' | 'shared' |
                                        'simultaneous_in_loop' |
                                        'simultaneous_original'
result['harmonic_orders']             : list[int]    # default [3, 4]
result['harmonics_shared']            : bool          # True iff non-'independent'
```

Optional, present only when the caller post-processed with
``subtract_outermost_sky_offset``:

```
result['sky_offsets']  : dict[str, float]   # per-band offset that was subtracted
```

Each isophote row carries shared columns and per-band-suffixed columns:

- Shared: `sma, x0, y0, eps, pa, x0_err, y0_err, eps_err, pa_err,
  stop_code, niter, rms, valid, use_eccentric_anomaly, ndata, nflag,
  tflux_e, tflux_c, npix_e, npix_c`. Under shared validity `ndata` is
  the per-ring kept-sample count and `nflag` the sigma-clipped count.
  Under loose validity the per-band kept sets differ, so both are sums
  over bands: `ndata` is the total kept samples used by the joint solve
  and `nflag` the total sigma-clipped samples (never negative).
- Per band `<b>`: `intens_<b>, intens_err_<b>, rms_<b>,
  n_valid_<b>` (the last is the per-band surviving-sample count after
  sigma clipping; under shared validity it equals the shared `ndata`).
- Per band per order `<n>`: `a<n>_<b>, b<n>_<b>, a<n>_err_<b>,
  b<n>_err_<b>` for each `n` in ``harmonic_orders`` (default `[3, 4]`,
  extensible to `[3, 4, 5, 6]` etc.). Under non-``'independent'``
  ``multiband_higher_harmonics`` modes these per-band columns all
  carry the **identical shared value** across bands at every isophote
  (the joint refit / in-loop solve produces one shared coefficient per
  order). Per-band Bender normalization at plotting time scales the
  shared raw value by ``-1/(sma·|dI/da_b|)``, which still produces
  band-distinct curves because per-band gradients differ.
- Debug-only per band: `grad_<b>, grad_error_<b>, grad_r_error_<b>`
  when `debug=True`.

The FITS writer (`isophote_results_mb_to_fits`) uses a 4-HDU layout
(`PrimaryHDU`, `ISOPHOTES`, `CONFIG`, `META`). The primary header
carries the multi-band identifiers (`MULTIBND`, `BANDS`, `REFBAND`,
`HARMCMB`, `VARMODE`). The CONFIG HDU mirrors the single-band layout:
the full `IsosterConfigMB` dump as JSON PARAM/VALUE rows keyed by
pydantic field name. The META HDU stores the remaining top-level
result keys (`forced_photometry_mode`, `template_n_isophotes`,
`sky_offsets`, `lsb_auto_lock*`, `first_isophote_*`) as JSON
PARAM/VALUE rows so they survive the round trip;
`isophote_results_mb_from_fits` merges them back into the result dict.

**ASDF I/O symmetry:** `isophote_results_mb_to_asdf(result, filename)` and
`isophote_results_mb_from_asdf(filename)` mirror the single-band ASDF
helpers and are the recommended interchange format for downstream
Python pipelines that already consume ASDF. The on-disk tree stores the
full result dict natively (no JSON workaround), preserving per-band
suffix columns (`intens_<b>`, `a3_<b>`, `cog_<b>`, …), the
`IsosterConfigMB` model dump, multi-band top-level keys, and optional
`lsb_auto_lock` / `lsb_locked` lock-state metadata. The reader
reconstructs an `IsosterConfigMB` (or returns `None` if the schema has
moved on, mirroring the FITS forward-compatibility contract). The
`asdf` package is an optional dependency; both helpers raise an
`ImportError` with install hint if it is unavailable.

**B=1 fallback:** when `len(bands) == 1`, `fit_image_multiband`
delegates to `fit_image` and returns the legacy single-band schema
unmodified, with an informational warning.

## Worked example

See `examples/example_asteris_denoised/run_isoster_multiband.py` for the
end-to-end Stage-1 demo: joint multi-band fit on the asteris denoised
HSC coadds of object 37484563299062823. The script loads all five HSC
bands of denoised cutouts, the existing object mask (built by
`build_object_mask.py` on the noisy cutout), and per-band uniform-
variance maps from the sigma-clipped sky RMS. It runs the joint
multi-band fit, writes a Schema-1 FITS result and the composite QA
PNG to `outputs/example_asteris_denoised/<id>/`, and prints a geometry
sanity check against the existing i-band single-band reference fit
(loaded from the same outputs directory).

Typical sanity-check output on object 37484563299062823:

```
geometry sanity vs i-band reference (median over valid rings):
  eps:  multi-band=0.123   i-band=0.118
  pa:   multi-band=139.65 deg   i-band=144.96 deg
```

The small offset between multi-band and i-band-only geometries is
expected: the joint fit pools harmonic-coefficient information across
all five bands, so the recovered `pa` is a band-weighted compromise
rather than the i-band-specific solution. The two are in family — a
real bias would manifest as a shift of several degrees or a larger
ellipticity discrepancy.

## Algorithm notes

### Joint design matrix (decision D2)

For one isophote with B bands and N kept samples, the joint solver
fits a single ``(B + 4)``-parameter least-squares system per iteration:

```
[ 1_g 0   0   sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_g]    [intens_g]
[ 0   1_r 0   sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_r]  = [intens_r]
[ 0   0   1_i sin(φ) cos(φ) sin(2φ) cos(2φ) ] [I0_i]    [intens_i]
                                            [A1   ]
                                            [B1   ]
                                            [A2   ]
                                            [B2   ]
```

Per-band weights `w_b` enter the joint normal equations as a diagonal
weight matrix `W` (each band's row block contributes `Aᵀ W A` with
`W = diag(w_b)` in OLS, or `W = diag(w_b / variance_<b>(pixel))` in
WLS). With B=1 and `w_b = 1` the joint solver reduces to the existing
single-band 5-parameter system bit-for-bit (verified by
`test_joint_solver_b1_matches_single_band_solver`).

### Combined gradient (decision D10)

The geometry-update math (Jedrzejewski 1987) requires a single radial
gradient. For multi-band the driver computes per-band gradients
separately and combines them with the same per-band weights:

```
gradient_joint = Σ_b w_b · grad_b / Σ_b w_b
σ²_joint       = Σ_b w_b² · σ_b² / (Σ_b w_b)²    (independent measurements)
```

Plugged into the standard geometry-update formulas; the gradient-error
gate (`maxgerr`) reads `σ_joint / |gradient_joint|`.

### Sample-validity rule (decision D9)

A sample on the ellipse is dropped from the joint solve if **any**
band's mask flags it, **any** band has NaN at that location, or
**any** band's variance is non-positive after sanitization. This
guarantees that every band's row block in the joint design matrix has
the same `N` samples, which the joint solve requires. Edge cases where
one bad band drops samples in all bands are a known revisit item.

### Sigma clipping (decision D9)

Each band is clipped independently against its own intensity
statistics; the surviving sample masks are AND-ed across bands and
applied uniformly. Reduces to single-band exactly when B=1.

### Loose validity (D9 backport)

When ``IsosterConfigMB.loose_validity=True`` (default ``False``), the
shared-validity AND is relaxed: each band keeps its own surviving
samples and the joint design matrix becomes block-diagonal in the
per-band intercept columns. The shared geometric coefficients
``(A1, B1, A2, B2)`` are still constrained jointly. A band that falls
below ``loose_validity_min_per_band_count`` (absolute count, default 6)
or ``loose_validity_min_per_band_frac`` (fraction of attempted samples,
default 0.2) at a given isophote is dropped from the joint solve at
that isophote — its ``intens_<b>``, ``intens_err_<b>``, and harmonic
columns are set to NaN, and the surviving bands still constrain the
shared geometry. The whole isophote is marked ``stop_code=3`` only
when fewer than 2 bands survive.

A new per-isophote column ``n_valid_<b>`` reports each band's actual
surviving-sample count (after the per-band sigma clip). The QA figure
auto-adds a small panel showing ``n_valid_<b> / n_attempted`` per band
when ``loose_validity=True``.

The optional knob ``loose_validity_band_normalization`` controls how
per-band sample counts feed the joint solve and the combined gradient:

- ``"none"`` (default): each band's row block contributes
  proportionally to its own ``N_b``; ``w_b`` multiplies every row;
  combined gradient is ``Σ w_b · grad_b / Σ w_b`` over surviving
  bands. Bands with more kept samples dominate.
- ``"per_band_count"``: each band's row block is renormalized by
  ``√(1/N_b)`` so its total contribution equals ``w_b`` regardless of
  ``N_b``; combined gradient is weighted by ``(w_b · N_b)``. Restores
  the user's "this band matters this much" mental model when
  per-band masks differ. Requires ``loose_validity=True``.

Loose validity composes cleanly with both ``harmonic_combination='ref'``
and ``fit_per_band_intens_jointly=False``.

### Higher-order harmonics modes (Section 6, locked 2026-05-02)

The ``IsosterConfigMB.multiband_higher_harmonics`` enum controls how
higher-order (n ≥ 3) harmonic deviations ``(A_n, B_n)`` are fit. The
default ``'independent'`` reproduces Stage-1 behavior bit-for-bit; the
other three values share higher-order coefficients across bands in
different ways. ``IsosterConfigMB.harmonic_orders`` (default ``[3, 4]``,
extensible to ``[3, 4, 5, 6]`` etc.) selects which orders are included
in any of the three non-``'independent'`` modes.

| Value | Where higher orders are solved | Cross-band coupling | Refits |
|-------|-------------------------------|---------------------|--------|
| ``independent`` (default) | Post-hoc, per band, per order via the single-band ``compute_deviations`` solver | None — each band independent | a3_b, b3_b, a4_b, b4_b ... |
| ``shared`` | Post-hoc joint solve over residuals | Shared (A_n, B_n) for n ≥ 3 across bands | (A_n, B_n) for n ≥ 3 only |
| ``simultaneous_in_loop`` | In every iteration's joint solve | Shared (A_n, B_n) for ALL n | (A1, B1, A2, B2, A_n, B_n) every iteration |
| ``simultaneous_original`` | 5-param in loop; one wider post-hoc joint solve over all orders | Shared (A_n, B_n) for ALL n | One post-hoc solve refits all orders |

#### `independent` (default)

Stage-1 behavior: at convergence, the driver calls
``_attach_per_band_harmonics`` which runs an independent
``compute_deviations`` per band per order. Each band's a_n / b_n
reflects its own ring's intensity, gradient, and noise. Per-band
``a_n_<b>`` and ``b_n_<b>`` typically differ across bands.

#### `shared` (NEW DEVELOPMENT)

Replaces the per-band post-hoc step with **one joint refit** of
higher-order coefficients shared across bands. The geometry-loop
values for ``(A1, B1, A2, B2)`` and per-band ``I0_b`` are frozen.
Design matrix is the smallest possible: ``(B·N, 2·L)`` where
``L = len(harmonic_orders)``. Per-band ``a_n_<b>`` columns all carry
the same shared value at every isophote. Errors come from the joint
covariance; per-band ``a_n_err_<b>`` is the same across bands too.

Astrophysical motivation: a multi-band joint fit is meant to produce
a single 1-D profile of the galaxy across bands; if the higher-order
geometric deviation genuinely differed band-to-band you should be
running ``harmonic_combination='ref'`` plus forced photometry instead.

#### `simultaneous_in_loop` and `simultaneous_original` (RECOVERED FEATURE)

Multi-band lifts of the single-band ``IsosterConfig.simultaneous_harmonics``
/ ``isofit_mode`` features. Both extend the joint design matrix to
``(B·N, B + 4 + 2·L)`` with shared higher-order columns; the
difference is when the wider solve fires:

- ``simultaneous_in_loop``: every iteration's joint solve uses the
  wider matrix. Geometry update reads ``coeffs[B..B+3]`` exactly as
  before — coefficient layout is unchanged at those indices.
- ``simultaneous_original``: standard 5-param iteration loop, then ONE
  wider joint solve over all orders runs after convergence (Ciambur
  2015 original variant).

Both modes emit a ``UserWarning`` at config construction citing
single-band benchmark concerns; validate carefully on PGC006669 and
asteris before trusting.

#### Validators

- ``shared`` / ``simultaneous_*`` × ``harmonic_combination='ref'``:
  hard-error. Ref-mode bypasses the joint solver entirely.
- ``shared`` / ``simultaneous_*`` × ``fit_per_band_intens_jointly=False``:
  silently allowed. Compose cleanly with the ring-mean intercept mode.
- ``shared`` / ``simultaneous_*`` × ``loose_validity=True``: silently
  allowed. The jagged builders ``build_joint_design_matrix_jagged`` and
  ``build_joint_design_matrix_jagged_higher`` carry the higher-order
  columns through the loose-validity path.

#### Performance bar (Stage-1 D17 / Section 6.1 Q11)

All four modes must stay within ≤ 2.5× single-band wall time on the
asteris perf benchmark. Measured on the B=5 768² cutout (2026-05-03):

| Configuration | Median (s) | Ratio vs SB |
|---|---:|---:|
| singleband (i-band) | 0.209 | 1.00× |
| multiband independent | 0.293 | 1.41× |
| multiband shared | 0.279 | 1.34× |
| multiband simultaneous_in_loop | 0.317 | 1.52× |
| multiband simultaneous_original | 0.272 | **1.30×** |

All four PASS. ``simultaneous_original`` is the cheapest non-default
mode because one post-hoc joint solve replaces ``B × L`` separate
``compute_deviations`` calls.

### Per-band intercept mode revisited (Section 6.12)

The boolean ``IsosterConfigMB.fit_per_band_intens_jointly`` (default
``True``, renamed from the deprecated
``fix_per_band_background_to_zero=False`` in the Section 6.12 cleanup)
governs how each isophote's per-band intercept ``I0_b`` is computed.
See the [Per-band intercept mode](#per-band-intercept-mode) section
above for the full mechanism description.

The two modes are **numerically equivalent on full rings** because
sin(nφ) and cos(nφ) are orthogonal to the constant column over [0, 2π].
They diverge only on partial rings (loose validity, masked sectors),
where the joint solve absorbs harmonic-shape coupling between bands.
This is when the joint mode adds information across bands relative to
a per-band ring statistic.

Sky correction is **not** part of the fit. The optional post-process
``subtract_outermost_sky_offset(result, n_outer=N)`` lives in
``isoster.multiband.plotting_mb`` and operates only on already-fit
results; it never modifies the fit. The library never invokes it
implicitly. Demo scripts call it explicitly between FITS write and
QA plot generation; raw FITS files always preserve the unmodified
``intens_<b>``.

### Outer-region regularization — damping mode (Stage-3 Stage-B)

Backport of the single-band ``outer_reg_*`` family in **damping**
mode. Use ``use_outer_center_regularization=True`` to enable a soft
Tikhonov-style step shrink in the LSB regime: per-iteration geometry
updates ``δ_param`` are multiplied by ``(1 - α(sma))``, where ``α``
ramps from 0 below ``outer_reg_sma_onset`` to a saturation set by
``outer_reg_strength``. Saturated clipped jumps in PA, ellipticity,
and center are suppressed; the fit still walks to its data-preferred
geometry — there is no pull toward the reference. The full Tikhonov
solver mode (with reference pull) lands in Stage E.

Six fields, mirroring single-band semantics:

| Field | Default | Notes |
|---|---|---|
| ``use_outer_center_regularization`` | ``False`` | Master toggle. |
| ``outer_reg_sma_onset`` | ``50.0`` | Sigmoid midpoint (pixels). |
| ``outer_reg_strength`` | ``2.0`` | Saturation amplitude of the ramp. |
| ``outer_reg_weights`` | ``{center: 1, eps: 0, pa: 0}`` | Per-axis weights; the multi-band default ``{1,0,0}`` damps the center only — the joint solve already stabilizes eps / pa across bands (see the callout below). |
| ``outer_reg_sma_width`` | ``None`` (auto = ``0.4 * onset``) | Sigmoid slope width. |
| ``outer_reg_ref_sma_factor`` | ``2.0`` | Inward isophotes within ``sma0 × factor`` feed the flux-weighted reference. |

The ``outer_reg_mode`` field is currently a one-value
``Literal["damping"]``. Stage E will widen it to
``Literal["damping", "solver"]`` and add the full Tikhonov
ref-pull. Users who set ``outer_reg_mode='solver'`` today get a
clear pydantic Literal validation error.

**Auto-enable.** With ``use_outer_center_regularization=True`` and
default ``geometry_convergence=False``, the validator emits a
``UserWarning`` and flips ``geometry_convergence=True`` automatically.
Without geometry convergence the fit runs to ``maxit`` on outer
isophotes with no change to recorded geometry — pure cost.
Suppress the warning by setting ``geometry_convergence=True``
explicitly.

**Sanity warnings.** The validator also emits warnings for:

- ``outer_reg_sma_onset < sma0`` (penalty fires from the first
  outward step; typical onset ~ ``sma0 * 3``).
- ``minsma >= sma0`` (no inward isophotes → reference falls back to
  the anchor's geometry).
- All ``outer_reg_weights`` zero (feature is inert).
- ``fix_<axis>=True`` with ``outer_reg_weights[<axis>]>0`` (axis is
  frozen so the penalty is identically zero).

**Reference geometry.** Built once per fit by
``_build_outer_reference_mb`` in the driver. Flux-weighted mean
(weights = ``intens_<reference_band>``) over inward isophotes with
``sma <= sma0 * outer_reg_ref_sma_factor`` plus the anchor; ``pa``
uses a circular mean on ``2*pa`` so axis-like angles around 0 and π
average sensibly. Falls back to the anchor's geometry if the inward
sweep is empty or all candidates fail finiteness checks.

**Asteris benchmark.**
``benchmarks/multiband/bench_outer_reg_damping_asteris.py`` compares
three configs (no outer-reg, center-only damper, all-axes damper) on
HSC g/r/i/z/y. Headline results (n_repeats=3, B=5, 768²):

| configuration | median (s) | ratio vs SB | outer eps MAD | outer pa MAD |
|---|---:|---:|---:|---:|
| singleband (i-band) | 0.333 | 1.00× | — | — |
| baseline (no outer-reg) | 0.425 | 1.28× | 1.17e-01 | 9.19e-02 |
| damping center-only | 0.436 | 1.31× | 5.39e-02 | 0.00 |
| damping all axes (default) | 0.388 | **1.16×** | 9.93e-06 | 0.00 |

All PASS the ≤ 2.5× single-band bar. Damping-vs-baseline overhead
is **0.91×** — the all-axes damper is faster than baseline because
the auto-enabled ``geometry_convergence`` lets damped fits exit
earlier than the harmonic-only criterion does in the LSB regime.

> **Multi-band default is `{1, 0, 0}` (center-only) — different from
> single-band's `{1, 1, 1}`.** The Stage-B follow-up strength ×
> weights sweep on PGC006669 (2026-05-04) showed that the per-axis
> Tikhonov blend factor ``α = λ·w·coeff² / (1 + λ·w·coeff²)``
> saturates above 99.99% for any positive eps / pa weight in the
> user-intuition range (0.25–1.0): in the outer LSB the per-axis
> Jacobian ``coeff = (1-eps)/grad`` reaches ~ 700 because the joint
> gradient ``→ 0``, so ``λ·w·coeff² ≫ 1`` and α is in the
> saturation regime regardless of `w`. **The eps / pa weight choice
> is therefore binary, not graded:** any positive value pins
> geometry to the inner reference; zero leaves it free.
>
> ``{1, 0, 0}`` is the safe default for general multi-band targets
> (barred / spiral systems, BCG → ICL transitions, anywhere outer-
> disc geometry might genuinely evolve). The multi-band joint
> solver already provides cross-band stability on eps / pa via the
> shared ``(A1, B1, A2, B2)`` harmonic constraint, so eps / pa
> damping is largely redundant here.
>
> ``{1, 1, 1}`` is an **advanced opt-in** for confirmed-static
> outer envelopes (cD galaxies, isolated massive ellipticals).
> The asteris benchmark — a massive elliptical with extended LSB
> envelope — is exactly that regime; on asteris ``{1, 1, 1}``
> tightens the recorded outer geometry without harm.
>
> Down-weighted variants like ``{1, 0.5, 0.5}`` or
> ``{1, 0.25, 0.25}`` are **empirically NOT a useful middle
> ground**: the PGC sweep showed PA bias ≈ +0.71 rad for all four
> down-weighted configs (16× spread in `w·strength`), versus
> -0.24 rad for center-only. Avoid.
>
> See ``outputs/benchmark_multiband/outer_reg_strength_sweep_pgc/
> alpha_saturation_diagnostic.png`` for the closed-form α-vs-weight
> curves at four ``λ·coeff²`` regimes, and the journal
> ``docs/agent/journal/2026-05-04_stage_b_followup_strength_sweep.md``
> for the full bias / MAD / convergence breakdown.

### LSB auto-lock (Stage-3 Stage-C)

Backport of the single-band ``lsb_auto_lock`` family. Outward growth
starts in free-geometry mode; once the **joint combined gradient**
(plan section 7 S3) degrades — a relative joint gradient error
above ``lsb_auto_lock_maxgerr`` for ``lsb_auto_lock_debounce``
consecutive isophotes, OR a ``stop_code=-1``, OR a non-negative
joint gradient — the lock commits. Remaining outward isophotes
inherit the **shared geometry** of the anchor (the isophote
immediately *before* the streak) and switch to the configured
``lsb_auto_lock_integrator``. Inward growth and the central pixel
are untouched.

Four fields:

| Field | Default | Notes |
|---|---|---|
| ``lsb_auto_lock`` | ``False`` | Master toggle. |
| ``lsb_auto_lock_maxgerr`` | ``0.3`` | Trigger threshold on \|grad_err / grad\| (joint scalars). Stricter than the free-fit ``maxgerr=0.5`` so the lock fires before a stop_code=-1 would have. |
| ``lsb_auto_lock_debounce`` | ``2`` | Consecutive triggered isophotes before commit. |
| ``lsb_auto_lock_integrator`` | ``'median'`` | Integrator used in the locked region. |

**Hard-error validators** (plan section 7.5 items 2–3):

1. ``lsb_auto_lock=True`` ∧ any of ``fix_center`` / ``fix_pa`` /
   ``fix_eps`` ⇒ ``ValueError``. The lock requires free geometry on
   the outward sweep (mirrors single-band).
2. ``lsb_auto_lock_integrator='median'`` ∧
   ``fit_per_band_intens_jointly=True`` ⇒ ``ValueError``. The lock-
   fire path would clone the cfg with ``integrator='median'`` on a
   matrix-mode joint LS, which Stage-A's S1 rejects. **Default
   ``lsb_auto_lock_integrator='median'`` therefore requires the
   user to also set ``fit_per_band_intens_jointly=False`` when
   enabling the lock.** To use the matrix-mode joint solve with the
   lock, set ``lsb_auto_lock_integrator='mean'``.

**Soft warning + auto-enable.** ``lsb_auto_lock=True`` with
``debug=False`` (the default) emits a ``UserWarning`` and flips
``debug=True`` internally — the lock trigger reads top-level
``grad`` / ``grad_error`` row keys that the fitter only writes
under debug.

**Top-level result keys** (when ``lsb_auto_lock=True``):

- ``result['lsb_auto_lock']`` — ``True``.
- ``result['lsb_auto_lock_sma']`` — SMA of the trigger isophote
  (``None`` if the lock never committed).
- ``result['lsb_auto_lock_anchor_sma']`` — SMA of the true geometry
  anchor (the last clean isophote before the debounce streak, whose
  geometry the lock froze to); ``None`` if the lock never committed.
- ``result['lsb_auto_lock_count']`` — number of isophotes carrying
  ``lsb_locked=True`` in their row dict.

**Per-isophote keys** (present on every isophote when ``lsb_auto_lock=True``;
inward isophotes and the central pixel always carry ``False``, keeping the
schema uniform for FITS round-trips):

- ``iso['lsb_locked']`` — ``True`` for outward isophotes after the
  lock fires (the trigger isophote and all subsequent), ``False``
  for the anchor and earlier isophotes.
- ``iso['lsb_auto_lock_anchor']`` — ``True`` only on the trigger isophote
  itself, marks the SMA at which the lock committed.

**Asteris benchmark.**
``benchmarks/multiband/bench_lsb_auto_lock_asteris.py`` compares
three configs (no lock, lock-mean, lock-median) on HSC g/r/i/z/y
(n_repeats=3, B=5, 768², 250k masked):

| configuration | median (s) | ratio vs SB | lock fires | lock SMA | n_locked | outer pa MAD |
|---|---:|---:|:---:|---:|---:|---:|
| singleband (i-band) | 0.219 | 1.00× | — | — | — | — |
| baseline (no lock) | 0.299 | 1.37× | no | — | 0 | 9.19e-02 |
| lock-mean | 0.280 | **1.28×** | yes | 238.4 | 9 | 6.31e-02 |
| lock-median | 0.313 | 1.43× | yes | 238.4 | 9 | **1.01e-02** |

All PASS the ≤ 2.5× single-band bar. ``lock-mean`` is actually
*faster* than ``baseline`` because the locked-region isophotes skip
the iteration loop (frozen geometry → forced photometry style).
Outer-tail PA MAD drops from 9.19e-02 (baseline) → 1.01e-02
(lock-median): the lock pins outer geometry at the clean anchor,
exactly the LSB stabilization the feature is designed for.

> **Galaxy-type caveat (2026-05-04).** The lock is designed for
> **massive ellipticals / cD galaxies with extended LSB envelopes**
> — the regime where the inner geometry is the best estimate of the
> outer envelope's geometry and a hard freeze is the right LSB
> stabilizer. **Do not enable it as a default on arbitrary galaxy
> types.** On systems with genuine outer-disc geometry evolution
> (barred galaxies, S0 / spiral systems with a real bulge → disc
> transition, BCG → ICL transitions where the envelope rounds off),
> a hard lock pins the outer envelope to the inner geometry and
> obscures real structural change — the same failure mode the
> Stage-B outer-reg ``{1, 1, 1}`` caveat documents. For those
> targets, either leave the lock off, or raise
> ``lsb_auto_lock_maxgerr`` (e.g. to ``0.5``, matching the free-fit
> ``maxgerr``) so the lock delays firing until the joint gradient
> is genuinely lost. The aggregate scatter metrics (eps MAD ≈ 0,
> pa MAD ≈ 0) on locked isophotes look great in benchmarks but are
> the geometric signature of "geometry pinned at anchor," not
> "geometry tracked." Use the QA mosaics in
> ``outputs/benchmark_multiband/lsb_auto_lock_{asteris,pgc}/`` to
> read the trajectory, not just the scatter.

### Central-region geometry regularization (Stage-3 Stage-F)

Verbatim port of the single-band ``central_reg_*`` family. Geometry
is shared in multi-band, so the penalty is band-agnostic — no
per-band design choice. Set ``use_central_regularization=True`` to
add a Gaussian-decaying penalty to the best-iteration selector
(``effective_amp``):

```
λ(sma) = central_reg_strength · exp(-(sma / central_reg_sma_threshold)²)

penalty = λ · ( w_eps · Δeps²  +  w_pa · Δpa²  +  w_center · (Δx0² + Δy0²) )
```

where ``Δ*`` are the per-iteration changes from the previous
isophote's geometry. PA residual wraps onto ``[-π, π]`` (single-band
convention). The gate is ``λ < 1e-6`` (penalty becomes exactly
zero past ~3× the threshold SMA).

This is a **selector-layer** penalty — it does not modify the
harmonic step. Iterations whose geometry jumped far from the
previous isophote at low SMA look worse to the selector and are
not chosen as the new ``best_geometry``. Mechanically distinct from
``outer_reg_*`` damping (which step-shrinks the harmonic update)
and from ``lsb_auto_lock`` (which hard-freezes geometry).

Four fields:

| Field | Default | Notes |
|---|---|---|
| ``use_central_regularization`` | ``False`` | Master toggle. |
| ``central_reg_sma_threshold`` | ``5.0`` | Gaussian decay scale (px). Penalty essentially zero beyond ~3× threshold. |
| ``central_reg_strength`` | ``1.0`` | Maximum strength at SMA=0. ``ge=0`` (zero is legal — feature on but inert). |
| ``central_reg_weights`` | ``{eps: 1, pa: 1, center: 1}`` | Per-axis weights. Unknown keys are rejected (validated unconditionally, even when the feature is off, to catch typos). |

Use case: stabilize fits in low-S/N central regions where the
harmonic amplitudes are noise-dominated and small geometry jumps
get amplified by ``coeff = (1-ε)/grad`` etc. The single-band
benchmark suite uses this on dim nuclei. Geometry is shared in
multi-band, so the penalty composes cleanly with all other
features (``loose_validity``, ``multiband_higher_harmonics``,
``outer_reg_*``, ``lsb_auto_lock``) without interaction validators.

### Forced-photometry mode (Stage-3 Stage-H)

Pass ``template_isophotes=...`` to ``fit_image_multiband`` to bypass
the iteration loop entirely and run forced multi-band extraction at
each template row's exact ``(sma, x0, y0, eps, pa)`` geometry.
Common workflow:

```python
from isoster import IsosterConfig, fit_image
from isoster.multiband import IsosterConfigMB, fit_image_multiband

# 1. Fit single-band on a deep band first.
sb_cfg = IsosterConfig(sma0=10.0, maxsma=384.0, astep=0.1, debug=True)
sb_result = fit_image(i_image, mask=mask, config=sb_cfg)

# 2. Pass the single-band result as a multi-band template.
mb_cfg = IsosterConfigMB(
    bands=["g", "r", "i", "z", "y"], reference_band="i",
    compute_cog=True,
)
mb_result = fit_image_multiband(
    [g_image, r_image, i_image, z_image, y_image],
    masks=mask, config=mb_cfg,
    template_isophotes=sb_result,
    variance_maps=[g_var, r_var, i_var, z_var, y_var],
)
```

**Use case:** the user wants per-band HSC profiles but is not yet
ready to trust the joint multi-band fit, OR specifically wants the
i-band geometry to drive everything. Forced mode runs the per-band
sampler once across the shared geometry and emits the standard
multi-band result-dict shape — more efficient than running
single-band ``B`` times since the geometry pre-resolution and per-
band sampling work happens in one pass.

**Distinct from `harmonic_combination='ref'`.** Ref-mode still runs
the iteration loop and lets the geometry walk under the reference
band's harmonic constraint. Forced mode bypasses the iteration loop
entirely — geometry is bit-identical to the input template. The
distinction matters when users invoke forced mode specifically
because they want guaranteed-pinned geometry from a separate fit.

**Accepted template-input forms:**

- A path to a Schema-1 multi-band FITS file (e.g. an earlier
  ``fit_image_multiband`` run saved via
  ``isophote_results_mb_to_fits``).
- A path to a single-band FITS file (single-band's
  ``isophote_results_to_fits`` output). Single-band templates work
  because Stage-H only needs the geometry columns.
- A multi-band result dict (``fit_image_multiband`` return value).
- A single-band result dict (``fit_image`` return value).
- A list of dicts with ``sma`` / ``x0`` / ``y0`` / ``eps`` / ``pa``
  keys.

The list is sorted by ``sma`` ascending before extraction. ``sma=0``
rows dispatch to ``_fit_central_pixel_mb`` (per-band central
record); other rows dispatch to ``extract_forced_photometry_mb``
(per-band ring extraction).

**Validators (warn-and-ignore).** When ``template_isophotes`` is
provided, the iteration-loop-only features are no-ops and emit a
single ``UserWarning`` listing what was ignored:

- ``lsb_auto_lock=True``
- ``use_outer_center_regularization=True``
- ``use_central_regularization=True``
- ``harmonic_combination='ref'``

These features all live inside the iteration loop; forced mode
skips that loop. ``compute_cog=True`` is fully legal — in fact the
primary use case on top of forced extraction (per-band CoG against
fixed geometry).

**Output schema** matches the standard multi-band schema with all
geometry columns carrying the template values exactly. Two extra
top-level keys signal the workflow:

- ``result['forced_photometry_mode']`` — ``True``.
- ``result['template_n_isophotes']`` — count of input rows.

Per-band ``intens_<b>`` / ``intens_err_<b>`` come from
``extract_forced_photometry_mb`` per template SMA.

**Per-band harmonic deviations are computed (Stage-H.1, 2026-05-04).**
When ``compute_deviations=True`` (the default), the orchestrator
runs a second pass after the per-row sampling to fill the per-band
harmonic columns ``a<n>_<b>`` / ``b<n>_<b>`` (and their error
variants) for every order in ``harmonic_orders``. The per-band
gradient required by the Bender normalization comes from
``np.gradient(intens_<b>_col, sma_col)`` over the full template —
neighbor-derived per-band gradients, the same fallback the audit
pipeline uses on tools that ship no usable ``grad`` column. Ring
data is captured during the first pass via the helper's
``return_ring_data=True`` flag, so harmonic computation does not
re-sample the rings. Set ``compute_deviations=False`` to skip the
harmonic-fill pass and keep the columns zero (the original
Stage-H behavior, matching single-band's forced-photometry
convention).

### Per-band curve-of-growth (Stage-3 Stage-D)

Set ``compute_cog=True`` to enable per-band cumulative-flux
photometry. The driver runs once over the assembled isophote list
(ascending sma) after the inward + outward sweeps complete and
stamps the following columns onto each row dict — and thereby into
the Schema-1 ``ISOPHOTES`` table, since
``Table(rows=isophotes)`` auto-infers columns from the row dicts:

| Column | Per-band? | Description |
|---|:---:|---|
| ``cog_<b>`` | yes | Cumulative flux through this isophote in band ``b`` (sum of annular fluxes from the central pixel out). |
| ``cog_annulus_<b>`` | yes | Annular flux at this isophote in band ``b`` (``intens_avg_<b> × area_annulus``). |
| ``area_annulus`` | no | Annular area (px²); shared because geometry is shared. Negative-annulus entries (isophote crossing) are zero-clamped. |
| ``flag_cross`` | no | Boolean; True where the geometry indicates a potential ellipse crossing. |
| ``flag_negative_area`` | no | Boolean; True where the raw annular area went negative before clamping. |

The shared columns mirror single-band semantics. Per-band columns
are unique to multi-band and use each band's ``intens_<b>``
trapezoidal-averaged across the inner / outer ring of the annulus
(the first annulus uses the innermost isophote's ``intens_<b>``
alone). NaN ``intens_<b>`` (e.g. a band dropped under
``loose_validity=True`` at a given isophote) propagates into
``cog_annulus_<b>`` / ``cog_<b>`` for that row — by design, so
downstream consumers can detect band-drops.

The B=1 multi-band CoG is bit-identical to the single-band CoG when
``intens_<b>`` matches the single-band ``intens`` (verified in
``tests/multiband/test_fitting_mb.py:test_compute_cog_mb_b1_matches_single_band_per_band_column``).

Schema-1 stays additive: when ``compute_cog=False`` (the default)
the new columns do not appear at all, and existing FITS files
round-trip unchanged. The plan-section 7 S7 decision (per-band
columns in the main per-isophote table, NOT a separate HDU) means
``compute_cog=True`` adds ``B × 2 + 3`` extra columns (e.g. 13
columns for B=5 HSC). The Schema-2 motivation if this becomes too
heavy is recorded in the plan doc.

## Uncertainty parity with single-band

Multi-band and single-band now follow the same four rules for building an
uncertainty. Each had drifted independently, and every drift reported a number
that belonged to a different estimator than the one printed beside it.

**1. An error belongs to the statistic actually reported.** Multi-band used the
variance of an inverse-variance-weighted mean everywhere, regardless of what was
reported. Under `integrator='median'` the reported value is a plain median, so
its uncertainty needs the median's own variance — the heteroscedastic form
`π·N / (2·(Σ 1/√vᵢ)²)` under WLS, or `(π/2)·σ²/N` under OLS. The mismatch was
not a subtle heteroscedastic effect: with a *uniform* variance map the retired
value was wrong by exactly `√(π/2) = 1.253`, because the median's penalty was
simply absent. This applies both to the per-band intensity error
(`_per_band_intercept_variance`, used by the four decoupled solvers and by
`intens_err_<b>`) and to the per-band gradient error (`compute_joint_gradient`,
which now calls the shared `_ring_statistic_and_variance`).

Measured on a Sérsic ring at `sma=20` with the noted variance map:

| variance map | integrator | quantity | new / retired |
|---|---|---|---|
| smooth ramp | mean | gradient error | 1.012 |
| smooth ramp | median | gradient error | 1.257 |
| azimuthal split | mean | gradient error | 5.007 |
| azimuthal split | median | gradient error | 1.634 |
| uniform | median | `intens_err_<b>` | 1.253 |
| smooth ramp | median | `intens_err_<b>` | 1.257 |
| azimuthal split | median | `intens_err_<b>` | 1.634 |

**2. The OLS scatter term uses residuals, not raw ring scatter.** A ring's real
m=1 / m=2 signal is deterministic structure the model fits, and it averages out
of the intercept, so leaving it in the scatter overstates the error. Single-band
uses `np.std(intens - model)`; the four decoupled solvers now evaluate their
fitted geometric block and score the intercept on the residuals the same way.

**3. The residual variance measures the complete fitted model.** Under
`simultaneous_in_loop` and `simultaneous_original` the solve carries shared
higher-order columns, and `_compute_joint_residual_variance` used to evaluate a
truncated model that omitted them — charging the fitted m=n structure to the
noise. On a noiseless synthetic built from its own coefficients, where the
residual variance must be exactly zero, it returned **30.5**; it now returns
`0.0`. The geometry path also counted `n_bands + 4` parameters while the
harmonic attacher counted `n_bands + 4 + 2L`, so one solve produced two
different error scales. Both now pass `harmonic_orders` and the full parameter
count. This is the multi-band port of the single-band truncated-model fix in
`fit_isophote`.

**4. One solve yields one error scale.** The `sigma_bg**2` floor is applied
inside `_compute_joint_residual_variance`, so every consumer of *that solve's
residual variance* shares it. Previously only the geometry errors carried it. An
exactly-determined or under-determined solve now returns `0.0` rather than
`None`: the model passes through every point, so it carries no information about
the noise. `None` used to make the harmonic attachers skip the OLS rescale and
publish a raw `(AᵀA)⁻¹` diagonal, which is an arbitrary scale, not an error.

The rule that decides which errors take the floor is about **what kind of
quantity each one is**, not which mode is active:

> A *fitted parameter* is scaled by its own solve's residual variance, floor
> included. A *plain ring statistic* carries that statistic's own variance and
> is never floored.

Measured by scaling `sigma_bg` from 5 to 50 (a 10× change in the error floor):

| error | coupled joint | decoupled | reference mode |
|---|---|---|---|
| `eps_err` (and the other geometry errors) | 10.0× | 10.0× | 10.0× |
| `intens_err_<b>`, reference band | — | — | 10.0× |
| `intens_err_<b>`, other bands | 10.0× | 1.0× | 1.0× |
| `a3_err_<b>` (`independent` mode) | 1.0× | 1.0× | 1.0× |

Reading the table through the rule: in coupled joint mode every `intens_<b>` is
a co-fitted intercept, so all of them are floored. In decoupled mode every one
is a ring statistic, so none are. **Reference mode is mixed**: the reference
band's `intens_<ref>` is overwritten with the fitted intercept `coeffs_ref[0]`
and takes the floor, while the other bands report a ring statistic and do not. The difference is real, not cosmetic — with incomplete angular
coverage the fitted intercept and the plain mean were measured at 100.0 against
101.32.

`independent` higher-order errors come from the shared `compute_deviations`,
which derives its own residual variance; single-band's non-ISOFIT harmonic
errors behave identically, so that row is parity rather than a gap. Single-band's
OLS `intens_err` (`rms/√N`) is likewise unfloored.

**Non-reference bands in reference mode report the same reducer as everywhere
else.** Their `intens_<b>` comes from `_per_band_mean_or_median`, so under WLS it
is the inverse-variance-weighted mean — the estimator whose variance
`intens_err_<b>` reports, and the one every other WLS path in the package uses.
Ref mode previously widened its coefficient vector with a plain `np.mean` while
reporting the weighted mean's error beside it: on a bimodal variance map the
reported error understated the reported value's own error by a factor of ten
(0.0904 against 0.9026). Under OLS both reducers give the plain mean, so nothing
changes there. Ref mode cannot select `integrator='median'` — two config
validators together exclude that combination — so this is always a mean.

**Reference mode takes its scale from the band it actually solved.** The
harmonic coefficients come from `fit_first_and_second_harmonics_ref`, which sees
only the reference band, so scaling their covariance with a residual variance
pooled over every band let a band *outside* the solve set its noise level.
Holding the reference band byte-identical and raising a second band's noise from
0.2 to 20 moved `eps_err` by 18×; at fixed geometry the pooled form spanned a
factor of 99 across a 1000× noise change. `_reference_residual_variance` now
computes the scale from the reference band's own residuals against the exact
five-parameter model it fitted, and it is passed through the
`residual_variance=` keyword — the same mechanism single-band uses in
`compute_parameter_errors`. The reference band's geometry errors and its
`intens_err_<ref>` therefore share one scale, and both are now bit-identical
under changes to any other band.

### One deliberate remaining difference

- **`intens_err` for a fitted intercept, and the `sigma_bg` floor.** Under OLS,
  single-band reports
  `intens_err = rms/√N`, computed outside the covariance and therefore *not*
  floored — even though its reported `intens` is also a fitted intercept
  (`y0_fit`). Multi-band instead derives `intens_err_<b>` from
  `cov[b, b] × var_residual` wherever the intercept was fitted — coupled joint
  mode, and the reference band under `harmonic_combination='ref'` — so it is
  floored along with the geometry errors from that same solve. The two
  implementations differ structurally here and cannot both be satisfied;
  multi-band keeps every error from one solve on one scale.

## Testing

Multi-band tests live under `tests/multiband/` (339 total, all green
as of the Phase-39 + review-pass merge). The per-module counts below
are approximate (collected at merge time):

| Module | Cases | Coverage |
|---|---|---|
| `test_config_mb.py` | ~96 | band-name regex, duplicate detection, reference-band membership, band_weights validation, integrator restriction, SMA/iteration consistency, loose-validity field defaults + normalization compatibility, multiband_higher_harmonics enum (4 values + parametrized variants), harmonic_orders validator (empty / <3 / duplicates / unique-sort), simultaneous_* experimental warning, shared-mode no-warn, ring-mean intercept × ref-mode incompatibility, Stage-3 fields (outer_reg / lsb_auto_lock / central_reg / compute_cog / forced-photometry validators). |
| `test_sampling_mb.py` | ~20 | B=1 numerical parity with single-band sampler, shared-validity (per-band masks, NaN), invalid-variance marking (non-finite and non-positive → NaN + warning, samples dropped, retired sentinels absent), all-masked degeneracy, mask broadcasting, variance all-or-nothing rejection, joint design matrix kernel parity. |
| `test_gradient_error_mb.py` | 10 | Per-band gradient error matches the reported ring statistic under both integrators; heteroscedastic mean error exceeds the retired inverse-variance form; uniform variance reproduces it; OLS median carries the √(π/2) penalty; B=1 gradient and error equal single-band's exactly. |
| `test_intercept_error_mb.py` | 14 | `_per_band_intercept_variance` against each reported statistic (IVW mean, plain mean, median); WLS median wrong by √(π/2) even for uniform variance under the retired pairing; OLS uses residual not raw scatter; empty band contributes 0.0; all four decoupled solvers write the matching `cov[b, b]`. |
| `test_residual_variance_mb.py` | 26 | Noiseless simultaneous model gives exactly zero residual variance (rectangular + jagged), and the truncated form is shown to charge the harmonics to the noise; noise-only residual recovered; `simultaneous_in_loop` geometry, harmonic and intensity errors share one scale end-to-end; `sigma_bg**2` floor applied inside the helper and reaching both geometry and per-band intensity errors; floor is a lower bound, not a rescale; exactly- and under-determined solves report `0.0` (not `None`) and still take the floor; harmonic attachers publish zero rather than an unscaled `(AᵀA)⁻¹` diagonal; reference-mode scale comes from the reference band alone (bit-identical across a 1000× noise change in an excluded band) and its fitted intercept shares that scale; non-reference bands report the inverse-variance-weighted mean matching their error, on a bimodal variance map where it cannot coincide with the plain mean. |
| `test_fitting_mb.py` | ~82 | joint solver recovery, B=1 single-band parity, WLS exact covariance, band-weight scaling, per-band sigma clip + AND, planted-galaxy recovery, ref-mode fallback, forced photometry, ring-mean intercept mode, loose-validity (band drop / n_valid columns / normalization), Stage-3 features (median intercept, outer-reg damping, lsb_auto_lock state machine, compute_cog wiring, central-reg penalty, forced-mode end-to-end + warn-list), **review-pass regressions** (B1 OLS rescale, B2 forced warn-list extensions, B3 forced-mode `grad_<b>`, H1 config immutability, H3 central-pixel WLS error, H4 SEM scaling). |
| `test_higher_harmonics.py` | ~20 | All four enum values; per-band-equality of shared higher orders; planted-m=4 recovery (shared + simultaneous_*); loose-validity × simultaneous (jagged kernel); simultaneous_original ≈ shared within tolerance; experimental UserWarning; FITS round-trip; harmonic_orders=[3,4,5,6] writes 16 expected per-band columns; D16 normalization separates curves under sharing; direct solver-level fit_simultaneous_joint{,_loose} planted-recovery. |
| `test_driver_mb.py` | ~17 | B=1 → single-band delegation, B=2 end-to-end recovery, WLS variance-mode tagging, band_weights passthrough, ref-mode end-to-end, error paths. |
| `test_utils_mb.py` | ~14 | FITS Schema-1 round-trip (per-band columns, loose validity, WLS variance mode, compute_cog), PrimaryHDU multi-band keywords, ASDF round-trip (Stage I — per-band columns, IsosterConfigMB recovery, compute_cog, lsb_auto_lock metadata, import guard), `load_bands_from_hdus`. |
| `test_plotting_mb.py` | 5 | Composite QA renders without exception, with SB constants, missing-bands error, image-count mismatch, loose-validity n_valid panel. |
| `test_cli_mb.py` | 10 | Stage-J: smoke runs (FITS / CSV / ASDF), banner output + `--quiet`, YAML-only bands, `--template` forced-photometry geometry parity, error paths (missing `--bands`, mismatched counts, `--reference-band` not in `--bands`, `--mask`/`--masks` mutual exclusion). |

Run with:

```bash
uv run pytest tests/multiband/ -v
```

## Related docs

- `docs/04-architecture.md` — multi-band module tree.
- `docs/agent/plan-2026-04-29-multiband-feasibility.md` — locked
  24-decision design record.
- `docs/02-configuration-reference.md` — single-band config (most
  multi-band fields share the same semantics).
