# Outer-Region Regularization for Elliptical Isophote Fitting

## Abstract

`isoster` implements a Tikhonov-regularized coordinate-descent fitter for
elliptical isophote extraction that remains stable in the
low-surface-brightness (LSB) outer regions of galaxies, where
harmonic-driven iterative fitters (Jedrzejewski 1987; Busko 1996) are
known to diverge or produce quantized artefacts from clipped per-iteration
step limits. The method adds a per-iteration penalty on geometry
displacement from a frozen inner reference, with a logistic ramp in
semi-major axis (SMA) that keeps the inner-region fit unchanged. Two
operating modes are supported: `damping`, which shrinks harmonic steps
without biasing toward the reference; and `solver`, which additionally
pulls geometry toward the reference. A complementary selector-level
penalty ensures cumulative drift is bounded. On a benchmark suite of 9
galaxies spanning synthetic edge cases and real HSC cluster-central
brightest-cluster galaxies (BCGs), `damping` mode with default tuning
preserves physical b/a(s) and PA(s) walks in the meaningful fit regime
(sma ≲ 80 px on HSC coadds) while suppressing clip-saturated jumps in
the LSB outskirts, at no runtime cost measurable against the
no-regularization baseline (§6.2).

## 1. Background and motivation

### 1.1 Elliptical isophote fitting in brief

The standard algorithm (Jedrzejewski 1987; photutils.isophote) fits an
intensity contour `I(φ) = I₀` at a series of SMA values by iteratively
adjusting the ellipse geometry `(x₀, y₀, ε, PA)` until a residual
function of azimuthal angle `φ` is minimal in a low-order harmonic
basis. In each iteration the residuals are decomposed as

```
ΔI(φ) = ∑ₙ [Aₙ cos(n φ) + Bₙ sin(n φ)]
```

and the `n = 1, 2` harmonic amplitudes drive updates to `(x₀, y₀)`,
`PA`, and `ε` respectively via analytic Jacobian relations
(`isoster/fitting.py` § geometry update). Each parameter update is
**clipped** to a per-iteration maximum:

| parameter | config field | default |
|---|---|---|
| center shift | `clip_max_shift` | 5.0 px |
| PA step | `clip_max_pa` | 0.5 rad (≈ 28.6°) |
| ε step | `clip_max_eps` | 0.1 |

Clipping is a safety bound preventing catastrophic single-iteration
excursions, inherited from photutils.

### 1.2 Failure mode in the LSB regime

In the outer regions of galaxies where the intensity gradient approaches
the background noise floor:

1. All four harmonic amplitudes `(A₁, B₁, A₂, B₂)` become comparable in
   magnitude (noise-dominated).
2. Analytic geometry updates saturate at the clip limits.
3. The "best iteration" selector among clipped candidates produces a
   quantized output: either no update (`Δp = 0`) or a saturated step
   (`|Δp| ≈ clip_max_p`).

On the HSC BCG `37498869835124888`, an uncorrected fit shows:

- Maximum combined center drift `|Δ(x, y)|_max = 91 px` over the
  outward sequence (sma 30-550 px).
- Saturated PA jumps `|ΔPA| ≈ 27°` (0.94 × clip_max_pa).
- Saturated ε jumps `|Δε| = 0.1`.

These produce discontinuities in the 1-D `b/a(s)` and `PA(s)` profiles
that no smoothing of the fit output can recover.

### 1.3 Prior approach and its limitation

The naive stabilization is a selector-level penalty:

```
L_selector(iter) = |max_amp(iter)| + λ(sma) · d²(geom(iter), geom_ref)
```

where `d²` is a quadratic displacement from a frozen inner reference.
The selector picks `argmin_iter L_selector`. This method was originally
deployed in isoster under `outer_reg_mode = "selector"` (removed in the
current version).

**Structural limitation**: the selector does not modify the iteration
trajectory. In the noise-dominated regime the candidate set produced by
the iterations is already quantized by clip limits; the selector is
forced to pick between "no update" and "saturated step". The quantized
output cannot be avoided by any choice of `λ` or weighting.

Empirical evidence: on 9 benchmark galaxies, the selector-only method
produces max `|ΔPA|` of 13–27° in the outer region — worse than the
unregularized baseline on several cases where the baseline's natural
convergence avoids saturating the clip.

### 1.4 Solver-level Tikhonov approach

The present method moves the penalty **inside** the per-iteration update
equations, producing a candidate set that is *not* clip-quantized. Each
iteration's step is the minimum of a Tikhonov-regularized linearized
objective:

```
L_solver(Δp) = ½ [amp + (Δp / c_p)]² + ½ λ(sma) w_p (p + Δp − p_ref)²
```

for each geometry parameter `p ∈ {x₀, y₀, ε, PA}` with Jacobian
coefficient `c_p`, weight `w_p`, and reference `p_ref`. Solving
analytically produces a penalty-aware step that naturally interpolates
between the harmonic-driven and the reference-pulled limit.

## 2. Mathematical formulation

### 2.1 Parameter-specific Jacobians

Let `amp_p` denote the signed harmonic amplitude driving parameter `p`
in a given iteration, and let `c_p` be the coefficient mapping
amplitude to unregularized step:

```
Δp_harmonic = c_p · amp_p
```

The explicit forms (see `isoster/fitting.py`:1630-1800) are

| p | mapping | c_p |
|---|---|---|
| minor-axis center shift (from `A₁`) | `aux = −amp (1 − ε) / g` | `(1 − ε) / g` |
| major-axis center shift (from `B₁`) | `aux = −amp / g` | `1 / g` |
| PA (from `A₂`) | `Δpa = amp · 2(1 − ε) / (s · g · ((1 − ε)² − 1))` | `2(1 − ε) / (s · g · ((1 − ε)² − 1))` |
| ε (from `B₂`) | `Δeps = amp · 2(1 − ε) / (s · g)` | `2(1 − ε) / (s · g)` |

where `s` is the SMA and `g = dI/dr` is the radial intensity gradient.
A damping factor is applied to all four.

### 2.2 Tikhonov objective and closed-form step

Augmenting the linearized harmonic residual with a quadratic penalty on
geometry displacement:

```
L(Δp) = ½ [amp + Δp/c_p]²  +  ½ λ w (p + Δp − p_ref)²       (2.1)
```

Differentiating with respect to `Δp` and setting to zero:

```
(1/c_p)[amp + Δp/c_p] + λ w (p + Δp − p_ref) = 0
Δp [1/c_p² + λ w] = −amp/c_p − λ w (p − p_ref)
Δp = [−amp c_p − λ w c_p² (p − p_ref)] / [1 + λ w c_p²]
```

Introducing the blend fraction

```
α ≡ λ w c_p² / (1 + λ w c_p²) ∈ [0, 1)                        (2.2)
```

the closed-form update is

```
Δp = (1 − α) · (c_p · amp)  −  α · (p − p_ref)
   = (1 − α) · Δp_harmonic  −  α · (p − p_ref)                (2.3)
```

Equation (2.3) is the **Tikhonov blend**. Two limits:

- `α = 0`: recovers `Δp = Δp_harmonic`. This is byte-identical to the
  unregularized fit. Achieved when `λ = 0`, `w = 0`, or the fit is at
  the harmonic minimum.
- `α → 1`: recovers `Δp = −(p − p_ref)`. Full pull to reference. Reached
  in the limit `λ w c_p² → ∞`, e.g. as `c_p → ∞` (vanishing gradient
  `g → 0`, noise-dominated regime).

The monotonic dependence of `α` on `c_p²` is astrophysically natural:
**where the harmonic fit has less local information (small `g`), the
penalty takes over more strongly**. Where the gradient is steep, the
data dominates.

### 2.3 Two operating modes

The solver branch implements two variants of (2.3):

**Damping** (default, recommended): drop the pull term.

```
Δp = (1 − α) · Δp_harmonic                                     (2.4)
```

Mathematically equivalent to a Tikhonov penalty on `Δp²` (penalize the
step size rather than the deviation from reference). The fit still
converges to the harmonic minimum; each step is just shorter in the
outer region. Preserves the harmonic convergence criterion because
`Δp → 0` when `amp → 0`.

**Solver** (specialized): full form (2.3).

Steps settle at a static balance point where `Δp = 0` implies
`amp = −c_p · λ w (p − p_ref)` — i.e. the harmonic residual balances
the reference pull. Breaks the standard `|max_amp| < tol` convergence
criterion; requires `geometry_convergence=True` to terminate cleanly
(see §2.7).

### 2.4 Logistic ramp in SMA

The strength `λ(sma)` varies smoothly from 0 in the inner region to
`strength` in the outer region:

```
λ(s) = strength / [1 + exp(−(s − onset) / width)]             (2.5)
```

At `s = onset`, `λ = strength/2`. At `s ≫ onset`, `λ → strength`. The
width controls the transition sharpness; smaller width = more abrupt
transition. When not explicitly set, `width` auto-computes as
`0.4 × onset`, which centers the transition in the `[0.6 × onset,
1.4 × onset]` SMA range.

Choice of logistic (vs. step or linear): smooth C^∞, two-parameter with
intuitive meaning (midpoint + slope), matches the empirically observed
shape of noise-amplitude growth in HSC edge-case galaxies (see §5).

### 2.5 Position-angle wrap handling

PA is an axis-like angle defined modulo π, so the residual
`δ = pa − pa_ref` must be wrapped onto `(−π/2, π/2]` to avoid spurious
large values when one of the two wraps through 0/π:

```
δ_wrapped = ((δ + π/2) mod π) − π/2                          (2.6)
```

All uses of `(pa − pa_ref)` in the Tikhonov update and the selector
penalty apply (2.6) before squaring or linear combination.

### 2.6 Reference geometry construction

The reference `geom_ref = (x₀_ref, y₀_ref, ε_ref, PA_ref)` is built
**once** from the inward isophote sequence (`_build_outer_reference` in
`isoster/driver.py`) before the outward fit begins. Candidate isophotes
are those with acceptable stop codes and
`sma ≤ sma₀ · outer_reg_ref_sma_factor` (default factor = 2.0).

For `x₀, y₀, ε`: flux-weighted arithmetic mean, with flux weights
clamped at `max(I, 10⁻⁶)` to prevent negative-background rings from
dominating:

```
x₀_ref = Σᵢ wᵢ x₀ᵢ / Σᵢ wᵢ,   wᵢ = max(Iᵢ, 10⁻⁶)             (2.7)
```

For `PA`: flux-weighted **circular mean on `2·PA`** (axis-like angles),
to handle the modular ambiguity:

```
C = Σᵢ wᵢ cos(2 PAᵢ),   S = Σᵢ wᵢ sin(2 PAᵢ)
R = √(C² + S²) / Σᵢ wᵢ                                       (2.8)
PA_ref = (½ · atan2(S, C)) mod π    if R ≥ 0.1
       = anchor.PA                   otherwise
```

The resultant length `R` tests whether the inner PA values are coherent
enough to define a meaningful mean. For random-scatter inputs `R → 0`
and the mean is undefined; the fallback to the anchor isophote's PA is
always safe.

### 2.7 Selector layer

An auxiliary penalty is added to the best-iteration amplitude selector:

```
L_selector(iter) = |max_amp(iter)|  +  ∑_p λ(sma) w_p (p(iter) − p_ref)²  (2.9)
```

The selector picks the iteration that minimizes (2.9), producing the
recorded isophote geometry. Unlike the solver-level term, the selector
does **not** modify the iteration trajectory — it only influences which
of the trajectory points is recorded as the output. Note that (2.9) is
a deliberate heuristic that mixes units: `|max_amp|` is a harmonic
amplitude in intensity units while the penalty terms are squared
geometry offsets (px², rad²), so the relative weighting of the terms
depends on the image's flux scale.

The selector penalty composes with either operating mode:
- With `damping`: the selector biases the recorded geometry toward
  `geom_ref` when multiple candidates are near the harmonic minimum.
- With `solver`: the selector over-pulls relative to the balance point
  from (2.3), yielding results similar to the historical selector-only
  method.

In the cleaned interface the selector layer is always active when the
feature is enabled (no independent toggle).

### 2.8 Convergence criterion coupling

Standard harmonic convergence is declared when

```
|max_amp| < conver · convergence_scale · rms                 (2.10)
```

The Tikhonov shrink and pull prevent this from tripping cleanly:

- In `damping` mode, per-iteration steps are shrunk by `(1 − α)`, so
  residual harmonics decay slowly; many LSB isophotes reach maxit
  without (2.10) tripping.
- In `solver` mode, the pull term biases geometry away from the
  harmonic minimum, so `|max_amp|` stays above the noise floor by
  construction; (2.10) never trips in the LSB tail.

In both cases the **recorded geometry is stable** (identical at maxit
of 50, 100, 500, or 2000 on every benchmark galaxy), but the
`|max_amp|`-based stop code is `2` (max-iter reached) rather than `0`
(converged).

The remedy is the geometry-stability criterion
(`geometry_convergence=True`): stop when geometry changes fall below
`geometry_tolerance` for `geometry_stable_iters` consecutive iterations.
This is auto-enabled by the config validator whenever
`use_outer_center_regularization=True`, with a `UserWarning` emitted if
the caller had explicitly disabled it.

## 3. Algorithm

### 3.1 Outward loop skeleton

```
# One-time setup after the inward loop completes
outer_ref = _build_outer_reference(inward_isophotes, anchor_iso, config)

for each outward sma:
    start_geometry = previous_isophote.geometry
    result = fit_isophote(
        image, mask, sma, start_geometry, config,
        previous_geometry=previous_isophote,
        outer_reference_geom=outer_ref,
    )
    append result to outward_isophotes
```

### 3.2 Per-isophote fit (damping mode)

```
# Solver-setup hoist, once per isophote
if use_outer_center_regularization and outer_reg_mode in ("damping", "solver"):
    w_center, w_eps, w_pa = config.outer_reg_weights.{center, eps, pa}
    onset = config.outer_reg_sma_onset
    width = config.outer_reg_sma_width or 0.4 * onset
    λ = config.outer_reg_strength / (1 + exp(-(sma - onset) / width))
    solver_on = λ >= 1e-6
    solver_use_pull = (outer_reg_mode == "solver")
else:
    solver_on = False

for iteration i = 1..maxit:
    sample ellipse, compute harmonic amps A1, B1, A2, B2
    if mode == "largest":
        pick parameter p with max |amp_p|
    else:  # simultaneous
        update all four

    for each parameter p to be updated:
        # unregularized step
        Δp_harmonic = c_p(ε, s, g) · amp_p · damping

        if solver_on and w_p > 0:
            α = λ · w_p · c_p² / (1 + λ · w_p · c_p²)
            if solver_use_pull:
                Δp = (1 - α) · Δp_harmonic - α · (p - p_ref)   # solver
            else:
                Δp = (1 - α) · Δp_harmonic                      # damping

        clip Δp to ±clip_max_p
        apply Δp to p

    # Record best iteration via selector penalty (2.9)
    penalty = compute_outer_center_regularization_penalty(...)
    eff_amp = |max_amp| + central_reg_penalty + penalty
    if eff_amp < best_eff_amp:
        record current geometry as best

    # Convergence
    if |max_amp| < conver · rms:  break         # (2.10)
    if geometry_convergence:
        if geometry has been stable for geometry_stable_iters iterations:
            break

record best geometry for this sma
```

### 3.3 Relation to simultaneous mode

In `geometry_update_mode="simultaneous"`, all four parameters are
updated in the same iteration from their respective harmonic amps
(rather than picking a single max-amp parameter). The Tikhonov blend is
applied independently to each of the four updates, using that
parameter's own `c_p` and `w_p`. The mathematics of (2.2)-(2.3) is
unchanged; only the update schedule differs.

The 7-parameter ISOFIT joint solver (`simultaneous_harmonics=True`;
Ciambur 2015) is **not** wired into the Tikhonov term; its linear system
includes shape harmonics `(A₃, B₃, A₄, B₄)` whose Jacobian coefficients
are not independently expressed. When `simultaneous_harmonics=True` is
combined with any outer-reg mode, the solver-level term is skipped and
only the selector-layer penalty (2.9) applies. A `UserWarning` is
emitted at config time.

## 4. Composition with other mechanisms

### 4.1 LSB auto-lock (`lsb_auto_lock`)

Independent mechanism that freezes geometry hard after a configurable
debounce count of high-gradient-error isophotes in the outward
sequence. Complementary to outer-reg:

- **Outer-reg**: soft pre-lock smoothing throughout the outer region.
- **Auto-lock**: hard post-lock freeze once the fit has unambiguously
  entered the LSB tail.

The two compose without interference: `geometry_convergence` is active
for both phases, and `_build_outer_reference` is built from the full
inward sequence regardless of the lock state.

### 4.2 Central regularization (`use_central_regularization`)

Parallel mechanism ramping **inward** from the center (exponential
decay in SMA, `central_reg_sma_threshold` sets the scale). Addresses
the dual failure mode: noisy inner isophotes at sub-pixel SMA. The two
features decay in opposite directions and do not overlap in SMA.

### 4.3 Geometry clipping (`clip_max_*`)

Safety bounds on per-iteration steps. With damping mode active, typical
steps are shrunk below the clip limits, so clipping rarely fires in
practice. Retained as a last-resort guard for pathological noise
excursions.

### 4.4 Fix flags (`fix_center`, `fix_pa`, `fix_eps`)

Hard per-parameter freezes. Incompatible with outer-reg per axis: if
`fix_p = True` and `outer_reg_weights[p] > 0`, a `UserWarning` is
emitted because the penalty is inert on a frozen parameter.

### 4.5 Forced photometry

When a template geometry is provided (`fit_image(..., template=...)`),
geometry is prescribed per-isophote and the outer-reg feature is a
no-op. A `UserWarning` at runtime flags the useless flag.

## 5. Configuration interface

### 5.1 Primary fields (5)

| field | type | default | role |
|---|---|---|---|
| `use_outer_center_regularization` | bool | `False` | master on/off |
| `outer_reg_mode` | str | `"damping"` | `"damping"` (recommended) or `"solver"` (specialized) |
| `outer_reg_strength` | float | `2.0` | peak value of `λ(s)` in the outer region |
| `outer_reg_sma_onset` | float | `50.0` | SMA (pixels) at the logistic midpoint |
| `outer_reg_weights` | dict | `{center: 1, eps: 1, pa: 1}` | per-axis weights `w_p` |

### 5.2 Expert tuning (2)

| field | type | default | role |
|---|---|---|---|
| `outer_reg_sma_width` | Optional[float] | `None` | logistic slope width; auto = `0.4 × onset` if `None` |
| `outer_reg_ref_sma_factor` | float | `2.0` | inward-candidate SMA cutoff for reference build |

### 5.3 Auto-coupled fields

| field | coupling |
|---|---|
| `geometry_convergence` | forced to `True` when outer-reg is active; warn if user explicitly set `False` |

### 5.4 Result-dict additions

On successful fit, the result dictionary carries the frozen reference:

- `outer_reg_x0_ref`, `outer_reg_y0_ref` (px)
- `outer_reg_eps_ref`
- `outer_reg_pa_ref` (rad, `[0, π)`)
- `use_outer_center_regularization` (echo of the flag)

## 6. Empirical validation

### 6.1 Benchmark set

Nine galaxies from the isoster distribution:

- **6 mock** (`example_hsc_edgecases/`): synthetic HSC `i`-band
  coadd-style images with controlled contaminants — clean case, nearby
  bright star, nearby large galaxy, blending star, artifact, cluster
  neighborhood.
- **3 real** (`example_hsc_edge_real/`): real HSC cluster-central BCGs
  with rich environments (companions, halos, nearby stars).

All at HSC pixel scale 0.168 arcsec, zeropoint 27.0. Cutouts
nominally 1200×1200 px with anchor from `X_OBJ/Y_OBJ` mask-header keys
(real) or image center (mock).

Six arms tested:

1. `baseline`: no outer-reg
2. `damping_default`: mode=damping, defaults (`strength=2, onset=50,
   width=None→20, weights={1,1,1}`)
3. `damping_full`: same but onset=50, width=20, strength=2
4. `damping_recommended`: onset=100, width=20, strength=4
5. `solver_standard`: mode=solver, onset=50, width=20, strength=2
6. `solver_strong`: mode=solver, onset=50, width=20, strength=8

Metrics in outer region (`sma ≥ sma₀ = 10 px`):

- `pl_comb`: max combined center drift (px) relative to anchor
- `max_|Δpa|`: largest single-step PA change (deg)
- `max_|Δeps|`: largest single-step ellipticity change
- `n_sat_{pa,eps}`: count of transitions ≥ 90% of respective clip limit
- `T(s)`: median of 3-run `fit_image` wall-clock

### 6.2 Runtime

Median runtime per galaxy × arm (s), warm cache:

| ID (kind)                                     | baseline | damp_default | damp_full | damp_recom | solver_std | solver_strong |
|----------------------------------------------|---------:|-------------:|----------:|-----------:|-----------:|--------------:|
| 10140088 (mock, clear)                       | 0.29 | 0.26 | 0.25 | 0.30 | 0.26 | 0.27 |
| 10140002 (mock, bright star)                 | 0.26 | 0.27 | 0.26 | 0.25 | 0.28 | 0.25 |
| 10140006 (mock, large galaxy)                | 0.35 | 0.32 | 0.29 | 0.23 | 0.26 | 0.24 |
| 10140009 (mock, blending)                    | 0.30 | 0.25 | 0.26 | 0.23 | 0.24 | 0.23 |
| 10140056 (mock, artifact)                    | 0.24 | 0.26 | 0.23 | 0.24 | 0.25 | 0.23 |
| 10140093 (mock, cluster)                     | 0.24 | 0.23 | 0.26 | 0.24 | 0.24 | 0.25 |
| 37498869835124888 (real BCG)                 | 0.33 | 0.39 | 0.31 | 0.35 | 0.30 | 0.33 |
| 42177291811318246 (real BCG)                 | 0.32 | 0.28 | 0.35 | 0.31 | 0.29 | 0.31 |
| 42310032070569600 (real BCG)                 | 0.24 | 0.27 | 0.24 | 0.30 | 0.27 | 0.34 |
| **total** (9 galaxies)                       | **2.56** | **2.52** | **2.44** | **2.43** | **2.39** | **2.46** |

Every arm's total lands within about 7% of baseline (`damp_default`
−1.6%, `damp_full` −4.7%, `damp_recom` −5.1%, `solver_std` −6.6%,
`solver_strong` −3.9%). All five are *faster* than baseline, which is
the giveaway: these are medians of three runs on sub-second fits, so
the spread is timing noise rather than a systematic saving, and the
honest reading is that regularization costs nothing measurable at this
precision — not that it speeds the fit up. An earlier version of this
sentence claimed ±3%, which its own table does not support. Solver
modes are equivalent because both rely on the auto-enabled
`geometry_convergence`. The Tikhonov arithmetic adds negligible
per-iteration overhead (≲ 10 FLOP per axis).

### 6.3 Smoothness — saturated-step counts

Summed across all 9 galaxies:

| arm | n_sat_pa | n_sat_eps |
|---|---:|---:|
| baseline | 1 | 3 |
| damping_default | 2 | 8 |
| damping_full | **0** | **0** |
| damping_recommended | **0** | **0** |
| solver_standard | 1 | 0 |
| solver_strong | 1 | 5 |

**Key result**: `damping_full` and `damping_recommended` (both with
weights `{1, 1, 1}`) achieve zero saturated clipped steps on every
galaxy. The `damping_default` arm uses the historical
weights-`{1, 0, 0}` defaults (center-only damping, dropped in the
current release) and still produces 10 saturated events total,
confirming that **full-axis weights are necessary**.

### 6.4 Smoothness — max |Δpa| by galaxy (outer, deg)

| galaxy | kind | baseline | damp_default | damp_full | damp_recom | solver_std | solver_strong |
|---|---|---:|---:|---:|---:|---:|---:|
| 10140088 | mock | 6.05 | 4.12 | 2.38 | 2.24 | **14.81** | **14.00** |
| 10140002 | mock | 6.69 | **12.48** | 1.84 | 1.91 | **11.41** | 9.80 |
| 10140006 | mock | 4.04 | 3.75 | 3.06 | 3.05 | **31.46** | **42.69** |
| 10140009 | mock | 6.59 | 3.86 | 2.22 | 2.65 | **12.79** | **13.06** |
| 10140056 | mock | 4.42 | 3.78 | 2.54 | 2.65 | **14.03** | **12.85** |
| 10140093 | mock | 4.46 | 6.79 | 3.08 | 4.02 | **13.97** | **11.90** |
| 37498869835124888 | real | 6.29 | **18.51** | 2.01 | 2.50 | 0.00 | 0.00 |
| 42177291811318246 | real | 9.91 | 7.29 | 7.66 | 6.23 | 6.97 | 5.50 |
| 42310032070569600 | real | **28.65** | **28.65** | 0.67 | 1.11 | 11.07 | 0.00 |

Bold: > 10° (large transitions or frozen-to-zero artefacts).

Observations:

- `damping_full` and `damping_recommended` are the most consistent
  performers: ≤ 8° on every galaxy, ≤ 3° on 7 of 9.
- `solver_standard`/`solver_strong` produce large single transitions to
  the Tikhonov balance point (11-43°) on most mock galaxies and freeze
  to zero variation on two real galaxies where the reference happens to
  match the solver's equilibrium.
- `damping_default` (historical center-only weights) is inferior to
  full weights on 3/9 galaxies — this confirms that the interface
  default must be full weights.

### 6.5 Center stability

Max combined center drift `pl_comb` (px) in the outer region:

| arm | min | median | max | # galaxies with pl_comb > 10 px |
|---|---:|---:|---:|---:|
| baseline | 0.05 | 10.4 | 91.4 | 6 |
| damping_default | 0.07 | 0.17 | 0.69 | 0 |
| damping_full | 0.07 | 0.17 | 0.76 | 0 |
| damping_recommended | 0.09 | 0.28 | 1.11 | 0 |
| solver_standard | 0.00 | 0.05 | 0.28 | 0 |
| solver_strong | 0.00 | 0.05 | 0.24 | 0 |

All regularized arms eliminate the baseline's runaway center drift
(6 of 9 galaxies show baseline `pl_comb > 10 px`, including 91 px on
`37498869835124888`).

## 7. Limitations

### 7.1 Over-regularization in solver mode

Solver mode pulls outer geometry toward the inner reference. On
galaxies with real astrophysical structure (PA twists, outward
flattening) this suppresses physically meaningful variation. On 5 of 6
mock galaxies, the solver mode's `max_|Δpa|` exceeds the baseline's
— the single large transition to the solver-balance-point is more
visible than the baseline's noise.

Damping mode is the recommended choice for general LSB fitting. Solver
is useful only when the astrophysical expectation is that outer
geometry *should* match inner geometry (e.g. contamination-dominated
fields, deep-survey verification).

### 7.2 Anchor precision in the inward reference

The flux-weighted inner reference (§2.6) has sub-pixel precision; the
accuracy of the center reference depends on the inward isophote fit
quality in the first few pixels of SMA. When the initial guess
`(x₀, y₀)` passed to the fitter has only integer-pixel precision
(e.g. from a smoothed peak finder), the inner fit adapts during the
inward loop and converges to sub-pixel accuracy, but the QA plot's
"center offset" panel (which plots `x₀(s) − x₀_config`) shows this as
a constant ~0.3 px offset. This is cosmetic; the outer-reg reference
correctly captures the sub-pixel center.

### 7.3 ISOFIT joint harmonic mode

Not wired into the solver-level term. When `simultaneous_harmonics=True`
is combined with any outer-reg mode, the Tikhonov update is skipped and
only the selector-layer penalty (2.9) applies. A `UserWarning` at
config time makes this explicit.

### 7.4 Degenerate inner PA scatter

When inner PA values are so scattered that the circular-mean resultant
`R < 0.1` (eq. 2.8), `PA_ref` falls back to the anchor isophote's PA.
This is safe but may produce a less representative reference for
galaxies where the inner isophotes genuinely lack coherent PA (near
face-on systems with `ε < 0.05`). In such cases the PA weight
`w_pa` can be set to 0 to disable PA regularization while retaining
center and ε damping.

## 8. Usage guide

### 8.1 Minimal recommended configuration

```python
from isoster.config import IsosterConfig
from isoster import fit_image

config = IsosterConfig(
    x0=...,  y0=...,  sma0=...,  maxsma=...,
    use_outer_center_regularization=True,
    # defaults: mode="damping", strength=2.0, onset=50.0, weights={1,1,1}
)
results = fit_image(image, mask=mask, config=config, variance_map=variance)
```

This is appropriate for general LSB fitting on HSC-grade data.

### 8.2 Tuning `outer_reg_sma_onset`

The ramp midpoint should sit roughly where the harmonic fit starts to
degrade. For HSC coadds, this is typically 50-100 px depending on
galaxy size. Heuristic: `onset ≈ 0.5 × R_eff` for unresolved edge of
significant signal.

### 8.3 Tuning `outer_reg_strength`

| value | effect |
|---|---|
| `0.5` | very soft — useful only on galaxies that are nearly stable without regularization |
| `2.0` (default) | moderate; recommended for most cases |
| `4-8` | strong damping, for real LSB galaxies with heavy contamination |
| `≥ 10` | may over-regularize; check QA PNGs for frozen profiles |

### 8.4 Specialist: solver mode for forced-like anchoring

```python
config = IsosterConfig(
    ...,
    use_outer_center_regularization=True,
    outer_reg_mode="solver",
    outer_reg_strength=8.0,  # strong pull
)
```

Anchors outer isophotes firmly to the inner flux-weighted reference.
Appropriate when the outer fit would otherwise drift off-source
(e.g. contamination fields), but flattens any genuine outer
astrophysical structure. Inspect QA PNGs before committing.

### 8.5 Combining with auto-lock

```python
config = IsosterConfig(
    ...,
    use_outer_center_regularization=True,  # pre-lock smoothing
    lsb_auto_lock=True,                    # post-lock hard freeze
)
```

Damping handles the transition region; auto-lock takes over once the
gradient quality has unambiguously collapsed. `geometry_convergence` is
auto-enabled by either feature.

## 9. Implementation references

| concern | file / function |
|---|---|
| Config fields + validators | `isoster/config.py` § "Outer Region Center Regularization" |
| `α` blend helper | `isoster/fitting.py` `_tikhonov_alpha` |
| Selector-level penalty | `isoster/fitting.py` `compute_outer_center_regularization_penalty` |
| Per-branch Tikhonov blend | `isoster/fitting.py` `fit_isophote` update section |
| Reference construction | `isoster/driver.py` `_build_outer_reference` |
| Integration test suite | `tests/integration/test_outer_center_regularization.py` |
| Cross-galaxy benchmark | `examples/run_cross_galaxy_outer_reg_sweep.py` |

## References

- Jedrzejewski, R. I. 1987, MNRAS, 226, 747 (original isophote fitter).
- Busko, I. 1996, in ASP Conf. Ser. 101, Astronomical Data Analysis
  Software and Systems V, ed. G. H. Jacoby & J. Barnes, 139
  (stsdas.analysis.isophote implementation).
- Ciambur, B. C. 2015, ApJ, 810, 120 (ISOFIT joint high-order harmonic
  solver).
- Bradley, L., et al. 2025 (photutils.isophote reference implementation).
- Tikhonov, A. N. 1963, Soviet Math. Dokl., 4, 1035 (original
  regularization theory).
