# LSB Regime Features: Design and Implementation

This document describes the two complementary features that stabilize
isoster isophote fits in the deep low-surface-brightness (LSB) regime:

1. **Automatic LSB Geometry Lock** (`lsb_auto_lock`) — a hard post-lock that
   freezes geometry once the outward fit crosses a gradient-quality
   threshold.
2. **Outer Region Center Regularization** (`use_outer_center_regularization`)
   — a soft pre-lock that nudges the centroid toward a flux-weighted inner
   reference while the fit is still free.

Both features target the same failure mode — outward isophotes drifting
away from the galaxy center once the radial gradient collapses into
background — but they attack it at different stages and compose cleanly.

For usage, see `docs/01-user-guide.md` ("Automatic LSB Geometry Lock" and
"Outer Region Center Regularization"). For parameter names and defaults,
see `docs/02-configuration-reference.md` sections 12 and 13. **For the
full mathematical derivation, algorithm pseudocode, benchmark tables,
and publication-grade reference on the outer-region Tikhonov mechanism,
see `docs/08-outer-regularization.md`.** This document (07) explains
*why* the features exist, *how* the algorithms were chosen, and
*where* the implementation lives file by file.

## 1. Motivation

Photutils/isofit-style elliptical isophote fitting iterates on the largest
absolute harmonic amplitude to update one geometry parameter per iteration
(the traditional coordinate descent). Convergence is signalled when the
largest amplitude drops below a tolerance. In the high-S/N regime this
works well; in the LSB regime it fails in two distinct ways:

- **Noise-driven amplitude updates**: once the local gradient collapses to
  zero, the harmonic amplitudes are dominated by noise. Each iteration's
  "largest" amplitude is random, so the geometry does a random walk outward
  rather than converging.
- **Contamination-driven amplitude updates**: masked bright neighbours, PSF
  wings, unresolved blends and detector artifacts add slowly varying
  harmonics that bias the amplitude selector. The fit walks toward the
  contaminant instead of staying on the galaxy centroid.

Both manifest in the HSC edge-case galaxies in
`examples/example_hsc_edgecases/`. Free fits on the 6-galaxy set show
pre-lock outward centroid drifts from ~1.7 px (clean case) to ~8 px
(blending bright star); `pl_rms` of the centroid trajectory runs
0.4-1.4 px, far above the sub-pixel noise floor of the inner region.

### Why not `fix_center=True`?

`fix_center=True` is the hard solution: freeze x0/y0 for the entire fit.
It works, but it is *too blunt*:

- It throws away the inner fit's own information. If the user does not
  supply an accurate `(x0, y0)`, the frozen center is worse than whatever
  the inner isophotes would have converged to.
- It hides centering errors. A drifting fit is a diagnostic signal; a
  fix_center fit silently bakes in whatever initial guess was used.
- It is incompatible with forced photometry in any useful way (the template
  already fixes geometry).

The two LSB features are softer and data-driven: they derive the reference
center from the inward isophotes, and they only *freeze* geometry once the
fit itself signals that it has entered the LSB regime.

### Why not just the auto-lock?

An auto-lock post-hoc is a hard freeze, but it only fires once the outward
fit has *already* accumulated drift in the pre-lock region. On noisy or
contaminated galaxies the pre-lock drift can reach several pixels before
the gradient quality check trips, and the locked tail is anchored to the
last pre-lock isophote's centroid — which may already be displaced.

Outer center regularization fills this gap: it damps pre-lock drift as the
fit walks outward, so the eventual anchor that the auto-lock commits to is
itself closer to the true centroid. The composition is the key design
point — soft pre-lock + hard post-lock — and is validated empirically in
§5.

## 2. Algorithmic considerations

### 2.1 Automatic LSB Geometry Lock

The detector and the lock mechanism are deliberately simple.

**Detector.** An outward isophote is classified as "LSB-regime" when any
of the following hold:

- Its `stop_code == -1` (isophote failed to converge).
- `grad >= 0` (positive or flat radial gradient — not a galaxy profile).
- `|grad_error / grad| > lsb_auto_lock_maxgerr` (gradient relative error
  exceeds the threshold, default 0.3).

See `isoster/driver.py` → `_is_lsb_isophote(iso, maxgerr_thresh)`.

The earlier session used a second threshold
`lsb_auto_lock_grad_snr` (`|grad / grad_error| < 3.0`); it was dropped in
this branch because it tested the same ratio as `maxgerr` — two knobs
that are in fact one.

**Debounce.** A single isophote crossing the threshold is not enough: the
detector counts consecutive trips (`lsb_auto_lock_debounce`, default 2).
Transient numerical glitches (non-finite gradients, single noisy samples)
therefore cannot commit the lock.

**Lock commit.** When the debounce counter reaches its threshold, the
driver:

1. Walks back `debounce` isophotes to select the *anchor* — the last clean
   outward isophote before the LSB tail started.
2. Builds a locked `IsosterConfig` clone (`_build_locked_cfg`) with
   `fix_center/fix_eps/fix_pa=True` at the anchor geometry, and swaps the
   integrator to `lsb_auto_lock_integrator` (default `"median"`, more
   robust against contamination).
3. Re-runs the remaining outward schedule under the locked clone.
4. Marks each locked isophote with `lsb_locked=True`, and the
   *committing trigger* isophote — the last of the debounce streak,
   which was itself fit free — with `lsb_auto_lock_anchor=True`. The
   geometry anchor selected in step 1 is not marked.

All four clone flags collectively disable geometry solvers for the locked
tail, so the remaining isophotes are effectively forced-photometry slices
on a single frozen ellipse. The median integrator avoids one additional
failure mode: a bright pixel inside the locked ellipse dragging the mean
intensity upward.

### 2.2 Outer Region Center Regularization

The soft pre-lock is a regularization penalty added to the isophote
fitting loop. There are two decisions worth calling out.

**Penalize the selector, not the solver.** isoster's per-iteration loop
picks the "best" iteration by minimum
`abs(max_harmonic_amp)` (the effective amplitude). The penalty adds

```
λ(sma) × ((x0 − x0_ref)² + (y0 − y0_ref)²)
```

to that selector, not to the geometry update equations. The geometry math
is untouched: each iteration walks in the direction the harmonic solver
says, but the selector prefers iterations that stay close to the reference
centroid.

This matters because it *cannot inject bias into the fit*. The iteration
sequence is identical with or without the penalty — only which iteration
gets recorded as "best" changes. In the clean case, where the free fit's
own best iteration already sits near the reference, the penalty adds
~0 and is a no-op. In the drifting case, it prefers iterations that
return toward the reference rather than the one with the nominally
smallest harmonic. The experimentally observed 30-50× drift reduction on
the HSC edge cases (§5) comes from this selector bias alone.

Implementing the penalty inside the geometry update equations — as a
Tikhonov-style term on `(Δx, Δy)` — was considered and rejected. It would
require changing every solver branch (5-param, ISOFIT 7-param, EA mode)
and would couple the center regularization to the iteration math, making
it harder to reason about convergence and harder to disable cleanly.

**Logistic ramp in sma.** The penalty is zero at small sma and saturates
at `outer_reg_strength` at large sma:

```
λ(sma) = outer_reg_strength / (1 + exp(−(sma − onset) / width))
```

The inner region is therefore untouched (the logistic is ~0 well inside
`onset`), the transition is smooth (`width` ≈ a few astep growth steps),
and the outer region gets the full strength. The logistic was picked over
a step or a linear ramp because it has a well-defined "center" and width,
is easy to tune from two parameters, and matches the empirically observed
shape of the transition — the drift starts ramping up near the half-light
radius and saturates a few scale lengths further out.

**Where does the reference geometry come from?** From a flux-weighted
mean of the *inward* isophote geometry, plus the anchor isophote at
`sma0` folded in unconditionally. See `_build_outer_reference` in
`isoster/driver.py`:

- Candidates: all inward isophotes with acceptable stop codes and
  `sma ≤ anchor.sma × outer_reg_ref_sma_factor`.
- Weights: `max(intens, 1e-6)`, so the brightest inner isophotes dominate
  and negative-background rings cannot flip the sign.
- `x0_ref`, `y0_ref`, `eps_ref`: flux-weighted arithmetic mean.
- `pa_ref`: flux-weighted **circular mean on `2·pa`** — pa is defined
  mod π (axis-like), so a naive arithmetic mean of pa=0 and pa≈π lands
  at π/2 (orthogonal to both). If the resultant length on the unit
  circle is < 0.1 (inner pa values too scattered for a useful mean),
  pa_ref falls back to the anchor's pa.
- Fallback: if no inward isophote qualifies (`minsma ≥ sma0`, for
  example), the reference is just the anchor's geometry and a
  `UserWarning` is emitted at config time.

The flux-weighted mean (as opposed to picking a single inner isophote)
is robust to individual bad inner isophotes, and is the standard trick
for collapsing a noisy inner trajectory into a single well-defined
reference.

The reference is built *once* per fit, before the outward loop runs,
and is frozen for the remainder of the fit. Letting it update per
iteration would recreate the drift problem the feature is meant to
solve.

**Selector asymmetry and the `outer_reg_weights` extension.** The
penalty as originally written was **center-only**: `λ(sma) · ((x0 −
x0_ref)² + (y0 − y0_ref)²)`. In empirical use on real HSC BCG data the
center-only form was found to introduce a selector asymmetry: in the
LSB outer regime, harmonic amplitudes in all four modes (x0, y0, eps,
pa) are noise-dominated and comparable in magnitude; the best-iteration
selector therefore systematically prefers iterations where eps or pa
was just updated, because those iterations leave `(x0, y0)` unmoved
and so carry zero center-penalty. The clipped per-iteration steps
(`clip_max_eps=0.1`, `clip_max_pa=0.5 rad`) then show up as saturated
jumps in the 1-D eps(sma) and PA(sma) profiles. The feature was
*redirecting* the outer random walk from `(x0, y0)` into `(eps, pa)`
rather than damping it.

The fix is the `outer_reg_weights` config dict, which mirrors the
existing `central_reg_weights` pattern:

```python
outer_reg_weights = {"center": 1.0, "eps": 1.0, "pa": 1.0}   # default
```

With all three weights positive the penalty becomes

```
λ(sma) · [ w_c · ((x0 − x0_ref)² + (y0 − y0_ref)²)
         + w_e · (eps − eps_ref)²
         + w_p · (pa_wrap − pa_ref)² ]
```

where `pa_wrap` folds the PA residual onto `(−π/2, π/2]`. The current
default weights `{center: 1, eps: 1, pa: 1}` close the selector
asymmetry by damping all geometry axes together. Set `eps` or `pa` to
`0.0` only when the fit should intentionally leave those axes undamped.
See §5 for empirical validation on the `37498869835124888` HSC BCG.

### 2.3 Inward-first loop ordering

The driver runs the *inward* loop before the outward loop. This is
unconditional in this branch and is not gated on the outer-reg flag,
because:

- The inner reference must exist before any outward isophote is fit with
  the penalty active.
- No existing test depends on the outward-first order; the only test that
  previously cared — a growth-path monkeypatch — was updated to filter by
  role not position.

See the note in `docs/04-architecture.md` ("Inward-first loop order").

### 2.4 Solver-level Tikhonov modes (damping and solver)

The selector-only form described in §2.2 was found empirically on real
HSC BCGs to suffer a different failure: in the noise-dominated outer
regime, the harmonic solver's per-iteration candidates are effectively
**quantized** by the clip limits (`clip_max_eps`, `clip_max_pa`). The
selector penalty picks among candidates but cannot produce a sub-clip
answer — so it is forced to choose between "no update" and "saturated
clipped step", producing visible 0.1-eps and 0.5-rad PA jumps in the
1-D profiles even when the center is anchored. This is a structural
limit of any selector-only mechanism.

The fix is to put the penalty **inside** the per-iteration harmonic
update (Tikhonov-style), so each iteration's candidate step is itself
penalty-aware and no longer fixed to the clip magnitude. The
`outer_reg_mode` config field picks one of two solver-level variants:

**`damping` (default).** Per-iteration step shrink only — no reference
pull. Each parameter's update equation becomes

```
Δparam  =  (1 − α) · Δparam_harmonic
```

with `α = λ(sma) · w · coeff² / (1 + λ(sma) · w · coeff²)` where
`coeff` is the harmonic-to-parameter Jacobian the existing solver
already computes. The fit still walks to its data-preferred geometry;
only the **step magnitude** is shrunk in the outer region, so
saturated clipped updates cannot occur. This mode preserves the real
astrophysical walks of eps(sma) and PA(sma) that the baseline free fit
captures (up to ~80-90 px on 37498869835124888), while suppressing the
clip-saturated jumps in the LSB regime. Harmonic convergence works
correctly: once the step is small enough, the `|max_amp|` criterion
trips normally.

**`solver`.** Full Tikhonov term — step shrink *plus* a pull toward
the frozen reference:

```
Δparam  =  (1 − α) · Δparam_harmonic  −  α · (param − param_ref)
```

The geometry settles at a static balance point where the harmonic
drive equals the Tikhonov pull. This point depends on α (i.e. on
`λ`, `w`, and the local gradient); it is neither the data-preferred
geometry nor `param_ref`. Use `solver` when you want the outer
isophotes anchored to the inner flux-weighted reference (for example,
when the outer fit would otherwise drift off-source due to heavy
contamination). The mode flattens genuine outer PA/eps structure by
design.

**Selector layer (always on).** Orthogonal to `outer_reg_mode`. The
selector-level penalty described in §2.2 is always applied on top of
the chosen solver-level mode whenever
`use_outer_center_regularization=True`; there is no independent
toggle for it. It acts as a cumulative anchor that the selector uses
to score candidate iterations.

**`geometry_convergence` auto-enable.** Both solver-level modes cause
`|max_amp|` to stay somewhat above the harmonic convergence threshold
— in `damping` because the shrunk step leaves residual harmonics that
decay slowly, in `solver` because the ref-pull intentionally biases
away from the harmonic minimum. Without a geometry-stability
convergence check, many outer isophotes run to `maxit` with zero
change to the recorded geometry (wasted cost). The config validator
therefore auto-enables `geometry_convergence=True` whenever
`use_outer_center_regularization=True`, with a `UserWarning` if the
user had explicitly set it `False`. With auto-enable on, runtime is
within a few percent of the no-regularization baseline.

**Removed: `outer_reg_mode='selector'`.** The original selector-only
mode was dropped from the valid values in the same change that
introduced damping/solver. The selector-level penalty still exists as
an always-on layer on top of any `outer_reg_mode`. Users whose old
configs set `outer_reg_mode="selector"`
will see a pydantic `ValidationError` at config time listing the
accepted values.

## 3. Composition

The two features are designed to compose without interaction side effects:

- **Pre-lock region** (`sma < transition_sma`): outer-reg is active (when
  enabled), auto-lock is idle. The selector-level penalty damps drift.
- **Lock commit** (one isophote): auto-lock walks back to the anchor, which
  may or may not be the last pre-lock isophote. The outer-reg feature does
  not touch the lock mechanism.
- **Locked tail** (`sma ≥ transition_sma`): geometry is frozen by the
  `_build_locked_cfg` clone. The outer-reg penalty is still *evaluated*
  per iteration but has no effect because `fix_center=True` means the
  solver never moves `x0, y0` anyway.

In practice this means enabling both flags is always safe: outer-reg is
silently inert in the locked tail, and auto-lock is silently inert before
its debounce trips. Neither feature is aware of the other.

The auto-lock hard freeze also has a subtle interaction with the
reference: because the anchor the lock commits to is *less drifted* when
outer-reg is on, the locked tail's fixed centroid is itself more accurate.
This is the composition win that the HSC edge-case benchmarks show (§5).

## 4. Guards and limitations

### Forced photometry

Both features are wired into the regular-mode driver only. In template
forced-photometry mode (`_fit_image_template_forced`), geometry is
prescribed by the template and the features are no-ops.

To avoid silent no-ops, the forced-photo driver emits a `UserWarning` at
runtime if either `lsb_auto_lock=True` or
`use_outer_center_regularization=True` is set together with a template.
This is intentional: a user who flips the flag in a config and later
reuses that config in forced-photo mode should see a clear message.

### `fix_center=True`

The outer-reg penalty operates on `(x0, y0)`; if those are already fixed,
it has nothing to do. The config validator emits a `UserWarning` when
`use_outer_center_regularization=True` is combined with `fix_center=True`
so the user knows the feature is inert.

The auto-lock is also inert under `fix_center=True` in the sense that
there is no drift to lock away — but the detector still fires and commits
a "lock" that simply re-freezes what was already frozen. No warning is
needed there.

### Sampling and harmonic modes

Neither feature touches the sampling path (`sampling.py`) or the harmonic
solvers (`fitting.py`'s coefficient math). Both are mode-agnostic by
construction:

- `use_eccentric_anomaly=True` (more even sampling around elongated isophotes, recommended
  for high ε) — the gradient diagnostics that drive the detector are
  computed the same way; the selector penalty is unchanged.
- `simultaneous_harmonics=True` (ISOFIT mode, Ciambur 2015 joint fit) —
  the harmonic amplitudes come out of a larger solver but the "largest
  amplitude" selector is the same, so the penalty slot is unchanged.

This is validated empirically in §5.

### Output keys

Both features only *add* new result-dict keys. They never modify the
per-isophote schema shape, so downstream consumers that don't know about
them keep working. New keys:

- `results["lsb_auto_lock"]` — bool, whether the detector fired.
- `results["lsb_auto_lock_sma"]` — commit sma, or None.
- `results["lsb_auto_lock_anchor_sma"]` — sma of the true geometry
  anchor (the last clean isophote before the debounce streak, whose
  geometry the lock froze to), or None. The per-row
  `lsb_auto_lock_anchor` marker goes on the committing trigger instead.
- `results["lsb_auto_lock_count"]` — locked tail length.
- `results["outer_reg_x0_ref"], ["outer_reg_y0_ref"],
  ["outer_reg_eps_ref"], ["outer_reg_pa_ref"]` — frozen reference
  geometry (only when `use_outer_center_regularization=True`).
  eps/pa keys were added alongside the `outer_reg_weights` extension;
  they are always populated for downstream QA even when the user
  leaves `weights["eps"] = weights["pa"] = 0`.
- Per-isophote: `iso["lsb_locked"]` and `iso["lsb_auto_lock_anchor"]`.

## 5. Empirical validation (HSC edge cases)

The full benchmark is
`examples/example_hsc_edgecases/run_lsb_mode_sweep.py`, summarized into
`outputs/example_hsc_edgecases/lsb_mode_sweep/_summary.md`. It crosses
three feature arms — `baseline` (free fit), `A` (`lsb_auto_lock=True`),
`B` (A + `use_outer_center_regularization=True` at strength 2.0) — with
three sampling/harmonic modes — `std`, `ea`, `isofit` — over 6 HSC
edge-case galaxies.

### Pre-lock combined drift (pixels)

| galaxy     | baseline | A_std | A_ea | A_isofit | B_std | B_ea | B_isofit |
|------------|---------:|------:|-----:|---------:|------:|-----:|---------:|
| 10140088   |     2.53 |  2.53 | 3.69 |     3.18 |  0.08 | 0.08 |     0.09 |
| 10140002   |     5.52 |  5.52 | 6.72 |    15.23 |  0.10 | 0.07 |     0.10 |
| 10140006   |     7.62 |  7.62 | 6.36 |     4.41 |  0.11 | 0.11 |     0.11 |
| 10140009   |     8.13 |  7.78 | 7.66 |     5.24 |  0.12 | 0.14 |     0.12 |
| 10140056   |     1.71 |  1.71 | 0.88 |     1.49 |  0.16 | 0.20 |     0.13 |
| 10140093   |     4.96 |  2.04 | 1.74 |     4.35 |  0.15 | 0.15 |     0.16 |

Observations:

- **A alone does not meaningfully reduce pre-lock drift** — that's expected,
  because A is a hard post-lock and the pre-lock region is still free.
- **B reduces pre-lock drift to 0.07-0.20 px on every galaxy and every
  mode** — a 30-50× reduction. The reduction ratio is consistent across
  std/EA/ISOFIT, confirming the feature is mode-agnostic.
- `A_isofit` on `10140002` is the dramatic regression case: pre-lock drift
  *triples* (5.52 → 15.23 px) when ISOFIT's higher-order solvers amplify
  noisy geometry updates in the LSB regime. `B_isofit` on the same galaxy
  is 0.10 px. The outer-reg penalty successfully contains what would
  otherwise be a runaway regression.

### Runtime (seconds per galaxy, total over 6 galaxies)

| arm          |  total (s) | per galaxy |
|--------------|-----------:|-----------:|
| baseline_std |        2.0 |       0.33 |
| A_std        |        2.1 |       0.35 |
| A_ea         |        2.2 |       0.37 |
| A_isofit     |        2.3 |       0.38 |
| B_std        |        2.1 |       0.35 |
| B_ea         |        2.2 |       0.37 |
| B_isofit     |        2.4 |       0.40 |

Observations:

- `lsb_auto_lock` itself adds ~5% runtime (B vs A is ~0%).
- `use_eccentric_anomaly=True` adds ~5-10%.
- `simultaneous_harmonics=True` adds ~10-20%.
- Combining all features adds ~20% over the baseline free fit — acceptable
  for the drift-reduction benefit.

### 5.1 Two-mode Tikhonov validation — HSC edge-real BCG `37498869835124888`

Outer-region metrics (sma ≥ 30 px) on the `37498869835124888` cluster
BCG. `pl_comb` is the pre-anchor combined center drift (max over x and
y); sat_Δ counts single-iteration transitions at ≥ 90% of their clip
limit (`clip_max_eps=0.1`, `clip_max_pa=0.5 rad ≈ 28.6°`).

| config                                    | med (s) | sc=2 | pl_comb | sat_Δeps | sat_ΔPA | max\|Δpa\| | eps range  | PA range (°) |
|-------------------------------------------|--------:|-----:|--------:|---------:|--------:|-----------:|-----------:|--------------:|
| baseline (no outer-reg)                   |   0.25  |   0  | 91.42   |    0     |    0    |   6.29°    | 0.19–0.53  | 50.4–69.1     |
| selector center-only (historical default) |   0.20  |   0  |  0.00   |    3     |    1    |  26.85°    | 0.21–0.41  | 40.8–67.6     |
| selector full weights                     |   0.21  |   0  |  0.00   |    1     |    0    |  13.74°    | 0.21–0.31  | 53.9–67.6     |
| **`damping` + `geomconv` (new default)**  | **0.27**|  **0**| **0.00**|  **0**   |  **0**  |  **1.42°** | **0.20–0.27** | **67.1–68.5** |
| `solver` + `geomconv`                     |   0.26  |   0  |  0.05   |    0     |    0    |   0.00°    | 0.17–0.21  | 61.4–61.4     |

(`sc=2` = count of outer isophotes that hit `maxit`. Selector rows
are shown for historical context: the `"selector"` value was removed
from `outer_reg_mode` in the same change that introduced the two
solver-level modes.)

Observations:

- **Baseline has meaningful astrophysical content out to sma ≈ 80 px**
  (b/a walks 1.0 → 0.5, PA walks 0° → 65°) then degrades into
  noise-driven drift (combined center drift reaches 91 px in the
  outermost isophotes).
- **Selector form (any weights)** produces visible saturated-step
  jumps in the 1-D profiles. Center is well-anchored (pl_comb ≈ 0)
  but eps/PA are quantized.
- **`damping` mode preserves the inner walk** (b/a down to ~0.75, PA
  to ~67°) and clamps the outermost noise-driven divergence. No
  saturated jumps; max |Δpa| = 1.42° in the outer is smoother than
  baseline's 6.29°.
- **`solver` mode flattens outer structure** toward the inner
  reference (b/a ≈ 0.21, PA ≈ 61° for all outer isophotes).
  Appropriate when you want outer isophotes anchored to the inner
  shape; not appropriate when you want the data-preferred walk.

Runtime column: auto-enabled `geometry_convergence=True` keeps both
solver-level modes within ~7% of the free-fit baseline. Without
geomconv, solver mode runs to maxit on ~26 isophotes (0.36s → 5.86s
at maxit=2000 with **no change to recorded geometry** — pure cost).

**Maxit sensitivity.** For `damping + geomconv`, maxit=50 vs
maxit=500 give bit-identical geometry at the same runtime. Raising
maxit is a no-op: geometry genuinely converges before 50 iterations
in every case. For `solver + geomconv`, same — geometry stabilizes
at the Tikhonov balance point by iteration ~10 on each outer
isophote.

Benchmark script:
`examples/example_hsc_edge_real/run_outer_reg_param_sweep.py` (arms
include `damping_recommended`, `solver_center`, `solver_full`,
`solver_strong`, `solver_full_noselector` among others). Summary
CSV/MD at `outputs/example_hsc_edge_real/outer_reg_param_sweep/`.

## 6. Implementation, file by file

### `isoster/config.py`

Section 12 (outer center regularization) and section 13 (LSB auto-lock)
define the config fields. Key pieces:

- Section 12 fields:
  - `use_outer_center_regularization: bool = False`
  - `outer_reg_sma_onset: float`
  - `outer_reg_sma_width: float`
  - `outer_reg_strength: float = 2.0` (new default; was 1.0 earlier in the
    branch)
  - `outer_reg_ref_sma_factor: float` — filters inward candidates when
    building the flux-weighted reference.
  - `outer_reg_weights: dict = {"center": 1.0, "eps": 1.0, "pa": 1.0}` —
    per-axis weights for the penalty. The default damps center,
    ellipticity, and PA together. Set an axis to `0.0` only when it
    should intentionally remain undamped.
  - `outer_reg_mode: str = "damping"` — solver-level variant. Accepts
    `"damping"` (default, step shrink only) or `"solver"` (full
    Tikhonov with ref pull). The historical `"selector"` value was
    removed; see §2.4. The selector-level penalty has no config field
    of its own; it is always active when the feature is enabled.

- Section 13 fields:
  - `lsb_auto_lock: bool = False`
  - `lsb_auto_lock_maxgerr: float = 0.3`
  - `lsb_auto_lock_debounce: int = 2`
  - `lsb_auto_lock_integrator: str = "median"`

- Validators in the section 12/13 block:
  - `outer_reg_sma_onset < sma0` → warn.
  - `use_outer_center_regularization=True` with `minsma >= sma0` → warn
    (inner reference will fall back to the anchor).
  - `fix_center=True` with `outer_reg_weights["center"] > 0` → warn
    (center penalty is inert; refined from the earlier blanket
    `fix_center` warning).
  - `fix_eps=True` with `outer_reg_weights["eps"] > 0` → warn.
  - `fix_pa=True` with `outer_reg_weights["pa"] > 0` → warn.
  - `use_outer_center_regularization=True` with all weights zero → warn
    (feature is a no-op).
  - `use_outer_center_regularization=True` with `simultaneous_harmonics=True`
    → warn (Tikhonov term is not wired into the ISOFIT 7-parameter
    joint solver; selector layer still applies if enabled).
  - `use_outer_center_regularization=True` with `geometry_convergence=False`
    → silently flips `geometry_convergence=True` and warns (required
    for either solver-level mode to finish cleanly; otherwise outer
    isophotes hit maxit with zero geometry change).
  - `lsb_auto_lock=True` with any of `fix_center/fix_pa/fix_eps=True` →
    raises, since the lock would conflict with prescribed geometry.
  - `lsb_auto_lock=True` with `debug=False` → silently flips `debug=True`
    (gradient diagnostics are required for the detector) and warns.

### `isoster/driver.py`

The driver owns the detector, the lock commit, and the reference-building
helpers. Key symbols:

- `_is_lsb_isophote(iso, maxgerr_thresh)` — the detector.
- `_build_locked_cfg(cfg, anchor_iso, locked_integrator)` — clones the
  config and freezes geometry for the locked tail.
- `_mark_lsb_lock_state(iso, locked, is_anchor=False)` — single write site
  for `lsb_locked` / `lsb_auto_lock_anchor`, so future loop reorderings
  cannot accidentally mutate the anchor dict twice.
- `_build_outer_reference(inwards_results, anchor_iso, cfg)` — the
  flux-weighted inner reference builder (renamed from
  `_build_outer_center_reference`). Returns a full geom dict
  `{x0, y0, eps, pa}`; pa uses a circular mean on `2·pa`.

In `fit_image` (the regular-mode path is inline in the driver):

1. After the inward loop, compute `outer_ref_geom` via
   `_build_outer_reference`. Pass it into the outward fit as
   `outer_reference_geom=outer_ref_geom`; which fields feed the penalty
   is controlled by `cfg.outer_reg_weights`.
2. Initialize `lsb_state = {"locked": False, "consec": 0, "transition_sma": None}`.
3. Mark the anchor with `_mark_lsb_lock_state(anchor_iso, locked=False)`.
4. In the outward loop, after each successful isophote:
   - `_mark_lsb_lock_state(next_iso, locked=lsb_state["locked"])`.
   - If not yet locked, run `_is_lsb_isophote`; increment or reset the
     debounce counter.
   - When the counter hits `lsb_auto_lock_debounce`, pick the anchor
     `debounce` isophotes back, build the locked clone, re-mark the trigger
     isophote with `is_anchor=True`, and switch the loop to the locked
     clone.
5. After the outward loop finishes, write result-dict keys:
   - `result["lsb_auto_lock"] = True`
   - `result["lsb_auto_lock_sma"] = lsb_state["transition_sma"]`
   - `result["lsb_auto_lock_count"] = sum(1 for iso in ... if iso.get("lsb_locked"))`
   - `result["outer_reg_x0_ref"] = outer_ref_x0` (when enabled)
   - `result["outer_reg_y0_ref"] = outer_ref_y0` (when enabled)

The forced-photo driver (`_fit_image_template_forced`) early-returns
before any of the above, and emits a `UserWarning` when either feature
flag is set together with a template.

### `isoster/fitting.py`

The per-isophote fitter owns both the selector-layer penalty and the
solver-level Tikhonov blend.

- `compute_outer_center_regularization_penalty(current_geom,
  reference_geom, sma, config)` — the selector-level penalty. Returns
  0.0 when the feature is off, when `reference_geom is None`, or when
  `λ(sma) < 1e-6`.

- `_tikhonov_alpha(coeff, lambda_sma, weight)` — closed-form blend
  fraction for the solver-level Tikhonov update. See §2.4 for the
  math. Returns `α ∈ [0, 1)`: 0 means fully harmonic-driven (no reg
  effect on the step), 1 means fully pulled to reference (reachable
  only in the limit of vanishing gradient or `λ·w → ∞`).

- `fit_isophote` hoists the solver-level setup out of the iteration
  loop:

  ```python
  outer_solver_on = (
      cfg.use_outer_center_regularization
      and cfg.outer_reg_mode in ("damping", "solver")
      and outer_reference_geom is not None
      and not simultaneous_harmonics
  )
  outer_solver_use_pull = cfg.outer_reg_mode == "solver"
  outer_solver_lambda = strength / (1 + exp(-(sma - onset) / width))
  ```

  Each of the four coord-descent update branches (A1, B1, A2, B2) and
  the simultaneous branch computes its parameter-specific Jacobian
  coefficient, calls `_tikhonov_alpha`, and blends:

  ```python
  if outer_solver_on and w_param > 0:
      alpha = _tikhonov_alpha(coeff, lambda_sma, w_param)
      if outer_solver_use_pull:   # mode = "solver"
          step = (1 - alpha) * step_harmonic - alpha * (param - param_ref)
      else:                        # mode = "damping"
          step = (1 - alpha) * step_harmonic
  ```

  Clipping (`clip_max_*`) is applied to the blended step, so the
  safety bounds still fire if a catastrophic proposed update sneaks
  through.

- Selector integration site stays as before:

  ```python
  outer_reg_penalty = compute_outer_center_regularization_penalty(
      current_geom, outer_reference_geom, sma, cfg
  )
  effective_amp = abs(max_amp) + reg_penalty + outer_reg_penalty
  ```

  The selector penalty still feeds the best-iteration picker; it does
  not enter the convergence criterion (`|max_amp| < conver·rms`),
  which uses the raw harmonic amplitude. Note that `effective_amp` is
  a deliberate heuristic that mixes units: `abs(max_amp)` is a
  harmonic amplitude in intensity units while the two penalty terms
  are squared geometry offsets (px², rad²), so the relative
  weighting of the terms depends on the image's flux scale.

### `tests/integration/test_lsb_auto_lock.py`

Integration test coverage for the auto-lock: detector trigger, debounce,
anchor mutation, marker keys, result-dict keys, forced-photo guard,
compatibility with existing modes. Sibling test
`test_outer_center_regularization.py` covers the soft pre-lock: penalty
shape, reference building, fallback on `minsma >= sma0`, `fix_center`
warning, forced-photo guard.

The full suite is 951 passing, 1 skipped (5 deselected) as of 2026-08-21; regenerate with `uv run pytest tests/ --collect-only -q | tail -2` rather than trusting this figure.

## 7. Future work

- **Dynamic reference update when the lock is late.** If the auto-lock
  commits far past the `outer_reg_sma_onset`, the frozen reference is
  measurably "stale" and the penalty is resisting the locked geometry. An
  option to re-evaluate the reference at the lock anchor is worth
  investigating for deep DES/LSST fits.
- **Penalty form ablation for DES/LSST.** The `"absolute"` form was picked
  because it dominated the `"normalized"` form at equivalent strength on
  HSC edge cases. A deeper,
  lower-S/N survey might invert the ordering; the ablation should be
  repeated when that dataset lands.
- **Auto-tuned `outer_reg_strength`.** The `strength=2.0` default was
  chosen from the HSC sweep. A scale-free auto-tune — e.g. set strength
  so the saturated penalty matches the median pre-lock |max_amp| in the
  outer half of the inward loop — would remove one more user knob.
