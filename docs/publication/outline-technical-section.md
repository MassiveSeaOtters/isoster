# Technical section outline — ISOSTER paper

Working outline for the technical / method section of the ISOSTER paper.
Two emphases, as requested:

1. **Performance and speed-up** relative to `photutils.isophote` (the
   direct Python predecessor) and, secondarily, to AutoProf.
2. **New features** not present in `photutils.isophote` (and where
   relevant, comparisons against Ciambur 2015 IRAF `Isofit` and
   Stone+ 2021 `AutoProf`).

## Framing decisions (resolved 2026-05-13)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Paper tone | Method paper, heavy benchmarking |
| 2 | Code in technical section | One canonical-call listing in §X.2; rest in appendix |
| 3 | "Where photutils is slow" detail level | Loops vs vectors abstraction (no specific function names) |
| 4 | AutoProf cross-validation placement | Brief in §X.3.4 (correctness anchor), depth in §X.5 (head-to-head comparison) |
| 5 | Running real-galaxy example for §X.4.3 | None in this paper; benchmarks are mocks from `isophote_test/` |
| 6 | Multi-band scope | Include in depth in this paper; remove "experimental" label |
| 7 | Limitations additions | Mask / variance-map quality; scattered light / wide-PSF; GPU roadmap |

### Scope refinement (resolved 2026-05-13, second pass)

The technical section is **explain + demo only**. No benchmark
numbers, no aggregate metrics, no cross-tool tables. Those live in
separate companion sections that draw from the
`isophote_test/` and `sga_isoster/` campaigns (Huang2013,
S4G, and SGA2020 — covered in their own chapters).

| # | Decision | Choice |
|---|----------|--------|
| 8 | §X.3 framing | Algorithmic-reasons-only: "Why ISOSTER is fast" with no numbers. Cite benchmark chapters by reference. |
| 9 | Demonstration cases | One inline demo per §X.4.* feature subsection. Cherry-picked but honest. |
| 10 | Demo depth | One figure (1–3 panels) + 150–250 words of explanation + a short `IsosterConfig(...)` recipe per demo. |
| 11 | Demo data mix | Mocks (from `isophote_test/`) where ground truth is needed to make the argument; real galaxies (from `sga_isoster/`) where the point is handling messy data. |

External resources confirmed:

- **`isophote_test/`** (`/Users/shuang/Dropbox/work/project/otters/isophote_test`)
  — MockGal: Sersic + Ferrer + PSF-convolved nucleus renderer with
  three backends (`libprofit` preferred, `astropy` fallback, `GALFIT`
  reference). Inputs under `inputs/huang2013/`, `inputs/cs4g/`
  (Spitzer S4G mocks), `inputs/demos/`. **Source of mocks for
  ground-truth-required demos in §X.4.1, §X.4.3, §X.4.5.**
- **`sga_isoster/`** (`/Users/shuang/Dropbox/work/project/otters/sga_isoster`)
  — 2,000-galaxy SGA2020 / LegacySurvey campaign. `data/demo/`
  carries a small curated subset for the paper's demonstration
  figures. **Source of real-galaxy illustrations for §X.4.2
  (WLS error bars) and §X.4.4 (batch robustness on messy data).**

### Per-feature demo plan (preliminary)

| Subsection | Demo type | Source | What it shows |
|---|---|---|---|
| §X.4.1 EA + ISOFIT | Mock | `isophote_test/inputs/demos/` synthetic edge-on disk | φ-sampling under-resolves major axis on `eps > 0.6`; switching to ψ recovers higher-order shape coefficients. |
| §X.4.2 Variance-aware fitting | Real | `sga_isoster/data/demo/` LegacySurvey galaxy with invvar map | Outer-isophote error bars differ meaningfully between OLS and WLS on real survey data. |
| §X.4.3 LSB outskirt strategy | Mock | `isophote_test/inputs/demos/` deep mock with contaminant | Soft + hard composition converges to the correct center vs free-fit drift. |
| §X.4.4 Batch robustness | Real | `sga_isoster/data/demo/` small failure-mode set | `max_retry_first_isophote` recovers anchor where the default would silently fall back to central pixel. |
| §X.4.5 Multi-band joint fit | Mock | `isophote_test/inputs/demos/` color-gradient mock | Joint free fit recovers per-band intensities that forced photometry biases. |
| §X.4.6 Diagnostics / QA | (no separate demo) | — | Stop-code semantics demonstrated implicitly across earlier demos. |

Notation in this file:

- `[BENCH: …]` — a number / plot we have or need from the benchmark
  campaigns under `benchmarks/`.
- `[CITE: …]` — a reference paper to cite.
- `[FIG: …]` — a figure to produce or reuse from existing QA output.
- `[OPEN: …]` — an open question for the user to decide before this
  section is written in full.

---

## §X. ISOSTER: design, performance, and features

Short opening paragraph (~1–2 sentences) restating the goal of the
software: an accelerated, configurable, scriptable Python pipeline
for elliptical-isophote analysis of resolved galaxy images, that
preserves the Jedrzejewski 1987 / Ciambur 2015 algorithmic family
while removing its scaling and feature-extensibility bottlenecks.

Tone: **method paper, heavy benchmarking**. §X.3 (performance) is the
single longest subsection but the paper's identity is "a new code for
elliptical isophote fitting" — closer to Ciambur 2015 / Stone+ 2021 in
shape than to a pure code-comparison paper.

### §X.1 Algorithmic foundation (brief)

Two-paragraph review of the classical formulation — enough that the
later subsections can refer back to it without re-deriving anything.

- Elliptical-isophote fitting as a 1-D Fourier-harmonic problem on
  resampled radial profiles [CITE: Jedrzejewski 1987].
- The two angular conventions (position angle φ vs eccentric anomaly
  ψ) and when each is the right basis [CITE: Ciambur 2015].
- Pointer forward to figures: image-level pipeline diagram, per-SMA
  fit-loop diagram. [FIG: reuse the two top diagrams from
  `docs/publication/html/algorithm.html`; export to static SVG.]
- Mention the canonical existing implementations: IRAF `Ellipse`,
  `photutils.isophote`, `Isofit`/`Cmodel`, `AutoProf`. [CITE: all four.]

### §X.2 Implementation overview

Compact map of the codebase as it relates to the paper. Probably
~½ page, mostly so the reader knows what to call when reproducing.

- `isoster.fit_image` as the single user entry point; mode-selection
  table (regular vs template-forced).
- Per-isophote layer (`isoster/fitting.py`), sampling layer
  (`isoster/sampling.py`), driver layer (`isoster/driver.py`).
- Vectorized core + optional Numba kernels; pure-NumPy fallback for
  reproducibility on systems without LLVM.
- Output schema: per-isophote dicts → FITS / ASDF / astropy.table
  serializers. 4-HDU FITS layout (`PrimaryHDU` / `ISOPHOTES` / `META` /
  `CONFIG` BinTableHDUs) [JUSTIFY: avoiding HIERARCH warnings].

**One canonical-call code listing** (≤10 lines) goes in §X.2 so the
reader has a mental model of the API:

```python
from isoster import fit_image, isophote_results_to_fits
from isoster.config import IsosterConfig

config = IsosterConfig(sma0=10.0, maxsma=200.0, full_photometry=True)
results = fit_image(image, mask=mask, config=config, variance_map=var)
isophote_results_to_fits(results, "galaxy_isophotes.fits")
```

All longer code examples (feature ablations, multi-band joint fit,
batch-processing patterns) live in an appendix.

---

## §X.3 Why ISOSTER is fast (algorithmic)

Tagline: "Four algorithmic choices that the Python predecessors do
not jointly make: end-to-end vectorized sampling, Numba-accelerated
linear-system kernels, lazy gradient evaluation, and a shared
sampling stack between fitting / forced extraction / model
reconstruction."

This subsection is **algorithmic only** — no benchmark numbers, no
cross-tool tables. Concrete speedups are quantified in the separate
benchmark chapters that draw from the Huang2013, S4G, and SGA2020
campaigns (`isophote_test/`, `sga_isoster/`); §X.3 cites them by
reference.

### §X.3.1 Where the time was being spent

A page or so explaining what made the predecessor slow, in
software-engineering terms an astronomer can follow.

**Level of detail: "loops vs vectors" abstraction**, not specific
function names. This keeps the comparison robust to upstream
refactors and avoids a citation that ages out within a release
cycle. Three bottleneck classes:

- Per-pixel sector integration in Python-level loops: each ellipse
  point is approximated by a small rectangular area integral,
  evaluated one isophote point at a time. Useful pedagogy and
  accuracy, but Python-loop heavy.
- Re-evaluating the radial gradient every iteration with no caching,
  even when the geometry has barely changed between iterations.
- Single-purpose sampling: independent code paths for fitting,
  forced-photometry extraction, and 2-D model reconstruction.

### §X.3.2 What ISOSTER does differently

Bulleted technical claims, each backed by a benchmark.

1. **Vectorized path-based sampling.** Replace the per-isophote
   `Integrator` with a single `scipy.ndimage.map_coordinates` call
   per isophote, with `n_samples = max(64, 2π·sma)`. Subpixel
   interpolation is done once per isophote, on the full path, in C.
   [BENCH: sampling-time microbenchmark; isolate from harmonic-fit
   time. Probably `benchmarks/profile/…`.]

2. **Numba-accelerated inner kernels.** Harmonic least-squares
   design-matrix builders (`_build_first_second_harmonics_matrix`,
   `build_joint_design_matrix*` for multi-band), gradient assembly,
   and the dominant-coefficient selector. NumPy fallback is kept
   byte-equivalent.
   [BENCH: with-/without-Numba comparison; report both warm and
   cold-start (first-call JIT cost).]

3. **Lazy gradient evaluation (Modified Newton).** Gradient computed
   once per isophote on iteration 0 and reused unless convergence
   stalls for three iterations (then re-evaluated). RETRACTED: the ~45% reduction
   in sampling calls.
   [BENCH: per-isophote iteration count and total sampling-call
   counts on Huang2013 (or wherever the 45% number was first
   reported). Confirm the campaign name.]

4. **Inward-first driver order with a single anchor.** No redundant
   re-fitting at `sma0`; the inner reference for outer-region
   regularization is built without an extra pass.

5. **Shared 1-D sampling stack** between regular fitting, forced
   extraction, and 2-D model reconstruction
   (`build_isoster_model`). One vectorized code path for what was
   three independent paths in the predecessor.

### §X.3.3 Algorithmic complexity and where it goes

A short subsection on what determines fitting cost as image size and
SMA grid density change, framed in terms of operation counts rather
than wall-clock seconds:

- Per-isophote cost: $O(N)$ harmonic-fit + $O(N)$ sampling where
  $N = \max(64, 2\pi a)$.
- Image-level cost: dominated by the outward loop over $K_\mathrm{out}$
  isophotes, each at most `maxit` iterations; total worst-case
  $O(K_\mathrm{out} \cdot \mathtt{maxit} \cdot N_\max)$.
- Lazy gradient turns the typical case from $O(3 N \cdot \mathtt{maxit})$
  per isophote to $O(N \cdot (\mathtt{maxit} + 2))$.
- Image area enters only through `map_coordinates` interpolation
  cost, which is sublinear in the image because each isophote samples
  $O(N)$ points, not $O(\mathrm{image~area})$.

[FIG: optional — a small figure showing where time goes in a typical
fit (sampling / linear system / geometry update / overhead). Pulled
from a profiler trace, no aggregate speedup claim. May be moved to
the benchmark chapter; placeholder here.]

### §X.3.4 Correctness guarantees alongside speed

A short paragraph defending the claim that speed is not bought at
the price of accuracy. Specific cross-validation *numbers* go in
the benchmark chapters; this subsection establishes the
*guarantees*:

- OLS path byte-identical when `variance_map=None`. Specific test
  in `tests/` that pins this.
- Round-trip test: write FITS / ASDF → read back → run-to-run
  identical.
- The Pydantic-validated config means a fit can be re-executed from
  the saved `CONFIG` HDU alone, with no implicit "default-value
  drift" between runs.

---

## §X.4 New features

Tagline: "Features that did not exist (or required separate tools)
in `photutils.isophote`, presented in approximate order of
scientific impact."

### §X.4.1 Eccentric-anomaly sampling and ISOFIT-style joint harmonics

- Faithful reimplementation of Ciambur 2015 inside the same Python
  call. Two flags: `use_eccentric_anomaly` and `simultaneous_harmonics`;
  `isofit_mode={'in_loop', 'original'}` for the two ISOFIT variants.
- Vectorized extended design matrix `[1, sin θ, cos θ, sin 2θ, cos 2θ,
  sin n₁θ, cos n₁θ, …]` solved once per iteration; ~25–35% overhead
  vs the 5-parameter default path.
- Normalized harmonic coefficients in all reported columns, but note the
  expression differs by path: single-band `compute_deviations` stores
  `a_n = A_n_raw / (a · |dI/da|)` at fit time (absolute gradient), while
  the plot-time helper used by the multi-band raw modes applies the
  signed Bender form `−A_n_raw / (a · dI/da)`. They agree wherever the
  gradient is negative [cite Bender et al. 1989 if we cite the
  normalization anywhere].
- [FIG: stylized comparison of φ-sampled vs ψ-sampled harmonic
  residuals on an edge-on disk — pull from the QA gallery or
  `examples/example_edge_on/`.]

### §X.4.2 Variance-aware fitting (WLS)

- `variance_map` argument; exact (`AᵀWA`)⁻¹ covariance, no
  residual-variance rescaling.
- Cosmic-ray and hot-pixel handling for free via the variance map.
- Per-pixel gradient error decoupled from galaxy-structure scatter.
- [FIG: error-bar comparison plot for an HSC or DECaLS galaxy: OLS
  vs WLS; show the systematic underestimate of OLS errors in the
  outskirts.]
- [CITE: pointer to `docs/01-user-guide.md` "Using Variance Maps".]

### §X.4.3 Low-surface-brightness outskirt strategy

Two new mechanisms, presented together because they are designed to
compose.

- **Outer-region Tikhonov center regularization** (soft, ramped).
  Logistic ramp `λ(sma) = strength / (1 + exp(-(sma - onset)/width))`
  damps the geometry update toward a flux-weighted inner reference
  built from the inward isophotes. Both a "damping" mode and a
  "solver" mode.
  [CITE: `docs/08-outer-regularization.md` for the math &
  benchmarks.]
- **Automatic LSB geometry lock** (hard, one-way). Detector reads
  `grad_r_error` and `stop_code` from each outward isophote;
  commits a center/PA/eps freeze after a debounced trigger; the
  anchor is the isophote immediately *before* the streak.
  [CITE: `docs/07-lsb-features.md`.]
- Composability: soft pre-lock, hard post-lock; either can run alone.
- [FIG: μ–sma profile of a **mock** edge case from
  `isophote_test/inputs/demos/` (or a purpose-built deep-LSB mock)
  showing the three outward growth modes overlaid
  (free / soft / soft+lock); plus a centroid-drift plot in the bottom
  panel. Mocks give us ground truth, which is what this figure needs
  to argue the soft mode does the right thing.]

No real-galaxy running example in this paper. A separate science
case-study paper will showcase isoster on real HSC edge cases.

### §X.4.4 Robustness primitives for batch and survey pipelines

A short subsection, since these are "operations-grade" features that
matter for the people running ISOSTER on thousands of galaxies but
are not exciting per-galaxy.

- `max_retry_first_isophote` (perturb `sma0`, `eps`, `pa` on anchor
  failure).
- `permissive_geometry` (do not poison subsequent SMAs after one
  bad fit).
- Adaptive integrator (`'mean'` inside, `'median'` outside an SMA
  threshold).
- Gradient-SNR-adaptive damping; per-iteration step clipping.
- `sigma_bg` noise floor in the convergence criterion.

[FIG: failure-rate-versus-mag plot from the Huang2013 campaign —
predecessor vs ISOSTER (defaults) vs ISOSTER (batch preset).]

### §X.4.5 Multi-band joint fitting

Full-depth subsection. Multi-band is presented as a first-class
isoster feature — the "experimental" framing comes off for the
paper. This is the largest single subsection inside §X.4 and likely
~2–3 pages in the typeset paper.

**Why a joint free fit, not forced photometry.** Forced photometry
applies a single-band template geometry to other bands. That is
biased when the bands have genuinely different morphology
(disk-bulge color gradients, dust lanes, age gradients). The joint
free fit instead solves for **one shared geometry per SMA with
per-band intensities and per-band higher-order harmonics** — the
right model when the user wants to measure color gradients without
imposing them.

**Subsection map:**

1. **Motivation and contrast with forced photometry.** Short prose
   + one figure showing forced-photometry bias on a synthetic
   color-gradient galaxy. [FIG: forced-vs-joint residual map.]
2. **The joint design matrix.** For `B` bands and `N` valid samples
   per SMA, the joint linear system has shape
   `(B·N × (B + 4))`: per-band intensity zero-points (`B` columns)
   plus the four shared geometry harmonic columns
   `[sin θ, cos θ, sin 2θ, cos 2θ]`. With shared higher harmonics of
   `L` orders, the matrix extends to `(B·N × (B + 4 + 2·L))`.
   [FIG: design-matrix schematic.]
3. **Four higher-harmonic modes** controlled by
   `multiband_higher_harmonics ∈ {'independent', 'shared',
   'simultaneous_in_loop', 'simultaneous_original'}`.
   `'independent'` reproduces band-by-band post-hoc harmonics;
   `'shared'` does one joint post-hoc fit across all bands;
   `'simultaneous_in_loop'` solves the wider matrix every
   iteration (full ISOFIT for multi-band); `'simultaneous_original'`
   does the wide solve once after convergence (Ciambur 2015
   variant). Same algorithmic ladder as the single-band §X.4.1.
4. **Loose-validity** for non-identical per-band masking. When
   `loose_validity=True`, the rectangular `(B, N)` validity layout
   is replaced by jagged per-band arrays, and the design matrix is
   built per-band via `build_joint_design_matrix_jagged*`. Useful
   when one band has a defect mask the others don't share.
5. **Per-band normalization at plotting time.** CORRECTED: `'shared'`
   does *not* produce band-identical raw `A_n`/`B_n`. It fits one
   dimensionless shape and writes it into each band scaled by that
   band's own gradient, so the raw columns are band-distinct and
   normalizing each by its own `dI/da_b` recovers the *same* value in
   every band. It is `simultaneous_*` that stores one identical raw
   amplitude across bands -- and that is why those modes are
   experimental: a shared raw residual is not a shared shape. Under
   `'independent'` (the default) the stored values are already
   normalized and are not touched at plot time.
6. **Output schema (Schema 1).** Per-band columns named
   `a<n>_<band>` / `b<n>_<band>` / `a<n>_<band>_err` /
   `b<n>_<band>_err`, plus the standard shared-geometry columns
   carried through. ASDF I/O via
   `isophote_results_mb_to_asdf` / `_from_asdf`.
7. **CLI**: parallel multi-band entry point
   `isoster-mb image_g.fits image_r.fits image_i.fits …`
   (independent from the single-band CLI; defined in
   `isoster/multiband/cli_mb.py`).
8. **Validation.** Single-band equivalence test:
   `fit_image_multiband([image], …)` returns the legacy single-band
   schema byte-identical to `fit_image(image, …)`. Cross-validation
   on multi-band Huang2013 mocks: per-band intensity profiles match
   independent single-band fits where the colors are truly uniform,
   and deviate predictably where colors differ. [BENCH: from
   `isophote_test/inputs/huang2013/` with multi-band rendering.]

[CITE: `docs/10-multiband.md`, `docs/04-architecture.md` for the
locked Section 6 modes.]

Implications for the rest of the paper:

- The §X.3 benchmark sweep gains a multi-band axis: time at
  `B = 1, 2, 4` bands, with `'independent'` vs
  `'simultaneous_in_loop'` modes.
- The §X.5 feature matrix now includes a "joint multi-band free
  fit" row, which only isoster claims.
- A new appendix may be needed for the Schema 1 column-name
  reference if `docs/10-multiband.md` content does not transfer
  cleanly into the paper format.

### §X.4.6 Diagnostics, QA, and reproducibility infrastructure

Smallest of the new-feature subsections; mostly a paragraph + a
figure.

- Stop-code semantics: `0/1/2/3/-1` with documented triggers;
  carried through to all output schemas.
- QA plotting (`plot_qa_summary`, `plot_comparison_qa_figure`) with
  cross-tool comparison (`isoster` / `photutils` / `autoprof`).
  CORRECTED: `isoster` and `photutils` share the normalization
  convention; whether `autoprof` does is **unverified** and its
  harmonics are excluded from published comparisons until an
  AutoProf run against a planted deviation settles it.
- 4-HDU FITS (PRIMARY / ISOPHOTES / CONFIG / META), ASDF, and
  astropy-table serialization; the `CONFIG` HDU records the resolved
  configuration so a fit can be re-run from it. Bit-for-bit
  reproduction is not claimed and has not been demonstrated.
- Exhausted-benchmark framework: tool-agnostic composite scoring
  for ablation studies. [CITE: `docs/09-exhausted-benchmark.md`.]

---

## §X.5 Comparison to existing tools

A short, fair comparison subsection (not a sales pitch). Two paragraphs
plus a table.

- **vs `photutils.isophote`**: feature parity + everything in §X.4;
  speed advantage from §X.3; identical OLS path when no variance map.
- **vs `AutoProf` (Stone+ 2021)**: similar scope (modern Python, LSB
  push). Different design choices: AutoProf uses ML-style fit
  stabilization; ISOSTER uses an explicit logistic-Tikhonov ramp +
  one-way lock state machine. Harmonic basis and normalization:
  **UNVERIFIED**, do not assert a difference. The benchmark adapter
  passes AutoProf's coefficients through unconverted (assuming they
  match) and nobody has tested that against a planted deviation; an
  earlier draft asserted a different basis, which was equally untested.
  Read AutoProf's paper §4 and, better, run the comparison before
  writing anything here. AutoProf advertises ≳2 mag
  arcsec⁻² deeper than predecessors; we should benchmark whether
  ISOSTER reaches the same depth.
- **vs IRAF `Isofit` / `Cmodel`**: same algorithm family (Ciambur
  2015), reimplemented in Python with vectorization, WLS, and the
  LSB infrastructure on top.

[FIG: feature-matrix table — rows = features, columns = tools,
cells = ✅/⚠️/❌. Pulled from the comparison table in
`docs/publication/html/decision-tree.html` plus the AutoProf paper.]

[BENCH: head-to-head ISOSTER-vs-AutoProf depth comparison on a
matched sample. Likely the exhausted-benchmark cross-tool score gives
this for free — confirm.]

---

## §X.6 Limitations and roadmap

Honest accounting paragraph plus a short roadmap.

**Current limitations:**

- 2-D images only; no IFU cube support. The pipeline assumes a
  single 2-D intensity array per band.
- Outer-region regularization defaults are still being validated
  beyond HSC; the recommended ranges in
  `docs/02-configuration-reference.md` should be treated as
  starting values, not survey-blind defaults.
- Numba is optional but its cold-start JIT cost is ~1 s; users
  running isoster on a handful of galaxies may not see the
  vectorized-kernel win. The pure-NumPy fallback is byte-equivalent
  but ~3–5× slower on hot kernels.
- **Quality of the user-supplied mask and variance map.** Isoster
  inherits whatever bias is in those two inputs and does not
  re-derive them from the image. Survey users coming from
  forced-photometry pipelines should treat mask construction
  (defect mask + companion mask + secondary-source mask) and
  variance-map construction (sky variance + Poisson term + read
  noise) as upstream responsibilities, not as something isoster
  fixes downstream.
- **Scattered light and wide-PSF wings.** Isoster does not
  deconvolve the PSF. It is therefore not the right tool for
  studying outskirts where extended PSF wings dominate (very
  nearby NGC galaxies, the inner few-percent halo of bright
  ellipticals, the wings of saturated stars near the target).
  Refer to specialized PSF-modeling work for those regimes.

**Roadmap:**

- **GPU acceleration of the hot kernels** for survey pipelines
  handling O(10⁶) targets. The vectorized design-matrix builders
  and `map_coordinates`-style sampling are natural candidates for
  CuPy / JAX. Marks isoster's growth path without overpromising a
  release date.
- IFU cube support as a longer-term extension; the per-isophote
  fit loop generalizes naturally to a per-(isophote, wavelength)
  loop but the sampling stack needs a 3-D path.
- Tighter PSF-aware modes for the small-galaxy regime where PSF
  wing contamination is non-negligible.

---

## Cross-references to other paper sections

To be filled in once the rest of the paper has section numbers:

- Reference to the §"Data and mocks" section for the Huang2013 sample.
- Reference to §"Applications / case studies" for HSC edge-case
  galaxies.
- Reference to appendix for the full config-parameter table.

---

## Figures and tables to commission

Tally of distinct deliverables this section needs:

| ID | Type | Source | Status |
|---|---|---|---|
| Fig X.1 | Pipeline diagram | `docs/publication/html/algorithm.html` (SVG export) | redrawable |
| Fig X.2 | Per-isophote loop diagram | same | redrawable |
| Fig X.3 | Speedup heatmap (image size × SMA density) | `benchmarks/profile/…` | needs benchmark |
| Fig X.4 | Feature-stack overhead bars | benchmark sweep | needs benchmark |
| Fig X.5 | EA vs φ harmonic residuals (edge-on disk) | QA gallery | reuse |
| Fig X.6 | OLS vs WLS error bars (HSC or DECaLS) | example | reuse |
| Fig X.7 | LSB strategy comparison (free / soft / soft+lock) | mock from `isophote_test/inputs/demos/` | needs mock + run |
| Fig X.8 | Failure-rate vs magnitude (Huang2013) | `isophote_test/inputs/huang2013/` | reuse |
| Fig X.9 | Multi-band forced-vs-joint residual map | new color-gradient mock from `isophote_test/` | needs mock + run |
| Fig X.10 | Multi-band joint design-matrix schematic | hand-drawn | new |
| Tbl X.1 | Feature matrix vs predecessors (incl. joint multi-band row) | hand-drawn | new |
| Tbl X.2 | Stop-code semantics | `docs/01-user-guide.md` | reuse |
| Tbl X.3 | `multiband_higher_harmonics` mode comparison | `docs/04-architecture.md` Section 6 | reuse |

---

## Open questions — resolved 2026-05-13

All seven framing questions from the v1 outline are answered in the
"Framing decisions" table at the top of this file. No outstanding
opens at the outline level.

Next questions naturally come at the per-subsection drafting level:

- §X.3 — exact set of mock galaxies, image sizes, and SMA grids for
  the benchmark sweep. Probably 12–16 cells total.
- §X.4.5 — which `multiband_higher_harmonics` modes to feature in
  the figures (likely just `'independent'` and
  `'simultaneous_in_loop'` for the main text, full ladder in an
  appendix).
- §X.5 — exact LSB-depth metric for the AutoProf head-to-head.
  Suggested: SMA at which `stop_code` first becomes ≠ 0 on a matched
  set of deep mocks.
