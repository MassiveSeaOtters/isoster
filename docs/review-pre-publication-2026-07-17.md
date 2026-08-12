# Pre-Publication Code and Documentation Review

**Date:** 2026-07-17
**Branch:** `review/pre-publication-audit`
**Scope:** Full read of the single-band package (`isoster/*.py`), canonical docs (`docs/00`–`11`, `SPEC.md`, `index.md`), repo hygiene; delegated full-file reviews of `isoster/plotting.py` (3 009 lines), `isoster/multiband/` (~8 000 lines), and `docs/technical/` + `docs/reference/`.
**Verification:** full test suite (654 passed, 5 deselected, 80 s), `example_basic_usage` end-to-end, targeted numerical experiments for every bug claim marked *confirmed*. Nothing was modified; this file is the only artifact.

**How to read the verdicts.**
- *confirmed* — reproduced numerically at runtime by the reviewer, or verified line-by-line against the code cited.
- *reported* — found by a delegated full-file review with precise line citations, plausible on inspection, but not independently re-run; treat as "very likely, verify while fixing".

---

## Executive summary

The codebase is in good shape overall: the core algorithm is carefully implemented, the test suite is extensive and green, and the configuration layer is unusually thorough. However, the review found:

- **One license contradiction that blocks publication** (B0).
- **Confirmed bugs in four areas with scientific impact:** harmonic coefficients (silently zeroed on clean data, B1; double-normalized in the main QA figure, P1), photometric error bars (sentinel-fragile formula, B2; mis-scaled multiband OLS errors, M2/M3), multiband option combinations that silently misbehave (M1, M4, M5), and FITS round-trip metadata corruption (B3).
- **A systematic documentation drift**: several docs describe an older parameter set, stale "Stage-1 scope" docstrings contradict the multiband code they head, and public docs reference untracked internal files. Most serious for the journal path: the `docs/technical/` 1.x series — which reads like the draft of the paper's technical chapter — contains **formula-level errors** (wrong interpolation order, a convergence criterion missing its scale factor, a Tikhonov formula with the wrong variable, WLS error formulas the code does not use, a wrong photutils basis claim; see Part V).

None of the bugs requires architectural change to fix. The single most valuable *design* decision to make before submission is one canonical harmonic-coefficient convention (stored vs. plotted), because three conventions currently coexist (see P1 and Question 6).

| # | Severity | Finding |
|---|----------|---------|
| B0 | **Blocker** | `LICENSE` file is GPL-2.0; all metadata declares BSD-3-Clause |
| B1 | High | `compute_deviations` silently returns `a_n=b_n=0` on noise-free/low-noise data |
| B2 | High | WLS `intens_err` (and `grad_error`) explode to ~1e12 with the driver's own variance sentinels |
| B3 | High | FITS round-trip corrupts `lsb_locked` / `lsb_auto_lock_anchor` (missing bools read back `True`) |
| P1 | High | `plot_qa_summary` harmonic panel divides *already-normalized* coefficients by `sma·\|grad\|` again |
| M1 | High | Multiband: `harmonic_combination='ref'` × `loose_validity=True` silently returns zero harmonics, or crashes |
| M4 | High | Multiband forced photometry never sigma-clips (single-band forced mode does) |
| B4 | Medium | Duplicated `except` clause in `compute_parameter_errors` (dead code) |
| B5 | Medium | Central-pixel dict hardcodes harmonic orders [3,4]; schema breaks for other `harmonic_orders` |
| M2 | Medium | Multiband OLS with non-unit `band_weights`: parameter errors mis-scaled by ~1/w |
| M5 | Medium | Multiband dropped bands: docs promise NaN harmonic columns, code writes 0.0 |
| M8 | Medium | Multiband `ndata`/`nflag` semantics break under loose validity; `nflag` goes negative |
| P3 | Medium | Comparison QA figure crashes (`KeyError`) on any method name not in `METHOD_STYLES` |
| V1 | High (docs) | `docs/technical/` formula-level errors: interpolation order, convergence scale, Tikhonov variable, WLS error formulas, photutils basis |

---

## Part I — Single-band package (reviewer-verified)

## B0 — License contradiction (publication blocker) — **FIXED 2026-07-18**

- `LICENSE` (339 lines, from the initial commit `c86bde6`) is the full **GNU GPL Version 2** text.
- `pyproject.toml:10` declares `license = "BSD-3-Clause"`.
- `README.md` shows a BSD-3 badge and states "BSD-3-Clause. See LICENSE for details."
- `CITATION.cff` declares `license: BSD-3-Clause`.

A reviewer or journal editor checking the repository will find a direct legal contradiction. Given the photutils-derived heritage (BSD-3) and all metadata, BSD-3-Clause is presumably intended and the `LICENSE` file is a template accident — but this must be fixed and made consistent everywhere before submission. **Question to the author: confirm the intended license.**

> **Status: FIXED (2026-07-18, branch `fix/pre-publication-review`).** The author confirmed BSD-3-Clause. `LICENSE` now contains the standard BSD-3-Clause text (Copyright (c) 2026, Song Huang), matching `pyproject.toml`, the README badge/section, and `CITATION.cff`. No other license-bearing file needed changes.
>
> **New related finding (out of B0's original scope, flagged for the author):** `reference/autoprof/LICENSE` is **GPL-3.0** text — the vendored AutoProf copy is GPL-licensed, unlike the BSD photutils copy alongside it. `/reference/` is gitignored-intended but its files are tracked. If the repo goes public for the journal, decide whether to untrack `reference/` entirely (the GPL copy especially) or document its dev-only status; shipping GPL-3.0 source inside a BSD-3 package is itself a license conflict.

## B1 — `compute_deviations` silently returns zero harmonics on clean data (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** The OLS branch now uses the explicit design-matrix solve (`np.linalg.lstsq` + inverse normal equations scaled by the residual variance) instead of `scipy.optimize.leastsq`, matching the WLS branch and `fit_higher_harmonics_simultaneous`. Planted `b4=+0.03` recovered exactly on noise-free data; noisy-data coefficients agree with the old `leastsq` result to ~1e-9. Regression tests: `tests/unit/test_fitting.py::test_compute_deviations_noise_free` and `tests/integration/test_isofit_integration.py::test_default_path_noise_free_recovers_b4`. Fit speed unchanged (median 0.039 s → 0.037 s on the standard 260×260 mock; checksum identical to 1e-9).

**Location:** `isoster/fitting.py:517-576` (OLS branch, lines 546-551).

**Mechanism.** The OLS branch fits the n-th harmonic with `scipy.optimize.leastsq(..., full_output=True)` and treats `cov_matrix is None` as total failure:

```python
solution = leastsq(residual, params_init, full_output=True)
coeffs = solution[0]          # <- correct even when cov is None
cov_matrix = solution[1]
if cov_matrix is None:
    return 0.0, 0.0, 0.0, 0.0 # <- silently discards good coefficients
```

`leastsq` returns `cov_x=None` when it converges essentially exactly — which happens systematically on **noise-free or very-low-noise data** (verified: exact synthetic profiles, uniform-angle sampling, tiny 1e-12 noise all trigger it; SNR=100 data does not). No warning is emitted; the returned `a{n}`, `b{n}` and their `_err` fields are exactly `0.0`, indistinguishable from a perfectly measured zero.

**Reproduction.** Noise-free mock with a planted 4th-order deviation `b4 = +0.03`:
- direct `compute_deviations` call → `a4=+0.00000, b4=+0.00000` while a manual projection of the same samples gives the expected cos(4φ) amplitude of 8.11 counts;
- end-to-end `fit_image` → `b4 = 0.0` on every converged isophote.

**Impact.** This is the *default* configuration path (`compute_deviations=True`, `simultaneous_harmonics=False`, no variance map). The affected data regime — noiseless mocks — is exactly what `docs/05-testing.md` recommends for validation and what `examples/example_basic_usage` produces. It also silently disables harmonic content in `build_isoster_model` (verified: harmonics-on and harmonics-off models become identical). Existing tests never catch it because all harmonic-recovery tests use noisy data (e.g. the "mild noise" comment in `tests/multiband/test_fitting_mb.py:2241`).

**Fix direction.** Drop the `leastsq` dependency in favor of the explicit design-matrix solve already used in the WLS branch and in `fit_higher_harmonics_simultaneous` (`lstsq` + `inv(AᵀA)·σ²_residual`). Failing that, return the valid coefficients with `NaN` errors and a warning when `cov_x` is `None`. Add a noise-free harmonic-recovery test (planted `b4`, assert non-zero recovery).

## B2 — WLS `intens_err` formula is inconsistent and sentinel-fragile (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** `fit_isophote`'s WLS `intens_err` now reads the fit's own covariance `(AᵀWA)⁻¹[0,0]` (exact for the reported weighted-fit intercept; `1/√Σ(1/σᵢ²)` fallback), and the `compute_gradient` mean errors use the inverse-variance form `1/Σ(1/σᵢ²)` — both are immune to the 1e30 sentinels *and* to sentinel values smeared into neighboring samples by bilinear interpolation (sentinel weight ≈ 0; a threshold filter was tried first and cannot work because interpolation produces arbitrarily large sub-sentinel values). New shared `VARIANCE_SENTINEL` constant in `_shared.py`. Reproduction (11 NaN pixels): `intens_err` 9.99e12 → 0.63 (clean-run value), spurious `stop_code=-1`s gone, results match a NaN-free run to ~3%. Regression tests: `TestVarianceSentinelRobustness` in `tests/unit/test_variance_map.py`. Fit speed unchanged (0.018 s vs 0.017 s pre-fix, WLS path). The multiband inheritance (M10) is handled as a separate follow-up.
>
> **Superseded (2026-08-12, branch `fix/gradient-error-ring-statistics`).** The `compute_gradient` claim above — that the inverse-variance-weighted-mean form `1/Σ(1/σᵢ²)` is what the mean errors use, and that this form is immune to the 1e30 sentinel — was true on 2026-07-17 and is no longer true. `compute_gradient` was subsequently changed to derive each ring's uncertainty from the statistic it actually reports (`Σv_i/N²` for the unweighted mean; the analogous median form), which does not have this immunity: a literal `1e30` variance entry is finite and strictly positive, so it is now valid input and is kept in the ring rather than dropped, inflating `grad_error` by many orders of magnitude. This entry is left as-is for the historical record; do not edit the status block above. See `docs/04-architecture.md`, section "Gradient error and ring statistics", for the current formulas and behavior.

**Location:** `isoster/fitting.py:1429-1430` (and the same `Var(mean)=Σσᵢ²/N²` pattern in `compute_gradient`, lines 871-880, 922-929).

**Two problems.**

1. **Inconsistency.** `fit_isophote` reports the WLS weighted-mean intensity (`y0_fit` from the weighted fit) but computes its error as `sqrt(Σσᵢ²/N²)` — the variance of the *unweighted* mean. `extract_forced_photometry` (fitting.py:240-244) uses the exact weighted-mean covariance `1/sqrt(Σ1/σᵢ²)`. Docs (`03-algorithm.md:108`, `04-architecture.md:127`) call the first formula "exact variance of the mean"; it is exact only for the unweighted estimator (equal only when all σᵢ are equal). The same mixed pattern exists in the multiband decoupled-WLS paths (see M10).
2. **Fragility.** `fit_image` itself converts `NaN`/`inf` variance values to a `1e30` sentinel (driver.py:452-471). One such pixel inside an annulus makes `Σσᵢ²/N² ≈ 1e30/N`, so `intens_err` explodes — **measured 2.9×10¹²** on a 15-NaN variance map — and the same sum in `grad_error` produces spurious `stop_code=-1` failures on neighboring isophotes. The exact-weighted formula used in forced photometry is immune to the sentinels (weight ≈ 0).

**Fix direction.** Use `1/sqrt(Σ 1/σᵢ²)` (exact for the weighted estimator) or the `(AᵀWA)⁻¹[0,0]` element from the fit's own covariance; optionally exclude sentinel-weight samples explicitly. Add a test with NaN-containing variance maps asserting finite, sane `intens_err`.

## B3 — FITS round-trip corrupts LSB-lock boolean fields (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** Uniform schema at the driver (the option chosen by the author): when `lsb_auto_lock=True`, `_mark_lsb_lock_state` now always writes both `lsb_locked` and `lsb_auto_lock_anchor`, and a post-assembly pass stamps `False` on inward isophotes and the central pixel — no masked bool cells ever reach FITS. Applied to both `driver.py` and `driver_mb.py`. The FITS readers (`utils.py`, `utils_mb.py`) now skip masked cells instead of leaking `np.ma.masked` into row dicts (verified: a heterogeneous `ndata` column round-trips as "key absent" again). Docs 01/02/04/10 updated to the uniform-schema description. Regression tests: `TestFitsRoundTrip::test_lock_flags_survive_fits_round_trip` in `tests/integration/test_lsb_auto_lock.py` and `test_fits_roundtrip_preserves_lsb_auto_lock_flags` in `tests/multiband/test_utils_mb.py` (exact per-row flag preservation, exactly one anchor). No fit-speed impact: the stamping is one O(n_isophotes) pass of dict writes, and only when the feature is on.

**Location:** `isoster/driver.py:107-116` (key writing) + `isoster/utils.py:86` (`Table(rows=...)`).

**Mechanism.** `lsb_locked` is written only on outward isophotes, and `lsb_auto_lock_anchor` on exactly one isophote (docs 01/02 even document "inward isophotes never carry these keys"). `Table(rows=list_of_dicts)` fills the missing cells as masked; FITS logical columns cannot represent masked values, so `numpy.ma` fills them with its default bool fill value — **`True`**.

**Reproduction.** A results dict with the documented key pattern written via `isophote_results_to_fits` and read back with `isophote_results_from_fits`:

```
sma=   2.0  lsb_locked=True      lsb_auto_lock_anchor=True   # inward: had NO keys
sma=   5.0  lsb_locked=True      lsb_auto_lock_anchor=True   # inward: had NO keys
sma= 10.0  lsb_locked=False     lsb_auto_lock_anchor=True
sma= 20.0  lsb_locked=True      lsb_auto_lock_anchor=True   # the only real anchor
sma= 40.0  lsb_locked=True      lsb_auto_lock_anchor=True
```

Every isophote reads back as an anchor; inward isophotes read back as locked. The LSB-lock metadata is effectively destroyed by a save/load cycle — including for template-based forced photometry, which loads geometry from these files. The multiband FITS reader has the same gap for heterogeneous rows (masked sentinels leak into row dicts, `utils_mb.py:172-186`).

**Fix direction.** Make the per-isophote schema uniform whenever `lsb_auto_lock=True`: write `lsb_locked=False` on every isophote and `lsb_auto_lock_anchor=False` on all but the committing one (update docs 01/02/07 accordingly). Alternatively, sanitize missing bools to `False` inside `isophote_results_to_astropy_tables`.

## B4 — Duplicated `except` clause (dead code, confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** The unreachable second `except (np.linalg.LinAlgError, ValueError)` block in `compute_parameter_errors` was removed; behavior unchanged (existing error-path tests pass).

**Location:** `isoster/fitting.py:500-513`. Two identical `except (np.linalg.LinAlgError, ValueError)` blocks in a row in `compute_parameter_errors`; the second is unreachable. Harmless at runtime but confusing in review; remove one. (Ruff does not flag this pattern. The multiband counterpart has a single clause — not inherited there.)

## B5 — Central-pixel schema hardcodes harmonic orders (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** `fit_central_pixel` now accepts an optional `config` and builds its harmonic keys from `config.harmonic_orders`, included only when `compute_deviations` or `simultaneous_harmonics` is enabled — matching the fitted-isophote schema exactly. Both driver call sites (regular fit and template forced mode) pass the config; callers without a config keep the legacy `[3, 4]` keys for backward compatibility. The multiband `_fit_central_pixel_mb` already followed `config.harmonic_orders` (no change needed). Regression tests in `tests/unit/test_driver.py`: orders `[3,4,5,6]` with `minsma=0` → central pixel carries `a5/b5/a6/b6`; `compute_deviations=False` → central pixel carries no harmonic keys. Suite: 668 passed.

**Location:** `isoster/driver.py:345-352` (`fit_central_pixel`).

The central pixel always gets `a3/b3/a3_err/b3_err/a4/b4/a4_err/b4_err` — regardless of `config.harmonic_orders` and regardless of `compute_deviations=False`. With `harmonic_orders=[3,4,5,6]` the fitted isophotes carry `a5/b5/a6/b6` but the central pixel does not (verified: two distinct key sets in one result list; FITS export then yields masked cells for the central pixel's `a5` columns). Relatedly, forced-mode `extract_forced_photometry` *does* follow `config.harmonic_orders`, so central and non-central rows disagree there too. Same hardcoded-orders family in multiband plotting (M11) and dead multiband helper `_per_band_column_names`.

**Fix direction.** Build the central-pixel harmonic keys from `cfg.harmonic_orders`, gated on `compute_deviations`/`simultaneous_harmonics` like the rest of the pipeline.

---

## Part II — `isoster/plotting.py` (delegated full-file review; verdicts marked)

## P1 — Harmonic panel double-normalizes single-band coefficients (confirmed, high) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** All plotters now split conventions by input source:
> - `plot_qa_summary` (single-band): plots the stored `a3/b3/a4/b4` directly (they are already Bender-normalized by `compute_deviations`); the second `_normalize_harmonic_for_plot` call is removed.
> - `plotting_mb._plot_harmonic`: gained a `normalize` flag driven by `result["harmonics_shared"]` — joint modes (raw shared coefficients, decision D16) still normalize per band at plot time; the per-band post-hoc path (stored normalized) is no longer re-normalized.
> - `benchmarks/exhausted/analysis/scenario_summary.py`: Prior-2 stats now use the stored values and errors directly; the dead `normalize_harmonic`/`effective_grad`/`profile_local_grad` helpers were removed.
> - Stale "normalization happens at plotting time" docstring in `fitting_mb._attach_per_band_harmonics` corrected (it stores normalized values via `compute_deviations`).
>
> **Question 8 resolved:** AutoProf's `a3/b3/a4/b4` are photutils coefficients (`Isophote_Extract.py` builds a photutils `Ellipse` list), and photutils stores `up_coeffs/(sma·|grad|)` — i.e. normalized. All scenario_summary inputs are therefore normalized, which validates the fix there.
>
> Regression tests: `tests/unit/test_qa_summary.py` (plotted `a3/b3/a4/b4` equal stored values; review gap #7) and two mode-split tests in `tests/multiband/test_plotting_mb.py` (independent = plotted-as-stored; shared = plotted raw/(sma·|grad|)). Suite: 663 passed. One convention remains deliberately split (MB joint modes store raw, D16) pending the author's answer to Question 6.

**Location:** `isoster/plotting.py:989-995` (panel at 1337-1376).

Single-band isoster stores harmonic coefficients **already Bender-normalized**: `compute_deviations` (`fitting.py:560-565`) and `fit_higher_harmonics_simultaneous` (`fitting.py:764-780`) both divide the raw flux coefficient by `sma·|grad|` before storing. `plot_qa_summary` then applies `_normalize_harmonic_for_plot` (`= -a_n/(sma·grad)`) a **second time** at `plotting.py:992-995`. Verified numerically: a planted raw `A4=2.0` (flux units) at `sma=20, grad=-5` is stored as `a4=0.02` (correct Bender form) and then plotted as `0.0002` — wrong by a factor `1/(sma·|grad|)`. Because `sma·|grad|` declines steeply with radius, this distorts both the scale **and the radial shape** of the harmonic profile in the figure (inner points suppressed ~20× on a real benchmark profile).

It is also internally inconsistent: the same figure's contour overlays (`contour_isoster_phi/psi`) correctly treat stored `a_n` as the dimensionless fractional-radius perturbation, matching `model.py:278-296`. The re-normalization is only correct for inputs holding **raw** coefficients — which is the multiband joint-fit path (`fitting_mb.py:2360-2363` stores raw shared coefficients), not single-band isoster or photutils output.

The same double-division exists in `benchmarks/exhausted/analysis/scenario_summary.py:289-315` and in the multiband per-band post-hoc path (normalized at fit time in `fitting_mb.py:2153-2164`, normalized again in `plotting_mb.py:410`).

**Fix direction.** Plot stored `a3/b3/a4/b4` directly for isoster/photutils inputs (they are already the Bender-normalized values), and split conventions explicitly per input source. See Question 6 — one canonical convention is needed.

## P2 — Extended figure fractional-residual label has the wrong sign (confirmed, medium) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** The extended figure's colorbar label and docstring now read `100 * (model - data) / data`, matching `compute_fractional_residual_percent` and the other two figures. Regression test: `test_p2_extended_figure_fractional_label_matches_sign` in `tests/unit/test_comparison_qa.py`.

`compute_fractional_residual_percent` returns `100·(model − data)/data` (`plotting.py:467-472`), but the extended figure labels the colorbar `"(data - model) / data [%]"` (`plotting.py:1667-1669`); the docstring at 1489-1492 repeats the wrong sign. `plot_qa_summary:1145` and `_compute_residual_map:2084` use the correct sign, so the two figures disagree. A reader will interpret red/blue backwards in the extended figure.

## P3 — Comparison figure crashes on unknown method names (confirmed, medium) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** All seven `style = styles.get(method_name, {})` sites in `plot_comparison_qa_figure` now inject a default color: `style = {"color": "black", **styles.get(method_name, {})}`, so unknown methods (or user `method_styles` entries without a `color` key) render in black instead of raising `KeyError`. This matches the generality the layout code already had (N methods, and one annotation path already used `style.get("color", "black")`). Regression tests: `test_unknown_method_name_no_keyerror` and `test_unknown_method_with_partial_user_styles` in `tests/unit/test_comparison_qa.py`. Suite: 676 passed.

`plot_comparison_qa_figure` does `style = styles.get(method_name, {})` and then `style["color"]` (`plotting.py:2522, 2529, 2536, 2550, 2643, 2657, 2714, 2791, 2910` — verified by grep) — a `KeyError` for any method not in `METHOD_STYLES` or user-supplied `method_styles`. The layout code is written generically (`n_left_rows = 1 + n_methods`, 2239-2248) and one annotation path already uses the safe `style.get("color", "black")` (2285-2287), so the crash is inconsistent with the intended generality. Fix: default via `style.get("color", "black")` everywhere.

## P4 — Further plotting findings — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** All seven items addressed:
> - **NaN stop codes**: all three `astype(int)`/`int()` sites (`plot_qa_summary`, extended, and `draw_isophote_overlays` — a third site the review did not list, found by the new test) now use `np.nan_to_num(..., nan=0)`.
> **NaN→True coercion**: `_build_isos_for_overlay` now drops NaN `use_eccentric_anomaly`/`tool` entries so the downstream phi-mode/isoster defaults apply.
> **Asinh SB scale**: `plot_qa_summary` resolves the softening once from the isoster intensities; the comparison figure resolves once from the reference method (`available[0]`) and passes it to every method — one shared scale, and the `I=0` line is consistent. (The extended figure takes a single input and was never affected.)
> **Psi-mode contour**: now draws the exact `sma/(1−δ)` (matching `model.py`'s `r·(1−δ)`), with the same guard as `contour_isoster_phi`; verified `max(r) = sma/(1−0.2)` on a planted `b4=0.2`.
> **Runtime annotation**: scalar `runtime_seconds`/`retries` now pass through `build_method_profile`'s dict path (per-row array preservation was deliberately avoided — the annotation consumes a scalar).
> **0.05 residual floor**: both floors (extended, comparison `_plot_residual_panel`) now apply only in fractional (percent) mode; absolute mode uses the 99th percentile unbounded.
> **NaN geometry overlays**: `draw_isophote_overlays` skips rows with non-finite `x0/y0/eps/pa`.
>
> Regression tests: seven new tests across `tests/unit/test_qa_summary.py` and `tests/unit/test_comparison_qa.py` (`TestP4PlottingBatch`). Suite: 691 passed.

The first four were confirmed line-by-line by the reviewer; the last three are reported by the delegated review with precise citations.

- **`.astype(int)` on stop codes raises on NaN (confirmed, edge case)** (`plotting.py:987, 1525`): `get_arr("stop_code", 0).astype(int)` raises `ValueError` if any isophote dict carries an explicit `stop_code=np.nan` (missing keys are fine — they default to 0). Fix: `np.nan_to_num(..., nan=0).astype(int)`.
- **NaN→True coercion in overlay reconstruction (confirmed)** (`plotting.py:2059-2062` with 335): `build_method_profile` fills missing `use_eccentric_anomaly` with `np.nan`, and `bool(np.nan)` is `True`, so rows missing the flag silently switch to eccentric-anomaly contours (isoster's default is phi-mode). Same for `tool`: `str(np.nan)` → `"nan"` falls through to the psi default.
- **Asinh SB scale differs between overlaid methods (confirmed)** (`plotting.py:2506-2513, 2562-2570, 2985-2992`): every method's `transform_sb_profile` call passes the same user `sb_asinh_softening` (default `None`), and `None` is resolved per-method from that method's own intensities — different methods in the same panel are therefore on different asinh scales, and the `I=0` line comes from the last method's softening. Same issue between isoster and the photutils overlay in `plot_qa_summary`. Fix: derive softening once (reference method or image) and pass it explicitly.
- **Psi-mode contour is first-order where the model is exact (confirmed)** (`plotting.py:646-648`): draws `r = sma·(1+δ)`; `model.py:296` reconstructs `r·(1−δ)`, whose exact isophote is `sma/(1−δ)` (what `contour_isoster_phi` draws — verified to match the model exactly). At the admitted gate limit `|δ|=0.5`, `1+δ=1.5` vs `1/(1−δ)=2.0` — a 33% radius mismatch between overlay and model image for noisy rows. The photutils dispatch is fine (matches photutils's own model builder); the mismatch is only against isoster's own model image.
- **Runtime annotation is dead for the standard path (reported)** (`plotting.py:2198, 2288-2294`): `build_method_profile` does not preserve `runtime_seconds`/`retries` keys, so `_runtime_lines` is always empty unless callers inject keys after building profiles (one example does). Preserve the keys or drop the docstring mention.
- **Absolute-residual panels get an arbitrary 0.05 floor in data units (reported)** (`plotting.py:1675-1681, 2111-2117`): extended and comparison clip the residual color range to ≥0.05 in *both* modes; `plot_qa_summary:1152-1154` clips only in relative mode. For a low-flux image this flattens the absolute residual map to uniform zero.
- **NaN geometry rows in overlays (nit, reported)** (`plotting.py:746-784`): `draw_isophote_overlays` guards `sma` but not `pa`/`eps`/`x0`/`y0`.

## P5 — Plotting redundancy and unclear definitions (reported) — **FIXED 2026-07-17/18**

> **Status: FIXED (small items 2026-07-17; triplication refactor 2026-07-18, author-approved after visual review).** Removed: the unreachable duplicate `elif not available:` early-return in `plot_comparison_qa_figure`, the unused `normalize_angle` helper (no callers repo-wide), the unused `make_arcsinh_display` wrapper (its `docs/06-qa-functions.md` table row now points to `make_arcsinh_display_from_parameters`), and the redundant local `Path`/`gridspec` imports. Fixed: the stale "base 5" panel-count comment (actual `n_panels = 4`) and the colorbar percentile mislabels (`p0.5` → `p0.05`, matching `derive_arcsinh_parameters`' `lower_percentile=0.05` — four sites).
>
> **Triplication refactor (2026-07-18):** the three image-block copies (~340 lines) are consolidated into two shared helpers — `_plot_arcsinh_image_panel` (data/model panels) and the extended `_plot_residual_panel` — used by all three public figures with every feature preserved (mask overlay, per-method style overlays, model contours, scale bar, blank-model fallback, tick/label conventions). Pixel-diff vs pre-refactor renders: `plot_qa_summary` and `plot_comparison_qa_figure` identical (0 changed pixels); `plot_qa_summary_extended` differs only in the residual colorbar label text, which was the intended consistency fix. Three deliberate micro-unifications: residual label `data - model` everywhere, fractional label `(model - data) / data [%]` everywhere, and the extended figure's fractional color range gains the 8.0 cap the other two already had. All consumers (examples, benchmarks, tests) call only the public functions, whose signatures and appearance are unchanged; suite 699 passed. The remaining convention inconsistencies (b/a vs eps across figures, `PA [deg]` vs `PA (deg)`, filled/none vs filled/open marker vocabulary, sma gates 1.0/1.5) stay as documented cosmetic differences.

- The left-column image block (arcsinh data/model/residual trio) is **triplicated** (`1057-1181`, `1587-1707`, `2296-2487`) with drift in clip rules and labels — the P2 sign bug is exactly the drift this causes; `_compute_residual_map`/`_plot_residual_panel` already factor it out but only the comparison figure uses them. The right-column profile panels are likewise duplicated between summary and extended with cosmetic drift (legend fontsize, ylabels, an unexplained open/filled marker convention in extended).
- Dead code: unreachable duplicate early-return (`2490-2495`); unused helpers `normalize_angle` (276-279, no callers repo-wide) and `make_arcsinh_display` (436-459, only a docs-table mention); redundant local imports (`2221-2223`, `2627`).
- Undocumented units: result-dict `pa` is in **radians** (panels display degrees); nowhere stated in the plotting docstrings. Same for `eps` (dimensionless), `x0/y0` (pixels), `intens` (image flux units).
- Harmonic vocabulary: extended's `normalize_harmonics=True` computes `A_n/I` (`plotting.py:1854-1855`) — a flux-ratio convention that is **not** the Bender form CLAUDE.md mandates; summary's ylabel `$A_n$ or $B_n$ (norm)` never states the formula. Three conventions now coexist (see Question 6).
- Colorbar percentile mislabel (nit): `derive_arcsinh_parameters` uses `lower_percentile=0.05`, colorbars say `p0.5` (`1105, 1139, 1631, 1663`).
- Inconsistent conventions across figures: summary plots `b/a`, comparison plots `eps`; `sma` gates of 1.0, 1.5 (photutils overlay), and 1.0 (overlays) in three places; `"PA [deg]"` vs `"PA (deg)"`; `"filled"/"none"` vs `"filled"/"open"` marker vocabularies.
- Stale comment `plotting.py:1011` ("base 5 + optional harmonics + optional CoG", actual `n_panels = 4`); comparison docstring describes exactly three methods while the code handles N generically.

## P6 — Plotting areas verified correct (no action)

- `transform_sb_profile` — all four scale branches, formulas, error propagation (`2.5σ/(I ln10)`, asinh derivative), edge-case guards — algebraically correct.
- `_validate_sb_inputs` two-input SB rule honored everywhere; `pixarea` computed internally; no pre-combined-zeropoint path exists.
- `_normalize_harmonic_for_plot` formula, masking, `np.gradient` fallback — correct *given raw-coefficient input*.
- `contour_isoster_phi` vs `model.py` (exact match); `contour_photutils`/`contour_isoster_psi` vs photutils's own model (correct to first order — but see the basis erratum in V1.6: photutils coefficients are polar-angle-basis, so the `contour_photutils` docstring premise is false); `contour_autoprof` honest pure-ellipse fallback; EA/phi dispatch logic.
- `normalize_pa_degrees` double-angle unwrap — verified continuous across 180°.
- `derive_arcsinh_parameters`, `robust_limits`, axis-limit helpers — empty/all-NaN/degenerate inputs handled; empty `isoster_res` does not raise.

---

## Part III — `isoster/multiband/` (delegated full-file review; verdicts marked)

## M1 — `harmonic_combination='ref'` × `loose_validity=True` is broken (confirmed, high) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** The ref branch now resolves the reference band's position within `surviving_idx` and fits its own `phi_solve[ref_pos]` / `intens_solve[ref_pos]` (previously it indexed the surviving-only arrays by full-list position — silently fitting the wrong band, or raising swallowed/uncaught errors). If the reference band is dropped at an isophote (or has fewer than `ref_min_points` samples), the isophote is skipped with a clear `stop_code=3`, consistent with the existing whole-isophote drop handling. The `coeffs`/`cov_full` widening loops now iterate surviving bands with correct positional mapping; dropped bands' `I0_b` stays `NaN`, matching the loose joint solver's convention. Regression tests (`tests/multiband/test_driver_mb.py`): ref band not first + one band dropped before it → outer isophotes recover the reference band's geometry (eps≈0.3, not the wrong band's 0.6); reference band itself dropped → `stop_code=3`, no crash. Suite: 665 passed. Note: docs `10-multiband.md:496` claimed this combination works — it now does (ref × loose is supported, matching the docs).

**Location:** `isoster/multiband/fitting_mb.py:1347-1349` (loose branch) vs `1396-1399` (ref branch).

In the loose-validity branch, `intens_per_band = intens_solve` is a list over **surviving bands only** (1340, 1349) and `angles = phi_solve[0]` is the first *surviving* band's angles (1347 — its own comment says "placeholder … (unused)", but it **is** used at 1399). The ref-mode branch then takes `ref_idx = bands.index(config.reference_band)` in the **full** band list and does `intens_ref = intens_per_band[ref_idx]` (1396-1397). Three verified failure modes:

1. Reference band ≠ first surviving band with unequal kept counts → `np.linalg.lstsq` raises `LinAlgError`, which `fit_first_and_second_harmonics_ref` (680-685) silently swallows into the `[mean, 0,0,0,0]` fallback — every iteration returns zero harmonics, geometry stays frozen at the seed, the fit "converges" with `stop_code=0` and **no warning** (reproduced at runtime: eps stayed at the seed value);
2. any band dropped before the reference band → uncaught `IndexError`, crashes the fit (reproduced);
3. a dropped band shifting indices such that `ref_idx` still exists → silently fits the **wrong band's** intensities.

The only passing configuration is the one in the test suite (reference band first, no drops, `nclip=0`). `docs/10-multiband.md:496` ("Loose validity composes cleanly with both `harmonic_combination='ref'` and …") is false.

**Fix direction.** In the loose path, resolve the reference band's position within `surviving_idx` and pass its own `phi_solve[pos]`/`intens_solve[pos]`; if the reference band was dropped, stop with a clear code or warn + fall back explicitly. Until fixed, a hard error at config/driver entry for ref × loose would protect users.

## M2 — OLS with non-unit `band_weights`: parameter errors mis-scaled (confirmed, medium) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** Both call sites now route through `_compute_joint_residual_variance` (the weight-aware Σ w_b r²/(n−p) estimator): `_compute_parameter_errors_from_joint`'s shared OLS branch (previously unweighted `np.var`) and the coupled OLS `intens_err_<b>` branch (joint mode uses the pooled weight-aware estimate; ref mode keeps the band's own since the ref solve is unweighted). Verified by the review's Monte-Carlo property: reported errors are now exactly invariant to uniform weight scaling (w=4 vs w=1 identical; pre-fix factor √w low). Regression tests: `test_m2_ols_band_weights_errors_weight_invariant` and `test_m2_intens_err_weight_invariant_end_to_end` in `tests/multiband/test_fitting_mb.py`. Suite: 672 passed.

`_compute_parameter_errors_from_joint` (`fitting_mb.py:1012-1023`) rescales the joint OLS covariance `(AᵀWA)⁻¹` by an **unweighted** residual variance (`np.var(residuals, ddof=B+4)`); the correct estimator is the weight-aware `Σ w_b r²/(n−p)`. Same flaw in the coupled OLS `intens_err_<b>` branch (`fitting_mb.py:1776-1781`). Monte-Carlo check: uniform `w=4`, code reports σ(A1)=0.0496 vs true 0.102 (factor √4 low); the weight-aware rescale gives 0.0992. The sibling helper `_compute_joint_residual_variance` (`fitting_mb.py:2366-2435`, used by the `simultaneous_*` modes) does the weighted version correctly — internal inconsistency. Default `w=1` is unaffected. Fix: route both call sites through `_compute_joint_residual_variance`.

## M3 — OLS × loose validity: geometric errors never rescaled at all (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** The loose OLS branch of `_compute_parameter_errors_from_joint` is now wired through `_compute_joint_residual_variance` with jagged inputs: the call site passes the surviving bands' `phi_solve`/`intens_solve` lists, the full `band_weights_arr`, and the new `band_indices` mapping (added to the helper) so dropped bands align correctly in the full `coeffs`/weights vectors; ddof is `n_surviving + 4`. The "skip the OLS rescale" fallback is gone — uncomputable variance now returns zero errors explicitly. Same-family fix in `_attach_simultaneous_higher_harmonics_from_coeffs`: its OLS rescale silently no-opped when a dropped band's NaN intercept poisoned the residual sum; it now restricts to surviving bands with positional mapping (found while verifying M3 — the review's M14 "properly wired" verdict holds only when no bands are dropped). Regression tests: `test_m3_ols_loose_validity_rescales_geometric_errors` (loose matches shared on identical data) and `test_m3_simultaneous_stamper_rescales_with_dropped_band`. Suite: 672 passed.

`_compute_parameter_errors_from_joint`'s loose branch (`fitting_mb.py:1024-1030`) deliberately "skip[s] the OLS rescale and use[s] the as-built (AᵀA)⁻¹" (the code comment says so verbatim), so `x0_err/y0_err/eps_err/pa_err` under OLS+loose are raw inverse-normal-equation diagonals — typically orders of magnitude too small. The comment's claim that "the geometric block is unaffected" is wrong: the geometric block is part of `cov_full` and stays unscaled. The jagged-capable `_compute_joint_residual_variance` exists in the same file but is not wired in.

## M4 — Forced-photometry mode never sigma-clips (confirmed, high) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** `extract_forced_photometry_mb` now runs the existing `_per_band_sigma_clip` (shared-validity variant, honoring `sclip`/`nclip`/`sclip_low`/`sclip_high`, no-op when `nclip=0`) on the ring data before the per-band statistics, mirroring the single-band forced path and the iteration loop. The ring data returned with `return_ring_data=True` is also the clipped ring, so the forced-mode post-hoc harmonics are computed on the same outlier-free samples. Empty-after-clip rings return the same `stop_code=3` record as empty extractions. Two H4 SEM-formula tests now pin `nclip=0` explicitly (they verify the error formula on the full ring; the default `nclip=1` now clips, as intended). Regression test: `test_m4_forced_mode_sigma_clips_outliers` — same ring with a 500-count cosmic ray through single-band and multiband forced photometry; both recover the ~100 level (pre-fix MB reported ~101.6 with `rms_<b>≈25`). Suite: 666 passed.

Single-band `extract_forced_photometry` applies `sigma_clip` with the caller's `sclip/nclip` before computing intensities (`fitting.py:231-236`). `extract_forced_photometry_mb` (`fitting_mb.py:2777-2868`) — read in full — uses all valid samples directly; `config.sclip`/`nclip` are never referenced in the function, so ~3σ outliers (unmasked companions, cosmic rays) enter per-band `intens_<b>`/`rms_<b>` where the single-band path would clip them. Nothing in `docs/10-multiband.md` mentions this reduction. Fix: run the existing `_per_band_sigma_clip` (shared-validity variant) on the ring data before the per-band statistics.

## M5 — Dropped bands: docs say NaN, code writes 0.0 (confirmed, medium) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** Dropped bands now get NaN for *all* per-band fields, matching the docs and the drop-detection design intent. Three sites: the placeholder loop in `fit_isophote_mb` (was 0.0 for harmonic columns), `_zero_init_per_band_higher_harmonics` (writes NaN for dropped bands; all three shared/simultaneous stampers pass the dropped set), and `_attach_per_band_harmonics` (now skips dropped bands so the independent-mode post-hoc pass no longer overwrites the NaN marker with computed values on sparse data — previously it silently computed harmonics for dropped bands too, a third inconsistent behavior on top of the 0.0-vs-NaN mismatch). Regression test: `test_m5_dropped_band_harmonic_columns_are_nan` (converged isophotes with band g dropped → NaN for all `a/b/err` columns of g, finite values for surviving bands). Suite: 673 passed.

`docs/10-multiband.md:472-474`: dropped bands' "`intens_<b>`, `intens_err_<b>`, and harmonic columns are set to NaN". `fitting_mb.py:1706-1717` sets `intens_/intens_err_/rms_<b>` to NaN but writes `a<n>_<b> = b<n>_<b> = a<n>_err_<b> = b<n>_err_<b> = 0.0` — contradicting the code's own comment at 1705 ("NaN every per-band field"). Downstream consumers using NaN to detect band drops (the stated design intent for `cog_<b>` NaN propagation) will read the zeros as real measurements.

## M6 — B=1 delegation silently drops features (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** `_delegate_single_band` now accepts `template_isophotes` and forwards it to single-band `fit_image` (the B=1 check previously ran before the forced-photometry dispatch, so templates were silently ignored). The delegation kwarg list now also forwards the shared feature families: `compute_cog`, `harmonic_orders`, `lsb_auto_lock*`, `use_outer_center_regularization` + all `outer_reg_*`, and `use_central_regularization` + all `central_reg_*` (verified 1:1 name parity between `IsosterConfigMB` and `IsosterConfig`). Regression tests in `tests/multiband/test_driver_mb.py`: B=1 + template produces forced-photometry rows (`niter=0`); B=1 with `compute_cog=True`/`harmonic_orders=[3,4,5,6]` produces `cog`/`a5` keys; B=1 with `lsb_auto_lock=True` produces the lock metadata. Suite: 682 passed.

`fit_image_multiband` checks `len(bands)==1` *before* the forced-photometry dispatch (`driver_mb.py:848-850` vs 868-875; verified) and `_delegate_single_band` is not given `template_isophotes` at all, so templates are silently ignored for single-band inputs. The delegation kwarg list (`driver_mb.py:367-410`; verified) also omits `compute_cog`, `harmonic_orders`, `lsb_auto_lock*`, `outer_reg_*`, and `central_reg_*` — all of which exist on `IsosterConfig`. A B=1 run with `compute_cog=True` or the lock/regularization enabled silently produces a fit without those features; the delegation warning only mentions schema. Fix: forward the mappable fields (single-band `fit_image` accepts `template_isophotes`), or hard-error when they are set with B=1.

## M7 — stop_code=2 fallback harmonics computed on pre-clip data (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** The stop-2 fallback now passes the last iteration's post-clip `angles`/`intens_per_band`/`variances_per_band` to `_attach_higher_harmonics_dispatch` (the same arrays the joint solve and the converged paths use), instead of `last_data.*` (the raw, pre-clip sampler output). The loose-validity fallback already used the post-clip jagged lists and is unchanged. Regression test: `test_m7_stop2_posthoc_harmonics_use_clipped_data` — a cosmic ray on the fitted ring leaves the stop-2 harmonics statistically indistinguishable from a clean control (outlier's direct impact ~0.03–0.08 if it leaked). Suite: 677 passed.

Single-band's maxit fallback passes the loop's *post-clip* `angles`/`intens` to `_compute_posthoc_harmonics` (`fitting.py:1786-1796`). The MB fallback (`fitting_mb.py:2059-2074`) passes `last_data.angles`/`last_data.intens` — `last_data` is the raw sampler output; `_per_band_sigma_clip` (1353) returns new arrays and never mutates it, so these are the pre-clip samples — while `last_joint_coeffs` was fit on clipped data. Post-hoc harmonics on stop-2 rows therefore include the exact outliers the pipeline removed.

## M8 — `ndata`/`nflag` semantics break under loose validity (confirmed, medium) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** Under loose validity `total_points` is now `Σ_b n_valid_per_band[b]` (post-sampling valid samples, summed over bands) so it matches `actual_points` (Σ_b kept after clipping): `nflag = total − actual` is the clipped count and can no longer go negative, and the `fflag` guard now compares like-for-like sums (it was inert before). Shared validity is unchanged (per-ring counts). The loose-validity QA panel (`plotting_mb._plot_n_valid_per_band`) now derives `n_attempted` from the ellipse-path sample grid `max(64, floor(2π·sma))` — the same denominator the sampler's drop fraction uses — instead of `ndata+nflag`, so per-band fractions no longer exceed 1 and the panel no longer degenerates without debug fields. Docs `10-multiband.md` now state the loose-validity semantics explicitly. Regression test: `test_m8_ndata_nflag_sane_under_loose_validity` (nflag ≥ 0; `ndata == Σ_b n_valid_<b>` on ring rows; clipping registers in nflag). Suite: 674 passed.

Under loose validity, `total_points` = intersection count (`fitting_mb.py:1296`) but `actual_points` = Σ_b per-band kept counts (1321); the row records `ndata = actual_points`, `nflag = total_points − actual_points` (1806-1808). Reproduced at runtime: `ndata=237, nflag=-157` (negative). So under loose validity `ndata` changes units (per-ring count → sum over bands, double-counting shared samples), `nflag` can go negative, and the fflag guard (1374) compares these mismatched quantities, effectively disabling fflag under loose validity. Docs describe `ndata` only as the shared-validity count. The loose-validity QA panel (`plotting_mb.py:488-527`) inherits the confusion (per-band fractions can exceed 1; degenerates to `n_attempted=1` when `debug=False`).

## M9 — Combined gradient lacks single-band's refinements (reported) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** `compute_joint_gradient` now ports both single-band refinements (the author chose porting over documenting the simplification): (a) EFF-1 — when the first joint gradient looks anomalous (`|grad_err/grad| ≥ 0.3` and `grad ≥ previous/3`), the annulus is resampled at 2×step and the longer baseline is used; (b) runaway decay — a joint gradient `≥ previous/3` decays to `0.8·previous_gradient` with the error cleared. The per-band two-point computation and the weight pooling are unchanged; the refinements wrap the pooled scalar exactly as in single-band. Regression tests: `test_m9_runaway_gradient_decayed` and `test_m9_anomalous_first_gradient_resamples_at_double_step` (plateau profile: the 2× baseline finds the slope the 1× annulus cannot see). Fit speed: no regression on the standard mock (the resample triggers only on anomalous gradients, matching single-band's lazy behavior). Suite: 693 passed.

Single-band `compute_gradient` (`fitting.py:882-935`) (a) resamples at 2×step with a longer baseline when the first gradient looks anomalous, and (b) decays runaway gradients. `compute_joint_gradient` (`fitting_mb.py:842-962`) has neither — a plain two-point difference. In the LSB regime this is exactly where spurious/near-zero gradients drive the geometry update and the `maxgerr` gate. Not mentioned in `docs/10-multiband.md`. Port the refinement or document the simplification as deliberate.

## M10 — Inheritance check of the single-band issues (mixed) — **B2 inheritance FIXED 2026-07-17**

> **Status: B2-inheritance FIXED (2026-07-17, branch `fix/pre-publication-review`).** The decoupled WLS paths now use the exact `1/Σ(1/varᵢ)` (shared helper `_weighted_mean_variance`, moved to `_shared.py` since both paths need it): the four `cov[b, b]` ring-mean intercept sites, `compute_joint_gradient`'s per-annulus mean variances (also gaining B2's sentinel immunity for the MB `BAD_PIXEL_VARIANCE` values), and both direct-SEM `intens_err_<b>` branches. Uniform-variance results are bit-identical (both formulas reduce to σ²/N); non-uniform maps no longer overestimate. Regression tests: `test_m10_decoupled_wls_intercept_cov_is_exact` and `test_m10_intens_err_decoupled_wls_exact` in `tests/multiband/test_fitting_mb.py`. Multiband fit timing unchanged (WLS 0.032 s vs 0.033 s pre-session; OLS slightly faster thanks to B1). The M12 "reverse drift" notes (forced-mode SEM, central-pixel WLS error) remain deliberate improvements on the MB side — see M12.

- B4 (duplicated `except`): **not** inherited — single clause at `fitting_mb.py:1062`.
- B2 (WLS unweighted-mean error formula): **partially inherited.** The coupled joint mode reads the exact covariance diagonal (1769-1770) and forced WLS uses exact `1/√Σ(1/var)` (2842), but the *decoupled* WLS paths pair an inverse-variance-weighted mean with the unweighted-mean variance `mean(var)/n` — at `fitting_mb.py:286, 430, 522, 618, 1755, 1764`, including the loose-validity joint path that routes through `use_direct_sem` even when the exact joint covariance exists. Exact form is `1/Σ(1/var)`; the used formula overestimates when per-pixel variances are non-uniform. Fix together with B2.
- D9 (model.py docstring sign): no MB copy exists — `plotting_mb.py:45` reuses single-band `build_isoster_model`, so there is nothing to drift. Good design.

## M11 — Multiband plotting hardcodes harmonic orders (confirmed) — **FIXED 2026-07-17**

> **Status: FIXED (2026-07-17, branch `fix/pre-publication-review`).** `_isophotes_to_per_band_singleband_lists` now takes `harmonic_orders` (passed from `result["harmonic_orders"]` by `plot_qa_summary_mb`) instead of the hardcoded `for n in (3, 4)`, with key-detection fallback (`a{n}_<b>` keys of the first row that has any) for hand-built result dicts. Orders ≥5 now reach `build_isoster_model` in the residual mosaic. Regression test: `test_m11_model_inputs_follow_harmonic_orders` in `tests/multiband/test_plotting_mb.py` (explicit orders and key-detection fallback). Suite: 678 passed.

`plotting_mb.py:218-220` builds per-band model inputs with `for n in (3, 4):` (verified by reading), so `harmonic_orders=[3,4,5,6]` results silently lose orders ≥5 in the residual mosaic even though `build_isoster_model` auto-detects them. Same family as B5. Fix: derive orders from `result['harmonic_orders']`.

## M12 — Multiband redundancy and drift vs single-band (reported) — **ACTIONABLE ITEMS FIXED 2026-07-17/18**

> **Status: actionable items FIXED (2026-07-17; forced-SEM backport added 2026-07-18 after author decision).** (a) `sampling_mb.py` now imports `_prepare_mask_float` from `_shared.py` instead of re-implementing it inline (all shared helpers now come from the single source). (b) Reverse drift: single-band `fit_central_pixel` now reports `intens_err = sqrt(variance_map[iy, ix])` under WLS (multiband parity), with a regression test; OLS keeps the 0.0 convention shared by both paths (a single sample admits no internal error estimate — the review's NaN suggestion is noted but 0.0 is the established two-path convention). (c) **Median-SEM backport (author decision A1, 2026-07-18):** the single-band median-integrator `intens_err` now carries the Gaussian-asymptotic `sqrt(π/2)` factor at both OLS sites (`extract_forced_photometry` and `fit_isophote`), mirroring the multiband H4 formula — the ~25% underestimate in the LSB median path is fixed while the mean path stays byte-identical to photutils (verified: photutils' `int_err = std(ddof=0)/√N` at `reference/isophote.py:124`). `ddof` stays 0 on the mean path (full A2 backport rejected: breaks photutils parity for a <0.2% gain). Regression test: `test_forced_photometry_median_sem_factor`; doc 02's median-integrator section documents the factor. The larger recorded-not-actioned items stand: the duplicated helper bodies and the gradient scatter-error formula for median (same underestimate family, not flagged by the review, and touching the maxgerr gate changes fit outcomes). Deliberate differences (outer_reg_weights defaults, debug auto-enable, `_build_locked_cfg_mb` additions) stand as documented.

- ~200 lines of `driver_mb.py` are verbatim or near-verbatim copies of `driver.py` helpers (`_first_isophote_perturbations`, `_is_lsb_isophote_mb`, `_mark_lsb_lock_state_mb`, `_build_outer_reference_mb`, `_validate_non_negative_error_fields`); ~400-450 lines of `fitting_mb.py` are adapted copies (geometry-update block, central-reg penalty, parameter errors, sigma-clip wrappers, forced photometry). The joint solvers, loose-validity machinery, and `simultaneous_*` modes (~1500+ lines) are genuinely new. The copies are where the drift bugs above (M4, M7, M9) came from — consider importing the single-band helpers where signatures allow.
- **Reverse drift** (MB has fixes single-band lacks, not backported): forced-mode SEM with `ddof=1` + median `sqrt(π/2)` factor (`fitting_mb.py:2844-2860`) vs single-band's `rms/sqrt(N)` for both integrators (`fitting.py:250-252`); central-pixel WLS error from the variance map (`driver_mb.py:528-544`) vs single-band's unconditional `intens_err=0.0` (`driver.py:340`). Worth flagging since single-band is the reference implementation.
- `sampling_mb.py:116-138` re-implements `_prepare_mask_float` inline instead of importing it from `_shared.py` (the other four shared helpers are properly imported).
- Deliberate and documented (no action): `outer_reg_weights` default `{1,0,0}` vs single-band `{1,1,1}`; MB's debug auto-enable mechanism for `lsb_auto_lock`; `_build_locked_cfg_mb` additions.
- Numba bilinear sampler edge parity nit (`numba_kernels_mb.py:201-221` clamps at the border where scipy yields NaN).

## M13 — Multiband docstring/doc mismatches (confirmed spot-checks; mostly concern-level) — **CODE ITEMS FIXED 2026-07-17**

> **Status: code items FIXED (2026-07-17, branch `fix/pre-publication-review`); docstring/doc parts handled with the doc-drift batch.**
> - **Round-trip key loss**: the FITS writer gains a `META` BinTableHDU (JSON PARAM/VALUE rows) persisting every non-structural top-level result key (`forced_photometry_mode`, `template_n_isophotes`, `sky_offsets`, `lsb_auto_lock*`, `first_isophote_*`, outer-reg references), and the reader merges it back. The ASDF writer/reader now pass through all remaining top-level keys instead of a five-key allowlist. Regression tests for both formats in `tests/multiband/test_utils_mb.py`.
> - **`ndata`/`nflag` presence**: now written unconditionally on fitted rows (was `debug`-gated) and on the central-pixel row — the docs list them unconditionally and mixed presence produced masked FITS cells.
> - **`n_valid_<b>`**: now stamped on forced-photometry rows and on the two mid-loop early returns (`_stamp_n_valid_per_band` helper shared with the end-of-fit path).
> - **Dead code removed**: `_TOP_LEVEL_MB_KEYS`, `_per_band_column_names` + `_PER_BAND_HARMONIC_KEYS`, `_ = extract_forced_photometry_mb  # noqa`, `_ = n`.
> - **CLI**: `--reference-band` help no longer claims "Required" (it defaults to the first band).
>
> Remaining for the doc batch: stale "Stage-1 scope" docstrings (config_mb/driver_mb/fitting_mb headers), `10-multiband.md:358` writer name + CONFIG-HDU description, `10-multiband.md:627` outer_reg_weights default table, `sampling_mb.py:497` "Named tuple" wording, `fit_image_multiband` docstring Returns omissions, EXPERIMENTAL marker on the library entry point.
>
> **Docstring/doc items FIXED (2026-07-18, branch `fix/pre-publication-review`):** the stale "Stage-1 scope" docstrings in `config_mb.py` (module header, class header, `fit_per_band_intens_jointly` integrator note, `integrator` stale reason), `driver_mb.py` and `fitting_mb.py` module headers, and `isoster/multiband/__init__.py` now describe the current implementation; `sampling_mb.py` says "slotted dataclass"; `fit_image_multiband` carries an EXPERIMENTAL marker on the docstring and a complete Returns section; `10-multiband.md:358` names the correct writer (`isophote_results_mb_to_fits`) and describes the 4-HDU layout (primary `MULTIBND/BANDS/REFBAND/HARMCMB/VARMODE`, ISOPHOTES, CONFIG as JSON PARAM/VALUE rows, META for remaining top-level keys); `10-multiband.md:627` shows the real `outer_reg_weights` default `{center: 1, eps: 0, pa: 0}`.

- **Stale "Stage-1 scope" docstrings** contradicted by the code they head: `config_mb.py:13-24` lists `lsb_auto_lock_*`, `outer_reg_*`, `compute_cog`, `central_reg_*` as "Excluded fields (deliberately not copied)" — all are implemented; `config_mb.py:121-129` claims `integrator` does not affect `intens_<b>` — contradicted by the field's own description (328-346) and the decoupled median path; `driver_mb.py:13-14` and `fitting_mb.py:11-13` make similar stale Stage-1 claims. These are actively misleading to a journal reader.
- `docs/10-multiband.md:358` names the FITS writer "`isophote_results_to_fits`" (the single-band name; actual: `isophote_results_mb_to_fits`) and describes CONFIG-HDU keywords that do not match the implementation (`utils_mb.py:124-140` stores pydantic field names as JSON PARAM/VALUE rows; the primary header carries `MULTIBND/BANDS/REFBAND/HARMCMB/VARMODE`).
- `docs/10-multiband.md:627` table still shows `outer_reg_weights` default `{1,1,1}`; the callout at line 680 and the code say `{1,0,0}`.
- `ndata`/`nflag` listed unconditionally in the Schema-1 column set (`10-multiband.md:341`) but written only when `debug=True` on fitted rows — while `_empty_isophote_dict` always writes them (mixed presence → masked FITS cells).
- `n_valid_<b>` missing on forced-mode ring rows (`extract_forced_photometry_mb` never stamps it) and on the mid-loop early returns (1331-1334, 1375-1378).
- Round-trip key loss: `isophote_results_mb_from_fits` drops `forced_photometry_mode`, `template_n_isophotes`, `sky_offsets`, `lsb_auto_lock*`, `first_isophote_*`; the ASDF path preserves only a subset, despite the doc's "stores the full result dict natively" claim (`10-multiband.md:368`).
- Dead code: `_TOP_LEVEL_MB_KEYS` (`utils_mb.py:77-84`), `_per_band_column_names` + `_PER_BAND_HARMONIC_KEYS` (`fitting_mb.py:50-96`, also hardcoded to orders 3/4), `_ = extract_forced_photometry_mb  # noqa` (`driver_mb.py:1150`), `_ = n` (`plotting_mb.py:298`).
- Nits: `sampling_mb.py:497` "Named tuple" (now a slotted dataclass); CLI `--reference-band` help says "Required" but silently defaults to the first band; single-ndarray variance broadcast accepted by the driver but unreachable from the CLI; `fit_image_multiband` docstring Returns section omits five written keys; EXPERIMENTAL flagging is in place on the CLI/banner but the library entry point carries no marker.

## M14 — Multiband areas verified correct (no action)

- Forced-mode WLS `1/√Σ(1/var)` and OLS `ddof=1`/median-`sqrt(π/2)` SEM; central-pixel WLS error from the variance map; config immutability via `model_copy`.
- Joint design-matrix kernels (rectangular, jagged, higher-order): correct layouts and offsets; `per_band_count` normalization mathematically identical to the kernels' row scaling.
- `_compute_joint_residual_variance` (weighted OLS rescale) correct and properly wired for the `simultaneous_*` error stamping; shared-mode post-hoc refit freezes coefficients as designed.
- All stated validators present and matching docs (band regex/duplicates/reference membership, weights, integrator×joint hard-error, ring-mean×ref hard-error, higher-harmonics×ref hard-error, outer-reg sanity, lsb `fix_*` hard-error, lock×median×jointly hard-error).
- `cog_mb` semantics match single-band `compute_cog` and reuse `compute_ellipse_area`/`detect_crossing`; B=1 parity test exists.
- `_build_outer_reference_mb` flux-weighted means with circular 2·pa mean and anchor fallbacks; lsb one-way state machine matches the documented design.
- 264 multiband tests collect cleanly; the doc's headline count is accurate.

---

## Part IV — Cross-cutting findings (single-band, from the reviewer's own pass)

## Redundancy

- **R1. `isoster/optimize.py`** — an 11-line docstring-only "facade" that re-exports nothing. `docs/04-architecture.md:56` describes it as "compatibility facade re-exporting driver/fitting APIs", which is not true. Either restore the re-exports or delete the module and the doc line. — **FIXED 2026-07-17: re-exports restored** (`fit_image`, `fit_isophote`, `extract_forced_photometry`, `extract_isophote_data`, `compute_deviations`, `compute_gradient`, `compute_parameter_errors`, `sigma_clip` with `__all__`), so the module now matches its docstring and the architecture doc. `test_optimize_facade_importable` strengthened to assert the re-exported names exist.
- **R2. `sigma_clip_fast`** (`isoster/numba_kernels.py:418-462`) — unused duplicate of `fitting.sigma_clip` with a different signature; no call sites anywhere in the repo. Delete or wire in. — **FIXED 2026-07-17: deleted** (no call sites; `fitting.sigma_clip` remains the single implementation).
- **R3. Duplicated doc heading** — `docs/02-configuration-reference.md` has `## Photometry Outputs` twice (lines 453 and 458); the first is a three-line stub ending in `...`. — **FIXED 2026-07-17: stub removed**, one heading kept.
- See also P5 (plotting triplication) and M12 (multiband copies).

## Unclear definitions (journal-reviewer bait)

> **Status (2026-07-18, branch `fix/pre-publication-review`):** U1: doc 07 §2.1 now matches §6 and the code — the marker goes on the committing trigger isophote; the geometry anchor (debounce steps before the streak) is a different, unmarked isophote. U3: doc 01 now states the stored convention `sma·|gradient|` (absolute value) and adds a canonical sentence relating it to the Bender form `-A_n/(a·dI/da)` (identical when grad < 0). U4: docs 07 and 08 each gained an explicit sentence on the heuristic units blend of `effective_amp` (harmonic amplitude + squared-geometry penalties; the relative weighting depends on the image's flux scale). U5: doc 02 gained a "Units and Conventions" box (sma/x0/y0 in pixels, pa in radians, eps dimensionless, intens/rms in image flux units, harmonic coefficients dimensionless via `sma·|gradient|`). U2 was fixed with the code nits (psi naming in `get_elliptical_coordinates`). The open author questions (anchor naming/recording, penalty-units wording for the paper) remain below.

- **U1. `lsb_auto_lock_anchor` semantics.** The flag marks the *committing trigger* isophote (the first locked one), but the geometry anchor is the isophote `debounce` steps *before* the streak (`driver.py:706-717`). Docs 01/02 say "first locked isophote" (matches code); doc 07 §2.1 says the *anchor* isophote gets the marker while doc 07 §6 says the *trigger* gets it (self-contradictory). The name itself invites the wrong reading. Also, the true geometry-anchor index (`lsb_state["anchor_index"]`) is computed but never recorded in the results — consider recording the anchor isophote's sma/identity for QA. — **RESOLVED 2026-07-18 (author chose to record the anchor):** both drivers now stamp `lsb_auto_lock_anchor_sma` (the true geometry anchor's sma; `None` when the lock never commits), documented in docs 01/02/04/07/10 and technical/1.4.3. Persistence: the single-band FITS writer gains a `META` HDU (JSON PARAM/VALUE rows for all non-structural top-level keys — this also preserves the previously-lost `lsb_auto_lock*`/`first_isophote_*`/outer-reg keys; layout is now 4 HDUs), the single-band ASDF writer passes them through natively, and both readers merge them back (shared helpers in `_shared.py`, used by the MB writer too). Regression tests: anchor identity verified against the isophote debounce steps before the trigger, plus FITS/ASDF round-trips (single-band and MB). Suite: 699 passed. The naming question itself (marker keeps marking the commit point) stands as documented.
- **U2. `phi` naming in `get_elliptical_coordinates`** (`isoster/sampling.py:51-89`) returns the *eccentric anomaly* but names it `phi` — colliding with the package-wide convention that φ is the position angle. The docstring admits it ("elliptical angle (eccentric anomaly)") but the variable name will mislead. — **FIXED 2026-07-17:** the local variable and the docstring now use `psi`, with an explicit note that it is the eccentric anomaly (`tan(phi) = (1-eps)·tan(psi)`). Return values unchanged (positional unpacking at the only caller, the test).
- **U3. Harmonic normalization sign.** Stored `a{n}/b{n}` are normalized by `sma·|gradient|` (absolute value). Docs (`01-user-guide.md:374`) write "sma × gradient" without the absolute value; `technical/1.2` claims outputs are "Bender-normalized `-A_n/(a·dI/da)`". These coincide only when `grad < 0` (the normal case): `-A_n/(a·grad) = A_n/(a·|grad|)`. For positive-gradient (pathological/LSB) rows the stored value and the Bender form differ in sign. One canonical sentence defining the stored convention and its relation to the Bender form would close this. (See also P1 — the plotting layer then applies the Bender division a second time.)
- **U4. Selector-penalty units.** `effective_amp = |max_amp| + reg_penalty + outer_reg_penalty` (fitting.py:1423) adds a geometry-squared penalty (px², rad²) to a harmonic amplitude (intensity units). The blend is a useful heuristic — the benchmark evidence in doc 08 supports it — but the relative weighting silently depends on the image's flux scale. Worth one explicit sentence in docs 07/08 (and in the paper) stating the heuristic nature.
- **U5. Units in result dicts** (from the plotting review): `pa` in radians, `eps` dimensionless, `x0/y0` pixels, `intens` in image flux units — none stated in the plotting or fitting docstrings. One "Units and conventions" box in doc 02 would settle all of these.

## Documentation inconsistencies — **FIXED 2026-07-18**

> **Status: FIXED (2026-07-18, branch `fix/pre-publication-review`).** D1: doc 02 now says 62 parameters / 14 groups, `minit=6`, `nclip=1`, inward-first growth, a new First Isophote Robustness section documenting `max_retry_first_isophote`/`first_isophote_fail_count`, and the corrected `build_isoster_model` limitation (filters non-finite values, not stop codes). D2: doc 01 gives the real retry default (3), the legacy-FITS `config=None` behavior (no header reconstruction), the `grad ≥ 0` LSB trigger, single-Table export, and the `utils.py` docstring example no longer uses the deprecated kwarg. D3: doc 07's four `outer_reg_use_selector` references rewritten (selector layer is always on, no toggle), the §2.1 anchor text now matches §6 (marker on the committing trigger), `_fit_image_free` → inline `fit_image`, test count 698, dangling journal references removed. D4: `_is_acceptable_stop_code` spelled correctly, legacy-CONFIG paragraph corrected, `docs/agent/` references reworded for external readers, archive path fixed; the `optimize.py` facade claim is now true (R1). D5: archive path and `build_isoster_model` filter claim corrected. D6: CLAUDE.md points to `docs/index.md`. D7: `config.py:22` points to `docs/01-user-guide.md`. D8: `fflag` description now says sigma-clipped points only (masked points are removed at sampling). D9: `model.py` docstring sign fixed (`r·(1 − Σ[...])`) with an explanatory sentence — the code was already correct. D10: fixed with the V-batch (62 fields, module inventory, inline `fit_image`). D11: test counts updated to 698 passed / 5 deselected.

- **D1. `docs/02-configuration-reference.md`** — stale in several places: claims "43 parameters organized into 11 functional groups" (actual: **62 fields**, and the doc itself lists 13 groups); `minit` default listed as `10` (code: **6**); `nclip` default listed as `0` (code: **1**); §2 says the driver grows "outward to maxsma, then inward" while §13 of the same document (correctly) says inward-first; `max_retry_first_isophote`/`first_isophote_fail_count` are not documented at all; Known-limitation #2 ("build_isoster_model accepts all rows with sma>0 regardless of stop code **or NaN content**") is half-stale — the code filters non-finite values (it does not filter stop codes).
- **D2. `docs/01-user-guide.md`** — `max_retry_first_isophote` documented as "(default `0`, disabled)" (code: **default 3**); claims legacy FITS files get "header-keyword reconstruction automatically" (code returns `config=None`; no such fallback exists — same wrong claim in `04-architecture.md:162`); the LSB detector description omits the `grad >= 0` trigger (present in `driver.py:82-83` and in doc 07); "returns one or more astropy.table.Table objects" (returns a single Table); the FITS example in `utils.py:157-158` still uses the deprecated `template_isophotes=` kwarg.
- **D3. `docs/07-lsb-features.md`** — documents a config field **`outer_reg_use_selector`** in four places; the field does not exist (doc 08 §2.7 correctly states the selector layer has "no independent toggle"). Self-contradiction on `lsb_auto_lock_anchor` (§2.1 vs §6, see U1). References a `_fit_image_free` function that does not exist (regular mode is inline in `fit_image`). "347 passing" test count is stale (654 today). References `docs/journal/2026-04-14_handover*.md`, which does not exist at that path.
- **D4. `docs/04-architecture.md`** — `optimize.py` claim (see R1); legacy-CONFIG claim (see D2); public doc references untracked internal files (`docs/agent/plan-2026-04-29-multiband-feasibility.md`), which external readers and reviewers cannot see; "Historical records live under `docs/archive/`" — that path does not exist (it is `docs/agent/archive/`, untracked); `is_acceptable_stop_code()` should be `_is_acceptable_stop_code`.
- **D5. `docs/03-algorithm.md`** — "Reference: `docs/archive/review/autoprof-3-variance-map-error-propagation.md`" dangles (real location: `docs/agent/archive/review/`, untracked); "build_isoster_model currently filters only on sma > 0" is stale (also filters non-finite values).
- **D6. `CLAUDE.md`** — the Documentation Index lists `docs/00-index.md`; the actual file is `docs/index.md` (and `mkdocs.yml` points to `index.md`).
- **D7. `isoster/config.py:22`** — class docstring points to "docs/user-guide.md"; the actual file is `docs/01-user-guide.md`.
- **D8. `config.py` field description for `fflag`** says "masked + clipped", but the code counts only sigma-clipped points against `fflag` (masked points are removed during sampling, before `total_points` is computed). Doc 02 §5 has the correct description; the config field description is the wrong one.
- **D9. `isoster/model.py` docstring (line 79)** — the Notes formula `r_corrected = r * (1 + Σ[aₙ sin + bₙ cos])` has the wrong sign: the code's `r·(1 − δ)` is **verified correct** (hand-built isophotes with planted `b4=+0.03`: harmonics-on residual 0.62% vs 8.2% harmonics-off; flipped-sign control 16.5%). Fix the docstring, not the code.
- **D10. `docs/technical/1.2`** — repeats the stale "43 parameters" claim (twice); "three auxiliary modules" followed by a list of four; mode table names a `_fit_image_regular` driver that does not exist; `numba_kernels.py` description credits it with a "joint multi-band design-matrix builder" and a "dominant-coefficient selector" that live elsewhere/do not exist.
- **D11. Test-count drift** — doc 07 says 347 tests; current suite is 654 collected / 654 passing.
- See also M13 (multiband docstring/schema drift, including the stale Stage-1 docstrings and the `10-multiband.md:358` writer-name error).

## Smaller code observations (nits)

> **Status (2026-07-17, branch `fix/pre-publication-review`):** `cli.py:95` fixed (single-HDU no-data files now raise a clear `ValueError` instead of `IndexError`; same for the mask path) and `cli.py:108` simplified (redundant conditional removed). `cog.py:50` dead clause removed (the per-isophote `fix_eps`/`fix_pa` keys never exist; the `fix_geometry` early return already covers the intent). `fitting.py` center-shift clip difference now has an explicit comment (simultaneous both-axes vector clip gets the `max(clip_max_shift, 0.05·sma)` floor; the single-axis 'largest' path uses the plain clip). `fit_central_pixel` `intens_err` is addressed via M12 (WLS now reports `sqrt(var)`; OLS keeps 0.0, the established two-path convention). **Masked central pixel row: CONFIRMED INTENDED by the author (2026-07-18) — no change.** The `intens=NaN`, `stop_code=-1`, `valid=False` central row is deliberate, self-describing metadata (documented at `docs/01-user-guide.md:481`). Repo hygiene (`/reference/` tracked despite the ignore rule; `AGENTS.md` gitignored) — both RESOLVED 2026-07-18: `reference/` untracked (commit `f59a328`, also removing the GPL-3.0 AutoProf copy from the public repo; files stay on disk) and `AGENTS.md` tracked.

- `cli.py:95` — `hdul[1]` fallback raises `IndexError` on a single-HDU file whose primary has no data; `cli.py:108` — `args.template if args.template else None` is redundant.
- `driver.py:723-727` — central pixel is prepended when `minsma <= 0.0` even if it is invalid (masked); documented behavior, but the resulting NaN `intens` row lands in the FITS table — fine, just confirm this is intended for publication examples.
- `cog.py:50` — `iso.get("fix_eps", ...)` / `iso.get("fix_pa", ...)` read per-isophote keys that never exist; the early-return branch is effectively dead (behavior is still correct via `fix_geometry`).
- Simultaneous vs largest update modes clip center shifts differently: `max(clip_max_shift, 0.05·sma)` (fitting.py:1598) vs plain `clip_max_shift` (fitting.py:1659,1673). If intentional, one comment saying so would help.
- `fit_central_pixel` reports `intens_err=0.0` — exact zero reads as "infinitely precise"; `NaN` would be more honest (multiband already does better here — see M12 reverse drift).
- `docs/02` line ~61: "Validated: maxsma > minsma" — the validator actually allows equality (raises only when `maxsma < minsma`).
- Repo hygiene: `/reference/` is gitignored yet 30 files under it are tracked (committed before the rule). Either untrack or drop the ignore rule; also note `AGENTS.md` itself is gitignored, so external reviewers will not see the conventions file.
- `.gitignore`d `docs/publication/` — good; just flagging that `docs/journal/` contains exactly one *tracked* file while `docs/agent/journal/` is untracked, which is slightly confusing.

---

## Part V — `docs/technical/` (1.x series) + `docs/reference/` (delegated review; spot-verified)

All 14 files in scope were read end-to-end by the delegated reviewer and every load-bearing claim was checked against the code. The reviewer independently re-verified the highest-priority items line-by-line (marked confirmed). These docs read like the draft of the paper's technical chapter — the formula-level errors below are exactly what a journal referee will check first, and they must not propagate into the manuscript.

### V1 — Formula-level errors (highest referee risk) — **FIXED 2026-07-18**

> **Status: FIXED (2026-07-18, branch `fix/pre-publication-review`).** All six items corrected in the technical docs: V1.1 (bilinear order=1 for intensity/variance, order=0 for mask — no cubic mode), V1.2 (convergence criterion now written as `|a_max| < conver·S(a)·σ_eff` with `S(a)` defined everywhere it appears, including the previously undefined `scale` in `algorithm-walkthrough.md`), V1.3 (Tikhonov formula rewritten as α = λ·w·J_g²/(1+λ·w·J_g²) with J_g the harmonic→geometry Jacobian; narrative corrected — α is step-size-independent and grows with gradient shallowness), V1.4 (intens_err documented as the fit covariance `(AᵀWA)⁻¹[0,0]` matching the B2 fix; gradient means are unweighted with `1/Σσᵢ⁻²` per-mean variances; σ_eff unchanged by WLS), V1.5 (one-sided gradient = one extra pass, lazy 2→1, not "three passes to one"), V1.6 (photutils samples/fits harmonics in polar angle φ; the `contour_photutils` docstring premise in `isoster/plotting.py` also corrected per the erratum).

- **V1.1 "Cubic interpolation" (confirmed).** `1.1:14-16` and `1.3:81-82` say sampling uses `map_coordinates` "with cubic interpolation". Code uses `order=1` (bilinear) for intensity and variance and `order=0` for the mask (`sampling.py:144,152,162`). No cubic mode exists anywhere.
- **V1.2 Convergence criterion omits the scale factor (confirmed).** Five files (`1.1:55,72-73`, `1.4.3:13`, `1.4.4:239-241`, `1.4.6:33`) write `|a_max| < c·σ_eff`. Code: `abs(max_amp) < conver * convergence_scale * effective_rms` (`fitting.py:1539`), with the default `'sector_area'` scale `max(1, sma·Δsma·angular_width)` (`fitting.py:1153-1162`) — a factor of ~10 at sma=100. `convergence_scaling` is mentioned in **none** of the 14 files; `algorithm-walkthrough.md:83,310` includes `scale` but never defines it, so the chapter is also internally inconsistent.
- **V1.3 Tikhonov shrinkage formula uses the wrong variable (confirmed).** `1.4.3:146-167` writes α = λ(a)·w_g·(Δg)²/(1 + λ(a)·w_g·(Δg)²), with the narrative "small steps barely affected; large steps damped most strongly". Code: α = λ·w·**coeff²**/(1 + λ·w·**coeff²**) where `coeff` is the harmonic→geometry Jacobian (`_shared.py:60-104`). α is **independent of the step size** and grows with gradient *shallowness* — i.e., it engages exactly in the LSB regime it was built for. The mechanism description is materially wrong.
- **V1.4 WLS formulas the code does not use (confirmed; same root cause as B2).** `1.4.2:110-122` claims the WLS `intens_err` is the exact `1/√(Σσᵢ⁻²)` — `fit_isophote` uses `√(Σσᵢ²/N²)` (see B2); only forced photometry uses the exact form. `1.4.2:128-165` describes a gradient from two *weighted* means with a weight-combined error "used internally when a variance map is present" — the code's gradient means are plain `np.mean`/`np.median` with per-mean variances `Σσ²/N²` (`fitting.py:836-875`). `1.4.6:85-86` says WLS changes σ_eff — σ_eff is the unweighted residual std regardless (`fitting.py:1536-1537`). `algorithm-walkthrough.md:196` has the correct `intens_err` formula, so the chapter contradicts itself.
- **V1.5 Gradient cost framing (confirmed).** `1.3:37-44,246-247` and `1.1:64-66` describe a gradient sampled at two SMAs offset *above and below* the current one, "tripling" the per-iteration sampling cost, with lazy mode reducing "three passes to one". Both isoster and classical photutils use a **one-sided** difference: the current sample is reused, so the gradient costs **one** extra pass (doubling, not tripling), and lazy evaluation goes 2→1 (`fitting.py:792-935`; `reference/sample.py:335-347`). The "~45%" lazy-gradient saving in `algorithm-walkthrough.md:111` is unsourced under either framing.
- **V1.6 photutils harmonic basis (confirmed — see erratum).** `1.4.6:121-134` claims photutils fits and reports harmonics in eccentric-anomaly space. The vendored copy samples in **polar angle** φ (adaptive polar-angle step; `radius(phi)` is the polar-radius law — `reference/sample.py:155-223`, `reference/geometry.py:279-295`) and fits harmonics on those polar angles (`reference/harmonics.py:27-56,97-135`, `fitter.py:174`). The doc contradicts `1.1:106-109` and the `1.5.4` feature matrix ("Eccentric-anomaly sampling: photutils —"), which are correct.

> **Erratum (the reviewer correcting one of its own delegated checks).** The plotting deep-dive had cleared `contour_photutils`'s premise ("Photutils stores harmonic radius perturbations in psi", `plotting.py:679-680`) as accurate. That was wrong on the basis point: photutils' stored coefficients are normalized by `sma·|grad|` (`reference/isophote.py:287-288`) but live in the **polar-angle** basis, not ψ. The numerical impact on overlays is modest (the bases coincide to first order), but the docstring premise is false and should be fixed alongside `1.4.6`.

### V2 — Behavior/mechanism descriptions that contradict the code — **FIXED 2026-07-18**

> **Status: FIXED (2026-07-18, branch `fix/pre-publication-review`).** All six items corrected: V2.1 (growth loops terminate only at radius limits; stop codes suspend geometry propagation in strict mode while growth continues), V2.2 (retry default 3, the real retry schedule with near-circular eps settings, `first_isophote_failure` only when anchor + all retries/probes fail, probe loop only on anchor failure), V2.3 (outer_reg_strength default 2.0, sma_width None → auto 0.4×onset, three-key weights, flux-weighted circular 2·PA mean, silent anchor fallback), V2.4 (anchor marker goes on the committing trigger isophote — the last of the debounce streak), V2.5 (single-band 'original' post-hoc matrix has no geometry columns; default post-hoc fits each order separately; multiband `simultaneous_original` differs deliberately), V2.6 (default pipeline uses the polar-radius law in φ; EA formula marked as the psi-mode variant).

- **V2.1 "Codes {3, −1} terminate the current growth direction" (confirmed).** `1.4.6:56-57`, `algorithm-walkthrough.md:318-319`. Neither growth loop terminates on a stop code: the inward loop breaks only at `max(minsma, 0.5)` and the outward loop only at `maxsma` (`driver.py:611-613, 668-670` — verified: these are the only `break` statements in the growth loops). Unacceptable codes suspend geometry *propagation* in strict mode; growth continues to the radius limit — the "cascade" behavior `1.4.4.2:90-92` itself describes. The two files contradict the code and, implicitly, §1.4.4.2.
- **V2.2 First-isophote robustness section (confirmed).** `1.4.4` says: `max_retry_first_isophote` default 0 (actual **3** — also falsifying "none of the primitives is enabled by default", and self-contradicting `1.4.4.4`/`1.4.4.7`, which state gradient-SNR damping and step clipping are always on); a retry schedule "×(1.5, 0.7, 2.0, 0.5, 3.0) with eps offset by a small delta" (actual `(0.8, same), (1.3, same), (0.6, eps=0.05), (1.5, pa+π/4), (0.5, eps=0.05, pa+π/2)` plus an extended cycle — eps is **set** near-circular, not offset; `driver.py:36-49`); `first_isophote_failure` set "whenever retries were necessary" (actual: only when the anchor **and** all retry/probe attempts failed; `driver.py:556-582, 743-744`); and `FIRST_FEW_ISOPHOTE_FAILURE` fires "even when the anchor itself converged" (actual: the probe loop is entered only when the anchor failed; `driver.py:558-591`).
- **V2.3 Outer-regularization defaults and mechanism (confirmed).** `1.4.3` says: `outer_reg_strength` default 0 (actual **2.0**, `config.py:234-235`); Δa_ramp default recommendation "0.1·a_max" (actual default `None`, auto-computed as 0.4·`outer_reg_sma_onset`, `config.py:271-276`); weights default "{x0:1, y0:1, ε:1, PA:1}" over four axes (actual three keys `{"center", "eps", "pa"}`, `config.py:243-244` — the doc's own demo and both reference files use the right keys, so it contradicts itself); and an arithmetic flux-weighted mean of PA for the inner reference (code uses a flux-weighted **circular** mean on 2·PA with a resultant-length fallback to the anchor PA, `driver.py:191-203` — the naive mean is wrong mod π). Also: the claimed `UserWarning` on an empty inner reference does not exist; the code falls back to the anchor silently.
- **V2.4 `lsb_auto_lock_anchor` placement (confirmed; extends U1).** `1.4.3:300-303` says the marker "appears only on the first locked isophote (the one immediately following the anchor)". Code marks the **committing trigger** isophote — the last of the debounce streak, which was itself fit free (`driver.py:701-713`). For debounce ≥ 2 that is not the isophote immediately following the anchor. A third doc variant of the U1 confusion.
- **V2.5 ISOFIT post-hoc solve descriptions (reported).** `1.4.1:203-211` says `'original'` post-hoc applies "the joint solve" on the extended matrix (defined to include the geometry columns); `fit_higher_harmonics_simultaneous`'s design matrix is `[1, sin(nθ), cos(nθ), …]` with **no geometry columns** (`fitting.py:727-736, 1053-1056`). The multiband `simultaneous_original` *does* solve the full system post-hoc (`fitting_mb.py:2520-2539`), so the two paths silently differ. Similarly `1.4.1:156-158` describes the default post-hoc path as "the same five-parameter fit with geometry fixed" — it actually fits each order separately with a 3-column matrix (`compute_deviations`).
- **V2.6 §1.3's sampling formula is the EA parametrization (reported).** `1.3:88-95` displays `x = x0 + a·cosθ·cosPA − a(1−ε)·sinθ·sinPA` — uniform θ there is uniform-in-ψ, whereas the default pipeline steps φ uniformly with the polar-radius law (`numba_kernels.py:248-299`). As the generic sampling formula in the speed section it misrepresents the default mode and clashes with `1.1:106-118`.

### V3 — Stale names, numbers, and paths — **FIXED 2026-07-18**

> **Status: FIXED (2026-07-18, branch `fix/pre-publication-review`).** All items corrected: 62 fields/16 groups everywhere (including `1.2` and the configuration decision tree items within the technical docs), numba kernel inventory corrected in `1.2`/`1.3`/`1.6` (MB design-matrix builder lives in `numba_kernels_mb.py`; dominant-coefficient selection is plain `np.argmax`), `_fit_image_regular` replaced with the inline `fit_image` path, MB CLI examples use positional images + `--variance-maps`, the nonexistent `validate_alignment` claim removed, the `1.4.5` demo config includes the required `reference_band`, loose-validity defaults stated as count 6 / frac 0.2, the photutils CoG feature-matrix entry corrected (no CoG exists in photutils), `1.4.5.9`/`1.4.5.6` now state the raw-stored (joint modes) vs fit-time-normalized (`independent`) split, `1.0`'s stale draft status updated, and the external data-source paths (`isophote_test/`, `sga_isoster/`) are flagged as external companion repositories rather than resolvable repo paths.

- The 43-parameter claim also lives in `1.2:14,115-120` and `configuration-decision-tree.md:363-364` (extends D1/D10); 1.2's 13-group list omits three real groups.
- Numba kernel inventory (extends D10): `1.2:16`, `1.3:115-136`, `1.6:259-261` credit `numba_kernels.py` with a joint multi-band design-matrix builder (lives in `multiband/numba_kernels_mb.py`) and a dominant-coefficient selector (a plain `np.argmax`, `fitting.py:1409` — never JIT-compiled).
- `_fit_image_regular` in `1.2:60-63` and `algorithm-walkthrough.md:51-54` (extends D10).
- The multiband CLI example (`1.4.5:416-424`) uses `--image`/`--variance`; the real CLI takes positional image arguments and `--variance-maps` — neither `--image` nor `--variance` exists (confirmed against `cli_mb.py`).
- `validate_alignment` helper (`1.4.5:444-447`, also `04-architecture.md:29`) — no such function anywhere in the package (confirmed: repo-wide grep).
- The `1.4.5` demo recipe omits the **required** `reference_band` field (`config_mb.py:63-64` `Field(...)`; the snippet raises `ValidationError` as written).
- `1.6:98-107` claims the loose-validity drop threshold defaults to "fewer than half the rectangular sample count"; actual `loose_validity_min_per_band_frac = 0.2` (`config_mb.py:157-158`, confirmed).
- Feature matrix "Curve-of-growth attachment: photutils ✓" (`1.5:298`) — no CoG exists in `photutils.isophote` or the vendored copy (confirmed).
- `1.4.5.9` says the MB output dict contains "Bender-normalized" coefficients; `1.4.5.3`/`1.4.5.5` and the code say the schema stores **raw** coefficients with normalization at plot/audit time (confirmed — the same raw-vs-normalized split as P1).
- Data-source paths (`isophote_test/inputs/demos/`, `sga_isoster/data/demo/`) in six places point to directories outside this repo — unresolvable for a referee (reported; `mockgal/SKILL.md` confirms `isophote_test` is an external sibling).
- Stale draft status in `1.0:52-54` ("§1.3–§1.6 pending" — all exist and are complete) (reported).

### V4 — Internal inconsistencies and smaller items (condensed, reported) — **FIXED 2026-07-18**

> **Status: FIXED (2026-07-18, branch `fix/pre-publication-review`).** Applicable items corrected in the technical docs: floor not ceil for sample counts, WLS/OLS error ratio stated once ("1.5 to 2"), byte-identical-to-photutils claim softened, `'solver'` pull relocated to the geometry-update equations, "WLS/OLS differ only in `*_err`" qualified for non-uniform variance maps, `1.4.5` now mentions `harmonic_combination`/`fit_per_band_intens_jointly`/`band_weights`, SNR-damping floor 0.1, σ_bg floor uses post-clip survivor count, the "silently falls back … emits RuntimeWarning" oxymoron, cross-terms claim scoped, EXPERIMENTAL banner no longer claims a schema version, "same serialized configuration content", forced mode's non-use of `ACCEPTABLE_STOP_CODES`, five debug fields (not three), and the `grad ≥ 0` trigger added to the walkthrough's LSB detector list. Two nits reported but not found in the technical files (NumPy-fallback slowdown range, fflag prose) live outside them and were left. **Leftovers closed 2026-07-18:** `docs/reference/configuration-decision-tree.md` now says 62 parameters / 14 groups, and `config.py`'s `outer_reg_strength` field description now gives the canonical 1–3 range (matching 1.4.3; the 2–8 outlier is gone). Note for the manuscript: `docs/publication/draft/section-1.2-implementation-overview.md` (gitignored draft) still says "43 parameters across 13 functional groups" — fix at manuscript time (see Q9).

- Sample count ceil vs floor: three files write ⌈2πa⌉; the walkthrough and the code use floor (`sampling.py:135`).
- WLS/OLS error-bar ratio "1.2–2.1" vs "1.5–2" in the *same* file (`1.4.2:104` vs `:349`).
- `1.5.2` claims the OLS path is "byte-identical to photutils … up to LAPACK determinism"; `1.3.4` explicitly disclaims any numerical-agreement claim — and the claim is shaky anyway (different defaults, `minit`, sampling).
- `outer_reg_strength` recommended range given three ways (1–3 in 1.4.3, 2–8 in `config.py`, 1.0–3.0 in the decision tree).
- `'solver'` pull location (`algorithm-walkthrough.md:271-273` says inside the linear system; actually applied in the geometry-update equations, `fitting.py:1584-1713`).
- `1.4.2` "WLS/OLS differ only in the `*_err` fields" — only approximately true for non-uniform variance maps.
- `1.4.5`'s algorithm description omits `harmonic_combination`, `fit_per_band_intens_jointly`, and `band_weights` entirely, though all three change the design matrix or the solve.
- Nits: SNR-damping floor is 0.1, not "decays toward zero" (`fitting.py:1571-1573`); σ_bg floor uses the post-clip survivor count; fflag prose vs its own table; "silently falls back … emits a RuntimeWarning" oxymoron; cross-terms vanish only for complete uniform rings; EXPERIMENTAL banner carries no schema version; NumPy-fallback slowdown "3–5×" vs "2–5×"; forced mode never consults `ACCEPTABLE_STOP_CODES` (`driver.py:760-821`); "same on-disk bytes regardless of format"; debug adds five fields, not three; the walkthrough's LSB trigger list omits the `grad ≥ 0` trigger (`driver.py:82-83`).

### V5 — Verified correct in the technical docs (no action)

- §1.4.1's math (r(φ), ds/dφ, ds/dψ, the a/b vs (a/b)² uniformity order-of-magnitude, tanθ = (1−ε)tanψ) — checked analytically.
- Sampling counts (`max(64, int(2π·sma))`), EA-mode conventions, geometry-update coefficients and clipping rules, the stop-code trigger table, best-iteration selector, lazy-gradient triggers, integrator semantics, WLS normal-equation algebra in `compute_parameter_errors`, the LSB-lock machinery, pipeline structure, serialization layout, ISOFIT design matrices, and the decision-tree presets — all match the code.
- The "10–15× faster" headline appears in **none** of the 14 technical files (they defer numbers to the benchmark chapters) — no internal contradiction there.

## Test coverage assessment

- 654 tests pass (80 s), organized into unit / integration / validation / multiband / real_data (5 deselected = real_data). Ruff and ruff-format are clean. 264 multiband tests collect cleanly.
- Gaps that would have caught this review's bugs:
  1. **Noise-free harmonic recovery** (B1) — plant a known `b4`, fit with defaults, assert non-zero recovery.
  2. **FITS round-trip of feature outputs** (B3) — fit with `lsb_auto_lock=True`, save, load, assert marker counts.
  3. **Variance-map sentinel robustness** (B2) — NaN/inf in variance map, assert finite sane `intens_err` and no spurious `stop_code=-1`.
  4. **Schema uniformity test** (B5, M5, M11) — assert every isophote dict in one result has the same key set under representative configs (custom `harmonic_orders`, central pixel, lock on, dropped bands).
  5. **ref × loose-validity combination test** (M1) — reference band not first, one band dropped; assert either a clear stop code or correct recovery.
  6. **Forced-mode parity test** (M4) — same ring with an outlier through single-band and multiband forced photometry; assert both clip.
  7. **QA-figure regression test** (P1) — fit a mock with a known `b4`, render `plot_qa_summary`, assert the plotted harmonic amplitude matches the stored value.

## Improvement opportunities (for discussion, not blockers)

1. Unify the WLS intensity-error path on the exact weighted-mean covariance (fixes B2 and M10, and the forced/free inconsistency in one move).
2. Enforce a uniform per-isophote schema at result-build time (one place in `driver._build_fit_result`): it prevents B3/B5/M5-class issues permanently and makes the FITS/Table contract trivial to document.
3. Replace `leastsq` in `compute_deviations` with the explicit design-matrix solve used elsewhere (fixes B1, removes a dependency quirk, aligns OLS/WLS structure).
4. Decide **one harmonic-coefficient convention** (stored = Bender-normalized, presumably) and make fitters, plotters, and docs follow it (fixes P1, the multiband raw-vs-normalized split, and the `A_n/I` third convention in the extended figure). Add a "Units and conventions" box to doc 02 (covers U3/U5 and the radians/degrees split). — **RESOLVED differently 2026-07-18: the author chose to keep the deliberate raw/normalized split** (see Question 6 below); P1 is fixed at the plotting layer, the extended figure's `A_n/I` mode is an explicitly labeled display option, and the "Units and Conventions" box exists in doc 02.
5. Route the three triplicated image panels in `plotting.py` through the existing `_compute_residual_map`/`_plot_residual_panel` helpers (P5) — the P2 sign bug is the drift this prevents.
6. Port the multiband improvements back to single-band (M12 reverse drift: forced-mode SEM, central-pixel WLS error) or state the difference deliberately in the docs.
7. The "10–15x faster than photutils" headline (README, CLAUDE.md, CITATION.cff) should be pinned to a reproducible benchmark table in the paper (dataset, sizes, versions) — currently it is an unqualified claim repeated in three places. — **PARTIALLY ADDRESSED 2026-07-18:** the README claim now points to `docs/09-exhausted-benchmark.md` as the reproducible anchor. CLAUDE.md and CITATION.cff still carry the bare claim (CLAUDE.md is an internal guide; the CITATION.cff abstract describes the project's nature). The remaining piece is the paper's own benchmark table with dataset/sizes/versions — author's call at manuscript time.
8. Version is `0.1.0` everywhere; decide the release version (presumably ≥ 1.0 for a journal release) and keep `CITATION.cff` in sync. — **RESOLVED 2026-07-18: bumped to `1.0.0`** in `pyproject.toml`, `CITATION.cff` (plus `date-released: 2026-07-18`), and the tracked `isoster.egg-info/PKG-INFO` (regenerated on next build anyway). The README headline now anchors the "10–15× faster" claim to the benchmark doc (see also the improvement-opportunity item above, which shares this concern).
9. Consider a light dead-code check in CI (e.g. `vulture` whitelist) for R1/R2/B4/M13-class artifacts.
10. `docs/02` duplicates `## Photometry Outputs`; merge.

## Questions for the author

1. **License:** confirm BSD-3-Clause is intended (fix `LICENSE`), or update all metadata to GPL-2.0. — **RESOLVED 2026-07-18: author confirmed BSD-3-Clause; `LICENSE` replaced (B0 fixed).**
2. **`lsb_auto_lock_anchor`:** should the flag mark the commit point (current behavior) or the geometry anchor (what the name says)? Should the true anchor isophote's identity be recorded in the result dict? — **RESOLVED 2026-07-18: the flag keeps marking the commit point (docs aligned), and the true anchor is now recorded as `lsb_auto_lock_anchor_sma` in both drivers, round-tripping through FITS (new META HDU) and ASDF.**
3. **Selector-penalty units (U4):** is the intensity/px² blend documented deliberately anywhere besides the code? Should the paper state the heuristic explicitly? — **Docs 07/08 now state it (2026-07-18); whether the paper should is OPEN.**
4. **Public-doc references to untracked `docs/agent/` files:** strip them for the publication snapshot, or move the referenced plans into tracked docs? — **RESOLVED 2026-07-18: public docs reworded to describe them as internal records, not linked paths.**
5. **Release versioning:** target version for the journal submission, and should `CITATION.cff` track it? — **RESOLVED 2026-07-18: `1.0.0`** in `pyproject.toml` and `CITATION.cff` (with `date-released: 2026-07-18`).
6. **Harmonic convention (P1/U3):** confirm the intended stored convention is "Bender-normalized, dimensionless fractional-radius perturbation" for *all* fitters (single-band stores this; the multiband joint path stores raw coefficients per `fitting_mb.py:2360-2363`). Should the extended figure's `A_n/I` option be renamed or removed to avoid a third convention? — **RESOLVED 2026-07-18 (author decision: keep the deliberate split).** Storage conventions stay as designed: single-band and MB `independent` store Bender-normalized values; MB joint modes store the raw shared coefficient (D16 — its identity across bands is the visible evidence of one shared number, and per-band normalization happens at plot/audit time). All plotters split correctly per source (P1). The extended figure's `A_n/I` mode stays: it is an explicitly labeled display option (`$A_n / I$` ylabel), not a hidden third convention.
7. **ref × loose validity (M1):** is this combination meant to be supported (docs say yes) or should it hard-error until fixed? — **RESOLVED 2026-07-17: supported and now correct (M1 fixed).**
8. **AutoProf harmonic convention:** the delegated review could not determine whether AutoProf's stored `a3/b3/a4/b4` are raw or normalized; if AutoProf rows are ever fed to `plot_qa_summary`'s harmonic panel, P1's direction depends on this. Do you know which it is? — **RESOLVED 2026-07-17: normalized — AutoProf builds a photutils `Ellipse` list and photutils stores `up_coeffs/(sma·|grad|)` (`reference/autoprof/.../Isophote_Extract.py:1008-1010`, `reference/isophote.py:287-288`).**
9. **Paper inheritance:** will the manuscript's technical section draw directly from `docs/technical/` (1.x series)? If so, Part V's formula-level items (V1.1–V1.6, V2.1–V2.4) should be fixed there *and* checked again in the manuscript text before submission. — **RESOLVED 2026-07-18:** the manuscript draft at `docs/publication/draft/` was verified byte-identical to the PRE-FIX technical docs (git `main`), so it carried every V1–V4 error. All 12 draft files were re-synced from the corrected `docs/technical/` sources (no manuscript-specific edits existed, so nothing was lost; draft stays gitignored/local). Verified no stale items remain (`43 parameters`, `cubic`, `_fit_image_regular`, `validate_alignment`, stale defaults). Also fixed the group-count wording collision it surfaced: doc 02 now says "14 sections" while `1.2` says "16 functional groups" (different structures, no apparent contradiction).

---

## Appendix A — Verification log

| Check | Command / method | Result |
|---|---|---|
| Full test suite | `uv run pytest tests/ -x -q` | **654 passed**, 5 deselected, 80.5 s |
| Lint | `uv run ruff check isoster/`; `ruff format --check` | clean; 24 files formatted |
| Example | `uv run python examples/example_basic_usage/basic_usage.py` | OK, 34/34 isophotes converged, figure written |
| Config field count | `len(IsosterConfig.model_fields)` | **62** (docs claim 43) |
| B1 direct | `compute_deviations` on noise-free planted `b4=0.03` | returns exact `0.0`; `leastsq` coeffs correct, `cov_x=None` |
| B1 end-to-end | `fit_image` on noise-free mock | `b4=0.0` on all converged isophotes |
| B2 | `fit_image` with 15-NaN variance map | `intens_err=2.9e12` at one isophote; spurious `stop_code=-1` nearby |
| B3 | FITS write/read of documented lsb-key pattern | all rows read `lsb_auto_lock_anchor=True`; inward rows read `lsb_locked=True` |
| B5 | `harmonic_orders=[3,4,5,6]`, `minsma=0` | central pixel lacks `a5/b5/a6/b6`; masked after FITS export |
| D9 model sign | hand-built isophotes `b4=+0.03` → `build_isoster_model` | harmonics-on 0.62% vs off 8.2% residual std; flipped control 16.5% ⇒ code correct, docstring wrong |
| P1 | code read (`plotting.py:989-995` vs `fitting.py:560-565`) + delegated numeric check | planted raw A4=2.0 → stored 0.02 → plotted 0.0002 (double division) |
| P2 | code read (`plotting.py:467-472` vs `1667-1669`) | label sign contradicts function |
| P3 | grep of `styles.get(...)/style["color"]` pattern | 9 crash sites confirmed |
| M1 | code read (`fitting_mb.py:1340-1399`) + delegated runtime repro | indexing mixes full/surviving band lists; LinAlgError swallowed; IndexError reproduced |
| M2 | delegated Monte-Carlo | σ(A1) factor √w low; weight-aware rescale matches truth |
| M4 | code read of `extract_forced_photometry_mb` in full | no sigma-clip call anywhere in the function |
| M5 | code read (`fitting_mb.py:1706-1717`) | harmonic columns written 0.0; contradicts own comment and doc |
| M8 | delegated runtime run | `ndata=237, nflag=-157` under loose validity |
| M11 | code read (`plotting_mb.py:218-220`) | `for n in (3, 4)` hardcoded |
| M3 | code read (`fitting_mb.py:1024-1030`) | loose OLS branch skips residual rescale; comment admits it |
| M7 | code read (`fitting_mb.py:2059-2074` vs `1353`) | fallback uses `last_data` (pre-clip); clip returns new arrays |
| P4 (asinh) | code read (`plotting.py:2506-2513`) | same `None` softening passed per method → per-method scales |
| P4 (psi contour) | code read (`plotting.py:646-648` vs `model.py:296`) | overlay draws `sma·(1+δ)`; model's exact isophote is `sma/(1−δ)` |
| P4 (NaN→True) | code read (`plotting.py:2059-2062`, `335`) | `bool(np.nan)` is `True`; key always present after `build_method_profile` |
| V1.1 | grep `order=` in `sampling.py` | `order=1` intensity/variance, `order=0` mask; no cubic mode |
| V1.2 | code read (`fitting.py:1153-1162, 1539`) | `conver * convergence_scale * effective_rms`; sector_area scale ≈ sma·Δsma·2π/N |
| V1.3 | code read (`_shared.py:60-104`) | α = λ·w·coeff²/(1+λ·w·coeff²); no dependence on step size |
| V1.5 | code read (`reference/sample.py:335-347`) | one-sided gradient at `sma·(1+step)`, 2× retry; 2 passes total |
| V1.6 | code read (`reference/sample.py:155-223`, `harmonics.py`, `geometry.py:279-295`) | photutils samples/fits in polar angle φ — agent-0's ψ-basis claim was wrong |
| V2.1 | grep `break` in `driver.py` growth loops | only radius-limit breaks (613, 670); no stop-code termination |
| V2.2 | code read (`driver.py:36-49, 556-591`) | retry schedule/probe-loop semantics as reported |
| V2.3 | grep defaults in `config.py` | strength 2.0; weights `{center,eps,pa}`; sma_width `None` |
| V3 (misc) | greps: `validate_alignment` (none), `reference_band = Field(...)` (required), loose frac 0.2, no `cog` in photutils copy | all as reported |

## Appendix B — Delegated deep-dive summary

Three delegated full-file reviews were run in parallel with the reviewer's own pass. All three are complete and integrated above:

- **`isoster/plotting.py` (3 009 lines, read in full):** 10 bug-class findings (P1–P4), 6 redundancy items (P5), 9 unclear-definition items (P5/U5), docstring mismatches, and a verified-correct list (P6). Cross-checked against `fitting.py`, `model.py`, `_shared.py`, the multiband modules, `reference/` (photutils), benchmarks, tests, and one real output FITS profile. One of its cleared items (photutils harmonic basis) was later found wrong by the technical-docs review and is corrected in the V1.6 erratum.
- **`isoster/multiband/` (~8 000 lines, all files read in full):** 9 bugs (M1–M9), inheritance check of the single-band issues (M10), duplication/drift inventory (M12), docstring/schema mismatches (M13), and a verified-correct list (M14). Three runtime checks reproduced in the project venv.
- **`docs/technical/` (14 files) + `docs/reference/`:** 24 code-contradicting statements (V1–V3), 19 internal inconsistencies/overclaims (V4), ~12 nits, and a verified-correct list (V5). High-priority items were independently re-verified line-by-line by the reviewer (see Appendix A).
