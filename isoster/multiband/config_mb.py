"""
Multi-band isoster configuration (``IsosterConfigMB``).

Sibling of :class:`isoster.config.IsosterConfig` with a deliberately
reduced field set, plus multi-band-only fields. There is **no
inheritance** between the two classes: changes to ``IsosterConfig`` do
not propagate automatically. This is intentional — the multi-band path
is free to evolve independently of the single-band path, and a field
that exists in both is a field whose multi-band semantics were decided
deliberately rather than inherited by accident. See
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` (decision D23)
for the rationale.

Excluded fields (deliberately not copied):

- ``simultaneous_harmonics`` and ``isofit_mode`` (single-band ISOFIT
  API; the multi-band path uses the ``multiband_higher_harmonics``
  enum instead, see plan section 6).
- ``lsb_sma_threshold`` and the ``adaptive`` integrator option
  (single-band switches to the median integrator below an SMA
  threshold; the multi-band path instead switches integrator on the
  LSB auto-lock via ``lsb_auto_lock_integrator``).

All other single-band feature families are implemented here with
multi-band semantics: LSB auto-lock (``lsb_auto_lock_*``), outer-center
regularization (``use_outer_center_regularization`` and
``outer_reg_*``), central regularization (``central_reg_*``), and
curve-of-growth photometry (``compute_cog``). See the field
descriptions below.
"""

import re
import warnings
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# Band-name regex: must start with a letter, then letters/digits/underscores
# only. Hyphens and other punctuation are rejected so that band strings
# can be used verbatim as FITS column suffixes (e.g. ``intens_g``,
# ``intens_HSC_G``). See decision D8.
_BAND_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class IsosterConfigMB(BaseModel):
    """
    Configuration for multi-band isoster.

    The fitter consumes multiple aligned same-pixel-grid images of the
    same target, fits a single shared geometry per SMA, and reports
    per-band intensities and per-band harmonic deviations. See
    ``docs/10-multiband.md`` for the full user-facing reference.

    **Maturity.** The default configuration — shared validity, joint
    per-band intercepts, ``geometry_parameterized_solve=True`` and
    ``multiband_higher_harmonics='independent'``, in OLS or WLS — is
    supported as of 2026-08-15. Two options sit outside that guarantee,
    in different ways, and the difference matters:

    * ``multiband_higher_harmonics`` in
      ``{'simultaneous_in_loop', 'simultaneous_original'}`` is
      **experimental** and emits a ``UserWarning`` when selected.
    * ``loose_validity=True`` is **supported but non-default**. It is
      correct and tested end to end and does *not* warn; it is off by
      default only because turning it on changes every result on data with
      per-band masks. It is a masking-policy choice, not an unfinished
      feature.

    The B=5 path has synthetic coverage only.

    **Stop codes** are inherited from single-band isoster (see
    :class:`isoster.config.IsosterConfig`).
    """

    # --- Multi-band-only fields ---

    bands: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of band names. Must match the input images "
        "list element-for-element. Each name must match the regex "
        "``^[A-Za-z][A-Za-z0-9_]*$`` (letter then letters/digits/underscores). "
        "No duplicates. The strings appear verbatim as FITS column suffixes, "
        "e.g. ``bands=['g', 'r']`` produces ``intens_g`` and ``intens_r``.",
    )
    reference_band: str = Field(
        ...,
        description="Name of the band used for diagnostic and driver scalar "
        "decisions: ``sma0`` initial-guess fallback, first-isophote-retry "
        "intensity interpretation, single-scalar gradient/SNR for QA "
        "plotting. Must be one of ``bands``. Does not affect joint geometry "
        "or per-band intensity columns.",
    )
    band_weights: Optional[Union[Dict[str, float], List[float]]] = Field(
        default=None,
        description="Per-band scalar weights ``w_b`` for the joint normal "
        "equations. Each band's row block contributes ``A_b^T W_b A_b`` with "
        "``W_b = w_b · I`` in OLS, or ``W_b = diag(w_b / variance_b(pixel))`` "
        "in WLS. Either a dict keyed by band name (every band must have a key) "
        "or a list aligned with ``bands`` (length ``B``). Each weight must be "
        "positive and finite. ``None`` (default) means uniform weights "
        "``w_b = 1.0``.",
    )
    harmonic_combination: Literal["joint", "ref"] = Field(
        default="joint",
        description="Strategy for combining per-band harmonic information into "
        "the geometry update. ``'joint'`` (default): solve a single "
        "``(B + 4)``-column design matrix once per iteration (B per-band "
        "intercepts plus four shared geometry columns), with per-band "
        "``I0_b`` nuisance parameters and shared ``(A1, B1, A2, B2)`` "
        "geometric coefficients. ``'ref'``: drive geometry from the reference "
        "band only; other bands are passive (post-hoc intensity and harmonic "
        "extraction). Use ``'ref'`` for debugging or when the joint solver "
        "is suspected of misbehaving.",
    )
    geometry_parameterized_solve: bool = Field(
        default=True,
        description="Fit shared *geometry steps* instead of a shared harmonic "
        "amplitude.\n\n"
        "A common geometry step ``delta`` produces a ring harmonic of amplitude "
        "``delta * grad_b`` in band ``b``, so one shared amplitude across bands "
        "is a misspecified model whenever the bands have different radial "
        "gradients: it fits ``delta`` times a weight-averaged gradient, which "
        "then has to be divided back out by a separately pooled gradient. With "
        "``True`` each band's harmonic columns are scaled by that band's "
        "gradient, so the shared parameters are the geometry steps themselves.\n\n"
        "Effect: the effective per-band weight becomes ``w_b * grad_b**2 / "
        "var_b`` (minimum-variance) rather than ``w_b / var_b``. Measured on "
        "synthetic truth, this cuts the scatter of the recovered geometry by "
        "roughly 25% wherever the two weightings disagree — on HSC g/r/i the "
        "amplitude weighting was close to inverted against the bands' "
        "information content — and changes nothing where they agree. It also "
        "improves OLS, where the amplitude form weights bands equally "
        "regardless of how much geometric signal each carries.\n\n"
        "The scaling uses the previous iteration's per-band gradients, which "
        "are slowly-varying nuisance scale factors; the first iteration of each "
        "isophote falls back to the amplitude form.\n\n"
        "Threaded into every solve path — shared and loose validity, joint and "
        "decoupled intercepts, plain and simultaneous higher harmonics — so the "
        "parameterisation does not depend on which other options are set.\n\n"
        "Default ``True`` since 2026-08-15. Set ``False`` to reproduce earlier "
        "results or to compare the two parameterisations.",
    )
    fit_per_band_intens_jointly: bool = Field(
        default=True,
        description="Controls how per-band ``intens_<b>`` (a.k.a. the "
        "per-band intercept ``I0_b``) is computed at each isophote.\n"
        "\n"
        "When ``True`` (default, ``'joint'`` / coupled mode), the joint "
        "design matrix carries ``B`` per-band intercept columns alongside "
        "the shared ``(A1, B1, A2, B2)`` geometric harmonics, giving a "
        "``(B*N, B + 4)`` weighted-least-squares system. ``intens_<b>`` "
        "comes directly from the matrix solve, so the per-band intercept "
        "is co-fit with the harmonic deformations: on partial rings "
        "(masked sectors, loose validity) the joint solve absorbs the "
        "harmonic-shape coupling between bands and corrects each band's "
        "intercept accordingly. On full rings, ``sin(nφ)`` and ``cos(nφ)`` "
        "are mathematically orthogonal to the constant column over "
        "``[0, 2π]`` and the joint coefficient is numerically equivalent "
        "to the per-band inverse-variance-weighted mean.\n"
        "\n"
        "When ``False`` (``'ring_mean'`` / decoupled mode), the per-band "
        "intercept columns are dropped from the design matrix. "
        "``intens_<b>`` is computed independently as the per-band "
        "inverse-variance-weighted ring mean (or simple mean under OLS) "
        "and the geometric ``(A1, B1, A2, B2)`` solve runs on "
        "``intens - mean`` residuals. Use this when you have already "
        "subtracted sky upstream and want to interpret each band's "
        "intercept as a pure ring statistic decoupled from the harmonic "
        "fit. Mutually exclusive with ``harmonic_combination='ref'`` "
        "(ref-mode side-steps the joint solve entirely)."
        "\n\n"
        "**Note on integrator scope:** with ``integrator='mean'`` "
        "(default), both modes report a (weighted) ring mean for "
        "``intens_<b>`` — full rings are numerically identical since "
        "``sin(nφ)``/``cos(nφ)`` are orthogonal to the constant "
        "column over ``[0, 2π]``. With ``integrator='median'``, "
        "``intens_<b>`` is the per-band ring median — but only in "
        "this decoupled mode: the median is non-linear and cannot "
        "ride inside the joint matrix solve, so the validator "
        "hard-errors on ``integrator='median'`` combined with "
        "``fit_per_band_intens_jointly=True``. See the ``integrator`` "
        "field description for the full scope.",
    )

    # --- D9 backport: per-band loose validity ---
    loose_validity: bool = Field(
        default=False,
        description="Relax the cross-band shared-validity AND. With "
        "``False`` (default), a sample is dropped from every band if any "
        "band fails (mask, NaN, non-positive variance) at that location. "
        "With ``True``, each band keeps its own surviving samples; the "
        "joint design matrix becomes block-diagonal in the per-band "
        "intercept columns and uses each band's own kept angles. A band "
        "that falls below the per-band thresholds at a given isophote is "
        "dropped from the joint solve at that isophote (its ``intens_<b>`` "
        "is reported as NaN); the surviving bands still constrain the "
        "shared geometry. Whole-isophote ``stop_code=3`` only fires when "
        "fewer than 2 bands survive. Decision D9 backport (locked "
        "2026-05-01).",
    )
    loose_validity_min_per_band_count: int = Field(
        default=6,
        ge=1,
        description="Per-band absolute minimum surviving-sample count "
        "under loose validity. A band with fewer than this many kept "
        "samples at a given isophote is dropped from the joint solve at "
        "that isophote. Default 6 mirrors the single-band 5-parameter "
        "minimum + 1 sample of slack. Ignored when ``loose_validity=False``.",
    )
    loose_validity_min_per_band_frac: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Per-band minimum surviving-sample fraction under "
        "loose validity (kept / sampled). A band below this fraction at a "
        "given isophote is dropped from the joint solve at that isophote. "
        "Default 0.2. Ignored when ``loose_validity=False``.",
    )
    loose_validity_band_normalization: Literal["none", "per_band_count"] = Field(
        default="none",
        description="Per-band normalization of the joint design matrix "
        "and the combined-gradient combiner under loose validity. "
        "``'none'`` (default): each band's row block contributes "
        "proportionally to its own ``N_b``; ``w_b`` multiplies every row. "
        "Combined gradient = ``Σ w_b · grad_b / Σ w_b`` over surviving "
        "bands. ``'per_band_count'``: each band's design-matrix block is "
        "row-scaled by ``√(1/N_b)`` so its total contribution equals "
        "``w_b`` regardless of ``N_b``; combined gradient is weighted by "
        "``(w_b · N_b)``. Use ``'per_band_count'`` to preserve the "
        "user-specified ``band_weights`` semantics across mask-induced "
        "per-band sample-count differences. Requires ``loose_validity=True`` "
        "(rejected at construction otherwise — meaningless under shared "
        "validity since all ``N_b`` are identical).",
    )

    # --- Geometry initialization (copied from IsosterConfig) ---
    x0: Optional[float] = Field(None, description="Initial center x coordinate. If None, uses image center.")
    y0: Optional[float] = Field(None, description="Initial center y coordinate. If None, uses image center.")
    eps: float = Field(0.2, ge=0.0, lt=1.0, description="Initial ellipticity.")
    pa: float = Field(0.0, description="Initial position angle in radians.")

    # --- SMA grid (copied) ---
    sma0: float = Field(10.0, gt=0.0, description="Starting semi-major axis length.")
    minsma: float = Field(0.0, ge=0.0, description="Minimum SMA to fit.")
    maxsma: Optional[float] = Field(None, gt=0.0, description="Maximum SMA to fit. If None, uses image hypotenuse/2.")
    astep: float = Field(0.1, gt=0.0, description="Step size for SMA growth.")
    linear_growth: bool = Field(False, description="If True, sma = sma + astep. Else sma = sma * (1 + astep).")

    # --- Iteration / convergence (copied) ---
    maxit: int = Field(50, gt=0, description="Maximum iterations per isophote.")
    minit: int = Field(6, gt=0, description="Minimum iterations per isophote.")
    conver: float = Field(0.05, gt=0.0, description="Convergence threshold (max harmonic amplitude / rms).")
    convergence_scaling: Literal["none", "sector_area", "sqrt_sma"] = Field(
        default="sector_area",
        description="Scale convergence threshold with SMA. See "
        ":class:`isoster.config.IsosterConfig` for option meanings.",
    )
    sigma_bg: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Explicit background noise level (sigma) shared across bands. "
        "If provided, establishes a hard lower bound on the convergence "
        "threshold to prevent overfitting in LSB regions. Stage 1 uses a "
        "single shared sigma_bg; per-band sigma_bg is a Stage-2+ revisit.",
    )
    use_corrected_errors: bool = Field(
        default=True,
        description="Include the gradient uncertainty term in geometric error "
        "propagation. Same semantics as single-band.",
    )
    use_lazy_gradient: bool = Field(
        default=True,
        description="Use lazy gradient evaluation (gradient cached across "
        "iterations unless convergence stalls). Same semantics as single-band.",
    )

    # --- Geometry update control (copied) ---
    geometry_damping: float = Field(0.7, gt=0.0, le=1.0, description="Damping factor for geometry updates.")
    clip_max_shift: Optional[float] = Field(5.0, gt=0.0, description="Max center shift per iteration (pixels).")
    clip_max_pa: Optional[float] = Field(0.5, gt=0.0, description="Max PA change per iteration (radians).")
    clip_max_eps: Optional[float] = Field(0.1, gt=0.0, description="Max ellipticity change per iteration.")
    geometry_convergence: bool = Field(
        default=False,
        description="Enable secondary convergence based on geometry stability.",
    )
    geometry_tolerance: float = Field(0.01, gt=0.0, description="Threshold for geometry convergence.")
    geometry_stable_iters: int = Field(3, ge=2, description="Consecutive stable iterations required.")
    geometry_update_mode: Literal["largest", "simultaneous"] = Field(
        default="largest",
        description="Geometry update strategy: ``'largest'`` (coordinate descent) "
        "or ``'simultaneous'`` (update all four geometry parameters per iteration).",
    )

    # --- Quality control (copied) ---
    sclip: float = Field(3.0, gt=0.0, description="Sigma clipping threshold.")
    nclip: int = Field(1, ge=0, description="Number of sigma clipping iterations.")
    sclip_low: Optional[float] = Field(None, description="Lower sigma clipping threshold.")
    sclip_high: Optional[float] = Field(None, description="Upper sigma clipping threshold.")
    fflag: float = Field(0.5, ge=0.0, le=1.0, description="Maximum fraction of flagged data.")
    maxgerr: float = Field(0.5, gt=0.0, description="Maximum relative error in local radial gradient.")

    # --- Geometry constraints (copied) ---
    fix_center: bool = Field(False, description="Fix center coordinates during fitting.")
    fix_pa: bool = Field(False, description="Fix position angle during fitting.")
    fix_eps: bool = Field(False, description="Fix ellipticity during fitting.")

    # --- Higher-order harmonics (multiband_higher_harmonics enum) ---
    multiband_higher_harmonics: Literal[
        "independent",
        "shared",
        "simultaneous_in_loop",
        "simultaneous_original",
    ] = Field(
        default="independent",
        description="Strategy for higher-order (n>=3) harmonic fitting "
        "across bands. ``'independent'`` (default): per-band post-hoc "
        "fits, one per band per order, uncoupled across bands "
        "(reproduces Stage-1 behavior bit-identically). ``'shared'``: "
        "after joint geometry has converged, run ONE post-hoc joint "
        "WLS/OLS solve for a higher-order coefficient shared across all "
        "bands; (A1, B1, A2, B2) and per-band ``I0_b`` stay frozen at "
        "their converged-loop values. ``'simultaneous_in_loop'``: "
        "extend the joint design matrix every iteration to "
        "``(B*N, B + 4 + 2*len(harmonic_orders))`` with shared "
        "higher-order columns; all coefficients fit jointly inside the "
        "iteration loop (Ciambur 2015 in-loop variant). "
        "``'simultaneous_original'``: standard 5-param iteration loop, "
        "then ONE post-hoc joint solve over all orders simultaneously "
        "(Ciambur 2015 original variant).\n\n"
        "**What the Schema-1 columns hold differs by mode**, and "
        "``result['harmonics_shared']`` is the flag that says which "
        "convention applies. Under ``'independent'`` the per-band "
        "``a3_<b>``, ... columns are already Bender-normalised (the "
        "single-band ``compute_deviations`` solver normalises at fit "
        "time) and must not be normalised again. Under ``'shared'`` they "
        "are raw and **band-distinct**: the solve fits one dimensionless "
        "shape (each band's columns scaled by ``-sma*grad_b``) and "
        "converts back through each band's own scale, so the invariant is "
        "that they all normalise to one value. Under ``'simultaneous_*'`` "
        "they are raw and identical across bands, which represents a "
        "shared raw intensity residual rather than a shared shape — one "
        "of the two reasons those modes keep their experimental warning. "
        "Section 6 of plan-2026-04-29-multiband-feasibility.md.",
    )
    harmonic_orders: List[int] = Field(
        default_factory=lambda: [3, 4],
        description="Higher-order harmonic orders fit for isophote "
        "deviations under all ``multiband_higher_harmonics`` modes. "
        "Default ``[3, 4]`` matches the Stage-1 hardcoded behavior. "
        "Each entry must be an int >= 3 (orders 1 and 2 are reserved "
        "for the geometric harmonics driving the iteration loop). "
        "Duplicates are rejected; the list is unique-sorted on "
        "construction. Mirrors the single-band "
        "``IsosterConfig.harmonic_orders`` field.",
    )

    # --- Outputs / features (copied, with multi-band restrictions) ---
    compute_errors: bool = Field(True, description="Calculate parameter errors.")
    compute_deviations: bool = Field(True, description="Calculate higher-order harmonic deviations (a3, b3, a4, b4).")
    full_photometry: bool = Field(False, description="Calculate flux integration metrics (tflux_e, etc.).")
    compute_cog: bool = Field(
        default=False,
        description="Calculate per-band curve-of-growth photometry. When "
        "True, the driver runs a multi-band CoG over the assembled "
        "isophote list after the inward + outward sweeps complete and "
        "stamps the following columns onto each row dict (and thereby "
        "into the Schema-1 ISOPHOTES table):\n"
        "\n"
        "  - **per-band**: ``cog_<b>`` (cumulative flux through this "
        "isophote in band ``b``), ``cog_annulus_<b>`` (annular flux for "
        "this isophote in band ``b``).\n"
        "  - **shared**: ``area_annulus`` (annular area in pixel², "
        "geometry-only), ``flag_cross`` (bool, isophote crossing flag), "
        "``flag_negative_area`` (bool, negative-annular-area flag).\n"
        "\n"
        "Per-band columns appear when ``compute_cog=True``; Schema 1 "
        "stays additive otherwise. The shared columns mirror single-"
        "band semantics; per-band columns are unique to multi-band and "
        "match the Stage-3 Stage-D decision (plan section 7 S7) to put "
        "CoG output in the main per-isophote table rather than a "
        "separate HDU.",
    )
    debug: bool = Field(False, description="Include debug info in results (per-band grad columns, ndata, nflag).")

    # --- Integrator (restricted: no 'adaptive'; the multi-band path switches
    # --- integrator on the LSB auto-lock via lsb_auto_lock_integrator instead) ---
    integrator: Literal["mean", "median"] = Field(
        default="mean",
        description="Integration method for flux statistics derived from a "
        "single ring of pixels. ``'adaptive'`` is intentionally not "
        "supported: single-band's SMA-threshold switch "
        "(``lsb_sma_threshold``) has no multi-band counterpart — the "
        "locked-region integrator is set by ``lsb_auto_lock_integrator`` "
        "instead.\n"
        "\n"
        "**Affects three distinct quantities** (Stage-3, plan Section 7 S1–S2):\n"
        "  (a) per-band gradient computation (``compute_joint_gradient``);\n"
        "  (b) forced-photometry fallback used by the driver for the "
        "central pixel and as a fail-safe when the iterative fit cannot "
        "converge;\n"
        "  (c) per-band ``intens_<b>`` at converged isophotes — but ONLY "
        "in the decoupled intercept mode (``fit_per_band_intens_jointly="
        "False``). The matrix-mode joint LS solve cannot host a median "
        "(non-linear), so ``integrator='median'`` is gated to the "
        "decoupled path: the validator hard-errors on "
        "``integrator='median' ∧ fit_per_band_intens_jointly=True``. In "
        "the decoupled path with ``integrator='median'``, "
        "``intens_<b>`` is the per-band median of the ring samples "
        "(plain ``np.median``; sample sigma-clipping has already been "
        "applied upstream by the sclip/nclip pipeline). With "
        "``integrator='mean'``, both intercept modes give a "
        "(weighted) ring mean — full rings are numerically identical "
        "since ``sin(nφ)``/``cos(nφ)`` are orthogonal to the constant "
        "column over ``[0, 2π]``.",
    )

    # --- Eccentric anomaly sampling (copied) ---
    use_eccentric_anomaly: bool = Field(
        default=False,
        description="Use eccentric anomaly for uniform ellipse sampling "
        "(recommended for ε > 0.3). Same semantics as single-band.",
    )

    # --- Permissive geometry mode (copied) ---
    permissive_geometry: bool = Field(
        default=False,
        description="Enable permissive geometry updates (photutils-style). Same semantics as single-band.",
    )

    # --- First-isophote robustness (copied) ---
    max_retry_first_isophote: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum retry attempts for the first isophote. Same semantics as single-band.",
    )
    first_isophote_fail_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive initial-isophote failures before declaring "
        "FIRST_FEW_ISOPHOTE_FAILURE. Same semantics as single-band.",
    )

    # --- Stage-3 Stage-F: central-region geometry regularization ---
    # Verbatim lift of the single-band central_reg_* family. Geometry is
    # shared in multi-band so the math is band-agnostic; the penalty
    # adds to the best-iteration selector (effective_amp) for SMA below
    # the threshold, discouraging large per-iteration geometry jumps in
    # the low-S/N central region. Plan section 7 S8.
    use_central_regularization: bool = Field(
        default=False,
        description="Enable geometry regularization for the central region "
        "to stabilize fitting at low SMA. Adds a Gaussian-decaying penalty "
        "``λ(sma) = strength · exp(-(sma/threshold)²)`` to the best-"
        "iteration selector (``effective_amp``); iterations whose "
        "geometry jumped far from the previous isophote then look worse "
        "to the selector and are not chosen. Mirrors single-band "
        "semantics; geometry is shared across bands so no per-band "
        "design choice is needed.",
    )
    central_reg_sma_threshold: float = Field(
        default=5.0,
        gt=0.0,
        description="SMA threshold (pixels) for central regularization. "
        "Penalty strength decays as ``exp(-(sma/threshold)²)`` and is "
        "essentially zero beyond ~3× threshold.",
    )
    central_reg_strength: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum regularization strength at SMA=0. 0=no "
        "regularization, 1=moderate, 10=strong. Typical range 0.1–10.",
    )
    central_reg_weights: Dict[str, float] = Field(
        default_factory=lambda: {"eps": 1.0, "pa": 1.0, "center": 1.0},
        description="Per-axis weights for the central-region penalty. "
        "Default ``{eps: 1, pa: 1, center: 1}`` damps eps / PA / center "
        "uniformly. Unknown keys are rejected; valid keys: ``eps``, "
        "``pa``, ``center``.",
    )

    # --- Stage-3 Stage-B: outer-region center regularization (damping mode) ---
    # Backport of the single-band ``outer_reg_*`` family with the ``damping``
    # mode only. ``solver`` mode lands in Stage E. See plan section 7 (S5).
    use_outer_center_regularization: bool = Field(
        default=False,
        description="Enable soft outer-region geometry damping in the LSB "
        "regime. When True, the outward sweep softly shrinks per-iteration "
        "geometry steps via a Tikhonov-style ``alpha`` blend in regions "
        "above ``outer_reg_sma_onset`` — saturated clipped jumps in PA / "
        "ellipticity / center are suppressed, but the fit still walks to "
        "its data-preferred geometry (no pull toward a frozen reference; "
        "that is ``outer_reg_mode='solver'``, which lands in Stage E). "
        "Requires the inward sweep to run before outward growth so the "
        "frozen inner reference geometry can be built. Disabled by default; "
        "recommended for outer LSB on real-data BCG / ICL fits.",
    )
    outer_reg_sma_onset: float = Field(
        default=50.0,
        gt=0.0,
        description="SMA (pixels) at the midpoint of the outer-regularization "
        "sigmoid. Below this the damping is near zero; above it the alpha "
        "blend ramps up toward a saturation set by ``outer_reg_strength``.",
    )
    outer_reg_strength: float = Field(
        default=2.0,
        ge=0.0,
        description="Peak strength of the outer-region damping ramp. Controls "
        "how aggressively per-iteration geometry steps are shrunk above the "
        "onset. Typical range 2–8 for HSC-grade LSB data; default 2.0 was "
        "the single-band benchmark winner on the HSC edge-case suite.",
    )
    outer_reg_weights: Dict[str, float] = Field(
        default_factory=lambda: {"center": 1.0, "eps": 0.0, "pa": 0.0},
        description="Per-axis weights for the outer-region damping. The "
        "**multi-band default is** ``{center: 1, eps: 0, pa: 0}`` "
        "(center-only) — different from the historical single-band "
        "``{1, 1, 1}`` default. Set unused axes to 0 to disable damping "
        "on that axis; unknown keys are rejected.\n"
        "\n"
        "**The eps / pa weight choice is binary, not graded.** A "
        "strength × weights sweep on PGC006669 (Stage-B follow-up, "
        "2026-05-04 journal) showed that down-weighting eps / pa from "
        "``1.0`` to ``0.25`` produces *essentially identical* outer-"
        "region pinning, because the per-axis Tikhonov blend factor "
        "``α = λ·w·coeff² / (1 + λ·w·coeff²)`` saturates above ~ 99.99 % "
        "for any positive weight in the user-intuition range "
        "(0.25–1.0). In PGC's outer LSB the per-axis Jacobian "
        "``coeff = (1-eps)/grad`` reaches ~ 700 because the joint "
        "gradient ``→ 0`` in the LSB; ``coeff² ≈ 5×10⁵``, so "
        "``λ·w·coeff² ≫ 1`` for any ``w ≳ 10⁻⁴``. The honest 'partial "
        "damping' regime would need ``w ~ 10⁻⁴`` — far below user "
        "intuition. So the eps / pa weights are effectively a binary "
        "switch: any positive value pins the geometry; zero leaves it "
        "free.\n"
        "\n"
        "**Recommendation by galaxy type:**\n"
        "  - **Default ``{1, 0, 0}``** — works on the broad range of "
        "multi-band targets including barred / spiral systems, BCG → "
        "ICL transitions, and any galaxy where the outer-disc "
        "geometry might genuinely evolve. The multi-band joint solver "
        "already provides cross-band stability on eps / pa via the "
        "shared ``(A1, B1, A2, B2)`` harmonic constraint, so eps / pa "
        "damping is largely redundant here.\n"
        "  - **``{1, 1, 1}`` — advanced opt-in** for confirmed-static "
        "outer envelopes (cD galaxies, isolated massive ellipticals "
        "like asteris where the LSB envelope is genuinely round and "
        "not evolving). On those targets the all-axes pin tightens "
        "the recorded outer geometry.\n"
        "  - **Down-weighted variants like ``{1, 0.5, 0.5}`` or "
        "``{1, 0.25, 0.25}``** — empirically NOT a useful middle "
        "ground; produce essentially the same pinning as ``{1, 1, 1}``. "
        "Avoid.\n"
        "\n"
        "Validator warning fires when ``fix_<axis>=True`` is set with "
        "a positive ``outer_reg_weights[<axis>]`` (the penalty is "
        "identically zero on a frozen axis); with the new ``{1, 0, 0}`` "
        "default this warning no longer triggers spuriously when "
        "users set ``fix_eps=True`` or ``fix_pa=True``.",
    )
    outer_reg_mode: Literal["damping"] = Field(
        default="damping",
        description="How the outer-region penalty enters the fit. Stage B "
        "ships ``'damping'`` only — step shrink ``(1 - alpha)`` on the "
        "harmonic update, no pull toward the reference. ``'solver'`` mode "
        "(full Tikhonov with ref-pull) lands in Stage E and will widen "
        "this Literal at that time.",
    )
    outer_reg_sma_width: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Sigmoid slope width (pixels) for the outer-region "
        "damping ramp. ``None`` (default) auto-computes as "
        "``0.4 * outer_reg_sma_onset``. Smaller values give a sharper "
        "transition at the onset; expert tuning only.",
    )
    outer_reg_ref_sma_factor: float = Field(
        default=2.0,
        gt=1.0,
        description="Inward isophotes with ``sma <= sma0 * this factor`` "
        "feed the flux-weighted reference geometry. Default 2.0 mirrors "
        "single-band; typical range 1.5–3. Expert tuning only.",
    )

    # --- Stage-3 Stage-C: automatic LSB geometry lock (lsb_auto_lock_*) ---
    # Backport of single-band lsb_auto_lock. Once the joint combined
    # gradient degrades, freeze the shared geometry and switch the
    # remaining outward isophotes to the configured locked integrator.
    # Mirrors single-band semantics; trigger surface is the joint
    # combined gradient (plan S3); lock anchor is the converged shared
    # geometry (S4) — no per-band lock state.
    lsb_auto_lock: bool = Field(
        default=False,
        description="Enable automatic LSB geometry lock on the outward "
        "sweep. When True, fitting starts in free-geometry mode; once "
        "the joint combined gradient quality degrades (relative joint "
        "gradient error above ``lsb_auto_lock_maxgerr`` for "
        "``lsb_auto_lock_debounce`` consecutive outward isophotes, or "
        "a stop_code=-1 hits), the lock commits: remaining outward "
        "isophotes inherit the anchor's shared geometry (frozen "
        "``x0``, ``y0``, ``eps``, ``pa``) and switch to the configured "
        "``lsb_auto_lock_integrator``. Inward growth and the central "
        "pixel are unaffected. Trigger surface is the **joint combined "
        "gradient** (plan section 7 S3): tune ``reference_band`` and "
        "``band_weights`` to influence which band(s) drive the trigger.\n"
        "\n"
        "**Designed for massive ellipticals / cD galaxies with extended "
        "LSB envelopes** — the regime where the inner geometry is the "
        "best estimate of the outer envelope's geometry and a hard lock "
        "is the right LSB stabilizer. **Do NOT enable by default on "
        "arbitrary galaxy types**: on systems with genuine outer-disc "
        "geometry evolution (barred galaxies, S0/spiral systems with "
        "a real bulge → disc transition, BCG → ICL transitions), the "
        "lock will pin the outer envelope to the inner geometry and "
        "obscure real structural change. Same failure mode as outer-reg "
        "``{1, 1, 1}`` damping (Stage-B caveat). For those targets, "
        "either leave the lock off, or raise ``lsb_auto_lock_maxgerr`` "
        "(e.g. to 0.5, matching the free-fit ``maxgerr``) so the lock "
        "delays firing until the gradient is genuinely lost. Default "
        "False.",
    )
    lsb_auto_lock_maxgerr: float = Field(
        default=0.3,
        gt=0.0,
        description="Trigger threshold on the relative joint gradient "
        "error ``|grad_err_joint / grad_joint|``. The lock commits when "
        "this threshold is exceeded on ``lsb_auto_lock_debounce`` "
        "consecutive outward isophotes. Default 0.3 is stricter than "
        "the free-fit ``maxgerr`` (0.5) so the lock fires before a "
        "stop_code=-1 would have.",
    )
    lsb_auto_lock_debounce: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Consecutive triggered outward isophotes required "
        "before the LSB lock commits. Debounces single-isophote noise "
        "spikes. The lock anchor is the isophote immediately BEFORE "
        "the streak. Default 2 mirrors single-band.",
    )
    lsb_auto_lock_integrator: Literal["mean", "median"] = Field(
        default="median",
        description="Integrator used for the locked-region isophotes. "
        "Default ``'median'`` for robustness against contaminants that "
        "the fixed-geometry path would otherwise average in. When set "
        "to ``'median'``, the validator requires "
        "``fit_per_band_intens_jointly=False`` (plan S1: median cannot "
        "ride inside the matrix-mode joint LS solve) so the lock-fire "
        "path produces a config the fitter accepts.",
    )

    # --- Read-only field surfaced after driver entry ---
    # ``variance_mode`` is auto-derived from whether ``variance_maps`` is
    # provided to ``fit_image_multiband``. It is exposed as a config field
    # so the resolved-config record (saved to the FITS CONFIG HDU)
    # captures it. Users do not set it directly; the driver assigns it
    # before serialization.
    variance_mode: Optional[Literal["wls", "ols"]] = Field(
        default=None,
        description="Auto-derived: 'wls' when variance maps are provided to "
        "the driver, 'ols' otherwise. Users should not set this directly.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed_fields(cls, data):
        """Hard-error on the old ``fix_per_band_background_to_zero`` field.

        Renamed to :attr:`fit_per_band_intens_jointly` (with inverted
        polarity) in the Section 6 cleanup. The multi-band path is
        pre-production so we take a clean break: the old name is no
        longer accepted and there's no auto-translation. Users see a
        clear error pointing at the new field rather than a silent
        config drop (pydantic's default ``extra='ignore'`` would otherwise
        silently swallow it).
        """
        if isinstance(data, dict) and "fix_per_band_background_to_zero" in data:
            old = data["fix_per_band_background_to_zero"]
            new = "fit_per_band_intens_jointly={}".format(not bool(old))
            raise ValueError(
                f"`fix_per_band_background_to_zero` has been renamed to "
                f"`fit_per_band_intens_jointly` with inverted polarity. "
                f"Replace `fix_per_band_background_to_zero={old!r}` with "
                f"`{new}`. See Section 6 of "
                f"docs/agent/plan-2026-04-29-multiband-feasibility.md "
                f"for the rationale."
            )
        return data

    @model_validator(mode="after")
    def _check_multiband_consistency(self):
        """Validate multi-band-specific consistency rules."""
        # --- bands: regex + uniqueness (D8) ---
        for band in self.bands:
            if not _BAND_NAME_RE.match(band):
                raise ValueError(
                    f"band name {band!r} does not match the required regex "
                    f"'^[A-Za-z][A-Za-z0-9_]*$'. Band names must start with a "
                    f"letter and contain only letters, digits, and underscores. "
                    f"Hyphens and other punctuation are not allowed because "
                    f"band names appear verbatim as FITS column suffixes."
                )
        if len(set(self.bands)) != len(self.bands):
            seen = set()
            duplicates = [b for b in self.bands if b in seen or seen.add(b)]
            raise ValueError(f"bands contains duplicates: {duplicates}. Each band name must be unique.")

        # --- reference_band must be in bands (D3) ---
        if self.reference_band not in self.bands:
            raise ValueError(f"reference_band {self.reference_band!r} must be one of bands {self.bands!r}.")

        # --- band_weights validation (D12) ---
        if self.band_weights is not None:
            if isinstance(self.band_weights, dict):
                missing = set(self.bands) - set(self.band_weights.keys())
                extra = set(self.band_weights.keys()) - set(self.bands)
                if missing:
                    raise ValueError(
                        f"band_weights is missing keys for bands "
                        f"{sorted(missing)}. When band_weights is a dict, "
                        f"every band in `bands` must have a corresponding key."
                    )
                if extra:
                    raise ValueError(
                        f"band_weights contains keys {sorted(extra)} that are not in `bands` {self.bands!r}."
                    )
                weight_values = [self.band_weights[b] for b in self.bands]
            else:
                if len(self.band_weights) != len(self.bands):
                    raise ValueError(
                        f"band_weights list has length {len(self.band_weights)} "
                        f"but `bands` has length {len(self.bands)}. The list "
                        f"form of band_weights must align with `bands` "
                        f"element-for-element."
                    )
                weight_values = list(self.band_weights)
            for band_name, w in zip(self.bands, weight_values):
                if not (isinstance(w, (int, float)) and w > 0 and w == w and w != float("inf")):
                    raise ValueError(
                        f"band_weights[{band_name!r}] = {w!r} is not a "
                        f"positive finite float. Each weight must be > 0 and "
                        f"finite."
                    )

        # --- iteration consistency (copied from single-band) ---
        if self.maxsma is not None and self.maxsma < self.minsma:
            raise ValueError(f"maxsma ({self.maxsma}) must be greater than minsma ({self.minsma}).")
        if self.minit > self.maxit:
            raise ValueError(f"minit ({self.minit}) must be <= maxit ({self.maxit}).")

        # Loose validity needs at least two bands. The joint solve requires two
        # surviving bands to be meaningful, so at B=1 every isophote would fail
        # with stop_code=3 partway through the fit. Rejecting here turns that
        # into an immediate, explanatory error instead of a silent whole-image
        # failure.
        if self.loose_validity and len(self.bands) < 2:
            raise ValueError(
                "loose_validity=True requires at least 2 bands; got "
                f"{len(self.bands)} ({self.bands}). Per-band validity only has "
                "meaning when bands can disagree, and the joint solve needs two "
                "surviving bands, so a single-band run would fail every isophote "
                "with stop_code=3. Use loose_validity=False, or fit this band "
                "with single-band isoster (isoster.fit_image)."
            )

        # --- D11 backport: the ``ring_mean`` intercept mode
        # (fit_per_band_intens_jointly=False) is incompatible with
        # ref-mode harmonic combination: ref-mode bypasses the joint
        # solver entirely so the column-drop is meaningless.
        if (not self.fit_per_band_intens_jointly) and self.harmonic_combination == "ref":
            raise ValueError(
                "fit_per_band_intens_jointly=False is incompatible with "
                "harmonic_combination='ref': ref-mode bypasses the joint "
                "design matrix entirely, so the per-band-intercept-column "
                "drop has nothing to act on. Either set "
                "fit_per_band_intens_jointly=True or pick "
                "harmonic_combination='joint'."
            )

        # --- Stage-3 S1: integrator='median' requires the decoupled
        # intercept mode. The matrix-mode joint LS cannot host a median
        # (non-linear); see plan section 7.2 S1 for the rejected Path-B
        # alternative (post-hoc median replacement) and why Path A was
        # chosen instead.
        if self.integrator == "median" and self.fit_per_band_intens_jointly:
            raise ValueError(
                "integrator='median' requires fit_per_band_intens_jointly="
                "False. The matrix-mode joint LS solve carries per-band "
                "intercept columns whose values come from a linear "
                "least-squares solve, which cannot represent a median. "
                "Use fit_per_band_intens_jointly=False to switch to the "
                "decoupled intercept mode where intens_<b> is computed "
                "as a per-band median of the ring samples, or stay on "
                "integrator='mean' to keep the matrix-mode solve."
            )

        # --- D9 backport: per-band-count normalization only makes sense
        # under loose validity (all N_b are identical under shared validity).
        if self.loose_validity_band_normalization == "per_band_count" and not self.loose_validity:
            raise ValueError(
                "loose_validity_band_normalization='per_band_count' requires "
                "loose_validity=True. Under shared validity all bands have "
                "the same N_b and the per-band-count renormalization is a "
                "no-op."
            )

        # --- Section 6: multiband_higher_harmonics validation ---
        # harmonic_orders must be a non-empty list of ints >= 3 with no
        # duplicates; unique-sort in place so downstream code can rely on
        # ascending order.
        if not isinstance(self.harmonic_orders, list) or len(self.harmonic_orders) == 0:
            raise ValueError(f"harmonic_orders must be a non-empty list of ints >= 3, got {self.harmonic_orders!r}.")
        for n in self.harmonic_orders:
            if not isinstance(n, int) or isinstance(n, bool) or n < 3:
                raise ValueError(
                    f"harmonic_orders entries must be ints >= 3 (orders 1 "
                    f"and 2 are reserved for geometric harmonics); got "
                    f"{self.harmonic_orders!r}."
                )
        if len(set(self.harmonic_orders)) != len(self.harmonic_orders):
            seen: set = set()
            duplicates = [n for n in self.harmonic_orders if n in seen or seen.add(n)]
            raise ValueError(f"harmonic_orders contains duplicates {duplicates}; each order must appear at most once.")
        # Unique-sort in place (mutating the validated list is fine — pydantic
        # has already accepted the field).
        self.harmonic_orders = sorted(self.harmonic_orders)

        # Hard-error: shared/simultaneous_* modes are incompatible with
        # ref-mode harmonic combination, which bypasses the joint solver.
        if self.multiband_higher_harmonics != "independent" and self.harmonic_combination == "ref":
            raise ValueError(
                f"multiband_higher_harmonics={self.multiband_higher_harmonics!r} "
                f"is incompatible with harmonic_combination='ref': ref-mode "
                f"bypasses the joint design matrix entirely, so the "
                f"shared/simultaneous higher-order modes have nothing to "
                f"act on. Either set multiband_higher_harmonics='independent' "
                f"or pick harmonic_combination='joint'."
            )

        # Soft warning when a simultaneous_* mode is selected: single-band
        # benchmarks have flagged in-loop joint harmonic fitting as
        # unreliable; users should validate on PGC006669 + asteris before
        # trusting the output.
        if self.multiband_higher_harmonics in (
            "simultaneous_in_loop",
            "simultaneous_original",
        ):
            warnings.warn(
                f"multiband_higher_harmonics={self.multiband_higher_harmonics!r} "
                f"is experimental: the single-band equivalent "
                f"(IsosterConfig.simultaneous_harmonics=True) has shown "
                f"benchmark regressions. Multi-band may behave better thanks "
                f"to the joint constraint, but validate on the asteris and "
                f"PGC006669 demos before trusting the output. Consider "
                f"multiband_higher_harmonics='shared' as the lower-risk "
                f"alternative.",
                UserWarning,
                stacklevel=2,
            )

        # --- soft warnings parallel to single-band V2/V3 ---
        if self.maxsma is not None and self.maxsma < self.sma0:
            warnings.warn(
                f"maxsma ({self.maxsma}) < sma0 ({self.sma0}): only one isophote + inward sweep will be produced.",
                UserWarning,
                stacklevel=2,
            )
        if self.minsma >= self.sma0:
            warnings.warn(
                f"minsma ({self.minsma}) >= sma0 ({self.sma0}): inward loop will not run.",
                UserWarning,
                stacklevel=2,
            )
        if self.geometry_update_mode == "simultaneous" and self.geometry_damping > 0.7:
            warnings.warn(
                "geometry_damping > 0.7 with geometry_update_mode='simultaneous' may cause oscillations; consider 0.5.",
                UserWarning,
                stacklevel=2,
            )
        if self.geometry_convergence and self.maxit < self.minit + self.geometry_stable_iters:
            warnings.warn(
                f"maxit ({self.maxit}) < minit + geometry_stable_iters "
                f"({self.minit + self.geometry_stable_iters}): "
                f"geometry convergence can never trigger.",
                UserWarning,
                stacklevel=2,
            )

        # --- Stage-3 Stage-F: central-region regularization sanity ---
        # Always-on hard error: central_reg_weights must use known keys
        # even when the feature is off (catches typos before the user
        # toggles use_central_regularization=True). Single-band
        # behavior matches.
        valid_central_axes = {"eps", "pa", "center"}
        unknown_central = set(self.central_reg_weights.keys()) - valid_central_axes
        if unknown_central:
            raise ValueError(
                f"central_reg_weights contains unknown keys: "
                f"{sorted(unknown_central)}. Valid keys: "
                f"{sorted(valid_central_axes)}."
            )

        # --- Stage-3 Stage-B: outer-region regularization sanity ---
        if self.use_outer_center_regularization:
            valid_axes = {"center", "eps", "pa"}
            unknown_axes = set(self.outer_reg_weights.keys()) - valid_axes
            if unknown_axes:
                raise ValueError(
                    f"outer_reg_weights contains unknown keys: {sorted(unknown_axes)}. "
                    f"Valid keys: {sorted(valid_axes)}."
                )
            if self.outer_reg_sma_onset < self.sma0:
                warnings.warn(
                    f"outer_reg_sma_onset ({self.outer_reg_sma_onset}) < sma0 "
                    f"({self.sma0}): the outer-regularization damping will "
                    f"fire from the first outward step and may over-constrain "
                    f"the mid-sma region. Typical onset ~ sma0 * 3.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.minsma >= self.sma0:
                warnings.warn(
                    f"use_outer_center_regularization=True with minsma "
                    f"({self.minsma}) >= sma0 ({self.sma0}): no inward "
                    f"isophotes to build the reference centroid, so the "
                    f"reference will fall back to the anchor isophote center.",
                    UserWarning,
                    stacklevel=2,
                )
            for axis in ("center", "eps", "pa"):
                fixed = {"center": "fix_center", "eps": "fix_eps", "pa": "fix_pa"}[axis]
                if getattr(self, fixed) and float(self.outer_reg_weights.get(axis, 0.0)) > 0.0:
                    warnings.warn(
                        f"outer_reg_weights[{axis!r}]>0 with {fixed}=True: "
                        f"the {axis} parameter is frozen so the {axis} "
                        f"penalty is identically zero. Set "
                        f"outer_reg_weights[{axis!r}]=0 or disable {fixed}.",
                        UserWarning,
                        stacklevel=2,
                    )
            if all(float(self.outer_reg_weights.get(k, 0.0)) <= 0.0 for k in ("center", "eps", "pa")):
                warnings.warn(
                    "use_outer_center_regularization=True but all "
                    "outer_reg_weights are zero: the damping is identically "
                    "zero and the feature is inert. Set at least one of "
                    "center/eps/pa to a positive value.",
                    UserWarning,
                    stacklevel=2,
                )
            # Auto-enable geometry_convergence: with damping the geometry
            # genuinely settles, but the harmonic criterion may keep tripping
            # near saturation; without geometry_convergence the fit runs to
            # maxit on many outer isophotes with no change to recorded
            # geometry. Pure cost, no benefit. Mirrors single-band behavior.
            if not self.geometry_convergence:
                warnings.warn(
                    "use_outer_center_regularization=True auto-enables "
                    "geometry_convergence=True because the outer damping "
                    "term otherwise prevents the harmonic criterion from "
                    "tripping in the LSB regime. To keep "
                    "geometry_convergence=False, disable "
                    "use_outer_center_regularization instead.",
                    UserWarning,
                    stacklevel=2,
                )
                self.geometry_convergence = True

        # --- Stage-3 Stage-C: lsb_auto_lock validators ---
        # The lock is meaningful only when starting from free geometry —
        # mirror single-band's hard-error on fix_*. Section 7.5 item 2.
        if self.lsb_auto_lock:
            conflicting = [name for name in ("fix_center", "fix_pa", "fix_eps") if getattr(self, name)]
            if conflicting:
                raise ValueError(
                    f"lsb_auto_lock=True conflicts with "
                    f"{', '.join(conflicting)}=True: the auto-lock requires "
                    f"free geometry at the start of outward growth. Either "
                    f"disable lsb_auto_lock or unfreeze the listed "
                    f"geometry parameter(s)."
                )
            # Section 7.5 item 3: the locked-region clone would set
            # integrator='median' AND keep fit_per_band_intens_jointly,
            # which Stage-A's S1 rejects. Catch the illegal combination
            # at config construction so the lock-fire path cannot trip
            # the validator at clone time.
            if self.lsb_auto_lock_integrator == "median" and self.fit_per_band_intens_jointly:
                raise ValueError(
                    "lsb_auto_lock_integrator='median' with "
                    "fit_per_band_intens_jointly=True is not supported: "
                    "the locked-region cfg clone would switch the "
                    "integrator to median, which Stage-A's S1 rejects "
                    "for matrix-mode joint LS (median cannot ride "
                    "inside the design matrix). Set "
                    "fit_per_band_intens_jointly=False to use the "
                    "decoupled intercept mode (which legally hosts the "
                    "median path), or set "
                    "lsb_auto_lock_integrator='mean' to keep the "
                    "matrix-mode joint solve."
                )
            # Soft warning + auto-enable debug: the lock trigger reads
            # top-level grad / grad_error scalars that the fitter only
            # writes when debug=True (mirrors single-band).
            if not self.debug:
                warnings.warn(
                    "lsb_auto_lock=True requires the joint gradient "
                    "diagnostics on each isophote; debug will be enabled "
                    "internally for this fit.",
                    UserWarning,
                    stacklevel=2,
                )
                self.debug = True

        return self

    def resolved_band_weights(self) -> Dict[str, float]:
        """
        Return ``band_weights`` as a normalized ``{band: weight}`` dict.

        When ``band_weights`` is ``None`` returns uniform 1.0 for every
        band. Useful for downstream code that wants a single canonical
        form regardless of how the user constructed the config.
        """
        if self.band_weights is None:
            return {b: 1.0 for b in self.bands}
        if isinstance(self.band_weights, dict):
            return {b: float(self.band_weights[b]) for b in self.bands}
        return {b: float(w) for b, w in zip(self.bands, self.band_weights)}
