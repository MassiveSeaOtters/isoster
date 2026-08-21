import warnings
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class IsosterConfig(BaseModel):
    """
    Configuration for isoster fitting.

    This Pydantic model defines all tunable parameters for the isophote fitting algorithm,
    including geometry initialization, fitting control, quality control, and output options.

    **Stop Codes:**
    Fitting results include a 'stop_code' field indicating termination condition:
    - 0: Success (converged)
    - 1: Too many flagged pixels (>fflag threshold)
    - 2: Maximum iteration count reached without convergence
    - 3: Too few points (<6)
    - -1: Gradient error (invalid or unreliable)

    See docs/01-user-guide.md (Stop Codes (Canonical Reference)) for detailed documentation.
    """

    # Geometry initialization
    x0: Optional[float] = Field(None, description="Initial center x coordinate. If None, uses image center.")
    y0: Optional[float] = Field(None, description="Initial center y coordinate. If None, uses image center.")
    eps: float = Field(0.2, ge=0.0, lt=1.0, description="Initial ellipticity.")
    pa: float = Field(0.0, description="Initial position angle in radians.")

    # SMA control
    sma0: float = Field(10.0, gt=0.0, description="Starting semi-major axis length.")
    minsma: float = Field(0.0, ge=0.0, description="Minimum SMA to fit.")
    maxsma: Optional[float] = Field(None, gt=0.0, description="Maximum SMA to fit. If None, uses image size/2.")
    astep: float = Field(0.1, gt=0.0, description="Step size for SMA growth.")
    linear_growth: bool = Field(False, description="If True, sma = sma + astep. Else sma = sma * (1 + astep).")

    # Fitting control
    maxit: int = Field(50, gt=0, description="Maximum iterations per isophote.")
    minit: int = Field(6, gt=0, description="Minimum iterations per isophote.")
    conver: float = Field(0.05, gt=0.0, description="Convergence threshold (max harmonic amplitude / rms).")
    convergence_scaling: str = Field(
        default="sector_area",
        pattern="^(none|sector_area|sqrt_sma)$",
        description="Scale convergence threshold with SMA. "
        "'sector_area': multiply by approximate sector area (matches photutils behavior, default). "
        "'sqrt_sma': multiply by sqrt(sma). "
        "'none': constant threshold (legacy behavior).",
    )
    sigma_bg: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Explicit background noise level (sigma). "
        "If provided, establish a hard lower bound on the convergence threshold: "
        "max(rms, sigma_bg / sqrt(N)). This prevents overfitting in LSB regions.",
    )
    use_corrected_errors: bool = Field(
        default=True,
        description="If True, include the gradient uncertainty term in geometric error propagation. "
        "This provides more realistic (larger) error bars in LSB regions.",
    )
    use_lazy_gradient: bool = Field(
        default=True,
        description="If True, use 'Lazy Gradient Evaluation' (Modified Newton Method). "
        "The radial gradient is computed only on the first iteration and reused "
        "for subsequent iterations unless convergence stalls. This significantly "
        "reduces image sampling overhead.",
    )

    # Geometry update control
    geometry_damping: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Damping factor for geometry updates (0 < d <= 1). "
        "Each geometry correction is multiplied by this factor. "
        "Default 0.7 stabilizes oscillations at outer isophotes. "
        "Use 1.0 for no damping (legacy behavior).",
    )
    clip_max_shift: Optional[float] = Field(
        default=5.0,
        gt=0.0,
        description="Maximum allowed center shift (in pixels) per iteration. Set to None to disable this safeguard.",
    )
    clip_max_pa: Optional[float] = Field(
        default=0.5,
        gt=0.0,
        description="Maximum allowed PA change (in radians) per iteration. Set to None to disable this safeguard.",
    )
    clip_max_eps: Optional[float] = Field(
        default=0.1,
        gt=0.0,
        description="Maximum allowed ellipticity change per iteration. Set to None to disable this safeguard.",
    )
    geometry_convergence: bool = Field(
        default=False,
        description="Enable secondary convergence based on geometry stability. "
        "Declares convergence when geometry changes fall below tolerance "
        "for consecutive iterations, even if harmonic criterion is not met.",
    )
    geometry_tolerance: float = Field(
        default=0.01,
        gt=0.0,
        description="Threshold for geometry convergence. Convergence declared when "
        "max(|delta_eps|, |delta_pa/pi|, |delta_x0/sma|, |delta_y0/sma|) "
        "< geometry_tolerance for geometry_stable_iters consecutive iterations.",
    )
    geometry_stable_iters: int = Field(
        default=3,
        ge=2,
        description="Number of consecutive iterations with small geometry changes "
        "required to trigger geometry-based convergence.",
    )

    # Quality control
    sclip: float = Field(3.0, gt=0.0, description="Sigma clipping threshold.")
    nclip: int = Field(1, ge=0, description="Number of sigma clipping iterations.")
    sclip_low: Optional[float] = Field(None, description="Lower sigma clipping threshold.")
    sclip_high: Optional[float] = Field(None, description="Upper sigma clipping threshold.")
    fflag: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of sigma-clipped data points. Masked points are removed during sampling and do not count.",
    )
    maxgerr: float = Field(0.5, gt=0.0, description="Maximum relative error in local radial gradient.")

    # Constraints
    fix_center: bool = Field(False, description="Fix center coordinates during fitting.")
    fix_pa: bool = Field(False, description="Fix position angle during fitting.")
    fix_eps: bool = Field(False, description="Fix ellipticity during fitting.")

    # Outputs & Features
    compute_errors: bool = Field(True, description="Calculate parameter errors.")
    compute_deviations: bool = Field(True, description="Calculate higher-order harmonic deviations (a3, b3, etc.).")
    full_photometry: bool = Field(False, description="Calculate flux integration metrics (tflux_e, etc.).")
    compute_cog: bool = Field(False, description="Calculate curve-of-growth photometry.")
    debug: bool = Field(False, description="Include debug info in results and enable verbose calculation.")

    # Integration Mode
    integrator: str = Field(
        default="mean", pattern="^(mean|median|adaptive)$", description="Integration method for flux calculation."
    )
    lsb_sma_threshold: Optional[float] = Field(
        None, gt=0.0, description="SMA threshold for switching to median integrator in adaptive mode."
    )

    # Eccentric Anomaly Sampling
    use_eccentric_anomaly: bool = Field(
        False,
        description="Step the eccentric anomaly psi uniformly instead of the position angle "
        "(recommended for eps > 0.3). This distributes samples more evenly around an "
        "elongated ellipse -- the arc length per unit psi varies by a factor of a/b, "
        "against a larger and less tractable variation per unit phi -- though it is not "
        "exactly uniform in arc length. Harmonics are then fitted in the psi basis and "
        "the geometry update consumes those coefficients directly. eps = 1 - b/a.",
    )

    # Higher-Order Harmonics Fitting
    simultaneous_harmonics: bool = Field(
        False,
        description="Enable true ISOFIT simultaneous harmonic fitting (Ciambur 2015). "
        "When True, higher-order harmonics (specified by harmonic_orders) are fitted "
        "jointly with the geometry harmonics (orders 1-2) inside the iteration loop, "
        "accounting for cross-correlations and yielding cleaner RMS estimates. "
        "Falls back to 5-param fit when insufficient sample points. "
        "When False, harmonics are fitted post-hoc after geometry convergence.",
    )

    harmonic_orders: List[int] = Field(
        default_factory=lambda: [3, 4],
        description="List of harmonic orders to fit for isophote deviations. "
        "Default [3, 4] fits 3rd and 4th order. Can extend to [3, 4, 5, 6] etc.",
    )

    isofit_mode: str = Field(
        default="in_loop",
        pattern="^(in_loop|original)$",
        description="ISOFIT algorithm variant (only meaningful when simultaneous_harmonics=True). "
        "'in_loop': fit all harmonic orders simultaneously inside the iteration loop "
        "(current isoster behavior, more aggressive than Ciambur 2015). "
        "'original': match Ciambur 2015 — use 5-param fit inside the loop for geometry, "
        "then fit all higher-order harmonics simultaneously post-hoc after convergence.",
    )

    # Geometry update strategy
    geometry_update_mode: str = Field(
        default="largest",
        pattern="^(largest|simultaneous)$",
        description="Geometry update strategy per iteration. "
        "'largest': update only the geometry parameter with the largest harmonic amplitude "
        "(coordinate descent, traditional isofit/photutils behavior). "
        "'simultaneous': update all four geometry parameters (x0, y0, PA, eps) each "
        "iteration using their respective harmonic corrections. Typically converges in "
        "fewer iterations but may need lower damping (e.g., 0.5).",
    )

    # Central Region Geometry Regularization
    use_central_regularization: bool = Field(
        False,
        description="Enable geometry regularization for central region to stabilize fitting at low SMA. "
        "Adds penalty for large geometry changes at SMA < threshold.",
    )

    central_reg_sma_threshold: float = Field(
        5.0,
        gt=0.0,
        description="SMA threshold for central regularization (pixels). "
        "Regularization strength decays exponentially with distance from center.",
    )

    central_reg_strength: float = Field(
        1.0,
        ge=0.0,
        description="Maximum regularization strength at SMA=0. "
        "0=no regularization, 1=moderate, 10=strong. Typical range: 0.1-10.",
    )

    central_reg_weights: dict = Field(
        default_factory=lambda: {"eps": 1.0, "pa": 1.0, "center": 1.0},
        description="Relative weights for regularization penalties on ellipticity, PA, and center. "
        "Dict with keys: 'eps', 'pa', 'center'.",
    )

    # Outer Region Center Regularization
    use_outer_center_regularization: bool = Field(
        False,
        description="Enable soft center-only regularization in the outer (LSB) region. "
        "When True, the outward fit is softly pulled toward a frozen inner reference "
        "center (x0_ref, y0_ref) built from the inward isophotes. Unlike lsb_auto_lock, "
        "this does NOT freeze geometry - it adds a best-iteration penalty that damps "
        "artificial drift but still lets a genuine lopsided outer center bleed through. "
        "Requires running inward growth before outward growth. Disabled by default; "
        "recommended when fitting into the LSB regime on real-data edge cases.",
    )
    outer_reg_sma_onset: float = Field(
        50.0,
        gt=0.0,
        description="SMA (pixels) at the midpoint of the outer regularization sigmoid. "
        "Below this the penalty is near zero; above it the penalty ramps up toward "
        "outer_reg_strength.",
    )
    outer_reg_strength: float = Field(
        2.0,
        ge=0.0,
        description="Peak strength of the outer regularization. Controls how "
        "aggressively the Tikhonov term shrinks per-iteration geometry steps "
        "in the outer region. Typical range: 1-3. "
        "Default 2.0 was the benchmark winner on the HSC edge-case and "
        "edge-real BCG suites.",
    )
    outer_reg_weights: dict = Field(
        default_factory=lambda: {"center": 1.0, "eps": 1.0, "pa": 1.0},
        description="Per-axis weights for the outer-region Tikhonov penalty. "
        "Default {1, 1, 1} damps center, ellipticity, and PA uniformly - "
        "required to prevent the selector asymmetry failure mode where a "
        "center-only penalty redirects the outer random walk from (x0, y0) "
        "into (eps, pa), producing saturated clipped steps. Set eps or pa "
        "to 0 to disable damping on that axis (rare). Units: center in "
        "pixel^2, eps dimensionless^2, pa radian^2.",
    )
    outer_reg_mode: str = Field(
        default="damping",
        pattern="^(damping|solver)$",
        description="How the outer-region Tikhonov penalty enters the fit. "
        "'damping' (default, recommended): step-shrink only - the "
        "per-iteration harmonic step is multiplied by (1-alpha) in the outer "
        "region, so saturated clipped jumps in PA/eps are suppressed but the "
        "fit still walks to its data-preferred geometry. Preserves harmonic "
        "convergence and runs at essentially baseline speed when combined "
        "with geometry_convergence=True (auto-enabled whenever outer-region "
        "regularization is active). "
        "'solver': full Tikhonov term (step shrink + pull toward the inner "
        "flux-weighted reference) in the update equations. Biases outer "
        "geometry toward the reference, which flattens genuine astrophysical "
        "PA/eps walks. Specialized use case; 'damping' is preferred for "
        "general LSB fitting.",
    )
    # --- Expert tuning knobs (defaults work for HSC-grade data) ---
    outer_reg_sma_width: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Sigmoid slope width (pixels) for the outer regularization "
        "ramp. When None (default), auto-computed as 0.4 * outer_reg_sma_onset. "
        "Expert tuning: smaller values give a sharper transition at the onset.",
    )
    outer_reg_ref_sma_factor: float = Field(
        2.0,
        gt=1.0,
        description="Expert tuning: inward isophotes with sma <= sma0 * this "
        "factor feed the flux-weighted reference geometry. Default 2.0 has "
        "never needed adjustment in benchmarks; typical range 1.5-3.",
    )

    # Permissive Geometry Mode (photutils-style)
    permissive_geometry: bool = Field(
        False,
        description="Enable permissive geometry updates (photutils-style). "
        "When True, geometry is always updated even from failed fits to prevent "
        "cascading failures, and None gradient errors are treated as acceptable. "
        "This matches photutils's 'best effort' approach vs isoster's stricter "
        "'convergence-required' default behavior.",
    )

    # First Isophote Robustness
    max_retry_first_isophote: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum number of retry attempts for the first isophote when it fails "
        "(stop_code not in {0, 1, 2}). Each attempt perturbs sma0 and/or initial "
        "geometry (eps, pa). 0 disables retry; default 3 uses a fixed schedule "
        "(0.8*sma0, 1.3*sma0, 0.6*sma0+near-circular) that adds near-zero overhead "
        "when the first isophote already succeeds.",
    )
    first_isophote_fail_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of consecutive initial isophotes that must all fail before "
        "declaring FIRST_FEW_ISOPHOTE_FAILURE. Default 3 means the first isophote "
        "at sma0 plus the next 2 growth steps must all have unacceptable stop codes.",
    )

    # Automatic LSB Geometry Lock
    lsb_auto_lock: bool = Field(
        False,
        description="Enable automatic LSB geometry lock. When True, outward growth starts "
        "with free geometry and automatically locks center/eps/pa to the last clean "
        "isophote (and switches to the configured integrator) once the gradient quality "
        "indicates the LSB regime. Inward growth and the central pixel are unaffected.",
    )
    lsb_auto_lock_maxgerr: float = Field(
        0.3,
        gt=0.0,
        description="Trigger threshold on the relative gradient error |grad_err / grad|. "
        "The lock commits when this threshold is exceeded on lsb_auto_lock_debounce "
        "outward isophotes in a row. Intentionally stricter than the free-fit maxgerr "
        "(default 0.5) so the lock fires before a stop_code=-1 would have.",
    )
    lsb_auto_lock_debounce: int = Field(
        2,
        ge=1,
        le=10,
        description="Number of consecutive triggered outward isophotes required before "
        "the LSB lock commits. Debounces single-isophote noise spikes. The lock anchor "
        "is the isophote immediately before the streak.",
    )
    lsb_auto_lock_integrator: str = Field(
        "median",
        pattern="^(mean|median)$",
        description="Integrator used for locked-region isophotes. Default 'median' for "
        "robustness against contaminants that the fixed-geometry path would otherwise "
        "average in.",
    )

    @model_validator(mode="after")
    def check_config_consistency(self):
        """Validate config consistency and emit warnings for likely misconfigurations."""
        # --- Hard errors ---
        if self.maxsma is not None and self.maxsma < self.minsma:
            raise ValueError(f"maxsma ({self.maxsma}) must be greater than minsma ({self.minsma})")
        if self.minit > self.maxit:
            raise ValueError(f"minit ({self.minit}) must be <= maxit ({self.maxit})")
        if self.integrator == "adaptive" and self.lsb_sma_threshold is None:
            raise ValueError("lsb_sma_threshold must be provided when integrator='adaptive'")
        if any(order < 3 for order in self.harmonic_orders):
            raise ValueError(
                f"harmonic_orders must all be >= 3 (orders 1 and 2 are used internally "
                f"for geometry fitting), got {self.harmonic_orders}"
            )
        # V7: central_reg_weights must use valid keys
        valid_reg_keys = {"eps", "pa", "center"}
        unknown_keys = set(self.central_reg_weights.keys()) - valid_reg_keys
        if unknown_keys:
            raise ValueError(f"central_reg_weights contains unknown keys: {unknown_keys}. Valid keys: {valid_reg_keys}")

        # --- Soft warnings for likely misconfigurations ---

        # V1: isofit_mode is a no-op without simultaneous_harmonics
        if self.isofit_mode != "in_loop" and not self.simultaneous_harmonics:
            warnings.warn("isofit_mode has no effect when simultaneous_harmonics=False", UserWarning, stacklevel=2)

        # V2: maxsma < sma0 means only one isophote + inward sweep
        if self.maxsma is not None and self.maxsma < self.sma0:
            warnings.warn(
                f"maxsma ({self.maxsma}) < sma0 ({self.sma0}): only one isophote + inward sweep will be produced",
                UserWarning,
                stacklevel=2,
            )

        # V3: minsma >= sma0 means inward loop never runs
        if self.minsma >= self.sma0:
            warnings.warn(
                f"minsma ({self.minsma}) >= sma0 ({self.sma0}): inward loop will not run", UserWarning, stacklevel=2
            )

        # V4: simultaneous geometry update with high damping may oscillate
        if self.geometry_update_mode == "simultaneous" and self.geometry_damping > 0.7:
            warnings.warn(
                "geometry_damping > 0.7 with geometry_update_mode='simultaneous' may cause oscillations; consider 0.5",
                UserWarning,
                stacklevel=2,
            )

        # V10: geometry convergence can never trigger if maxit too small
        if self.geometry_convergence and self.maxit < self.minit + self.geometry_stable_iters:
            warnings.warn(
                f"maxit ({self.maxit}) < minit + geometry_stable_iters "
                f"({self.minit + self.geometry_stable_iters}): "
                f"geometry convergence can never trigger",
                UserWarning,
                stacklevel=2,
            )

        # V11: lsb_sma_threshold with non-adaptive integrator is ignored
        if self.integrator != "adaptive" and self.lsb_sma_threshold is not None:
            warnings.warn(
                f"lsb_sma_threshold is set but integrator='{self.integrator}' "
                f"(not 'adaptive'); threshold will be ignored",
                UserWarning,
                stacklevel=2,
            )

        # Automatic LSB geometry lock: the lock is meaningful only when
        # starting from free geometry. If the caller already fixed anything,
        # the transition would be a no-op — fail loudly instead.
        if self.lsb_auto_lock:
            conflicting = [name for name in ("fix_center", "fix_pa", "fix_eps") if getattr(self, name)]
            if conflicting:
                raise ValueError(
                    f"lsb_auto_lock=True conflicts with {', '.join(conflicting)}=True: "
                    f"the auto-lock requires free geometry at the start of outward growth."
                )
            if self.debug is False:
                warnings.warn(
                    "lsb_auto_lock=True requires gradient diagnostics; debug will be enabled internally for this fit.",
                    UserWarning,
                    stacklevel=2,
                )

        # Outer center regularization sanity warnings.
        if self.use_outer_center_regularization:
            if self.outer_reg_sma_onset < self.sma0:
                warnings.warn(
                    f"outer_reg_sma_onset ({self.outer_reg_sma_onset}) < sma0 ({self.sma0}): "
                    f"the outer regularization penalty will fire from the first outward step "
                    f"and may over-constrain the mid-sma region. Typical onset ~ sma0 * 3.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.minsma >= self.sma0:
                warnings.warn(
                    f"use_outer_center_regularization=True with minsma ({self.minsma}) >= "
                    f"sma0 ({self.sma0}): no inward isophotes to build the reference centroid, "
                    f"so the reference will fall back to the anchor isophote center.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.fix_center and float(self.outer_reg_weights.get("center", 0.0)) > 0.0:
                warnings.warn(
                    "use_outer_center_regularization=True with fix_center=True and "
                    "outer_reg_weights['center']>0: the center is frozen, so the "
                    "center penalty is identically zero. Set outer_reg_weights['center']=0 "
                    "or disable fix_center to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.fix_eps and float(self.outer_reg_weights.get("eps", 0.0)) > 0.0:
                warnings.warn(
                    "outer_reg_weights['eps']>0 with fix_eps=True: ellipticity is "
                    "frozen, so the eps penalty is identically zero. Set "
                    "outer_reg_weights['eps']=0 or disable fix_eps.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.fix_pa and float(self.outer_reg_weights.get("pa", 0.0)) > 0.0:
                warnings.warn(
                    "outer_reg_weights['pa']>0 with fix_pa=True: position angle is "
                    "frozen, so the pa penalty is identically zero. Set "
                    "outer_reg_weights['pa']=0 or disable fix_pa.",
                    UserWarning,
                    stacklevel=2,
                )
            all_zero = all(float(self.outer_reg_weights.get(k, 0.0)) <= 0.0 for k in ("center", "eps", "pa"))
            if all_zero:
                warnings.warn(
                    "use_outer_center_regularization=True but all outer_reg_weights "
                    "are zero: the penalty is identically zero and the feature is "
                    "inert. Set at least one of center/eps/pa to a positive value.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.simultaneous_harmonics:
                warnings.warn(
                    f"outer_reg_mode='{self.outer_reg_mode}' combined with "
                    "simultaneous_harmonics=True is not supported: the "
                    "Tikhonov term is not wired into the 7-parameter ISOFIT "
                    "joint solver. The solver-level modification is skipped "
                    "for this fit; only the selector-layer penalty applies.",
                    UserWarning,
                    stacklevel=2,
                )
            # Auto-enable geometry_convergence when outer-region Tikhonov is
            # active. In damping mode the geometry genuinely settles; in
            # solver mode the Tikhonov pull creates a static balance point
            # that geometry_convergence detects cleanly. Without this the fit
            # runs to maxit on many outer isophotes with zero change to the
            # recorded geometry - pure cost, no benefit. See
            # docs/07-lsb-features.md section on convergence behavior.
            if not self.geometry_convergence:
                warnings.warn(
                    "use_outer_center_regularization=True auto-enables "
                    "geometry_convergence=True because the outer Tikhonov "
                    "term otherwise prevents the harmonic criterion from "
                    "tripping. Runtime would be several times slower with "
                    "no change to recorded geometry. To keep "
                    "geometry_convergence=False, disable "
                    "use_outer_center_regularization instead.",
                    UserWarning,
                    stacklevel=2,
                )
                self.geometry_convergence = True

        return self
