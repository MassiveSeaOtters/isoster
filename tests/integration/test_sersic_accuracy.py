"""
Integration tests for isoster with QA visualization.

These tests verify the main fit_image() entry point using synthetic Sersic models
and generate QA figures following CLAUDE.md guidelines (vertical layout, centered models).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.gridspec import GridSpec

from isoster.config import IsosterConfig
from isoster.driver import fit_image
from isoster.output_paths import resolve_output_directory
from tests.fixtures import create_sersic_model


def get_qa_output_directory():
    """Return output directory for integration QA artifacts."""
    return resolve_output_directory("tests_integration", "sersic_accuracy")


def plot_qa_figure(image, results, true_profile, params, output_path):
    """
    Create comprehensive QA figure following CLAUDE.md guidelines.

    Per CLAUDE.md:
    - Vertical layout for 1-D profiles sharing same X-axis
    - Surface brightness profile should be larger
    - Proper PA normalization (no >90 degree jumps)
    - Color-code by stop code

    Args:
        image: Original image
        results: isoster fit results dict
        true_profile: Function for true 1D intensity profile
        params: Dict with model parameters
        output_path: Path to save figure
    """
    isophotes = results["isophotes"]

    # Extract data
    sma = np.array([iso["sma"] for iso in isophotes])
    intens = np.array([iso["intens"] for iso in isophotes])
    intens_err = np.array([iso["intens_err"] for iso in isophotes])
    x0 = np.array([iso["x0"] for iso in isophotes])
    y0 = np.array([iso["y0"] for iso in isophotes])
    eps_fit = np.array([iso["eps"] for iso in isophotes])
    pa_fit = np.array([iso["pa"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Normalize PA: wrap to [0, pi] and avoid jumps > pi/2
    pa_normalized = pa_fit.copy()
    pa_normalized = np.mod(pa_normalized, np.pi)  # Wrap to [0, pi]

    # Fix jumps larger than pi/2
    for i in range(1, len(pa_normalized)):
        diff = pa_normalized[i] - pa_normalized[i - 1]
        if diff > np.pi / 2:
            pa_normalized[i:] -= np.pi
        elif diff < -np.pi / 2:
            pa_normalized[i:] += np.pi

    # Wrap back to [0, pi]
    pa_normalized = np.mod(pa_normalized, np.pi)

    # Separate by stop code
    good = stop_codes == 0
    minor = stop_codes == 2
    flagged = stop_codes == 1
    failed = (stop_codes == -1) | (stop_codes == 3)

    # Compute true profiles
    true_intens = true_profile(sma)
    # Relative residuals
    rel_residual = (intens - true_intens) / true_intens

    # SMA^0.25 for X-axis (compress outer profile)
    x_axis = sma**0.25
    x_label = r"$\mathrm{SMA}^{0.25}$ (pixels$^{0.25}$)"

    # Create figure with 2 columns: images (left) + profiles (right, vertical)
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, 2, figure=fig, hspace=0.05, wspace=0.25, width_ratios=[1, 1], height_ratios=[2, 1, 1, 1, 1])

    # ===== LEFT COLUMN: Images =====
    # Original image with isophotes
    ax_img = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image[image > 0], [1, 99.9])
    im = ax_img.imshow(
        np.log10(np.clip(image, vmin, None)), cmap="gray", origin="lower", vmin=np.log10(vmin), vmax=np.log10(vmax)
    )
    plt.colorbar(im, ax=ax_img, label=r"$\log_{10}$(Intensity)")

    # Plot selective isophotes (every 5th, color-coded by stop code)
    from matplotlib.patches import Ellipse as MPLEllipse

    for i, iso in enumerate(isophotes[::5]):
        color = {0: "green", 1: "orange", 2: "yellow", -1: "red", 3: "red"}.get(iso["stop_code"], "gray")
        ellipse = MPLEllipse(
            (iso["x0"], iso["y0"]),
            2 * iso["sma"],
            2 * iso["sma"] * (1 - iso["eps"]),
            angle=np.degrees(iso["pa"]),
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.8,
        )
        ax_img.add_patch(ellipse)

    ax_img.set_title(f"Original (n={params['n']}, Re={params['R_e']:.1f}, eps={params['eps']:.2f})")
    ax_img.set_xlabel("X (pixels)")
    ax_img.set_ylabel("Y (pixels)")

    # Residual
    ax_resid = fig.add_subplot(gs[1:, 0])
    from isoster.model import build_isoster_model

    model = build_isoster_model(image.shape, isophotes)
    residual = image - model
    vmax_res = np.percentile(np.abs(residual), 99)
    im = ax_resid.imshow(residual, cmap="RdBu_r", origin="lower", vmin=-vmax_res, vmax=vmax_res)
    plt.colorbar(im, ax=ax_resid, label="Residual")
    ax_resid.set_title("Residual (Data - Model)")
    ax_resid.set_xlabel("X (pixels)")
    ax_resid.set_ylabel("Y (pixels)")

    # ===== RIGHT COLUMN: 1-D Profiles (VERTICAL LAYOUT, SHARED X-AXIS) =====

    # Row 1: Surface brightness (larger - 2 rows)
    ax_sb = fig.add_subplot(gs[0, 1])

    # Plot truth as dashed line
    ax_sb.plot(x_axis, np.log10(true_intens), "k--", linewidth=2, label="True Sersic", zorder=10)

    # Plot data as scatter with errorbars
    if good.any():
        ax_sb.errorbar(
            x_axis[good],
            np.log10(intens[good]),
            yerr=intens_err[good] / (intens[good] * np.log(10)),
            fmt="o",
            color="green",
            markersize=5,
            capsize=2,
            alpha=0.7,
            label="Converged (0)",
            zorder=3,
        )
    if minor.any():
        ax_sb.errorbar(
            x_axis[minor],
            np.log10(intens[minor]),
            yerr=intens_err[minor] / (intens[minor] * np.log(10)),
            fmt="s",
            color="yellow",
            markersize=5,
            capsize=2,
            alpha=0.7,
            label="Minor issues (2)",
            zorder=2,
        )
    if flagged.any():
        ax_sb.scatter(
            x_axis[flagged],
            np.log10(intens[flagged]),
            marker="^",
            color="orange",
            s=30,
            alpha=0.7,
            label="Flagged (1)",
            zorder=1,
        )
    if failed.any():
        ax_sb.scatter(
            x_axis[failed],
            np.log10(intens[failed]),
            marker="x",
            color="red",
            s=40,
            alpha=0.7,
            label="Failed (-1, 3)",
            zorder=0,
        )

    ax_sb.set_ylabel(r"$\log_{10}$(Intensity)", fontsize=11)
    ax_sb.set_title("Surface Brightness Profile")
    ax_sb.legend(loc="upper right", fontsize=9)
    ax_sb.grid(alpha=0.3)
    ax_sb.tick_params(labelbottom=False)  # Hide x-axis labels

    # Set Y-axis limits without errorbars
    valid_intens = intens[(good | minor) & (intens > 0)]
    if len(valid_intens) > 0:
        y_min = np.log10(np.min(valid_intens)) - 0.3
        y_max = np.log10(np.max(valid_intens)) + 0.2
        ax_sb.set_ylim(y_min, y_max)

    # Row 2: Relative residual
    ax_res = fig.add_subplot(gs[1, 1], sharex=ax_sb)
    ax_res.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    if good.any():
        ax_res.scatter(x_axis[good], rel_residual[good] * 100, color="green", s=20, alpha=0.7)
    if minor.any():
        ax_res.scatter(x_axis[minor], rel_residual[minor] * 100, color="yellow", s=20, alpha=0.7)
    ax_res.set_ylabel("Rel. Residual (%)", fontsize=10)
    ax_res.grid(alpha=0.3)
    ax_res.tick_params(labelbottom=False)

    # Row 3: Ellipticity
    ax_eps = fig.add_subplot(gs[2, 1], sharex=ax_sb)
    ax_eps.axhline(params["eps"], color="k", linestyle="--", linewidth=2, label="True", zorder=10)
    if good.any():
        ax_eps.scatter(x_axis[good], eps_fit[good], color="green", s=20, alpha=0.7, label="Fitted")
    if minor.any():
        ax_eps.scatter(x_axis[minor], eps_fit[minor], color="yellow", s=20, alpha=0.7)
    ax_eps.set_ylabel("Ellipticity", fontsize=10)
    ax_eps.legend(fontsize=8, loc="best")
    ax_eps.grid(alpha=0.3)
    ax_eps.tick_params(labelbottom=False)

    # Set Y-axis limits
    valid_eps = eps_fit[(good | minor)]
    if len(valid_eps) > 0:
        eps_margin = 0.1
        ax_eps.set_ylim(max(0, np.min(valid_eps) - eps_margin), min(1, np.max(valid_eps) + eps_margin))

    # Row 4: Position angle (with Y-axis label as requested)
    ax_pa = fig.add_subplot(gs[3, 1], sharex=ax_sb)
    ax_pa.axhline(params["pa"], color="k", linestyle="--", linewidth=2, label="True", zorder=10)
    if good.any():
        ax_pa.scatter(x_axis[good], pa_normalized[good], color="green", s=20, alpha=0.7, label="Fitted")
    if minor.any():
        ax_pa.scatter(x_axis[minor], pa_normalized[minor], color="yellow", s=20, alpha=0.7)
    ax_pa.set_ylabel("Position Angle (rad)", fontsize=10)
    ax_pa.legend(fontsize=8, loc="best")
    ax_pa.grid(alpha=0.3)
    ax_pa.tick_params(labelbottom=False)

    # Set Y-axis limits
    valid_pa = pa_normalized[(good | minor)]
    if len(valid_pa) > 0:
        pa_range = np.max(valid_pa) - np.min(valid_pa)
        if pa_range > 0:
            ax_pa.set_ylim(np.min(valid_pa) - 0.3 * pa_range, np.max(valid_pa) + 0.3 * pa_range)

    # Row 5: Centroid offset
    ax_center = fig.add_subplot(gs[4, 1], sharex=ax_sb)
    center_offset = np.sqrt((x0 - params["x0"]) ** 2 + (y0 - params["y0"]) ** 2)
    if good.any():
        ax_center.scatter(x_axis[good], center_offset[good], color="green", s=20, alpha=0.7)
    if minor.any():
        ax_center.scatter(x_axis[minor], center_offset[minor], color="yellow", s=20, alpha=0.7)
    ax_center.set_xlabel(x_label, fontsize=11)
    ax_center.set_ylabel("Center Offset (pix)", fontsize=10)
    ax_center.grid(alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"QA figure saved to: {output_path}")


# ============================================================================
# Integration Tests
# ============================================================================


def test_sersic_n4_noiseless():
    """
    Test Sersic n=4 (de Vaucouleurs) noiseless profile.

    High Sersic index requires higher oversampling.
    """
    # Model parameters
    R_e, n, I_e = 20.0, 4.0, 2000.0
    eps, pa = 0.4, np.pi / 4

    # Create model (centered, properly sized, high oversample for n=4)
    image, true_profile, params = create_sersic_model(
        R_e=R_e,
        n=n,
        I_e=I_e,
        eps=eps,
        pa=pa,
        oversample=10,  # High for steep central profile
    )

    # Configure fitting
    cfg = IsosterConfig(
        x0=params["x0"],
        y0=params["y0"],
        sma0=10.0,
        minsma=3.0,
        maxsma=min(80, params["x0"] - 10),
        astep=0.12,
        eps=eps,
        pa=pa,
        minit=10,
        maxit=50,
        conver=0.03,  # Tight for noiseless
        fix_center=True,
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results["isophotes"]

    # Generate QA figure
    output_dir = get_qa_output_directory()
    plot_qa_figure(image, results, true_profile, params, output_dir / "test_sersic_n4_noiseless.png")

    # Validation: Check accuracy in valid range (0.5*R_e to 8*R_e, sma>=3)
    sma = np.array([iso["sma"] for iso in isophotes])
    intens = np.array([iso["intens"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    valid = (sma >= 3) & (sma >= 0.5 * R_e) & (sma <= 8 * R_e) & (stop_codes == 0)

    valid_count = int(valid.sum())
    assert valid_count > 0, "No valid converged points found in the n=4 noiseless metric window"

    true_intens = true_profile(sma[valid])
    rel_diff = np.abs((intens[valid] - true_intens) / true_intens)

    max_rel_diff = np.max(rel_diff)
    median_rel_diff = np.median(rel_diff)

    print("✓ Sersic n=4 noiseless test:")
    print(f"  Image size: {params['shape']}, center: ({params['x0']}, {params['y0']})")
    print(f"  Valid range: {sma[valid].min():.1f} - {sma[valid].max():.1f} pixels")
    print(f"  Valid points: {valid_count}")
    print(f"  Max rel diff: {max_rel_diff * 100:.3f}%")
    print(f"  Median rel diff: {median_rel_diff * 100:.3f}%")
    print(f"  Converged: {(stop_codes == 0).sum()} / {len(isophotes)}")

    assert max_rel_diff < 0.01, f"Max relative difference {max_rel_diff * 100:.2f}% exceeds 1% threshold"


def test_sersic_n1_high_ellipticity():
    """
    Test high-ellipticity (eps=0.7) exponential disk with EA mode.

    Requires relaxed maxgerr for high ellipticity.
    """
    R_e, n, I_e = 25.0, 1.0, 1500.0
    eps, pa = 0.7, np.pi / 3

    # Create model (centered, properly sized)
    image, true_profile, params = create_sersic_model(R_e=R_e, n=n, I_e=I_e, eps=eps, pa=pa, oversample=5)

    # Add noise
    rng = np.random.RandomState(123)
    snr_at_re = 100
    noise_level = I_e / snr_at_re
    image += rng.normal(0, noise_level, image.shape)

    # Configure fitting with EA mode and RELAXED maxgerr for high ellipticity
    cfg = IsosterConfig(
        x0=params["x0"],
        y0=params["y0"],
        sma0=8.0,
        minsma=3.0,
        maxsma=min(100, params["x0"] - 10),
        astep=0.15,
        eps=eps,
        pa=pa,
        minit=10,
        maxit=50,
        conver=0.05,
        maxgerr=1.0,  # RELAXED for high ellipticity (default 0.5 too strict)
        use_eccentric_anomaly=True,  # Better for high ellipticity
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results["isophotes"]

    # Generate QA figure
    output_dir = get_qa_output_directory()
    plot_qa_figure(image, results, true_profile, params, output_dir / "test_sersic_n1_high_eps.png")

    # Validation
    sma = np.array([iso["sma"] for iso in isophotes])
    intens = np.array([iso["intens"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Window ends at 3 R_e: on this S/N_e = 100 exponential disk the ring S/N is about 75
    # there and 17 at 4 R_e, where a 5 % check on one ring fails by noise alone (the
    # default cubic read is about 20 % noisier than bilinear and did so at sma = 99).
    valid = (sma >= 3) & (sma >= 0.5 * R_e) & (sma <= 3 * R_e) & (stop_codes == 0)

    valid_count = int(valid.sum())
    assert valid_count > 0, "No valid converged points found in the high-ellipticity metric window"

    true_intens = true_profile(sma[valid])
    rel_diff = np.abs((intens[valid] - true_intens) / true_intens)
    max_rel_diff = np.max(rel_diff)
    median_rel_diff = np.median(rel_diff)

    print("✓ High ellipticity test (eps=0.7, S/N=100, maxgerr=1.0):")
    print(f"  Image size: {params['shape']}, center: ({params['x0']}, {params['y0']})")
    print(f"  Valid range: {sma[valid].min():.1f} - {sma[valid].max():.1f} pixels")
    print(f"  Valid points: {valid_count}")
    print(f"  Max intensity rel diff: {max_rel_diff * 100:.3f}%")
    print(f"  Median intensity rel diff: {median_rel_diff * 100:.3f}%")
    print(f"  Converged: {(stop_codes == 0).sum()} / {len(isophotes)}")

    assert max_rel_diff < 0.05, f"Max intensity difference {max_rel_diff * 100:.2f}% exceeds 5% threshold"


def test_sersic_n4_extreme_ellipticity():
    """
    Test VERY EXTREME case: Sersic n=4 with eps=0.6.

    Combines steep central profile with high ellipticity - most challenging case.
    Requires high oversampling AND relaxed maxgerr.
    """
    R_e, n, I_e = 20.0, 4.0, 2000.0
    eps, pa = 0.6, np.pi / 6

    # Create model (centered, properly sized, VERY high oversample)
    image, true_profile, params = create_sersic_model(
        R_e=R_e,
        n=n,
        I_e=I_e,
        eps=eps,
        pa=pa,
        oversample=15,  # Very high for n=4 + high eps
    )

    # Add moderate noise
    rng = np.random.RandomState(456)
    snr_at_re = 80
    noise_level = I_e / snr_at_re
    image += rng.normal(0, noise_level, image.shape)

    # Configure fitting - RELAXED settings for extreme case
    cfg = IsosterConfig(
        x0=params["x0"],
        y0=params["y0"],
        sma0=10.0,
        minsma=3.0,
        maxsma=min(80, params["x0"] - 10),
        astep=0.15,
        eps=eps,
        pa=pa,
        minit=10,
        maxit=60,  # More iterations
        conver=0.08,  # Relaxed convergence
        maxgerr=1.2,  # VERY relaxed for extreme ellipticity
        use_eccentric_anomaly=True,  # Essential for high ellipticity
    )

    # Run fit
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results["isophotes"]

    # Generate QA figure
    output_dir = get_qa_output_directory()
    plot_qa_figure(image, results, true_profile, params, output_dir / "test_sersic_n4_eps0.6_extreme.png")

    # Validation - more relaxed for extreme case
    sma = np.array([iso["sma"] for iso in isophotes])
    intens = np.array([iso["intens"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    valid = (sma >= 5) & (sma >= 0.5 * R_e) & (sma <= 5 * R_e) & (stop_codes == 0)

    valid_count = int(valid.sum())
    assert valid_count > 0, "No valid converged points found in the extreme-case metric window"

    true_intens = true_profile(sma[valid])
    rel_diff = np.abs((intens[valid] - true_intens) / true_intens)

    max_rel_diff = np.max(rel_diff)
    median_rel_diff = np.median(rel_diff)

    print("✓ EXTREME case (n=4, eps=0.6, S/N=80, maxgerr=1.2):")
    print(f"  Image size: {params['shape']}, center: ({params['x0']}, {params['y0']})")
    print(f"  Valid range: {sma[valid].min():.1f} - {sma[valid].max():.1f} pixels")
    print(f"  Valid points: {valid_count}")
    print(f"  Max rel diff: {max_rel_diff * 100:.3f}%")
    print(f"  Median rel diff: {median_rel_diff * 100:.3f}%")
    print(f"  Converged: {(stop_codes == 0).sum()} / {len(isophotes)}")

    # Relaxed threshold for extreme case: <10% error acceptable
    assert max_rel_diff < 0.10, f"Max relative difference {max_rel_diff * 100:.2f}% exceeds 10% threshold"
    assert (stop_codes == 0).sum() > 5, "Should have at least 5 converged isophotes for extreme case"


if __name__ == "__main__":
    # Run tests and generate QA figures
    pytest.main([__file__, "-v", "-s"])
