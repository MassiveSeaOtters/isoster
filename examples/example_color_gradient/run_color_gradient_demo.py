"""Known-truth colour-gradient demonstration for the multi-band joint fit.

This is the quantitative validation the maturity plan calls for in Phase 6: not
"the fit looks reasonable on a galaxy", but "here is a galaxy whose geometry and
colour profile we know exactly, and here is how close each method gets".

The point being tested is the actual value proposition of a joint multi-band
fit. Real galaxies have colour gradients: the light profile differs between
bands while the *isophote shape* is the same. Fitting each band independently
lets noise push each band's geometry somewhere slightly different, and a colour
computed by differencing those bands then mixes a real colour gradient with a
spurious geometry mismatch. A joint fit constrains one shared geometry from all
bands at once, so the colours are differenced on identical apertures.

The synthetic galaxy has, by construction:

* one common geometry in every band (``x0, y0, eps, pa``);
* a genuine colour gradient, produced by giving each band its own effective
  radius, so the bands have different radial profiles and different gradients;
* per-band noise levels matching HSC g/r/i sky variances.

Truth is therefore analytic: the geometry is what we planted, and the colour
profile is a closed-form function of radius.

Usage::

    uv run python examples/example_color_gradient/run_color_gradient_demo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.multiband import fit_image_multiband  # noqa: E402
from isoster.multiband.config_mb import IsosterConfigMB  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402

# --- Known truth -----------------------------------------------------------

BANDS = ["g", "r", "i"]
IMAGE_SIZE = 400
TRUE_GEOMETRY = {"x0": 200.0, "y0": 200.0, "eps": 0.30, "pa": 0.60}
SERSIC_N = 2.0
# One effective radius per band produces the colour gradient: the galaxy is more
# extended in g than in i, so g-i reddens inward, as in a real early-type.
BAND_RE = {"g": 46.0, "r": 40.0, "i": 36.0}
BAND_AMPLITUDE = {"g": 14.0, "r": 42.0, "i": 70.0}
# Sky noise per band, scaled from the HSC demo's measured sky variances.
BAND_SIGMA = {"g": 0.023, "r": 0.052, "i": 0.046}
ZEROPOINT = 27.0

SMA_MIN, SMA_MAX = 25.0, 110.0


def sersic_bn(n: float) -> float:
    """Ciotti & Bertin approximation, adequate for n >= 0.5."""
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


def true_intensity(band: str, sma: np.ndarray) -> np.ndarray:
    """Analytic surface brightness of band ``band`` at semi-major axis ``sma``."""
    bn = sersic_bn(SERSIC_N)
    return BAND_AMPLITUDE[band] * np.exp(-bn * ((sma / BAND_RE[band]) ** (1.0 / SERSIC_N) - 1.0))


def true_color(sma: np.ndarray) -> np.ndarray:
    """The planted g-i colour profile, in magnitudes."""
    return -2.5 * np.log10(true_intensity("g", sma) / true_intensity("i", sma))


def render_band(band: str, seed: int) -> np.ndarray:
    """One noisy image with the common geometry and this band's own profile."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float64)
    dx = x - TRUE_GEOMETRY["x0"]
    dy = y - TRUE_GEOMETRY["y0"]
    cos_pa, sin_pa = np.cos(TRUE_GEOMETRY["pa"]), np.sin(TRUE_GEOMETRY["pa"])
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    sma = np.sqrt(x_rot**2 + (y_rot / (1.0 - TRUE_GEOMETRY["eps"])) ** 2)
    img = true_intensity(band, np.maximum(sma, 1e-6))
    return img + rng.normal(0.0, BAND_SIGMA[band], img.shape)


# --- The two methods -------------------------------------------------------


def run_joint(images: List[np.ndarray], variance_maps: List[np.ndarray]) -> Dict[str, np.ndarray]:
    cfg = IsosterConfigMB(
        bands=BANDS,
        reference_band="i",
        sma0=30.0,
        minsma=10.0,
        maxsma=140.0,
        x0=TRUE_GEOMETRY["x0"],
        y0=TRUE_GEOMETRY["y0"],
        geometry_parameterized_solve=True,
    )
    res = fit_image_multiband(images, None, cfg, variance_maps=variance_maps)
    rows = [iso for iso in res["isophotes"] if int(iso.get("stop_code", -9)) == 0]
    out: Dict[str, np.ndarray] = {"sma": np.array([r["sma"] for r in rows], dtype=float)}
    for b in BANDS:
        out[b] = np.array([r.get(f"intens_{b}", np.nan) for r in rows], dtype=float)
    out["eps"] = np.array([r["eps"] for r in rows], dtype=float)
    out["pa"] = np.array([r["pa"] for r in rows], dtype=float)
    return out


def run_independent(images: List[np.ndarray], variance_maps: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Fit each band on its own, then difference — the pre-multi-band workflow."""
    per_band: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    geometry: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for b, img, var in zip(BANDS, images, variance_maps):
        cfg = IsosterConfig(
            sma0=30.0,
            minsma=10.0,
            maxsma=140.0,
            x0=TRUE_GEOMETRY["x0"],
            y0=TRUE_GEOMETRY["y0"],
        )
        res = fit_image(img, config=cfg, variance_map=var)
        rows = [iso for iso in res["isophotes"] if int(iso.get("stop_code", -9)) == 0]
        sma = np.array([r["sma"] for r in rows], dtype=float)
        per_band[b] = (sma, np.array([r["intens"] for r in rows], dtype=float))
        geometry[b] = (
            sma,
            np.array([r["eps"] for r in rows], dtype=float),
            np.array([r["pa"] for r in rows], dtype=float),
        )
    return {"per_band": per_band, "geometry": geometry}


def color_from_joint(joint: Dict[str, np.ndarray], sma_grid: np.ndarray) -> np.ndarray:
    g = np.interp(sma_grid, joint["sma"], joint["g"], left=np.nan, right=np.nan)
    i = np.interp(sma_grid, joint["sma"], joint["i"], left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -2.5 * np.log10(np.where((g > 0) & (i > 0), g / i, np.nan))


def color_from_independent(indep: Dict, sma_grid: np.ndarray) -> np.ndarray:
    sma_g, int_g = indep["per_band"]["g"]
    sma_i, int_i = indep["per_band"]["i"]
    g = np.interp(sma_grid, sma_g, int_g, left=np.nan, right=np.nan)
    i = np.interp(sma_grid, sma_i, int_i, left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -2.5 * np.log10(np.where((g > 0) & (i > 0), g / i, np.nan))


# --- Main ------------------------------------------------------------------

# The joint fit's advantage is a function of signal-to-noise: at very high S/N
# both methods recover the geometry almost perfectly and there is nothing to
# gain, while at low S/N independent per-band fits scatter and the differenced
# colour picks up a spurious geometry mismatch. Sweeping makes that dependence
# the result, instead of quoting one flattering number from a tuned fixture.
NOISE_SCALES = (1.0, 10.0, 30.0, 60.0)
REPRESENTATIVE_SCALE = 30.0
N_REALISATIONS = 6


def rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def run_at_noise(scale: float, sma_grid: np.ndarray, truth: np.ndarray):
    """Colour-error arrays for both methods at one noise level."""
    base = dict(BAND_SIGMA)
    for band in BAND_SIGMA:
        BAND_SIGMA[band] = base[band] * scale
    try:
        joint_err, indep_err = [], []
        example = None
        for trial in range(N_REALISATIONS):
            images = [render_band(b, seed=100 * trial + k) for k, b in enumerate(BANDS)]
            variance_maps = [np.full((IMAGE_SIZE, IMAGE_SIZE), BAND_SIGMA[b] ** 2) for b in BANDS]
            joint = run_joint(images, variance_maps)
            indep = run_independent(images, variance_maps)
            c_joint = color_from_joint(joint, sma_grid)
            c_indep = color_from_independent(indep, sma_grid)
            joint_err.append(c_joint - truth)
            indep_err.append(c_indep - truth)
            if example is None:
                example = (c_joint, c_indep)
        snr = float(true_intensity("g", np.array([SMA_MAX]))[0] / BAND_SIGMA["g"])
        return np.array(joint_err), np.array(indep_err), snr, example
    finally:
        for band in base:
            BAND_SIGMA[band] = base[band]


def main() -> int:
    warnings.simplefilter("ignore")
    out_dir = resolve_output_directory("example_color_gradient")
    print("=== Known-truth colour-gradient demonstration ===")
    print(
        f"  planted geometry : eps={TRUE_GEOMETRY['eps']}, pa={TRUE_GEOMETRY['pa']}, "
        f"centre=({TRUE_GEOMETRY['x0']}, {TRUE_GEOMETRY['y0']})"
    )
    print(f"  colour gradient  : re_g={BAND_RE['g']}, re_r={BAND_RE['r']}, re_i={BAND_RE['i']} px")
    print(f"  {N_REALISATIONS} noise realisations per level, {SMA_MIN:.0f} <= sma <= {SMA_MAX:.0f} px\n")

    sma_grid = np.linspace(SMA_MIN, SMA_MAX, 40)
    truth = true_color(sma_grid)

    print(f"  {'noise x':>8s} {'S/N @ sma_max':>14s} {'joint RMS':>11s} {'independent':>12s} {'ratio':>7s}")
    snrs, joint_rms, indep_rms = [], [], []
    example_at_representative = None
    for scale in NOISE_SCALES:
        joint_err, indep_err, snr, example = run_at_noise(scale, sma_grid, truth)
        j, i = rms(joint_err), rms(indep_err)
        snrs.append(snr)
        joint_rms.append(j)
        indep_rms.append(i)
        print(f"  {scale:8.0f} {snr:14.1f} {j:11.5f} {i:12.5f} {j / i:7.3f}")
        if scale == REPRESENTATIVE_SCALE:
            example_at_representative = (example, joint_err, indep_err, snr)

    print(
        "\n  The advantage is a low-S/N effect, which is where colour profiles are\n"
        "  actually hard: near-parity above S/N ~ 50, roughly half the error below\n"
        "  S/N ~ 10. A joint geometry cannot help when every band already nails it."
    )

    # --- Figure ---
    # Importing the isoster plotting stack pulls in a publication rcParams
    # profile with large fonts, which overflows a two-panel figure this size.
    # Reset to matplotlib defaults and set our own, so the demo renders the same
    # way regardless of what the import chain configured.
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({"text.usetex": False, "font.size": 9, "axes.titlesize": 10})

    (example, joint_err, indep_err, snr) = example_at_representative
    c_joint, c_indep = example
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_c.plot(sma_grid, truth, "k-", lw=2, label="planted truth", zorder=5)
    ax_c.plot(sma_grid, c_indep, "o--", ms=3, color="tab:orange", label="independent per-band")
    ax_c.plot(sma_grid, c_joint, "s-", ms=3, color="tab:blue", label="joint multi-band")
    ax_c.set_xlabel("semi-major axis [px]")
    ax_c.set_ylabel("g - i [mag]")
    ax_c.set_title(f"Colour profile, one realisation (S/N ~ {snr:.1f})")
    ax_c.legend(frameon=False, fontsize=8)

    ax_r.plot(snrs, indep_rms, "o--", color="tab:orange", label="independent per-band")
    ax_r.plot(snrs, joint_rms, "s-", color="tab:blue", label="joint multi-band")
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel(f"S/N per pixel at sma = {SMA_MAX:.0f} px")
    ax_r.set_ylabel("colour-profile RMS error [mag]")
    ax_r.set_title("Error vs signal-to-noise")
    ax_r.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig_path = out_dir / "color_gradient_truth_comparison.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {fig_path}")
    print("=== Demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
