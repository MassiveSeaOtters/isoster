"""
I/O Format Validation: FITS (4-HDU) and ASDF
=============================================

Fits a real galaxy (NGC3610 r-band), saves to both FITS and ASDF,
reloads from each, and generates QA figures side-by-side to confirm
that format and content are identical.

Usage
-----
::

    uv run python examples/example_io_validation/validate_fits_asdf.py

Output
------
``outputs/example_io_validation/``

- ``ngc3610_results.fits``      — 4-HDU FITS output
- ``ngc3610_results.asdf``      — ASDF output
- ``qa_from_original.png``      — QA from in-memory results
- ``qa_from_fits.png``          — QA from reloaded FITS
- ``qa_from_asdf.png``          — QA from reloaded ASDF
- ``validation_report.txt``     — numerical comparison summary
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits as astropy_fits

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

sys.path.insert(0, str(_HERE.parent / "example_ls_highorder_harmonic"))

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import plot_qa_summary_extended
from isoster.utils import (
    isophote_results_to_fits,
    isophote_results_from_fits,
    isophote_results_to_asdf,
    isophote_results_from_asdf,
)


def resolve_output_directory(name: str) -> Path:
    """Create and return outputs/<name>/."""
    out = _PROJECT_ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_ngc3610_rband() -> np.ndarray:
    """Load NGC3610 r-band (index 1) from the 3D FITS cube."""
    fits_path = _DATA_DIR / "ngc3610.fits"
    if not fits_path.exists():
        sys.exit(f"ERROR: FITS data not found at {fits_path}")

    with astropy_fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)

    if data.ndim == 3:
        return data[1]  # r-band
    return data


def compare_isophotes(iso_a: list[dict], iso_b: list[dict], label: str) -> list[str]:
    """Compare two isophote lists, return list of discrepancy strings."""
    issues = []
    if len(iso_a) != len(iso_b):
        issues.append(f"  {label}: length mismatch ({len(iso_a)} vs {len(iso_b)})")
        return issues

    max_intens_diff = 0.0
    max_sma_diff = 0.0
    max_eps_diff = 0.0
    max_pa_diff = 0.0

    for i, (a, b) in enumerate(zip(iso_a, iso_b)):
        max_sma_diff = max(max_sma_diff, abs(a['sma'] - b['sma']))
        max_intens_diff = max(max_intens_diff, abs(a['intens'] - b['intens']))
        max_eps_diff = max(max_eps_diff, abs(a['eps'] - b['eps']))
        max_pa_diff = max(max_pa_diff, abs(a['pa'] - b['pa']))

    for name, diff in [('sma', max_sma_diff), ('intens', max_intens_diff),
                       ('eps', max_eps_diff), ('pa', max_pa_diff)]:
        if diff > 1e-10:
            issues.append(f"  {label}: max |delta {name}| = {diff:.2e}")

    if not issues:
        issues.append(f"  {label}: PERFECT match ({len(iso_a)} isophotes)")

    return issues


def compare_configs(config_orig, config_loaded, label: str) -> list[str]:
    """Compare two IsosterConfig objects field by field."""
    issues = []
    if config_loaded is None:
        issues.append(f"  {label}: config is None (not recovered)")
        return issues

    if not isinstance(config_loaded, IsosterConfig):
        issues.append(f"  {label}: config type = {type(config_loaded)} (expected IsosterConfig)")
        return issues

    orig_dict = config_orig.model_dump()
    loaded_dict = config_loaded.model_dump()

    mismatches = []
    for key in orig_dict:
        if key not in loaded_dict:
            mismatches.append(f"    {key}: missing in loaded")
            continue
        v_orig = orig_dict[key]
        v_load = loaded_dict[key]
        if isinstance(v_orig, float) and isinstance(v_load, float):
            if abs(v_orig - v_load) > 1e-10:
                mismatches.append(f"    {key}: {v_orig} vs {v_load}")
        elif v_orig != v_load:
            mismatches.append(f"    {key}: {v_orig!r} vs {v_load!r}")

    if mismatches:
        issues.append(f"  {label}: config mismatches:")
        issues.extend(mismatches)
    else:
        issues.append(f"  {label}: config PERFECT match ({len(orig_dict)} fields)")

    return issues


def generate_qa(image, results, title, output_path, mask=None):
    """Generate an extended QA figure from results dict."""
    isophotes = results['isophotes']
    config = results.get('config', None)

    model = build_isoster_model(
        image.shape, isophotes, use_harmonics=True,
    )

    harmonic_orders = None
    if config is not None and hasattr(config, 'simultaneous_harmonics'):
        if config.simultaneous_harmonics:
            harmonic_orders = config.harmonic_orders

    plot_qa_summary_extended(
        title=title,
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        harmonic_orders=harmonic_orders,
        mask=mask,
        filename=str(output_path),
    )
    print(f"  QA figure saved: {output_path}")


def inspect_fits_structure(fits_path: str) -> list[str]:
    """Return a summary of the FITS HDU structure."""
    lines = []
    with astropy_fits.open(fits_path) as hdulist:
        lines.append(f"  FITS file: {fits_path}")
        lines.append(f"  Number of HDUs: {len(hdulist)}")
        for i, hdu in enumerate(hdulist):
            hdu_type = type(hdu).__name__
            name = hdu.name
            if hasattr(hdu, 'columns') and hdu.columns is not None:
                ncols = len(hdu.columns)
                nrows = hdu.data.shape[0] if hdu.data is not None else 0
                lines.append(f"    HDU {i}: {name:12s} ({hdu_type}, {nrows} rows x {ncols} cols)")
            else:
                lines.append(f"    HDU {i}: {name:12s} ({hdu_type})")

        # Check for any HIERARCH keywords in any HDU
        hierarch_count = 0
        for hdu in hdulist:
            for card in hdu.header.cards:
                if 'HIERARCH' in str(card):
                    hierarch_count += 1
        lines.append(f"  HIERARCH keywords found: {hierarch_count}")

    return lines


def main():
    output_dir = resolve_output_directory("example_io_validation")
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("I/O Format Validation Report")
    report_lines.append("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load image and fit
    # ------------------------------------------------------------------
    print("Loading NGC3610 r-band ...")
    image = load_ngc3610_rband()
    print(f"  Image shape: {image.shape}")

    config = IsosterConfig(
        sma0=6.0,
        use_eccentric_anomaly=True,
        simultaneous_harmonics=True,
        harmonic_orders=[3, 4, 5, 6, 7],
        convergence_scaling='sector_area',
        geometry_damping=0.7,
        permissive_geometry=True,
    )

    print("Fitting isophotes ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = isoster.fit_image(image, mask=None, config=config)

    isophotes = results['isophotes']
    n_iso = len(isophotes)
    n_conv = sum(1 for r in isophotes if r.get('stop_code', -99) == 0)
    print(f"  {n_iso} isophotes, {n_conv} converged")

    report_lines.append(f"\nGalaxy: NGC3610 r-band, shape={image.shape}")
    report_lines.append(f"Config: EA=True, sim_harm=True, orders=[3,4,5,6,7]")
    report_lines.append(f"Result: {n_iso} isophotes, {n_conv} converged")

    # ------------------------------------------------------------------
    # 2. Save to FITS (PRIMARY / ISOPHOTES / CONFIG / META)
    # ------------------------------------------------------------------
    fits_path = str(output_dir / "ngc3610_results.fits")
    print(f"\nSaving FITS: {fits_path}")

    with warnings.catch_warnings(record=True) as fits_warnings:
        warnings.simplefilter("always")
        isophote_results_to_fits(results, fits_path)

    verify_warnings = [w for w in fits_warnings
                       if issubclass(w.category, astropy_fits.verify.VerifyWarning)]
    print(f"  VerifyWarning count: {verify_warnings}")

    report_lines.append(f"\n--- FITS Output ---")
    report_lines.append(f"  VerifyWarning count: {len(verify_warnings)}")
    report_lines.extend(inspect_fits_structure(fits_path))

    # ------------------------------------------------------------------
    # 3. Save to ASDF
    # ------------------------------------------------------------------
    asdf_path = str(output_dir / "ngc3610_results.asdf")
    print(f"\nSaving ASDF: {asdf_path}")
    isophote_results_to_asdf(results, asdf_path)

    import os
    fits_size = os.path.getsize(fits_path)
    asdf_size = os.path.getsize(asdf_path)
    report_lines.append(f"\n--- ASDF Output ---")
    report_lines.append(f"  File sizes: FITS={fits_size:,} bytes, ASDF={asdf_size:,} bytes")

    # ------------------------------------------------------------------
    # 4. Reload from both formats
    # ------------------------------------------------------------------
    print("\nLoading from FITS ...")
    loaded_fits = isophote_results_from_fits(fits_path)
    print(f"  isophotes: {len(loaded_fits['isophotes'])}, "
          f"config: {type(loaded_fits['config']).__name__}")

    print("Loading from ASDF ...")
    loaded_asdf = isophote_results_from_asdf(asdf_path)
    print(f"  isophotes: {len(loaded_asdf['isophotes'])}, "
          f"config: {type(loaded_asdf['config']).__name__}")

    # ------------------------------------------------------------------
    # 5. Numerical comparison
    # ------------------------------------------------------------------
    report_lines.append(f"\n--- Numerical Comparison ---")

    report_lines.extend(compare_isophotes(
        results['isophotes'], loaded_fits['isophotes'], "FITS isophotes"))
    report_lines.extend(compare_configs(
        results['config'], loaded_fits['config'], "FITS config"))

    report_lines.extend(compare_isophotes(
        results['isophotes'], loaded_asdf['isophotes'], "ASDF isophotes"))
    report_lines.extend(compare_configs(
        results['config'], loaded_asdf['config'], "ASDF config"))

    # Cross-check: FITS vs ASDF should also match
    report_lines.extend(compare_isophotes(
        loaded_fits['isophotes'], loaded_asdf['isophotes'], "FITS vs ASDF isophotes"))

    # ------------------------------------------------------------------
    # 6. Generate QA figures from all three sources
    # ------------------------------------------------------------------
    print("\nGenerating QA figures ...")

    generate_qa(image, results,
                "NGC3610 — Original (in-memory)",
                output_dir / "qa_from_original.png")

    generate_qa(image, loaded_fits,
                "NGC3610 — Loaded from FITS",
                output_dir / "qa_from_fits.png")

    generate_qa(image, loaded_asdf,
                "NGC3610 — Loaded from ASDF",
                output_dir / "qa_from_asdf.png")

    # ------------------------------------------------------------------
    # 7. Write report
    # ------------------------------------------------------------------
    report_lines.append(f"\n--- QA Figures ---")
    report_lines.append(f"  qa_from_original.png  — from in-memory results")
    report_lines.append(f"  qa_from_fits.png      — from reloaded FITS")
    report_lines.append(f"  qa_from_asdf.png      — from reloaded ASDF")
    report_lines.append(f"\nAll three QA figures should be visually identical.")
    report_lines.append("=" * 60)

    report_path = output_dir / "validation_report.txt"
    report_text = "\n".join(report_lines)
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\nReport saved: {report_path}")
    print(f"All outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()
