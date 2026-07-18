"""
Multi-band isoster (experimental).

Joint elliptical isophote fitting on multiple aligned same-pixel-grid
images. Produces a single shared geometry per SMA with per-band
intensities and per-band harmonic deviations, replacing the traditional
forced-photometry workflow.

This is a parallel codebase: it does not modify any module under the
top-level ``isoster`` package. See ``docs/10-multiband.md`` for the
user-facing reference and
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` for the locked
design record.
"""

from .config_mb import IsosterConfigMB
from .driver_mb import fit_image_multiband
from .plotting_mb import (
    plot_qa_summary_mb,
    subtract_outermost_sky_offset,
)
from .utils_mb import (
    isophote_results_mb_from_asdf,
    isophote_results_mb_from_fits,
    isophote_results_mb_to_asdf,
    isophote_results_mb_to_astropy_table,
    isophote_results_mb_to_fits,
    load_bands_from_hdus,
)

__all__ = [
    "IsosterConfigMB",
    "fit_image_multiband",
    "isophote_results_mb_to_fits",
    "isophote_results_mb_from_fits",
    "isophote_results_mb_to_asdf",
    "isophote_results_mb_from_asdf",
    "isophote_results_mb_to_astropy_table",
    "load_bands_from_hdus",
    "plot_qa_summary_mb",
    "subtract_outermost_sky_offset",
]
