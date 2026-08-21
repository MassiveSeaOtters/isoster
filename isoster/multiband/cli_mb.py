"""
Command-line interface for multi-band isoster (``isoster-mb``).

This is a **parallel CLI** to the single-band ``isoster`` entry point —
deliberately not sharing implementation with :mod:`isoster.cli` so that
multi-band CLI changes cannot regress the single-band path. Multi-band
carries operational failure modes with no single-band counterpart —
per-band mask asymmetry, missing variance maps in some bands, alignment
mismatches — and the split keeps each path's handling of them local to
its own implementation. The argument layout mirrors the
single-band CLI for user familiarity (``--config``, ``--output``,
``--x0``, ``--y0``, ``--sma0``, ``--fix-center``, etc.) but takes one
positional FITS path per band plus a ``--bands`` flag listing the band
names in the same order.

Usage::

    isoster-mb image_g.fits image_r.fits image_i.fits \\
        --bands g r i --reference-band i \\
        --output isophotes_mb.fits --config config.yaml

A YAML ``--config`` file accepts any field from
:class:`isoster.multiband.IsosterConfigMB`. CLI flags override YAML
values. Output extension drives the writer:

* ``.fits``  → :func:`isoster.multiband.isophote_results_mb_to_fits`
* ``.asdf``  → :func:`isoster.multiband.isophote_results_mb_to_asdf`
* anything else → astropy ``Table.write`` of the per-isophote table
  (``.csv``, ``.ecsv``, …).

The CLI prints an interface-stability banner per invocation
(suppressible with ``--quiet``): the fit itself is supported in its
default configuration, but the CLI arguments and the Schema-1 output
layout may still change between releases.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

import numpy as np
import yaml
from astropy.io import fits

from .config_mb import IsosterConfigMB
from .driver_mb import fit_image_multiband
from .utils_mb import (
    isophote_results_mb_to_asdf,
    isophote_results_mb_to_astropy_table,
    isophote_results_mb_to_fits,
)

INTERFACE_STABILITY_BANNER = (
    "============================================================\n"
    " isoster-mb: multi-band CLI — UNSTABLE INTERFACE\n"
    " The default configuration is supported; the CLI arguments and\n"
    " the Schema-1 output layout may still change between releases.\n"
    " simultaneous_* harmonics are experimental and warn when used.\n"
    " See docs/10-multiband.md.\n"
    "============================================================"
)


def _load_yaml_config(path: str) -> dict:
    """Load a YAML config dict from ``path`` (returns ``{}`` if missing)."""
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config YAML at {path} did not parse to a mapping")
    return loaded


def _load_image(path: str) -> np.ndarray:
    """Load image data from FITS, falling back to extension 1 if PrimaryHDU is empty."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None and len(hdul) > 1:
            data = hdul[1].data
    if data is None:
        raise ValueError(f"No image data found in {path}")
    return np.asarray(data, dtype=np.float64)


def _load_mask(path: str) -> np.ndarray:
    """Load a boolean mask from a FITS file (PrimaryHDU or extension 1)."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None and len(hdul) > 1:
            data = hdul[1].data
    if data is None:
        raise ValueError(f"No mask data found in {path}")
    return np.asarray(data).astype(bool)


def _load_variance(path: str) -> np.ndarray:
    """Load a variance map from FITS as float64."""
    return _load_image(path)


def build_parser() -> argparse.ArgumentParser:
    """Construct the multi-band CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="isoster-mb",
        description=(
            "Multi-band isoster CLI (unstable interface). Fits a single "
            "shared elliptical-isophote geometry across multiple aligned, "
            "same-pixel-grid images and reports per-band intensities and "
            "harmonics."
        ),
    )
    parser.add_argument(
        "images",
        nargs="+",
        help=(
            "Input image FITS files, one per band, listed in the same "
            "order as --bands. Each must share the same (H, W) grid."
        ),
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        help=("Band names aligned with the positional images. Required if the YAML config does not provide ``bands``."),
    )
    parser.add_argument(
        "--reference-band",
        dest="reference_band",
        help=(
            "Diagnostic reference band; must appear in --bands. "
            "Defaults to the first band when unset (overrides the YAML "
            "config's ``reference_band``)."
        ),
    )
    parser.add_argument(
        "--mask",
        help=("Single mask FITS broadcast to every band. Mutually exclusive with --masks."),
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        help=("Per-band mask FITS files (must match --bands length). Mutually exclusive with --mask."),
    )
    parser.add_argument(
        "--variance-maps",
        dest="variance_maps",
        nargs="+",
        help=("Per-band variance maps for WLS (all-or-nothing: must match --bands length)."),
    )
    parser.add_argument("--config", help="YAML configuration file for IsosterConfigMB.")
    parser.add_argument(
        "--output",
        default="isophotes_mb.fits",
        help=(
            "Output file. ``.fits`` and ``.asdf`` use multi-band Schema-1 "
            "writers; any other extension uses astropy Table.write."
        ),
    )
    parser.add_argument(
        "--template",
        help=(
            "Template isophotes file for forced photometry (Schema-1 "
            "multi-band FITS or single-band FITS). Bypasses the iteration "
            "loop and extracts per-band intensities at the template's "
            "exact geometry."
        ),
    )

    # CLI overrides (mirror single-band; underscore aliases match user habits).
    parser.add_argument("--x0", type=float, help="Initial center x.")
    parser.add_argument("--y0", type=float, help="Initial center y.")
    parser.add_argument("--sma0", type=float, help="Initial semi-major axis.")
    parser.add_argument(
        "--fix-center",
        "--fix_center",
        dest="fix_center",
        action="store_true",
        help="Fix the center coordinates during fitting.",
    )
    parser.add_argument(
        "--fix-eps",
        "--fix_eps",
        dest="fix_eps",
        action="store_true",
        help="Fix the ellipticity during fitting.",
    )
    parser.add_argument(
        "--fix-pa",
        "--fix_pa",
        dest="fix_pa",
        action="store_true",
        help="Fix the position angle during fitting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the interface-stability banner.",
    )
    return parser


def _resolve_bands(
    cfg_dict: dict,
    args_bands: Optional[Sequence[str]],
    n_images: int,
) -> List[str]:
    """Resolve the band list from CLI overrides, falling back to YAML config."""
    bands = list(args_bands) if args_bands else list(cfg_dict.get("bands", []) or [])
    if not bands:
        raise SystemExit("isoster-mb: --bands is required (or set ``bands`` in --config YAML)")
    if len(bands) != n_images:
        raise SystemExit(
            f"isoster-mb: expected {len(bands)} positional images (one per band in --bands={bands}); got {n_images}"
        )
    return bands


def _resolve_reference_band(
    cfg_dict: dict,
    args_ref: Optional[str],
    bands: Sequence[str],
) -> str:
    """Resolve the reference band, defaulting to the first band when unset."""
    ref = args_ref or cfg_dict.get("reference_band")
    if not ref:
        # Mirror the YAML-friendly default: first band wins if the user did
        # not specify, matching the docs guidance for "diagnostic only".
        ref = bands[0]
    if ref not in bands:
        raise SystemExit(f"isoster-mb: --reference-band={ref!r} is not in --bands={list(bands)}")
    return ref


def _resolve_masks(args, n_bands: int):
    """Return masks in a form acceptable to ``fit_image_multiband``."""
    if args.mask and args.masks:
        raise SystemExit("isoster-mb: --mask and --masks are mutually exclusive")
    if args.mask:
        return _load_mask(args.mask)
    if args.masks:
        if len(args.masks) != n_bands:
            raise SystemExit(f"isoster-mb: --masks expected {n_bands} files; got {len(args.masks)}")
        return [_load_mask(p) for p in args.masks]
    return None


def _resolve_variance_maps(args, n_bands: int):
    """Return variance_maps in a form acceptable to ``fit_image_multiband``."""
    if not args.variance_maps:
        return None
    if len(args.variance_maps) != n_bands:
        raise SystemExit(f"isoster-mb: --variance-maps expected {n_bands} files; got {len(args.variance_maps)}")
    return [_load_variance(p) for p in args.variance_maps]


def _apply_cli_overrides(cfg_dict: dict, args) -> dict:
    """Layer CLI override flags on top of the YAML config dict."""
    if args.x0 is not None:
        cfg_dict["x0"] = args.x0
    if args.y0 is not None:
        cfg_dict["y0"] = args.y0
    if args.sma0 is not None:
        cfg_dict["sma0"] = args.sma0
    if args.fix_center:
        cfg_dict["fix_center"] = True
    if args.fix_eps:
        cfg_dict["fix_eps"] = True
    if args.fix_pa:
        cfg_dict["fix_pa"] = True
    return cfg_dict


def _write_output(result: dict, path: str) -> None:
    """Dispatch ``result`` to the writer matching the output extension."""
    lower = path.lower()
    if lower.endswith(".fits") or lower.endswith(".fit"):
        isophote_results_mb_to_fits(result, path)
    elif lower.endswith(".asdf"):
        isophote_results_mb_to_asdf(result, path)
    else:
        tbl = isophote_results_mb_to_astropy_table(result)
        tbl.write(path, overwrite=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the ``isoster-mb`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.quiet:
        print(INTERFACE_STABILITY_BANNER, file=sys.stderr)

    cfg_dict: dict = _load_yaml_config(args.config) if args.config else {}
    cfg_dict = _apply_cli_overrides(cfg_dict, args)

    bands = _resolve_bands(cfg_dict, args.bands, n_images=len(args.images))
    cfg_dict["bands"] = bands
    cfg_dict["reference_band"] = _resolve_reference_band(
        cfg_dict,
        args.reference_band,
        bands,
    )

    images = [_load_image(p) for p in args.images]
    masks = _resolve_masks(args, len(bands))
    variance_maps = _resolve_variance_maps(args, len(bands))

    config = IsosterConfigMB(**cfg_dict)

    print(
        f"isoster-mb: fitting {len(bands)} bands={bands} reference={config.reference_band} → {args.output}",
        file=sys.stderr,
    )
    result = fit_image_multiband(
        images=images,
        masks=masks,
        config=config,
        variance_maps=variance_maps,
        template_isophotes=args.template,
    )

    _write_output(result, args.output)
    print(f"isoster-mb: saved results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
