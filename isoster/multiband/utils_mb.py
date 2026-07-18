"""
FITS I/O for multi-band isoster results (Schema 1).

Decision D14: a single ``ISOPHOTES`` BinTableHDU with shared-geometry
columns plus per-band-suffixed columns (``intens_<b>``, ``a3_<b>``, ...).
The CONFIG HDU mirrors the existing single-band layout (PARAM/VALUE
rows, JSON-serialized values), with all multi-band-specific fields
(``bands``, ``reference_band``, ``band_weights``,
``harmonic_combination``, ``variance_mode``) recorded automatically by
``IsosterConfigMB.model_dump()``.

Provides :func:`load_bands_from_hdus` as the recommended way to extract
``(images, masks, variance_maps, bands)`` tuples from a list of FITS
HDUs whose ``FILTER`` headers carry the band identity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.typing import NDArray

from .._shared import _build_config_hdu, _build_meta_hdu, _config_to_dict, _parse_meta_hdu
from .config_mb import IsosterConfigMB

logger = logging.getLogger(__name__)


# Top-level keys already persisted via the ISOPHOTES table, the CONFIG HDU,
# or the primary header — everything else lands in the META HDU (M13).
_META_SKIP_KEYS = {
    "isophotes",
    "config",
    "bands",
    "multiband",
    "harmonic_combination",
    "reference_band",
    "band_weights",
    "variance_mode",
    "fit_per_band_intens_jointly",
    "loose_validity",
    "multiband_higher_harmonics",
    "harmonic_orders",
    "harmonics_shared",
}


def _build_meta_hdu_mb(results: dict) -> fits.BinTableHDU:
    """Build a META BinTableHDU for non-structural top-level result keys.

    Stores keys like ``forced_photometry_mode``, ``template_n_isophotes``,
    ``sky_offsets``, ``lsb_auto_lock*`` and ``first_isophote_*`` as JSON
    PARAM/VALUE rows so the FITS round-trip preserves them (M13).
    """
    meta = {key: value for key, value in results.items() if key not in _META_SKIP_KEYS}
    return _build_meta_hdu(meta)


def _parse_meta_hdu_mb(hdu: fits.BinTableHDU) -> dict:
    """Parse a META HDU (JSON PARAM/VALUE rows) back into result keys."""
    return _parse_meta_hdu(hdu)


# ---------------------------------------------------------------------------
# Multi-band CONFIG parser (overrides the single-band parse to reconstruct
# IsosterConfigMB instead of IsosterConfig)
# ---------------------------------------------------------------------------


def _parse_config_hdu_mb(hdu: fits.BinTableHDU) -> Optional[IsosterConfigMB]:
    """
    Reconstruct an :class:`IsosterConfigMB` from a CONFIG BinTableHDU.

    Mirrors :func:`isoster.utils._parse_config_hdu` but targets the
    multi-band config class. Returns ``None`` on failure (forward-
    compatibility with newer schemas).
    """
    config_tbl = Table.read(hdu)
    if len(config_tbl) == 0:
        return None
    parsed: dict = {}
    for row in config_tbl:
        key = str(row["PARAM"])
        try:
            parsed[key] = json.loads(row["VALUE"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping unparseable multi-band config key '%s'", key)
            continue
    known = set(IsosterConfigMB.model_fields.keys())
    filtered = {k: v for k, v in parsed.items() if k in known}
    try:
        return IsosterConfigMB(**filtered)
    except Exception as e:  # noqa: BLE001 — defensive
        logger.warning("Could not reconstruct IsosterConfigMB from FITS: %s", e)
        return None


# ---------------------------------------------------------------------------
# Schema 1 writer / reader
# ---------------------------------------------------------------------------


def isophote_results_mb_to_astropy_table(results: dict) -> Table:
    """
    Convert a multi-band isoster result dict into an astropy Table.

    Per-isophote rows include shared-geometry columns and per-band
    suffixed columns. Astropy infers column dtypes; rows missing a
    field are filled with masked / NaN values.
    """
    isophotes = results["isophotes"]
    if not isophotes:
        return Table()
    # Use Table.from_pandas-style row construction. astropy's
    # Table(rows=...) handles missing keys per row by emitting masked
    # entries.
    return Table(rows=isophotes)


def isophote_results_mb_to_fits(
    results: dict,
    filename: Union[str, Path],
    overwrite: bool = True,
) -> None:
    """
    Save multi-band isoster results to a multi-HDU FITS file (Schema 1).

    HDU layout:

    * 0: ``PrimaryHDU`` with ``MULTIBND=True`` and ``BANDS=<comma-list>`` headers.
    * 1: ``ISOPHOTES`` BinTable with shared-geometry + per-band-suffix columns.
    * 2: ``CONFIG`` BinTable (PARAM/VALUE rows from IsosterConfigMB.model_dump()).
    * 3: ``META`` BinTable (JSON PARAM/VALUE rows for the remaining
      top-level result keys: ``forced_photometry_mode``, ``sky_offsets``,
      ``lsb_auto_lock*``, ``first_isophote_*``, ...).
    """
    tbl = isophote_results_mb_to_astropy_table(results)
    iso_hdu = fits.table_to_hdu(tbl)
    iso_hdu.name = "ISOPHOTES"

    config_hdu = _build_config_hdu(results)
    meta_hdu = _build_meta_hdu_mb(results)

    primary = fits.PrimaryHDU()
    primary.header["MULTIBND"] = (True, "Multi-band schema 1 result")
    bands = list(results.get("bands", []))
    if bands:
        primary.header["BANDS"] = (",".join(bands), "comma-separated band names")
    primary.header["REFBAND"] = (
        str(results.get("reference_band", "")),
        "diagnostic reference band",
    )
    primary.header["HARMCMB"] = (
        str(results.get("harmonic_combination", "joint")),
        "harmonic combination mode",
    )
    primary.header["VARMODE"] = (
        str(results.get("variance_mode", "ols")),
        "variance mode (wls or ols)",
    )

    hdulist = fits.HDUList([primary, iso_hdu, config_hdu, meta_hdu])
    hdulist.writeto(filename, overwrite=overwrite)


def isophote_results_mb_from_fits(filename: Union[str, Path]) -> dict:
    """
    Load multi-band isoster results from a Schema-1 FITS file.

    Inverse of :func:`isophote_results_mb_to_fits`. The returned dict
    has the same shape as the in-memory result dict (`'isophotes'`
    list of dicts plus the multi-band top-level keys), suitable for
    feeding back into plotting routines or other downstream tools.
    """
    with fits.open(filename) as hdulist:
        try:
            iso_tbl = Table.read(hdulist["ISOPHOTES"])
        except KeyError:
            iso_tbl = Table.read(hdulist[1])

        config: Optional[IsosterConfigMB] = None
        if "CONFIG" in hdulist:
            config = _parse_config_hdu_mb(hdulist["CONFIG"])

        # Non-structural top-level keys (forced_photometry_mode,
        # sky_offsets, lsb_auto_lock*, first_isophote_*, ...; M13).
        meta: dict = {}
        if "META" in hdulist:
            meta = _parse_meta_hdu_mb(hdulist["META"])

        # Fall back to PrimaryHDU headers if CONFIG is missing or stale.
        primary_hdr = hdulist[0].header
        bands_from_hdr = primary_hdr.get("BANDS")
        ref_from_hdr = primary_hdr.get("REFBAND")
        harm_from_hdr = primary_hdr.get("HARMCMB")
        var_from_hdr = primary_hdr.get("VARMODE")

    isophotes: List[dict] = []
    for row in iso_tbl:
        out: dict = {}
        for col in iso_tbl.colnames:
            val = row[col]
            # Heterogeneous row keys produce masked cells; skip them so the
            # loaded dict keeps the original "key absent" shape (review B3).
            if val is np.ma.masked:
                continue
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            elif isinstance(val, np.str_):
                val = str(val)
            out[col] = val
        isophotes.append(out)

    fit_jointly = True
    loose_validity = False
    higher_mode = "independent"
    higher_orders: List[int] = [3, 4]
    if config is not None:
        bands = list(config.bands)
        ref_band = config.reference_band
        harm = config.harmonic_combination
        variance_mode = config.variance_mode if config.variance_mode is not None else "ols"
        band_weights = config.resolved_band_weights()
        fit_jointly = bool(config.fit_per_band_intens_jointly)
        loose_validity = bool(config.loose_validity)
        higher_mode = config.multiband_higher_harmonics
        higher_orders = list(config.harmonic_orders)
    else:
        bands = bands_from_hdr.split(",") if bands_from_hdr else []
        ref_band = ref_from_hdr or (bands[0] if bands else "")
        harm = harm_from_hdr or "joint"
        variance_mode = var_from_hdr or "ols"
        band_weights = {b: 1.0 for b in bands}

    result: dict = {
        "isophotes": isophotes,
        "config": config,
        "bands": bands,
        "multiband": True,
        "harmonic_combination": harm,
        "reference_band": ref_band,
        "band_weights": band_weights,
        "variance_mode": variance_mode,
        "fit_per_band_intens_jointly": fit_jointly,
        "loose_validity": loose_validity,
        "multiband_higher_harmonics": higher_mode,
        "harmonic_orders": higher_orders,
        "harmonics_shared": higher_mode != "independent",
    }
    result.update(meta)
    return result


# ---------------------------------------------------------------------------
# Bands loader helper (decision D8)
# ---------------------------------------------------------------------------


def load_bands_from_hdus(
    hdus: Sequence[Union[fits.PrimaryHDU, fits.ImageHDU]],
    *,
    filter_keyword: str = "FILTER",
    drop_prefix: str = "HSC-",
) -> Tuple[
    List[NDArray[np.floating]],
    List[Optional[NDArray[np.bool_]]],
    Optional[List[NDArray[np.floating]]],
    List[str],
]:
    """
    Convenience helper to extract aligned multi-band inputs from FITS HDUs.

    Each HDU's ``FILTER`` header value is sanitized (strip an optional
    ``HSC-`` prefix, lowercase) and used as the band name. Returns the
    image arrays, a list of all-``None`` masks (placeholder), an all-
    ``None`` variance-map list (placeholder), and the band-name list.

    This is intentionally minimal: callers that want masks or variance
    maps should construct them separately and pass them to
    :func:`fit_image_multiband`. The helper exists primarily so the
    asteris demo can do
    ``images, _, _, bands = load_bands_from_hdus(hdus)``.

    Parameters
    ----------
    hdus
        Sequence of HDUs; each must carry the FILTER header keyword.
    filter_keyword
        Header keyword to read for the band name. Default ``"FILTER"``.
    drop_prefix
        Prefix to strip from the filter value (case-insensitive).
        Default ``"HSC-"``.

    Returns
    -------
    images, masks, variance_maps, bands
        Aligned for direct unpacking into the multi-band driver.
    """
    images: List[NDArray[np.floating]] = []
    bands: List[str] = []
    for i, hdu in enumerate(hdus):
        if filter_keyword not in hdu.header:
            raise KeyError(f"hdus[{i}] is missing the required '{filter_keyword}' header keyword.")
        raw = str(hdu.header[filter_keyword]).strip()
        if drop_prefix and raw.upper().startswith(drop_prefix.upper()):
            raw = raw[len(drop_prefix) :]
        # Replace remaining hyphens with underscores so the resulting
        # band name matches IsosterConfigMB's regex.
        raw = raw.replace("-", "_")
        bands.append(raw.lower())
        images.append(np.asarray(hdu.data, dtype=np.float64))
    masks: List[Optional[NDArray[np.bool_]]] = [None] * len(hdus)
    variance_maps: Optional[List[NDArray[np.floating]]] = None
    return images, masks, variance_maps, bands


# ---------------------------------------------------------------------------
# ASDF support (optional dependency) — Stage I
# ---------------------------------------------------------------------------


def _coerce_native(value):
    """Convert a numpy scalar to its Python equivalent; pass through otherwise."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def isophote_results_mb_to_asdf(
    results: dict,
    filename: Union[str, Path],
) -> None:
    """
    Save multi-band isoster results to an ASDF file (Schema 1 mirror).

    ASDF stores the full result dict natively, so per-band-suffixed columns,
    nested config parameters, and lock-state metadata round-trip without any
    manual encoding. The on-disk tree contains:

    * ``format_version`` — always ``1``.
    * ``isophotes`` — list of per-row dicts (shared geometry + per-band
      suffix columns + optional ``lsb_locked`` / ``lsb_auto_lock_anchor``
      / ``cog_<b>`` / ``flag_*`` fields).
    * ``config`` — :class:`IsosterConfigMB` model dump (all multi-band
      fields, including ``bands``, ``reference_band``, ``band_weights``,
      ``harmonic_combination``, ``variance_mode``).
    * Top-level multi-band keys mirroring
      :func:`isophote_results_mb_from_fits` for downstream readers that
      bypass the config parse.

    Parameters
    ----------
    results
        The dict returned by :func:`fit_image_multiband`.
    filename
        Output filename (conventionally ending in ``.asdf``).

    Raises
    ------
    ImportError
        If the ``asdf`` package is not installed.
    """
    try:
        import asdf
    except ImportError:
        raise ImportError(
            "The 'asdf' package is required for ASDF I/O. "
            "Install it with: pip install 'asdf>=3.0'  or  uv pip install 'asdf>=3.0'"
        )

    isophotes = list(results.get("isophotes", []))
    config_dict = _config_to_dict(results.get("config", None))

    tree: dict = {
        "format_version": 1,
        "multiband": True,
        "isophotes": isophotes,
        "config": config_dict,
        "bands": list(results.get("bands", [])),
        "reference_band": results.get("reference_band", ""),
        "harmonic_combination": results.get("harmonic_combination", "joint"),
        "variance_mode": results.get("variance_mode", "ols"),
        "band_weights": dict(results.get("band_weights", {}) or {}),
        "fit_per_band_intens_jointly": bool(results.get("fit_per_band_intens_jointly", True)),
        "loose_validity": bool(results.get("loose_validity", False)),
        "multiband_higher_harmonics": results.get(
            "multiband_higher_harmonics",
            "independent",
        ),
        "harmonic_orders": list(results.get("harmonic_orders", [3, 4])),
        "harmonics_shared": bool(results.get("harmonics_shared", False)),
    }

    # Optional lsb_auto_lock metadata (Stage-3 Stage-C).
    for opt_key in (
        "lsb_auto_lock",
        "lsb_auto_lock_sma",
        "lsb_auto_lock_count",
        "first_isophote_failure",
        "first_isophote_retry_log",
    ):
        if opt_key in results:
            tree[opt_key] = results[opt_key]

    # M13: preserve any remaining top-level keys natively
    # (forced_photometry_mode, template_n_isophotes, sky_offsets,
    # outer-reg references, ...).
    for opt_key, opt_val in results.items():
        if opt_key not in tree and opt_key not in ("isophotes", "config"):
            tree[opt_key] = opt_val

    af = asdf.AsdfFile(tree)
    af.write_to(str(filename))


def isophote_results_mb_from_asdf(filename: Union[str, Path]) -> dict:
    """
    Load multi-band isoster results from an ASDF file.

    Inverse of :func:`isophote_results_mb_to_asdf`. The returned dict has
    the same shape as the in-memory result dict produced by
    :func:`fit_image_multiband`, including per-band suffixed columns,
    optional lock-state metadata, and a reconstructed
    :class:`IsosterConfigMB` (or ``None`` if reconstruction fails for
    forward-compatibility).

    Parameters
    ----------
    filename
        Path to the ASDF file.

    Returns
    -------
    results
        Dict with ``isophotes``, ``config``, and the multi-band top-level
        keys (``bands``, ``reference_band``, ``harmonic_combination``,
        ``variance_mode``, ``band_weights``, ``loose_validity``,
        ``multiband_higher_harmonics``, ``harmonic_orders``,
        ``harmonics_shared``). Optional ``lsb_auto_lock*`` keys are
        present only when the original fit had them set.

    Raises
    ------
    ImportError
        If the ``asdf`` package is not installed.
    """
    try:
        import asdf
    except ImportError:
        raise ImportError(
            "The 'asdf' package is required for ASDF I/O. "
            "Install it with: pip install 'asdf>=3.0'  or  uv pip install 'asdf>=3.0'"
        )

    with asdf.open(str(filename)) as af:
        tree = af.tree
        raw_isophotes = list(tree.get("isophotes", []))
        config_dict = dict(tree.get("config", {}) or {})
        bands = list(tree.get("bands", []) or [])
        ref_band = tree.get("reference_band", "") or (bands[0] if bands else "")
        harm = tree.get("harmonic_combination", "joint") or "joint"
        variance_mode = tree.get("variance_mode", "ols") or "ols"
        band_weights_raw = dict(tree.get("band_weights", {}) or {})
        fit_jointly = bool(tree.get("fit_per_band_intens_jointly", True))
        loose_validity = bool(tree.get("loose_validity", False))
        higher_mode = tree.get("multiband_higher_harmonics", "independent")
        higher_orders = list(tree.get("harmonic_orders", [3, 4]))
        harmonics_shared = bool(tree.get("harmonics_shared", higher_mode != "independent"))
        # Pull any non-structural top-level keys (lock-state metadata,
        # first-isophote diagnostics, forced-mode/sky-offset bookkeeping;
        # M13 — previously only lsb_* and first_isophote_* survived).
        optional = {}
        _structural_tree_keys = {
            "format_version",
            "multiband",
            "isophotes",
            "config",
            "bands",
            "reference_band",
            "harmonic_combination",
            "variance_mode",
            "band_weights",
            "fit_per_band_intens_jointly",
            "loose_validity",
            "multiband_higher_harmonics",
            "harmonic_orders",
            "harmonics_shared",
        }
        for opt_key in tree:
            if opt_key not in _structural_tree_keys:
                optional[opt_key] = tree[opt_key]

    # Sanitize numpy scalar types in isophote rows for downstream consumers
    # (matches the FITS reader contract).
    isophotes: List[dict] = []
    for iso in raw_isophotes:
        clean = {k: _coerce_native(v) for k, v in iso.items()}
        isophotes.append(clean)

    # Coerce band-weight values to plain floats (asdf may surface np types).
    band_weights = {str(k): float(v) for k, v in band_weights_raw.items()}

    config: Optional[IsosterConfigMB] = None
    if config_dict:
        known_fields = set(IsosterConfigMB.model_fields.keys())
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        try:
            config = IsosterConfigMB(**filtered)
        except Exception as e:  # noqa: BLE001 — defensive
            logger.warning(
                "Could not reconstruct IsosterConfigMB from ASDF: %s",
                e,
            )

    # If config reconstruction succeeded, prefer its values for accuracy.
    if config is not None:
        bands = list(config.bands)
        ref_band = config.reference_band
        harm = config.harmonic_combination
        variance_mode = config.variance_mode if config.variance_mode is not None else variance_mode
        band_weights = config.resolved_band_weights()
        fit_jointly = bool(config.fit_per_band_intens_jointly)
        loose_validity = bool(config.loose_validity)
        higher_mode = config.multiband_higher_harmonics
        higher_orders = list(config.harmonic_orders)
        harmonics_shared = higher_mode != "independent"

    result: dict = {
        "isophotes": isophotes,
        "config": config,
        "bands": bands,
        "multiband": True,
        "harmonic_combination": harm,
        "reference_band": ref_band,
        "band_weights": band_weights,
        "variance_mode": variance_mode,
        "fit_per_band_intens_jointly": fit_jointly,
        "loose_validity": loose_validity,
        "multiband_higher_harmonics": higher_mode,
        "harmonic_orders": higher_orders,
        "harmonics_shared": harmonics_shared,
    }
    result.update(optional)
    return result
