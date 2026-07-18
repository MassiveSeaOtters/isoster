import json
import logging

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Shared helpers (re-imported here for backward compatibility — historical
# call sites continue to use ``isoster.utils._NumpyEncoder``,
# ``._config_to_dict``, ``._build_config_hdu``). Single source of truth
# lives in ``isoster._shared``.
from ._shared import (  # noqa: F401
    _build_config_hdu,
    _build_meta_hdu,
    _config_to_dict,
    _NumpyEncoder,
    _parse_meta_hdu,
)

logger = logging.getLogger(__name__)


def _parse_config_hdu(hdu):
    """
    Reconstruct an IsosterConfig from a CONFIG BinTableHDU.

    Returns the config object, or None if reconstruction fails
    (e.g. unknown fields from a newer version of isoster).
    """
    from .config import IsosterConfig

    config_tbl = Table.read(hdu)
    if len(config_tbl) == 0:
        return None

    parsed = {}
    for row in config_tbl:
        key = str(row["PARAM"])
        try:
            parsed[key] = json.loads(row["VALUE"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping unparseable config key '%s'", key)
            continue

    # Filter to known fields only (forward compatibility)
    known_fields = set(IsosterConfig.model_fields.keys())
    filtered = {k: v for k, v in parsed.items() if k in known_fields}

    try:
        return IsosterConfig(**filtered)
    except Exception:
        logger.warning("Could not reconstruct IsosterConfig from FITS; returning None")
        return None


def isophote_results_to_astropy_tables(results):
    """
    Convert isophote results to an Astropy Table.

    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image(), containing 'isophotes' and 'config'.

    Returns
    -------
    table : astropy.table.Table
        Astropy Table containing the isophote data with standard columns:
        - sma: Semi-major axis length
        - intens: Mean intensity
        - intens_err: Intensity error
        - eps: Ellipticity
        - pa: Position angle (radians)
        - x0, y0: Center coordinates
        - stop_code: Fitting status code
        - niter: Number of iterations
    """
    # Handle case where results is just the list (backward compatibility or direct list usage)
    if isinstance(results, list):
        isophotes = results
    else:
        isophotes = results.get("isophotes", [])

    if not isophotes:
        return Table()

    # Create table from list of dicts
    tbl = Table(rows=isophotes)

    # Reorder columns for better readability
    common_cols = ["sma", "intens", "intens_err", "eps", "pa", "x0", "y0", "rms", "stop_code", "niter"]
    all_cols = tbl.colnames
    new_order = [c for c in common_cols if c in all_cols] + [c for c in all_cols if c not in common_cols]
    tbl = tbl[new_order]

    return tbl


def isophote_results_to_fits(results, filename, overwrite=True):
    """
    Save isophote results to a multi-HDU FITS file.

    The output layout is:
    - HDU 0: PrimaryHDU (empty)
    - HDU 1: BinTableHDU 'ISOPHOTES' (isophote data columns)
    - HDU 2: BinTableHDU 'CONFIG' (two columns: PARAM, VALUE with JSON-serialized config)
    - HDU 3: BinTableHDU 'META' (JSON PARAM/VALUE rows for the remaining
      top-level result keys: lsb_auto_lock*, first_isophote_*, outer-reg
      references, ...)

    Config is stored in a dedicated table extension rather than as header
    keywords, avoiding HIERARCH promotion warnings for long parameter names.

    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image().
    filename : str
        Output filename.
    overwrite : bool
        Whether to overwrite existing file.
    """
    tbl = isophote_results_to_astropy_tables(results)

    isophote_hdu = fits.table_to_hdu(tbl)
    isophote_hdu.name = "ISOPHOTES"

    config_hdu = _build_config_hdu(results)

    # Non-structural top-level keys (lsb_auto_lock*, first_isophote_*,
    # outer-reg references) ride in a META HDU so the round-trip preserves
    # them (same pattern as the multi-band writer, M13/Q2).
    meta = {k: v for k, v in results.items() if k not in ("isophotes", "config")}
    meta_hdu = _build_meta_hdu(meta)

    hdulist = fits.HDUList([fits.PrimaryHDU(), isophote_hdu, config_hdu, meta_hdu])
    hdulist.writeto(filename, overwrite=overwrite)


def isophote_results_from_fits(filename):
    """
    Load isophote results from a FITS table file.

    This is the inverse of isophote_results_to_fits(). It reconstructs
    the results dict from a saved FITS file, enabling template-based
    forced photometry workflows where geometry from one band is applied
    to other bands.

    Supports both the new 3-HDU layout (with CONFIG extension) and legacy
    files where config was stored as header keywords (returns config=None
    for legacy files).

    Parameters
    ----------
    filename : str
        Path to FITS file containing isophote table.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'isophotes': list of dicts, each containing isophote data
        - 'config': IsosterConfig if CONFIG HDU is present, else None

    Examples
    --------
    >>> template = isophote_results_from_fits('galaxy_gband.fits')
    >>> results_r = fit_image(image_r, mask_r, config,
    ...                       template=template)
    """
    with fits.open(filename) as hdulist:
        # Read isophotes: try named HDU first, fall back to index 1
        try:
            iso_tbl = Table.read(hdulist["ISOPHOTES"])
        except KeyError:
            iso_tbl = Table.read(hdulist[1])

        # Read config from CONFIG HDU (new format); None for legacy files
        config = None
        if "CONFIG" in hdulist:
            config = _parse_config_hdu(hdulist["CONFIG"])

        # Non-structural top-level keys (lsb_auto_lock*, first_isophote_*,
        # outer-reg references); absent in files written before the META
        # HDU existed.
        meta = {}
        if "META" in hdulist:
            meta = _parse_meta_hdu(hdulist["META"])

    # Convert table rows to list of dicts
    isophotes = []
    for row in iso_tbl:
        iso_dict = {}
        for colname in iso_tbl.colnames:
            value = row[colname]
            # Heterogeneous row keys produce masked cells; skip them so the
            # loaded dict keeps the original "key absent" shape (review B3).
            # CAUTION: this cannot protect bool columns — FITS logical columns
            # have no null value, so a conditionally-present bool key fills as
            # True on write; bool keys must be written on every row (like the
            # lsb lock flags).
            if value is np.ma.masked:
                continue
            # Convert numpy types to Python types for consistency
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            iso_dict[colname] = value
        isophotes.append(iso_dict)

    result = {"isophotes": isophotes, "config": config}
    result.update(meta)
    return result


# ---------------------------------------------------------------------------
# ASDF support (optional dependency)
# ---------------------------------------------------------------------------


def isophote_results_to_asdf(results, filename):
    """
    Save isophote results to an ASDF file.

    ASDF natively supports nested dicts, lists, and numpy arrays, so
    config parameters are stored without any serialization workarounds.

    Parameters
    ----------
    results : dict
        The dictionary returned by fit_image().
    filename : str
        Output filename (conventionally ending in .asdf).

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

    if isinstance(results, list):
        isophotes = results
        config_dict = {}
    else:
        isophotes = results.get("isophotes", [])
        config_dict = _config_to_dict(results.get("config", None))

    tree = {
        "format_version": 1,
        "isophotes": isophotes,
        "config": config_dict,
    }
    # Preserve non-structural top-level keys (lsb_auto_lock*,
    # first_isophote_*, outer-reg references) so the round-trip is
    # lossless (Q2).
    if isinstance(results, dict):
        for key, value in results.items():
            if key not in ("isophotes", "config") and key not in tree:
                tree[key] = value

    af = asdf.AsdfFile(tree)
    af.write_to(filename)


def isophote_results_from_asdf(filename):
    """
    Load isophote results from an ASDF file.

    Parameters
    ----------
    filename : str
        Path to ASDF file.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'isophotes': list of dicts
        - 'config': IsosterConfig if config data is present, else None

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

    from .config import IsosterConfig

    with asdf.open(filename) as af:
        isophotes = list(af.tree.get("isophotes", []))
        config_dict = dict(af.tree.get("config", {}))
        # Non-structural top-level keys (lsb_auto_lock*, first_isophote_*,
        # outer-reg references) written by isophote_results_to_asdf (Q2).
        extra = {k: v for k, v in af.tree.items() if k not in ("format_version", "isophotes", "config")}

    # Convert isophote values from numpy to native Python types
    clean_isophotes = []
    for iso in isophotes:
        clean = {}
        for k, v in iso.items():
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            clean[k] = v
        clean_isophotes.append(clean)

    config = None
    if config_dict:
        known_fields = set(IsosterConfig.model_fields.keys())
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        try:
            config = IsosterConfig(**filtered)
        except Exception:
            logger.warning("Could not reconstruct IsosterConfig from ASDF; returning None")

    result = {"isophotes": clean_isophotes, "config": config}
    result.update(extra)
    return result
