"""Huang2013 scenario-grid adapter (depth x redshift mock matrix).

Expected on-disk layout (configurable via the campaign YAML ``root``)::

    <root>/
      <galaxy_name>/
        <galaxy_name>_noiseless_z005.fits
        <galaxy_name>_wide_z005.fits
        <galaxy_name>_wide_z020.fits
        <galaxy_name>_wide_z035.fits
        <galaxy_name>_wide_z050.fits
        <galaxy_name>_deep_z005.fits
        <galaxy_name>_deep_z020.fits
        <galaxy_name>_deep_z035.fits
        <galaxy_name>_deep_z050.fits

``galaxy_id`` is emitted as ``<galaxy>/<depth>_z<zzz>`` so
:func:`safe_galaxy_id` yields one flat output directory per scenario
(e.g. ``IC1459__wide_z020``). The generator does not ship every
depth x redshift combination (``clean`` and ``noiseless`` typically only
exist at ``z005``); the adapter walks the disk to enumerate what is actually
present.

The generator (``isophote_test/scripts/generate_mocks.py``, libprofit
engine) writes per-component truth keys ``ELLIP{i}``, ``PA{i}``,
``APPMAG{i}``, ``RE_PX{i}``, ``SERSIC{i}``, plus scalar header keys
``PIXSCALE``, ``MAGZERO``, ``PSFFWHM``, ``SKY_SBL``, ``REDSHIFT``,
``CFGNAME``. Those header-written photometric scales override the
adapter defaults when present so scenario setups with different
instruments still pick up the correct zeropoint and pixel scale.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from .base import DatasetAdapter, GalaxyBundle, GalaxyMetadata
from .huang2013 import Huang2013Adapter, _maybe_float

SCENARIO_DEPTHS = ("clean", "noiseless", "wide", "deep")
# Superset of redshift tags emitted by the mockgal generator across the
# Huang2013 and S4G-like campaigns. huang2013 uses {005, 020, 035, 050};
# newer S4G-like regens use {005, 010}. Each adapter's ``list_galaxies``
# walks the actual disk layout so missing combinations for a given
# dataset are silently absent, not errors.
SCENARIO_REDSHIFT_TAGS = ("005", "010", "020", "035", "050")

SCENARIO_SUFFIX_RE = re.compile(
    rf"^(?P<depth>{'|'.join(SCENARIO_DEPTHS)})"
    rf"_z(?P<z>{'|'.join(SCENARIO_REDSHIFT_TAGS)})$"
)


class Huang2013ScenariosAdapter(Huang2013Adapter):
    """Huang2013 mocks organized by a (depth x redshift) scenario grid.

    Inherits the multi-Sersic truth decoding from
    :class:`Huang2013Adapter` but overrides the on-disk file layout
    (one file per scenario, not one per mock index) and records the
    scenario axes in ``GalaxyMetadata.extra``.
    """

    dataset_name = "huang2013"

    def __init__(
        self,
        root: str | Path,
        pixel_scale_arcsec: float | None = None,
        sb_zeropoint: float | None = None,
        depths: Iterable[str] | None = None,
        redshift_tags: Iterable[str] | None = None,
    ) -> None:
        # Parent defaults match the HSC coadd convention (0.168", 27.0).
        # Accept None here so the campaign YAML can omit both keys.
        kwargs: dict[str, float] = {}
        if pixel_scale_arcsec is not None:
            kwargs["pixel_scale_arcsec"] = pixel_scale_arcsec
        if sb_zeropoint is not None:
            kwargs["sb_zeropoint"] = sb_zeropoint
        super().__init__(root=root, **kwargs)

        self.depths: tuple[str, ...] | None = (
            tuple(depths) if depths is not None else None
        )
        self.redshift_tags: tuple[str, ...] | None = (
            tuple(redshift_tags) if redshift_tags is not None else None
        )
        self._validate_filters()

    def _validate_filters(self) -> None:
        if self.depths is not None:
            bad = [d for d in self.depths if d not in SCENARIO_DEPTHS]
            if bad:
                raise ValueError(
                    f"{type(self).__name__}: unknown depth tag(s) {bad!r}; "
                    f"valid = {SCENARIO_DEPTHS}"
                )
        if self.redshift_tags is not None:
            bad = [z for z in self.redshift_tags if z not in SCENARIO_REDSHIFT_TAGS]
            if bad:
                raise ValueError(
                    f"{type(self).__name__}: unknown redshift tag(s) {bad!r}; "
                    f"valid = {SCENARIO_REDSHIFT_TAGS}"
                )

    def list_galaxies(self) -> list[str]:
        """Return ``<galaxy>/<depth>_z<zzz>`` identifiers found on disk."""
        galaxy_ids: list[str] = []
        for galaxy_dir in sorted(self.root.iterdir()):
            if not galaxy_dir.is_dir() or galaxy_dir.name.startswith("."):
                continue
            galaxy_name = galaxy_dir.name
            prefix_len = len(galaxy_name) + 1  # "<galaxy>_"
            for fits_path in sorted(galaxy_dir.glob(f"{galaxy_name}_*.fits")):
                suffix = fits_path.stem[prefix_len:]
                match = SCENARIO_SUFFIX_RE.match(suffix)
                if match is None:
                    continue
                if self.depths is not None and match["depth"] not in self.depths:
                    continue
                if (
                    self.redshift_tags is not None
                    and match["z"] not in self.redshift_tags
                ):
                    continue
                galaxy_ids.append(f"{galaxy_name}/{suffix}")
        return galaxy_ids

    def load_galaxy(self, galaxy_id: str) -> GalaxyBundle:
        fits_path = self._resolve_fits_path(galaxy_id)
        with fits.open(fits_path) as hdul:
            image = np.asarray(hdul[0].data, dtype=np.float64)
            header = hdul[0].header.copy()
        if image.ndim != 2:
            raise ValueError(
                f"{fits_path}: expected 2D image, got shape {image.shape}"
            )

        initial_geometry = self._infer_initial_geometry(header, image.shape)
        effective_Re = self._read_effective_Re_pix(header)

        galaxy_name, scenario = galaxy_id.split("/", 1)
        match = SCENARIO_SUFFIX_RE.match(scenario)
        depth = match["depth"] if match else None
        redshift_tag = match["z"] if match else None

        pixel_scale = _maybe_float(header, "PIXSCALE") or self.pixel_scale_arcsec
        sb_zeropoint = _maybe_float(header, "MAGZERO")
        if sb_zeropoint is None:
            sb_zeropoint = self.sb_zeropoint

        metadata = GalaxyMetadata(
            galaxy_id=galaxy_id,
            dataset=self.dataset_name,
            pixel_scale_arcsec=float(pixel_scale),
            sb_zeropoint=float(sb_zeropoint),
            effective_Re_pix=effective_Re,
            redshift=_maybe_float(header, "REDSHIFT"),
            extra={
                "galaxy_name": galaxy_name,
                "scenario": scenario,
                "depth": depth,
                "redshift_tag": redshift_tag,
                "ncomp": int(header.get("NCOMP", 0)),
                "psf_fwhm_arcsec": _maybe_float(header, "PSFFWHM"),
                "sky_sb_limit": _maybe_float(header, "SKY_SBL"),
                "fits_path": str(fits_path),
                "truth_components": _extract_truth_components(header),
            },
        )
        return GalaxyBundle(
            metadata=metadata,
            image=image,
            variance=None,
            mask=None,
            initial_geometry=initial_geometry,
            truth_profile=None,
        )

    def _resolve_fits_path(self, galaxy_id: str) -> Path:
        try:
            galaxy_name, scenario = galaxy_id.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"Scenario galaxy_id must be '<galaxy>/<depth>_z<zzz>', "
                f"got '{galaxy_id}'"
            ) from exc
        if SCENARIO_SUFFIX_RE.match(scenario) is None:
            raise ValueError(
                f"Scenario token must match '<depth>_z<zzz>', got '{scenario}'"
            )
        fits_path = self.root / galaxy_name / f"{galaxy_name}_{scenario}.fits"
        if not fits_path.is_file():
            raise FileNotFoundError(f"Scenario mock FITS not found: {fits_path}")
        return fits_path


def _extract_truth_components(header: fits.Header) -> list[dict[str, float]]:
    """Pull per-component (ELLIP, PA, RE_PX, APPMAG, SERSIC, RE_KPC) into plain dicts."""
    ncomp = int(header.get("NCOMP", 0))
    components: list[dict[str, float]] = []
    for i in range(1, ncomp + 1):
        entry: dict[str, float] = {"index": float(i)}
        for key in ("ELLIP", "PA", "RE_PX", "APPMAG", "SERSIC", "RE_KPC"):
            value = _maybe_float(header, f"{key}{i}")
            if value is not None:
                entry[key.lower()] = value
        components.append(entry)
    return components


ADAPTER_CLASS: type[DatasetAdapter] = Huang2013ScenariosAdapter
