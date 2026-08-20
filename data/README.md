# data/

Shared FITS datasets used by tests, benchmarks, and examples.

## Files

> **Provenance policy.** Every fact below is either read from the file's own FITS
> header or explicitly marked as *adopted* (a value the project assumes for
> analysis because the header does not record it). Where an earlier version of
> this file asserted a survey, instrument or image size that the header does not
> support, the header wins.

### IC3370_mock2.fits

- **Source**: Synthetic mock from Huang et al. (2013), generated with the
  external `mockgal.py` workflow using the `libprofit` engine.
- **Galaxy type**: Elliptical (mock Sérsic model); header `OBJECT = 'IC 3370'`
- **Image size**: 1133 × 1133 pixels (from the file; a previous version of this
  README said 256 × 256, which was wrong)
- **Photometric zeropoint**: 27.0 (header `MAGZERO`)
- **Pixel scale**: 0.168 arcsec/px — *adopted*, HSC coadd convention. The header
  carries no WCS scale keyword.
- **Notes**: Primary benchmark target. Used in `benchmarks/ic3370_exhausted/` (39-config sweep)
  and `benchmarks/performance/bench_vs_autoprof.py`.
- **Rights**: Synthetic data, no restrictions.

### eso243-49.fits

- **Source**: DESI Legacy Imaging Surveys (header `SURVEY = 'LegacySurvey'`)
- **Galaxy type**: Edge-on S0 galaxy
- **Image size**: 256 × 256 pixels, 3-band cube (header `BANDS = 'grz'`)
- **Pixel scale**: 0.25 arcsec/px (from `CD1_1 = -6.944e-05` deg/px)
- **Notes**: Used in EA harmonics comparison tests and AutoProf benchmark.
- **Rights**: Public survey data. The Legacy Surveys release their imaging into
  the public domain; confirm the current statement at the survey's data-access
  page before redistributing.

### ngc3610.fits

- **Source**: SDSS (header `SURVEY = 'SDSS'`). An earlier version of this README
  attributed it to the Legacy Survey; the header does not support that.
- **Galaxy type**: Boxy-bulge elliptical galaxy
- **Image size**: 256 × 256 pixels, 3-band cube (header `BANDS = 'gri'`)
- **Pixel scale**: 1.0 arcsec/px (from `CD1_1 = -2.778e-04` deg/px). Note this is
  coarser than SDSS's native 0.396 arcsec/px, so the cube has been resampled.
- **Notes**: Used in EA harmonics comparison tests and AutoProf benchmark.
  NGC 3610 exhibits strong a4/b4 boxiness signatures.
- **Rights**: Public survey data. SDSS imaging is publicly released; cite the
  appropriate SDSS data-release paper if used in publication.

### m51/M51.fits

- **Source**: Ground-based archival exposure written by the IRAF FITS kernel
  (header `ORIGIN = 'NOAO-IRAF FITS Image Kernel July 2003'`,
  `DATE-OBS = '05/04/87'`). An earlier version of this README described it as an
  HST/ACS mosaic at ~0.05 arcsec/px; the header does not support that, and the
  file is a single 512 × 512 frame rather than a mosaic.
- **Galaxy type**: Grand-design spiral galaxy (M51 / NGC 5194)
- **Image size**: 512 × 512 pixels
- **Band / exposure**: B band, 600 s (header `OBJECT = 'm51  B  600s'`)
- **Pixel scale**: not recorded in the header; none is assumed by the tests that
  use this file.
- **Notes**: Canonical basic real-data test dataset.
  Referenced by `tests/real_data/test_m51.py`.
- **Rights**: Archival data of uncertain provenance — the header carries no
  proposal or observatory rights statement. Treat as unverified before
  redistributing or publishing figures made from it.

## Usage

Load a file from any script or test using a path relative to the project root:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # adjust depth as needed
data_path = PROJECT_ROOT / "data" / "IC3370_mock2.fits"
```

Or from a test file two levels deep (`tests/real_data/`):

```python
DATA_DIR = Path(__file__).parent.parent.parent / "data"
```

## Notes

- These files are git-tracked (FITS are git-ignored by default; check `.gitignore`
  if a file fails to appear after clone).
- External Huang2013 data (full 20-galaxy set) lives at
  `/Users/mac/work/hsc/huang2013/<GALAXY>/` and is **not** tracked in this repo.
