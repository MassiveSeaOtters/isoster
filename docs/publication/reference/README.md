# Reference papers for the isoster publication

Local copies of the two most directly relevant existing tools for
non-parametric 1-D isophotal analysis of galaxy images. These papers
set the scientific context for the isoster method paper: what the
state of the art looks like, what its strengths and limitations are,
and where isoster contributes something new.

The PDFs and their converted text are **untracked on purpose**: they are
third-party published papers, not ours to redistribute, and together they are
about 19 MB. `.gitignore` excludes `reference/*.pdf` and `reference/*.txt`
while tracking this README, so the repository records *what* we consulted
without carrying the papers themselves. Fetch them from the publishers or
arXiv if you need them locally.

## Files in this folder

For each paper we keep three artifacts:

- `*.pdf`     — original arXiv PDF (source of truth)
- `*.txt`     — reading-order plain text via `pdftotext`
                (best for sequential reading and grep)

To regenerate the text:

```bash
cd docs/publication/reference
pdftotext isofit_ciambur_2015.pdf isofit_ciambur_2015.txt
pdftotext autoprof_stone_2021.pdf autoprof_stone_2021.txt
```

## Catalogue

### 1. Ciambur (2015) — Isofit / Cmodel

- **Short handle**: `isofit_ciambur_2015`
- **Full title**: *Beyond Ellipse(s): Accurately Modelling the Isophotal
  Structure of Galaxies with Isofit and Cmodel*
- **Author**: B. C. Ciambur (Swinburne)
- **Journal**: ApJ, 810, 120 (2015)
- **DOI**: `10.1088/0004-637X/810/2/120`
- **arXiv**: `1507.02691` (v1, 9 Jul 2015)
- **Publisher URL**: https://iopscience.iop.org/article/10.1088/0004-637X/810/2/120
- **arXiv URL**: https://arxiv.org/abs/1507.02691
- **PDF URL**: https://arxiv.org/pdf/1507.02691

**Core idea**: Replace the polar angle `theta` (the natural variable in
`Ellipse` / `photutils.isophote`) with the **eccentric anomaly** when
sampling quasi-elliptical isophotes. The eccentric-anomaly parameterization
gives a much better description of deviations from ellipticity for
edge-on disks and X-shaped / peanut bulges, eliminates the cross-shaped
residual artefacts seen in conventional `Ellipse` model subtractions,
and makes high-order Fourier moments (`n > 4`) physically meaningful.

**Relevance to isoster**:

- Direct prior art for the Fourier-harmonic description of isophote shape
  (third- and fourth-order `A_n`, `B_n`); same physical motivation
  (boxy vs disky, X-shaped bulges).
- Identifies the failure mode (cross-shaped residuals) that motivates
  alternative parameterizations.
- Implemented as IRAF tasks (legacy); not a modern Python pipeline.

### 2. Stone, Arora, Courteau & Cuillandre (2021) — AutoProf I

- **Short handle**: `autoprof_stone_2021`
- **Full title**: *AutoProf — I. An automated non-parametric light
  profile pipeline for modern galaxy surveys*
- **Authors**: Connor J. Stone, Nikhil Arora, Stéphane Courteau,
  Jean-Charles Cuillandre
- **Journal**: MNRAS, 508, 1870 (2021)
- **DOI**: `10.1093/mnras/stab2709`
- **arXiv**: `2106.13809` (v2, 7 Oct 2021)
- **Publisher URL**: https://academic.oup.com/mnras/article/508/2/1870/6373481
- **arXiv URL**: https://arxiv.org/abs/2106.13809
- **PDF URL**: https://arxiv.org/pdf/2106.13809
- **Code**: https://github.com/ConnorStoneAstro/AutoProf

**Core idea**: A modern, fully automated Python pipeline for
non-parametric elliptical-isophote fitting on survey-scale imaging.
Improves on previous non-parametric ellipse fitters with
fit-stabilization procedures borrowed from machine-learning
(regularization, smoothing across radii), explicit support for
alternative slice-based profiles, smooth axisymmetric model
reconstruction, and decision-tree pipelining. Quantitative
comparison against `photutils`, `XVISTA`, and `GALFIT` shows
AutoProf reaches surface brightnesses typically >2 mag arcsec⁻²
fainter than the comparison codes on the same images.

**Relevance to isoster**:

- Most direct comparable in scope: modern Python non-parametric
  pipeline for survey imaging. Sets the LSB-depth bar isoster
  must meet or exceed.
- Independent confirmation that elliptical-isophote analysis (not
  multi-component parametric fits) is what is actually needed for
  late-type galaxies — supports isoster's design philosophy.
- AutoProf is the most natural side-by-side benchmark for the
  exhausted-campaign framework, alongside `photutils.isophote`.

## How these are used in the paper

- Section: **Method comparison** — isoster vs `photutils.isophote`
  vs AutoProf vs Isofit. Tabulate algorithmic differences (sampling
  scheme, harmonic basis, regularization, LSB handling, multi-band
  support).
- Section: **Benchmarks** — the exhausted-benchmark campaign already
  compares isoster, photutils, autoprof on mock galaxies; results
  feed directly into the paper.
- Section: **Novel contributions** — vectorized path-based sampling,
  outer-region Tikhonov regularization, multi-band joint free fit,
  exhausted-campaign QA scoring framework.
