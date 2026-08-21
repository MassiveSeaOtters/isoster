# Historical galaxy-isophote PDF review

Date: 2026-08-11  
Source folder: `~/Desktop/isophote_contour`  
Scope: 29 PDF files; `microdensitometer/` intentionally excluded

## Purpose and method

This is a literature-selection document, not an introduction outline. Its
purpose is to establish what each supplied paper contributed to the history
of galaxy isophotal analysis and whether ISOSTER should cite it.

Each PDF was matched to ADS where possible. For old image-only scans, the
first pages and, where needed, conclusions were rendered with Ghostscript and
read with targeted Tesseract OCR. ADS abstracts were preferred when present;
OCR was used for papers with no ADS abstract or when the methodological detail
was not captured by the abstract. The citation tiers mean:

- **CORE**: directly establishes the method lineage or a modern comparator;
  the ISOSTER paper should cite it somewhere.
- **SUPPORTING**: useful for a particular historical, scientific, or technical
  point; cite only where that point is made.
- **OMIT**: not needed for the main ISOSTER argument, although the paper may be
  scientifically interesting.

The microdensitometer prehistory remains important even though that subfolder
was excluded. Photographic plates were converted into numerical density maps
with scanning microdensitometers; isophotes could then be measured as contours
and fitted geometrically. Fraser (1967) is the clearest bridge in the reviewed
set between that instrumental origin and later numerical isophote analysis.

## Paper-by-paper triage

### 1967-1979: from photographic isophotometry to Fourier shape coefficients

#### `fraser1967.pdf`

- **Identity:** C. W. Fraser, “Isophotometry of Galaxies,” *The Observatory*,
  87, 29-31 (1967), ADS `1967Obs....87...29F`.
- **Evidence:** Image-only scan; no ADS abstract. Targeted OCR was clear.
- **Contribution:** A methodological note on scanning isodensitometers and
  whether their stepping distorts small-scale structure. It describes a Joyce
  Loebl microdensitometer with a Tech/Ops isodensitometer attachment used for a
  systematic bright-galaxy program.
- **Historical role:** Documents the instrument-to-isophote transition before
  general digital image analysis.
- **Tier:** **SUPPORTING** - cite if the paper explicitly mentions the
  microdensitometer origin of isophotal analysis.

#### `barbon1975a.pdf`

- **Identity:** R. Barbon, M. Capaccioli & M. Tarenghi, “Photographic
  Photometry of Bright Galaxies II: NGC 3384,” *A&A*, 38, 315-321 (1975),
  ADS `1975A&A....38..315B`.
- **Evidence:** Image-only scan; ADS abstract and targeted OCR agree.
- **Contribution:** B-band photographic photometry to about 26.6 mag
  arcsec^-2, separating nucleus, lens, and exponential disk.
- **Historical role:** Early component interpretation from surface-brightness
  profiles, but a single-galaxy study.
- **Tier:** **SUPPORTING** - useful for early lens/disk decomposition.

#### `barbon1975b.pdf`

- **Identity:** R. Barbon & M. Capaccioli, “Photographic Photometry of Bright
  Galaxies III: NGC 1023,” *A&A*, 42, 103-108 (1975),
  ADS `1975A&A....42..103B`.
- **Evidence:** Image-only scan; ADS abstract and targeted OCR agree.
- **Contribution:** Photographic surface photometry and morphological analysis
  of an early lenticular galaxy and a likely dwarf companion.
- **Historical role:** Another early profile-based morphology case study, but
  no major new method relative to the surrounding Barbon series.
- **Tier:** **OMIT** - redundant in a concise method history.

#### `barbon1976.pdf`

- **Identity:** R. Barbon, L. Benacchio & M. Capaccioli, “Photometric Study by
  a Numerical Mapping Technique of a Trio of Galaxies in Leo,” *A&A*, 51,
  25-29 (1976), ADS `1976A&A....51...25B`.
- **Evidence:** Image-only scan; ADS abstract and targeted OCR agree.
- **Contribution:** Uses numerical two-dimensional mapping to derive faint
  isophotes together with ellipticity, position-angle, and centre profiles.
- **Historical role:** A useful transition from one-dimensional photographic
  profiles to numerical two-dimensional isophote geometry.
- **Tier:** **SUPPORTING** - one of the best citations for the numerical-mapping
  transition before Carter and Jedrzejewski.

#### `kormendy1977.pdf`

- **Identity:** J. Kormendy, “Brightness Distributions in Compact and Normal
  Galaxies I: Surface Photometry of Red Compact Galaxies,” *ApJ*, 214,
  359-382 (1977), ADS `1977ApJ...214..359K`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Major- and minor-axis profiles for compact and normal
  galaxies; treats lenses as structural components distinct from spheroids and
  disks.
- **Historical role:** Strong early example of physical component inference
  from one-dimensional profiles, not an isophote-fitting algorithm paper.
- **Tier:** **SUPPORTING** - cite for structural interpretation, not method
  origin.

#### `fraser1977.pdf`

- **Identity:** C. W. Fraser, “Photographic Surface Photometry of Galaxies in
  the Virgo Cluster,” *A&AS*, 29, 161-194 (1977),
  ADS `1977A&AS...29..161F`.
- **Evidence:** Scan with usable text; ADS abstract available.
- **Contribution:** Homogeneous photographic parameters for 48 Virgo galaxies,
  including brightness, size, concentration, and disk trends with morphology.
- **Historical role:** Demonstrates scale-up and standardization of
  surface-photometry measurements.
- **Tier:** **SUPPORTING** - useful for historical survey context.

#### `carter1978.pdf`

- **Identity:** D. Carter, “The Structure of the Isophotes of Elliptical
  Galaxies,” *MNRAS*, 182, 797-799 (1978),
  ADS `1978MNRAS.182..797C`.
- **Evidence:** Image-only scan and no ADS abstract; the paper contains a clear
  extended summary, inspected directly.
- **Contribution:** Fits five-parameter ellipses to measured isophotes,
  rectifies each ellipse to a circle, and Fourier-expands radial deviations.
  It identifies the fourth-order cosine term as zero for an ellipse, negative
  for a boxy contour, and positive for a disky contour.
- **Historical role:** The direct astronomical foundation of quantitative
  higher-order isophote-shape analysis.
- **Tier:** **CORE** - essential; it predates and directly motivates later
  boxy/disky and harmonic work.

#### `williams1979.pdf`

- **Identity:** T. B. Williams & M. Schwarzschild, “A Photometric
  Determination of Twists in Three Early-Type Galaxies,” *ApJ*, 227, 56-63
  (1979), ADS `1979ApJ...227...56W`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Measures radial position-angle twists and connects them to
  the possibility of triaxial intrinsic shapes.
- **Historical role:** Establishes isophote twist as a physical diagnostic
  alongside ellipticity and higher-order shape.
- **Tier:** **SUPPORTING** - a strong scientific-use citation for PA profiles.

#### `young1979.pdf`

- **Identity:** P. J. Young et al., “CCD Photometry of the Nuclei of Three
  Supergiant Elliptical Galaxies,” *ApJ*, 234, 76-85 (1979),
  ADS `1979ApJ...234...76Y`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** CCD nuclear profiles and dynamical model fitting, including
  evidence for a central mass in NGC 6251.
- **Historical role:** Illustrates the early CCD transition, but not the
  contour-fitting lineage central to ISOSTER.
- **Tier:** **OMIT** - unless the manuscript discusses nuclear photometry.

### 1982-1988: CCDs, automation, and the canonical ellipse/Fourier workflow

#### `barbon1982.pdf`

- **Identity:** R. Barbon, M. Capaccioli & R. Rampazzo, “Photographic
  Photometry of Galaxies Using the INMP,” *A&A*, 115, 388 (1982),
  ADS `1982A&A...115..388B`.
- **Evidence:** Image-only scan; ADS abstract and targeted OCR agree.
- **Contribution:** Uses PDS scans and the Interactive Numerical Mapping
  Package to measure S0 galaxies and resolve bulge, lens, and disk components.
- **Historical role:** Matures the numerical-mapping line represented by
  Barbon et al. (1976), but remains photographic.
- **Tier:** **SUPPORTING** - cite for INMP or early interactive mapping.

#### `kuhl_giardina1982_elliptical_fourier_features.pdf`

- **Identity:** F. P. Kuhl & C. R. Giardina, “Elliptic Fourier Features of a
  Closed Contour,” *Computer Graphics and Image Processing*, 18, 236-258
  (1982). This paper is not reliably indexed in ADS.
- **Evidence:** Image-only scan; title page and abstract visually inspected.
- **Contribution:** Develops normalized elliptic Fourier descriptors for
  chain-coded closed contours, with translation, rotation, scale, and
  starting-point invariance.
- **Historical role:** A generic computer-vision contour method, not the source
  of the astronomical intensity-harmonic or Bender-normalized conventions.
- **Tier:** **OMIT** - do not imply that this paper is in the direct galaxy
  isophote lineage.

#### `kent1983.pdf`

- **Identity:** S. M. Kent, “CCD Photometry of the Center of M31,” *ApJ*, 266,
  562-567 (1983), ADS `1983ApJ...266..562K`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** CCD surface-brightness, PA, and ellipticity profiles for
  the M31 nucleus and bulge, followed by dynamical modelling.
- **Historical role:** Demonstrates what CCD dynamic range and resolution made
  possible, but does not define a general fitting algorithm.
- **Tier:** **OMIT** - unnecessary for a concise method history.

#### `lauer1985.pdf`

- **Identity:** T. R. Lauer, “High-Resolution Surface Photometry of Elliptical
  Galaxies,” *ApJS*, 57, 473-502 (1985), ADS `1985ApJS...57..473L`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** High-resolution CCD profiles for 42 E/S0 galaxies and a
  model-independent hybrid Fourier deconvolution for seeing correction.
- **Historical role:** Important for central-profile fidelity and PSF effects,
  but not primarily a higher-order isophote-shape paper.
- **Tier:** **SUPPORTING** - cite for seeing/central-resolution limitations.
- **Important distinction:** This is **not** Lauer’s separate 1985 MNRAS paper,
  “Boxy Isophotes, Discs and Dust Lanes in Elliptical Galaxies,”
  ADS `1985MNRAS.216..429L`.

#### `michard1985.pdf`

- **Identity:** R. Michard, “Detailed Surface Photometry of 36 E-S0
  Galaxies,” *A&AS*, 59, 205-228 (1985),
  ADS `1985A&AS...59..205M`.
- **Evidence:** Image-only scan; ADS abstract available.
- **Contribution:** Deep tabulated ellipse geometry and brightness profiles,
  documenting systematic radial changes and departures from simple
  interpolation laws.
- **Historical role:** Strong empirical basis for interpreting ellipticity and
  PA profiles across E/S0 populations.
- **Tier:** **SUPPORTING** - observational foundation, not the canonical
  algorithm.

#### `bender1987.pdf`

- **Identity:** R. Bender & C. Möllenhoff, “Morphological Analysis of Massive
  Early-Type Galaxies in the Virgo Cluster,” *A&A*, 177, 71-83 (1987),
  ADS `1987A&A...177...71B`.
- **Evidence:** Image-only scan; ADS abstract and targeted OCR agree.
- **Contribution:** Fits ellipses to CCD isophotes, analyzes deviations with
  Fourier terms, and connects boxy/disky signatures to disks, dust, and
  interactions.
- **Historical role:** A major early physical interpretation and
  popularization of fourth-order isophote deviations.
- **Tier:** **CORE** - pair with Carter (1978) when discussing the physical
  meaning of boxy/disky terms.

#### `cawson1987a.pdf`

- **Identity:** M. G. M. Cawson et al., “Automated Galaxy Surface Photometry I:
  Technique, Calibration and Validation,” *MNRAS*, 224, 557-566 (1987),
  ADS `1987MNRAS.224..557C`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Automated two-dimensional photometry of large numbers of
  galaxies from Schmidt plates with APM-to-CCD calibration and validation.
- **Historical role:** An early throughput and automation milestone, separate
  from detailed iterative ellipse fitting.
- **Tier:** **SUPPORTING** - useful if the paper traces survey-scale automation.

#### `jedrzejewski1987.pdf`

- **Identity:** R. I. Jedrzejewski, “CCD Surface Photometry of Elliptical
  Galaxies I: Observations, Reduction and Results,” *MNRAS*, 226, 747-768
  (1987), ADS `1987MNRAS.226..747J`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Formalizes the practical iterative method in which
  low-order intensity harmonics along a trial ellipse correct centre,
  ellipticity, and PA, while higher-order residuals describe non-elliptical
  shape.
- **Historical role:** The canonical algorithm behind IRAF `Ellipse`,
  `photutils.isophote`, and ISOSTER’s classical path.
- **Tier:** **CORE** - indispensable.

#### `mizuno1987.pdf`

- **Identity:** T. Mizuno & K. Hamajima, “Twist and Axis Ratio of Isophotes in
  the Central Region of Disk Galaxies,” *PASJ*, 39, 221-235 (1987),
  ADS `1987PASJ...39..221M`.
- **Evidence:** Image-only scan; ADS abstract available.
- **Contribution:** Measures coupled PA and axis-ratio changes in 19 early-type
  disks and classifies their central geometry.
- **Historical role:** A scientific use of radial ellipse geometry, not a new
  fitting framework.
- **Tier:** **SUPPORTING** - cite for PA/ellipticity applications.

#### `michard_simien1988.pdf`

- **Identity:** R. Michard & F. Simien, “Isophotal Contours of Early-Type
  Galaxies I: The Data,” *A&AS*, 74, 25-51 (1988),
  ADS `1988A&AS...74...25M`.
- **Evidence:** Image-only scan; ADS abstract and targeted OCR agree.
- **Contribution:** Applies a modified Fourier shape analysis to 29 E/S0
  galaxies and publishes smoothed two-dimensional brightness distributions
  suitable for model fitting.
- **Historical role:** An important parallel implementation and consolidation
  of the late-1980s Fourier-isophote approach.
- **Tier:** **SUPPORTING** - strong historical support, especially if multiple
  coefficient conventions or implementations are discussed.

### 1989-2021: comparison, standardization, geometric refinement, and pipelines

#### `prugniel1989.pdf`

- **Identity:** P. Prugniel, “Surface Photometry of Elliptical Galaxies in
  Pairs,” ESO workshop proceedings (1989), ADS `1989ESOC...31..161P`.
- **Evidence:** Image-only scan; ADS abstract available.
- **Contribution:** Reviews classical image models, presents a moment-analysis
  alternative, compares methods on a real galaxy, and applies the alternative
  to paired ellipticals.
- **Historical role:** Evidence that alternative image representations were
  already being compared alongside ellipse fitting.
- **Tier:** **SUPPORTING** - useful for method-comparison history.

#### `michard_simien1990.pdf`

- **Identity:** F. Simien & R. Michard, “Isophotal Contours of Early-Type
  Galaxies II: Axisymmetric-Bulge and Thin-Disk Approximations,” *A&A*, 227,
  11 (1990), ADS `1990A&A...227...11S`.
- **Evidence:** Image-only scan; ADS abstract available.
- **Contribution:** Interprets Paper I’s isophote-shape parameters with a thin
  inclined disk plus a variable-flattening bulge.
- **Historical role:** Connects measured non-elliptical contours to explicit
  bulge/disk structure.
- **Tier:** **SUPPORTING** - cite for physical interpretation, not algorithm
  origin.

#### `stiavelli1991.pdf`

- **Identity:** M. Stiavelli, P. Prugniel & W. W. Zeilinger, “Comparison
  Between Different Isophote-Fitting Programs,” ESO proceedings (1991),
  ADS `1991ESOC...38..231S`.
- **Evidence:** Image-only scan and no ADS abstract; abstract and conclusions
  were recovered with targeted OCR.
- **Contribution:** Compares PLEINPOT/BOXI, a modified INMP/CONFIT, and
  Bender’s ELLFIT on controlled images. Geometry agrees outside the inner two
  pixels and at high S/N, but fine structure in the fourth-order coefficient
  varies by about 0.005 with centering and implementation.
- **Historical role:** A direct early precedent for cross-code validation and
  for separating stable global trends from implementation-sensitive details.
- **Tier:** **SUPPORTING** - particularly valuable in ISOSTER’s benchmark or
  comparison section.

#### `morse1998_isophote_based_interpolation.pdf`

- **Identity:** B. S. Morse & D. Schwartzwald, “Isophote-Based Interpolation,”
  IEEE ICIP (1998), ADS `1998icip.conf..142M`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Computer-vision interpolation by evolving image level-set
  contours.
- **Historical role:** Shares the word “isophote,” but it is not galaxy
  surface photometry or astronomical ellipse fitting.
- **Tier:** **OMIT**.

#### `milvang-jensen1999.pdf`

- **Identity:** B. Milvang-Jensen & I. Jørgensen, “Galaxy Surface Photometry,”
  *Baltic Astronomy*, 8, 535-574 (1999),
  ADS `1999BaltA...8..535M`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** A pedagogical synthesis of ellipse fitting, Fourier
  deviations, relationships between coefficient definitions, disk
  luminosities/inclinations, colours, and the Fundamental Plane.
- **Historical role:** Consolidates the mature classical workflow and, most
  importantly for ISOSTER, explains how different harmonic conventions map to
  one another.
- **Tier:** **CORE** - an excellent review/notation citation even though it is
  not the primary algorithm source.

#### `rahman2021.pdf`

- **Identity:** Despite its filename, this is N. Rahman & S. F. Shandarin,
  “Measuring Shapes of Galaxy Images I: Ellipticity and Orientation,” *MNRAS*,
  343, 933-948 (**2003**), ADS `2003MNRAS.343..933R`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Uses Minkowski functionals, geometric moments, auxiliary
  ellipses, and contour smoothing as non-parametric shape descriptors.
- **Historical role:** A survey-scalable alternative contour-morphology
  formalism, not part of the Carter-Jedrzejewski intensity-harmonic lineage.
- **Tier:** **OMIT** from the main history; cite only in a broader alternatives
  discussion.

#### `schombert2007_archangel.pdf`

- **Identity:** J. Schombert, “ARCHANGEL Galaxy Photometry System,” arXiv
  e-print (2007), ADS `2007astro.ph..3646S`; later software record
  `2011ascl.soft07011S`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Packages established galaxy-photometry tools into an
  accessible workflow for large nearby galaxies and demonstrates how a
  speed-oriented survey pipeline can bias structural measurements.
- **Historical role:** A bridge from individual programs to user-facing
  surface-photometry systems before AutoProf.
- **Tier:** **SUPPORTING** - cite for pipeline history or reduction-systematics
  motivation.

#### `ciambur2015.pdf`

- **Identity:** B. C. Ciambur, “Beyond Ellipse(s): Accurately Modelling the
  Isophotal Structure of Galaxies with ISOFIT and CMODEL,” *ApJ*, 810, 120
  (2015), ADS `2015ApJ...810..120C`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Replaces polar angle with eccentric anomaly for
  quasi-elliptical contours, improving highly flattened/non-elliptical shape
  recovery and two-dimensional model reconstruction; implements the method in
  `Isofit` and `Cmodel`.
- **Historical role:** The principal geometric extension of the classical
  Carter-Jedrzejewski lineage and the direct ancestor of ISOSTER’s
  eccentric-anomaly/ISOFIT modes.
- **Tier:** **CORE** - indispensable.

#### `mitsuda2017.pdf`

- **Identity:** K. Mitsuda et al., “Isophote Shapes of Early-Type Galaxies in
  Massive Clusters at z ~ 1 and 0,” *ApJ*, 834, 109 (2017),
  ADS `2017ApJ...834..109M`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Develops a carefully cross-checked high-redshift shape
  analysis and compares the fourth-order coefficient for 130 z~1 and 355 local
  cluster early-type galaxies.
- **Historical role:** Demonstrates that isophote-shape analysis remains a
  useful physical diagnostic in heterogeneous survey data.
- **Tier:** **SUPPORTING** - strong modern science example, not method origin.

#### `stone2021_autoprof.pdf`

- **Identity:** C. J. Stone et al., “AutoProf I: An Automated Non-Parametric
  Light Profile Pipeline for Modern Galaxy Surveys,” *MNRAS*, 508, 1870-1887
  (2021), ADS `2021MNRAS.508.1870S`.
- **Evidence:** Good embedded text and ADS abstract.
- **Contribution:** Provides an automated surface-photometry pipeline with
  regularized geometry, robust outer-profile extraction, alternative profile
  products, model reconstruction, and comparisons with Photutils, XVISTA, and
  GALFIT.
- **Historical role:** The most relevant modern survey-era comparator to
  ISOSTER.
- **Tier:** **CORE** - indispensable for positioning and benchmarks.

## Citation conclusions

### Minimum method-lineage set

These five papers carry the irreducible historical and modern argument:

1. **Carter (1978)** - direct astronomical Fourier description of deviations
   from fitted ellipses and the boxy/disky fourth-order sign.
2. **Jedrzejewski (1987)** - canonical iterative ellipse-fitting algorithm.
3. **Bender & Möllenhoff (1987)** - influential physical interpretation and
   application of higher-order deviations.
4. **Ciambur (2015)** - eccentric-anomaly reformulation and accurate
   non-elliptical model reconstruction.
5. **Stone et al. (2021)** - modern automated, low-surface-brightness pipeline
   and the closest current software comparator.

### Expanded historical spine

Add these when the manuscript gives more than a compressed method genealogy:

- **Fraser (1967)** for the scanning-microdensitometer origin.
- **Barbon et al. (1976)** for numerical two-dimensional mapping before the
  canonical iterative algorithm.
- **Milvang-Jensen & Jørgensen (1999)** for the mature workflow and conversion
  among harmonic-coefficient conventions. This is especially relevant because
  ISOSTER distinguishes stored raw coefficients from Bender-normalized values.
- **Stiavelli et al. (1991)** in the benchmark/comparison discussion: it is a
  direct historical precedent showing that global geometry can agree while
  inner pixels and fine fourth-order structure remain code-sensitive.
- **Schombert (2007)** for the transition to packaged, user-facing galaxy
  surface-photometry systems.

### Scientific-use citations, chosen only when needed

- **Williams & Schwarzschild (1979):** isophote twists and triaxiality.
- **Michard (1985), Mizuno & Hamajima (1987):** radial ellipticity/PA profiles.
- **Michard & Simien (1988), Simien & Michard (1990):** independent Fourier
  representation and bulge/disk interpretation.
- **Mitsuda et al. (2017):** high-redshift use of the fourth-order coefficient.
- **Lauer (1985 ApJS):** seeing correction and central-profile fidelity.
- **Cawson et al. (1987):** early large-sample automation.

### Papers not needed for the main ISOSTER history

- Barbon & Capaccioli (1975, NGC 1023), Young et al. (1979), and Kent (1983)
  are case studies without a central method contribution.
- Kuhl & Giardina (1982) and Morse & Schwartzwald (1998) are computer-vision
  contour papers, not direct astronomical precedents.
- Rahman & Shandarin (2003) is a useful alternative morphology formalism but
  not part of the ellipse/intensity-harmonic lineage.

## Important adjacent papers not represented by these PDFs

The folder maps the history well but should not define the bibliography by
itself. At least three adjacent papers found in the ADS review deserve
consideration:

- T. R. Lauer, “Boxy Isophotes, Discs and Dust Lanes in Elliptical Galaxies,”
  *MNRAS*, 216, 429 (1985), ADS `1985MNRAS.216..429L`. This is distinct from
  the supplied Lauer ApJS PDF and is directly relevant to boxy/disky history.
- R. Bender, S. Doebereiner & C. Moellenhoff, “Isophote Shapes of Elliptical
  Galaxies I: The Data,” *A&AS*, 74, 385 (1988),
  ADS `1988A&AS...74..385B`.
- R. Bender et al., “Isophote Shapes of Elliptical Galaxies II: Correlations
  with Global Optical, Radio and X-ray Properties,” *A&A*, 217, 35 (1989),
  ADS `1989A&A...217...35B`.

These strengthen the link between measured fourth-order shape and the
physical dichotomy of early-type galaxies. The final citation choice should be
made after discussing how much historical and physical context belongs in the
introduction versus the algorithm/comparison sections.

## Main historical conclusions

1. The story begins with **instrumentation**, not with an abstract algorithm:
   photographic plates were scanned by microdensitometers to obtain numerical
   density maps and isophote contours.
2. The 1970s introduced **numerical two-dimensional mapping** and systematic
   radial profiles of ellipticity, PA, and centre.
3. **Carter (1978)** introduced the direct astronomical Fourier analysis of
   deviations from fitted ellipses and the operational boxy/disky fourth-order
   sign.
4. The arrival of CCDs made accurate central and low-contrast structure
   practical; **Jedrzejewski (1987)** then codified the iterative
   intensity-harmonic ellipse-fitting algorithm used today.
5. **Bender & Möllenhoff (1987)** and related late-1980s work turned harmonic
   residuals into physical diagnostics of disks, boxiness, dust, and assembly.
6. Early cross-program work already showed an important modern lesson:
   agreement in smooth global geometry does not guarantee agreement in inner
   pixels or small higher-order coefficients.
7. Later developments standardized notation, packaged workflows, corrected
   the angular coordinate for flattened systems, and finally emphasized
   automated robustness for survey-scale low-surface-brightness work.

No introduction outline or prose has been inferred from these conclusions;
they are the evidence base for the next discussion with the author.
