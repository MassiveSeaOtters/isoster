# Introduction outline discussion

Date: 2026-08-11  
Target journal: *The Astrophysical Journal*  
Status: provisional discussion notes; not manuscript prose

## Proposed central argument

One-dimensional isophotal analysis remains a compact, non-parametric way to
measure how galaxy structure changes with radius. ISOSTER modernizes the
classical ellipse-fitting lineage for deep, large, and multi-band imaging by
combining high throughput with explicit robustness controls, flexible harmonic
analysis, uncertainty-aware fitting, and reproducible outputs.

## Recommended four-paragraph structure

### Paragraph 1: historical lineage and present software landscape

Purpose: establish the method and its continuity without turning the
introduction into a catalogue of old papers.

- Mention the photographic-plate and scanning-microdensitometer origin in one
  clause; Fraser (1967) is sufficient if a citation is desired.
- Use Carter (1978) for Fourier deviations from fitted ellipses and the
  operational boxy/disky fourth-order sign.
- Use Jedrzejewski (1987) for the canonical iterative intensity-harmonic
  algorithm and the IRAF `Ellipse` implementation.
- Use Bender & Möllenhoff (1987), or Bender et al. (1988, 1989), for the move
  from coefficients to physical interpretation.
- Use Ciambur (2015) for eccentric-anomaly sampling and the `Isofit`/`Cmodel`
  extension.
- End with the modern code landscape: community-maintained IRAF, the active
  Astropy `photutils.isophote` implementation, and the automated AutoProf
  pipeline.
- Do not include the many single-galaxy historical case studies here.

### Paragraph 2: why one-dimensional isophotal analysis remains necessary

Recommended main ideas, in priority order:

- **It often provides the useful balance between fidelity and efficiency.** A
  single S\'ersic component is fast and interpretable but cannot represent the
  full diversity of galaxy structure; a highly flexible two-dimensional model,
  such as a large multi-Gaussian expansion or many-component decomposition,
  can achieve higher fidelity at substantially greater computational and
  interpretive cost. A one-dimensional sequence of flexible isophotes occupies
  a productive middle ground for many measurements.
- **The radial variation is itself a scientific observable.** Surface
  brightness, ellipticity, position angle, centroid, and higher-order
  deviations trace components and asymmetries such as disks, bars, lenses,
  isophote twists, boxiness/diskiness, and disturbed outskirts.
- **It is complementary to global two-dimensional parametric modeling.** A
  parametric decomposition provides compact component parameters under an
  assumed model family; isophotal analysis makes fewer assumptions about the
  radial form and exposes structure that a small set of analytic components
  can absorb, average over, or leave in the residuals.
- **Deeper imaging makes the balance more important.** Faint stellar haloes and
  intracluster light are poorly described by a rigid inner-galaxy model, but
  adding components makes a two-dimensional fit more expensive, degenerate,
  and sensitive to sky subtraction, scattered light, masks, and PSF wings.
  Robust one-dimensional profiles remain useful in this regime because their
  assumptions and failure modes are comparatively easy to inspect.
- **The method is already active at survey scale.** Representative examples
  should include the HSC measurement of individual massive-galaxy stellar
  haloes to 100 kpc (Huang et al. 2018), the SGA-2020 Legacy Survey atlas
  (Moustakas et al. 2023), and a recent stellar-halo or BCG+ICL application
  such as Kluge et al. (2020). These demonstrate continuing scientific use
  rather than merely promising future relevance.

The introduction should present isophotal and parametric methods as
complementary, not as competitors. The strongest transition is that survey
data make the classical measurement more valuable while also exposing the
limitations of its traditional implementations.

### Paragraph 3: limitations motivating ISOSTER

The cleanest framing is not that existing packages are inadequate, but that
they occupy different parts of a speed--flexibility--robustness trade space and
no single implementation integrates all requirements needed by this project.

Recommended main issues:

- **Throughput is the leading problem:** the iterative method repeatedly
  samples the image and radial gradient at many radii. Python implementations
  with fine-grained sampling
  loops become costly for large samples, configuration sweeps, and repeated
  multi-band measurements. ISOSTER's benchmark results show that this is an
  implementation bottleneck rather than an unavoidable cost of the method.
- **Per-pixel uncertainty information is not used by the classical fit:** the
  traditional ordinary-least-squares harmonic solution gives all valid samples
  equal weight and cannot use a supplied variance or RMS map. This discards
  information already produced by modern imaging pipelines and weakens both
  coefficient estimation and uncertainty propagation.
- **Outer-profile stability remains a practical problem:** the geometry
  corrections divide by the local radial intensity gradient. At low surface
  brightness the gradient becomes
  uncertain, so noise, imperfect masks, nearby sources, and image artifacts
  can cause large geometry steps, drift, premature termination, or sensitivity
  to starting conditions.
- **Fragmented scientific capabilities:** eccentric-anomaly sampling,
  simultaneous higher-order harmonics, variance-aware fitting, low-surface-
  brightness controls, model reconstruction, and multi-band workflows exist in
  different combinations across the available tools.
- **Multi-band information is incompletely used:** independent fits can give
  band-dependent geometry, while forced photometry transfers the geometry of
  one reference band without allowing all bands to inform it.
- **Reproducibility and diagnosis:** large campaigns need validated
  configurations, explicit stop conditions, saved configuration provenance,
  standardized coefficient conventions, and QA products that make failures
  visible rather than silently returning a profile.

For a concise ApJ introduction, throughput and per-pixel variance information
should carry the paragraph. Low-surface-brightness instability, fragmented
capabilities, the IRAF legacy environment, and reproducibility can be grouped
as the practical consequences and remaining integration gap.

### Paragraph 4: aims and scope of ISOSTER

Group the goals rather than listing every configuration option:

- Provide a fast, function-based Python implementation of the established
  Jedrzejewski ellipse-fitting workflow using vectorized image sampling.
- Preserve the classical path while integrating controlled extensions for
  eccentric-anomaly sampling, higher-order harmonics, variance-aware fitting,
  and low-surface-brightness robustness.
- Support reproducible single-band and multi-band workflows, including forced
  photometry, joint multi-band fitting (supported in its default
  configuration; the `simultaneous_*` harmonic modes remain experimental),
  two-dimensional model reconstruction, serialized configuration, and
  diagnostic plots.
- Validate both speed and scientific fidelity against established tools on
  controlled mocks and representative survey imaging.
- Close with a short map of the paper only after the final section order is
  fixed.

Do not place a numerical speedup in the introduction until the publication
benchmark has fixed the hardware, package versions, image sets, and matched
configurations.

## Suggested emphasis for a concise ApJ introduction

- Historical detail: one compact paragraph.
- Primary motivation: the fidelity--efficiency balance, strengthened by deep
  imaging and demonstrated by recent survey and diffuse-light applications.
- Primary technical problems: computational efficiency and failure of the
  classical fit to use per-pixel variance information.
- Low-surface-brightness instability: an important practical consequence of
  deeper imaging and noisy radial gradients, not the sole motivation.
- ISOSTER positioning: an integrated modern implementation, not a claim that
  every individual feature is unprecedented and not a replacement for every
  existing package.

## Decisions for discussion with the author

1. Should the opening lead with scientific observables or with the historical
   lineage? Scientific observables provide the stronger motivation; history
   can follow immediately afterward.
2. How much detail on low-surface-brightness systematics belongs here after
   efficiency and variance-aware fitting are established as the leading code
   limitations?
3. Is joint multi-band fitting a headline contribution of this paper or a
   capability mentioned briefly and developed more fully elsewhere? Note the
   default configuration is supported, so maturity is no longer the argument
   against featuring it; the open question is scope and length.
4. Should the introduction name IRAF `Isofit` separately from `Ellipse`, or
   reserve that distinction for the method section?
5. Should reproducible configuration, output, and QA be presented as a
   scientific requirement in the introduction or left to the software section?

## Evidence checked for this outline

- `docs/publication/literature/historical-isophote-pdf-review.md`
- `docs/publication/literature/introduction-literature-notes.md`
- `README.md`
- `docs/03-algorithm.md`
- `docs/04-architecture.md`
- `docs/technical/1.5-comparison.md`
- `isoster/config.py`, `isoster/driver.py`, `isoster/fitting.py`,
  `isoster/sampling.py`, and `isoster/model.py`
- Development journals covering variance-map fitting, ISOFIT, low-surface-
  brightness controls, robustness sweeps, exhausted benchmarks, and joint
  multi-band fitting.
